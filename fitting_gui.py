"""Standalone transmittance-resonance fitting window for the THz-TDS GUI.

The fitting window receives one already-computed transmittance spectrum from
the main application.  This keeps fitting independent from the FFT pipeline
and provides a small, extensible home for future resonance models.
"""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.optimize import curve_fit


DEFAULT_FIT_MIN_THZ = 0.5
DEFAULT_FIT_MAX_THZ = 2.5


def lorentzian(frequency: np.ndarray, offset: float, amplitude: float, center: float, hwhm: float) -> np.ndarray:
    """Constant-background Lorentzian peak; ``hwhm`` is half the FWHM."""

    return offset + amplitude * (hwhm**2 / ((frequency - center) ** 2 + hwhm**2))


def gaussian(frequency: np.ndarray, offset: float, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """Constant-background Gaussian peak; ``sigma`` is its standard deviation."""

    return offset + amplitude * np.exp(-0.5 * ((frequency - center) / sigma) ** 2)


def lorentzian_component(frequency: np.ndarray, amplitude: float, center: float, hwhm: float) -> np.ndarray:
    """One Lorentzian contribution without a background offset."""

    return amplitude * (hwhm**2 / ((frequency - center) ** 2 + hwhm**2))


def cumulative_lorentzian(frequency: np.ndarray, offset: float, *peak_parameters: float) -> np.ndarray:
    """Constant background plus a sum of Lorentzian peak contributions."""

    if len(peak_parameters) == 0 or len(peak_parameters) % 3 != 0:
        raise ValueError("Cumulative Lorentzian parameters must contain amplitude, center, and HWHM per peak.")
    fitted = np.full_like(frequency, offset, dtype=float)
    for amplitude, center, hwhm in np.asarray(peak_parameters, dtype=float).reshape(-1, 3):
        fitted += lorentzian_component(frequency, amplitude, center, hwhm)
    return fitted


@dataclass(frozen=True)
class FitModel:
    name: str
    function: object
    width_to_fwhm: float
    width_name: str


FIT_MODELS: dict[str, FitModel] = {
    "Lorentzian": FitModel("Lorentzian", lorentzian, 2.0, "HWHM"),
    "Gaussian": FitModel("Gaussian", gaussian, 2.0 * np.sqrt(2.0 * np.log(2.0)), "sigma"),
}
MODEL_OPTIONS = ("Lorentzian", "Gaussian", "Fano (planned)", "Drude-Smith (planned)")
MODEL_DESCRIPTIONS = {
    "Lorentzian": (
        "T(f) = b + A*Gamma^2 / ((f - f0)^2 + Gamma^2)\n"
        "b: baseline, A: peak height, f0: resonance frequency, Gamma: HWHM\n"
        "FWHM = 2*Gamma; Q = (f @ selected extremum) / FWHM"
    ),
    "Gaussian": (
        "T(f) = b + A*exp(-(f - f0)^2 / (2*sigma^2))\n"
        "b: baseline, A: peak height, f0: center, sigma: standard deviation\n"
        "FWHM = 2*sqrt(2*ln(2))*sigma; Q = (f @ selected extremum) / FWHM"
    ),
    "Fano (planned)": "Fano asymmetric resonance model — planned for a future update.",
    "Drude-Smith (planned)": "Drude-Smith carrier-response model — planned for a future update.",
}


@dataclass(frozen=True)
class SingleFitResult:
    model_name: str
    parameters: np.ndarray
    frequency_thz: np.ndarray
    fitted_transmittance: np.ndarray
    tmax: float
    frequency_at_tmax_thz: float
    tmin: float
    frequency_at_tmin_thz: float
    fwhm_thz: float
    q_factor: float
    r_squared: float
    adjusted_r_squared: float


@dataclass(frozen=True)
class CumulativePeakResult:
    index: int
    amplitude: float
    center_thz: float
    width_thz: float
    fwhm_thz: float
    q_factor: float


@dataclass(frozen=True)
class CumulativeFitResult:
    model_name: str
    frequency_thz: np.ndarray
    fitted_transmittance: np.ndarray
    component_transmittances: np.ndarray
    background: float
    peaks: tuple[CumulativePeakResult, ...]
    tmax: float
    frequency_at_tmax_thz: float
    tmin: float
    frequency_at_tmin_thz: float
    r_squared: float
    adjusted_r_squared: float


def _select_fit_data(
    transmittance: pd.DataFrame,
    frequency_min_thz: float,
    frequency_max_thz: float,
) -> tuple[np.ndarray, np.ndarray]:
    if frequency_min_thz >= frequency_max_thz:
        raise ValueError("Fit minimum frequency must be smaller than the maximum frequency.")
    if not {"freq", "mag"}.issubset(transmittance.columns):
        raise ValueError("Transmittance data must contain 'freq' and 'mag' columns.")

    all_frequency = transmittance["freq"].to_numpy(dtype=float)
    all_transmittance = transmittance["mag"].to_numpy(dtype=float)
    valid = (
        np.isfinite(all_frequency)
        & np.isfinite(all_transmittance)
        & (all_frequency >= frequency_min_thz)
        & (all_frequency <= frequency_max_thz)
    )
    frequency = all_frequency[valid]
    values = all_transmittance[valid]
    if len(frequency) < 8:
        raise ValueError("At least eight finite transmittance points are required in the fit band.")
    return frequency, values


def _fit_quality(values: np.ndarray, fitted_values: np.ndarray, parameter_count: int) -> tuple[float, float]:
    residual_sum_squares = float(np.sum((values - fitted_values) ** 2))
    total_sum_squares = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else float("nan")
    adjusted_r_squared = (
        1.0 - (1.0 - r_squared) * (len(values) - 1) / (len(values) - parameter_count - 1)
        if len(values) > parameter_count + 1 and np.isfinite(r_squared)
        else float("nan")
    )
    return r_squared, adjusted_r_squared


def fit_single_transmittance(
    transmittance: pd.DataFrame,
    model_name: str,
    frequency_min_thz: float,
    frequency_max_thz: float,
    extremum: str = "maximum",
) -> SingleFitResult:
    """Fit one peak (maximum) or dip (minimum) in the selected transmittance band.

    The input DataFrame must have the same ``freq`` and ``mag`` columns used
    by ``AnalysisResult.transmittance`` in the main FFT application.
    """

    if extremum not in ("maximum", "minimum"):
        raise ValueError("extremum must be maximum or minimum.")
    sign = 1.0 if extremum == "maximum" else -1.0
    if model_name not in FIT_MODELS:
        raise ValueError(f"{model_name} is not available for fitting yet.")
    frequency, values = _select_fit_data(transmittance, frequency_min_thz, frequency_max_thz)

    model = FIT_MODELS[model_name]
    peak_index = int(np.argmax(sign * values))
    baseline_guess = float(np.percentile(values, 10 if sign > 0 else 90))
    amplitude_guess = sign * max(sign * float(values[peak_index] - baseline_guess), max(abs(float(values[peak_index])) * 1e-3, 1e-6))
    center_guess = float(frequency[peak_index])
    minimum_width = max(float(np.median(np.diff(np.sort(frequency)))), 1e-6)
    width_guess = max((frequency_max_thz - frequency_min_thz) / 10.0, minimum_width * 2.0)

    lower_bounds = (-np.inf, 0.0 if sign > 0 else -np.inf, frequency_min_thz, minimum_width / 2.0)
    upper_bounds = (np.inf, np.inf if sign > 0 else 0.0, frequency_max_thz, frequency_max_thz - frequency_min_thz)
    parameters, _ = curve_fit(
        model.function,
        frequency,
        values,
        p0=(baseline_guess, amplitude_guess, center_guess, width_guess),
        bounds=(lower_bounds, upper_bounds),
        maxfev=20_000,
    )

    fitted_at_samples = model.function(frequency, *parameters)
    r_squared, adjusted_r_squared = _fit_quality(values, fitted_at_samples, len(parameters))

    dense_frequency = np.linspace(frequency_min_thz, frequency_max_thz, 4_001)
    fitted_values = model.function(dense_frequency, *parameters)
    max_index = int(np.argmax(fitted_values))
    min_index = int(np.argmin(fitted_values))
    fwhm = float(model.width_to_fwhm * parameters[3])
    frequency_at_tmax = float(dense_frequency[max_index])
    return SingleFitResult(
        model_name=model.name,
        parameters=parameters,
        frequency_thz=dense_frequency,
        fitted_transmittance=fitted_values,
        tmax=float(fitted_values[max_index]),
        frequency_at_tmax_thz=frequency_at_tmax,
        tmin=float(fitted_values[min_index]),
        frequency_at_tmin_thz=float(dense_frequency[min_index]),
        fwhm_thz=fwhm,
        q_factor=float(dense_frequency[max_index if sign > 0 else min_index]) / fwhm if fwhm > 0 else float("nan"),
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
    )


def fit_cumulative_resonances(
    transmittance: pd.DataFrame,
    model_name: str,
    clicked_peaks: list[tuple[float, float]],
    frequency_min_thz: float,
    frequency_max_thz: float,
    extremum: str = "maximum",
) -> CumulativeFitResult:
    """Simultaneously fit selected symmetric resonance components from click seeds."""

    if extremum not in ("maximum", "minimum"):
        raise ValueError("extremum must be maximum or minimum.")
    sign = 1.0 if extremum == "maximum" else -1.0
    if model_name not in FIT_MODELS:
        raise ValueError(f"{model_name} is not available for cumulative fitting yet.")
    if not clicked_peaks:
        raise ValueError("Click at least one peak before running a cumulative fit.")
    if any(not frequency_min_thz <= clicked_frequency <= frequency_max_thz for clicked_frequency, _ in clicked_peaks):
        raise ValueError("Every clicked peak must remain inside the current fit band.")
    frequency, values = _select_fit_data(transmittance, frequency_min_thz, frequency_max_thz)
    model = FIT_MODELS[model_name]
    peak_count = len(clicked_peaks)
    parameter_count = 1 + 3 * peak_count
    if len(frequency) <= parameter_count + 1:
        raise ValueError(
            f"{peak_count} clicked peaks need more than {parameter_count + 1} finite data points in the fit band."
        )

    baseline_guess = float(np.percentile(values, 10 if sign > 0 else 90))
    minimum_width = max(float(np.median(np.diff(np.sort(frequency)))), 1e-6)
    width_guess = max((frequency_max_thz - frequency_min_thz) / max(8.0, 4.0 * peak_count), minimum_width * 2.0)
    initial_parameters: list[float] = [baseline_guess]
    lower_bounds: list[float] = [-np.inf]
    upper_bounds: list[float] = [np.inf]
    amplitude_floor = max(abs(float(np.max(values))) * 1e-3, 1e-6)
    for clicked_frequency, clicked_value in clicked_peaks:
        amplitude_guess = sign * max(sign * (clicked_value - baseline_guess), amplitude_floor)
        initial_parameters.extend([amplitude_guess, clicked_frequency, width_guess])
        lower_bounds.extend([0.0 if sign > 0 else -np.inf, frequency_min_thz, minimum_width / 2.0])
        upper_bounds.extend([np.inf if sign > 0 else 0.0, frequency_max_thz, frequency_max_thz - frequency_min_thz])

    def cumulative_function(frequency_values: np.ndarray, offset: float, *peak_parameters: float) -> np.ndarray:
        fitted = np.full_like(frequency_values, offset, dtype=float)
        for amplitude, center, width in np.asarray(peak_parameters, dtype=float).reshape(-1, 3):
            fitted += model.function(frequency_values, 0.0, amplitude, center, width)
        return fitted

    parameters, _ = curve_fit(
        cumulative_function,
        frequency,
        values,
        p0=initial_parameters,
        bounds=(lower_bounds, upper_bounds),
        maxfev=200_000,
    )
    fitted_at_samples = cumulative_function(frequency, *parameters)
    r_squared, adjusted_r_squared = _fit_quality(values, fitted_at_samples, len(parameters))

    dense_frequency = np.linspace(frequency_min_thz, frequency_max_thz, 4_001)
    fitted_values = cumulative_function(dense_frequency, *parameters)
    component_parameters = parameters[1:].reshape(-1, 3)
    components = np.vstack(
        [model.function(dense_frequency, 0.0, amplitude, center, width)
         for amplitude, center, width in component_parameters]
    )
    peaks = tuple(
        CumulativePeakResult(
            index=index,
            amplitude=float(amplitude),
            center_thz=float(center),
            width_thz=float(width),
            fwhm_thz=float(model.width_to_fwhm * width),
            q_factor=float(center / (model.width_to_fwhm * width)),
        )
        for index, (amplitude, center, width) in enumerate(component_parameters, start=1)
    )
    max_index = int(np.argmax(fitted_values))
    min_index = int(np.argmin(fitted_values))
    return CumulativeFitResult(
        model_name=model.name,
        frequency_thz=dense_frequency,
        fitted_transmittance=fitted_values,
        component_transmittances=components,
        background=float(parameters[0]),
        peaks=peaks,
        tmax=float(fitted_values[max_index]),
        frequency_at_tmax_thz=float(dense_frequency[max_index]),
        tmin=float(fitted_values[min_index]),
        frequency_at_tmin_thz=float(dense_frequency[min_index]),
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
    )


class TransmittanceFittingWindow(tk.Toplevel):
    """Separate fitting UI for one transmittance spectrum."""

    def __init__(self, parent: tk.Misc, transmittance: pd.DataFrame, source_name: str) -> None:
        super().__init__(parent)
        self.title(f"Transmittance fitting — {source_name}")
        self.geometry("1450x900")
        self.minsize(1100, 700)
        self.transmittance = transmittance.copy()
        self.source_name = source_name
        self.fit_min = tk.StringVar(value=f"{DEFAULT_FIT_MIN_THZ:g}")
        self.fit_max = tk.StringVar(value=f"{DEFAULT_FIT_MAX_THZ:g}")
        self.extremum = tk.StringVar(value="maximum")
        self.fit_mode = tk.StringVar(value="single")
        self.status = tk.StringVar(value="Select a resonance model, then run a single fit.")
        self.model_formula = tk.StringVar()
        self.clicked_peak_summary = tk.StringVar(value="No peaks selected.")
        self.clicked_peaks: list[tuple[float, float]] = []
        self.result_values = {
            "Tmin": tk.StringVar(value="\u2014"),
            "f @ Tmin": tk.StringVar(value="\u2014"),
            "Tmax": tk.StringVar(value="—"),
            "f @ Tmax": tk.StringVar(value="—"),
            "FWHM": tk.StringVar(value="—"),
            "Q": tk.StringVar(value="—"),
            "R²": tk.StringVar(value="—"),
            "Adjusted R²": tk.StringVar(value="—"),
        }

        self._build_layout()
        self._draw_data()

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        controls = ttk.Frame(root, padding=(0, 0, 12, 0))
        controls.grid(row=0, column=0, sticky="nsw")
        plot_frame = ttk.Frame(root)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Single-spectrum fitting", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        ttk.Label(controls, text=f"Source: {self.source_name}", wraplength=290).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )

        ttk.Label(controls, text="Fit mode").grid(row=2, column=0, sticky="w")
        mode_frame = ttk.Frame(controls)
        mode_frame.grid(row=2, column=1, sticky="w")
        ttk.Radiobutton(mode_frame, text="Single", value="single", variable=self.fit_mode, command=self._on_mode_changed).pack(side="left")
        ttk.Radiobutton(mode_frame, text="Cumulative", value="cumulative", variable=self.fit_mode, command=self._on_mode_changed).pack(
            side="left", padx=(8, 0)
        )

        direction_frame = ttk.Frame(mode_frame)
        direction_frame.pack(side="bottom", anchor="w")
        for label, value in (("Maximum", "maximum"), ("Minimum", "minimum")):
            ttk.Radiobutton(direction_frame, text=label, value=value, variable=self.extremum,
                            command=self._on_extremum_changed).pack(side="left")

        ttk.Label(controls, text="Fit band [THz]", font=("Segoe UI", 10, "bold")).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(14, 3)
        )
        ttk.Label(controls, text="Minimum").grid(row=4, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.fit_min, width=12).grid(row=4, column=1, sticky="w")
        ttk.Label(controls, text="Maximum").grid(row=5, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(controls, textvariable=self.fit_max, width=12).grid(row=5, column=1, sticky="w", pady=(4, 0))

        ttk.Label(controls, text="Resonance model", font=("Segoe UI", 10, "bold")).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(14, 3)
        )
        model_frame = ttk.Frame(controls)
        model_frame.grid(row=7, column=0, columnspan=2, sticky="ew")
        self.model_list = tk.Listbox(model_frame, height=5, exportselection=False, width=31)
        scrollbar = ttk.Scrollbar(model_frame, orient="vertical", command=self.model_list.yview)
        self.model_list.configure(yscrollcommand=scrollbar.set)
        self.model_list.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        for model_name in MODEL_OPTIONS:
            self.model_list.insert(tk.END, model_name)
        self.model_list.selection_set(0)
        self.model_list.bind("<<ListboxSelect>>", self._on_model_changed)

        ttk.Label(controls, text="Model equation", font=("Segoe UI", 10, "bold")).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(10, 2)
        )
        ttk.Label(controls, textvariable=self.model_formula, wraplength=300, justify="left").grid(
            row=9, column=0, columnspan=2, sticky="w"
        )

        ttk.Label(controls, text="Cumulative peak selection", font=("Segoe UI", 10, "bold")).grid(
            row=10, column=0, columnspan=2, sticky="w", pady=(10, 2)
        )
        ttk.Label(controls, textvariable=self.clicked_peak_summary, wraplength=300).grid(
            row=11, column=0, columnspan=2, sticky="w")
        click_buttons = ttk.Frame(controls)
        click_buttons.grid(row=12, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.undo_peak_button = ttk.Button(click_buttons, text="Undo last click", command=self._undo_last_peak)
        self.undo_peak_button.pack(side="left")
        self.clear_peaks_button = ttk.Button(click_buttons, text="Clear clicks", command=self._clear_clicked_peaks)
        self.clear_peaks_button.pack(side="left", padx=(6, 0))

        self.run_button = ttk.Button(controls, text="Run single fit", command=self._run_fit)
        self.run_button.grid(row=13, column=0, columnspan=2, sticky="ew", pady=(12, 6))
        ttk.Label(controls, textvariable=self.status, wraplength=300).grid(
            row=14, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )

        ttk.Label(controls, text="Fit result", font=("Segoe UI", 10, "bold")).grid(
            row=15, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        for row, (label, value) in enumerate(self.result_values.items(), start=16):
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(controls, textvariable=value).grid(row=row, column=1, sticky="w", pady=2)

        peak_table_row = 16 + len(self.result_values)
        ttk.Label(controls, text="Cumulative peak parameters", font=("Segoe UI", 10, "bold")).grid(
            row=peak_table_row, column=0, columnspan=2, sticky="w", pady=(10, 3)
        )
        peak_table_frame = ttk.Frame(controls)
        peak_table_frame.grid(row=peak_table_row + 1, column=0, columnspan=2, sticky="ew")
        self.peak_table = ttk.Treeview(
            peak_table_frame,
            columns=("peak", "f0", "fwhm", "q"),
            show="headings",
            height=6,
        )
        for column, heading, width in (("peak", "#", 34), ("f0", "f0 [THz]", 76), ("fwhm", "FWHM [THz]", 94), ("q", "Q", 62)):
            self.peak_table.heading(column, text=heading)
            self.peak_table.column(column, width=width, anchor="center", stretch=False)
        peak_scrollbar = ttk.Scrollbar(peak_table_frame, orient="vertical", command=self.peak_table.yview)
        self.peak_table.configure(yscrollcommand=peak_scrollbar.set)
        self.peak_table.grid(row=0, column=0, sticky="ew")
        peak_scrollbar.grid(row=0, column=1, sticky="ns")

        self.figure = Figure(figsize=(10.5, 7.5), dpi=100, constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self._update_model_formula()
        self._update_cumulative_controls()

    def _selected_model_name(self) -> str | None:
        selection = self.model_list.curselection()
        return self.model_list.get(selection[0]) if selection else None

    def _on_model_changed(self, _event=None) -> None:
        model_name = self._selected_model_name()
        self._update_model_formula()
        if model_name not in FIT_MODELS:
            self.status.set(f"{model_name} is reserved for a future implementation.")
        elif self.fit_mode.get() == "cumulative":
            self.status.set(f"Cumulative {model_name}: click peaks on the graph, then run the fit.")
        else:
            self.status.set(f"{model_name} selected. Run single fit to calculate resonance metrics.")

    def _update_model_formula(self) -> None:
        if self.fit_mode.get() == "cumulative":
            model_name = self._selected_model_name()
            model = FIT_MODELS.get(model_name)
            if model is None:
                self.model_formula.set(f"{model_name} is not available for cumulative fitting yet.")
                return
            self.model_formula.set(
                f"T(f) = b + sum_j {model.name}_j(f)\n"
                f"Each graph click seeds one {model.name} center f0_j.\n"
                "All clicked peaks and the shared background are optimized together."
            )
            return
        model_name = self._selected_model_name()
        self.model_formula.set(MODEL_DESCRIPTIONS.get(model_name, "Select a model to see its fitting equation."))

    def _on_extremum_changed(self) -> None:
        self.clicked_peaks.clear()
        self._on_mode_changed()
        self.status.set(f"{self.extremum.get().title()} selected. Click dips for cumulative minimum fitting.")

    def _on_mode_changed(self) -> None:
        if self.fit_mode.get() == "cumulative":
            self.status.set("Cumulative mode: select Lorentzian or Gaussian, then click peaks on the graph.")
        else:
            self.status.set("Single mode fits one selected model across the chosen frequency band.")
        self._reset_fit_results()
        self._update_model_formula()
        self._update_cumulative_controls()
        self._draw_data(band=self._read_band_if_valid())

    def _update_cumulative_controls(self) -> None:
        cumulative = self.fit_mode.get() == "cumulative"
        self.run_button.configure(text="Run cumulative fit" if cumulative else "Run single fit")
        self.undo_peak_button.configure(state="normal" if cumulative and self.clicked_peaks else "disabled")
        self.clear_peaks_button.configure(state="normal" if cumulative and self.clicked_peaks else "disabled")
        if cumulative:
            self.clicked_peak_summary.set(
                f"{len(self.clicked_peaks)} peak(s) selected. Left-click a peak/dip inside the fit band."
            )
        else:
            self.clicked_peak_summary.set("Switch to Cumulative mode to select peaks on the graph.")

    def _read_band_if_valid(self) -> tuple[float, float] | None:
        try:
            return self._read_band()
        except ValueError:
            return None

    def _on_plot_click(self, event) -> None:
        if self.fit_mode.get() != "cumulative" or event.button != 1 or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        band = self._read_band_if_valid()
        if band is None:
            self.status.set("Enter a valid fit band before selecting cumulative peaks.")
            return
        if not band[0] <= event.xdata <= band[1]:
            self.status.set("Click inside the highlighted fit band.")
            return

        minimum_separation = max((band[1] - band[0]) / 1_000.0, 1e-6)
        if any(abs(event.xdata - peak_frequency) < minimum_separation for peak_frequency, _ in self.clicked_peaks):
            self.status.set("That peak position is already selected. Click a distinct peak.")
            return
        self.clicked_peaks.append((float(event.xdata), float(event.ydata)))
        self.clicked_peaks.sort(key=lambda peak: peak[0])
        self._reset_fit_results()
        self._update_cumulative_controls()
        self.status.set(f"Peak {len(self.clicked_peaks)} added. Add more peaks or run the cumulative fit.")
        self._draw_data(band=band)

    def _undo_last_peak(self) -> None:
        if not self.clicked_peaks:
            return
        self.clicked_peaks.pop()
        self._reset_fit_results()
        self._update_cumulative_controls()
        self.status.set("Removed the last selected peak.")
        self._draw_data(band=self._read_band_if_valid())

    def _clear_clicked_peaks(self) -> None:
        self.clicked_peaks.clear()
        self._reset_fit_results()
        self._update_cumulative_controls()
        self.status.set("Cleared all selected peaks.")
        self._draw_data(band=self._read_band_if_valid())

    def _reset_fit_results(self) -> None:
        for value in self.result_values.values():
            value.set("—")
        for item in self.peak_table.get_children():
            self.peak_table.delete(item)

    def _show_cumulative_peak_results(self, result: CumulativeFitResult) -> None:
        for item in self.peak_table.get_children():
            self.peak_table.delete(item)
        for peak in result.peaks:
            self.peak_table.insert(
                "",
                tk.END,
                values=(peak.index, f"{peak.center_thz:.6g}", f"{peak.fwhm_thz:.6g}", f"{peak.q_factor:.6g}"),
            )

    def _read_band(self) -> tuple[float, float]:
        try:
            minimum = float(self.fit_min.get())
            maximum = float(self.fit_max.get())
        except ValueError as exc:
            raise ValueError("Fit-band limits must be numbers.") from exc
        if minimum >= maximum:
            raise ValueError("Fit minimum frequency must be smaller than the maximum frequency.")
        return minimum, maximum

    def _draw_data(
        self,
        result: SingleFitResult | CumulativeFitResult | None = None,
        band: tuple[float, float] | None = None,
    ) -> None:
        self.ax.clear()
        frequency = self.transmittance["freq"].to_numpy(dtype=float)
        values = self.transmittance["mag"].to_numpy(dtype=float)
        valid = np.isfinite(frequency) & np.isfinite(values)
        self.ax.plot(frequency[valid], values[valid], color="0.65", linewidth=1.0, label="Measured transmittance")
        if band is not None:
            self.ax.axvspan(band[0], band[1], color="tab:blue", alpha=0.08, label="Fit band")
        if self.fit_mode.get() == "cumulative" and self.clicked_peaks:
            clicked_frequency, clicked_value = np.asarray(self.clicked_peaks, dtype=float).T
            self.ax.scatter(
                clicked_frequency,
                clicked_value,
                color="black",
                marker="x",
                s=58,
                linewidths=1.8,
                zorder=5,
                label="Clicked peak seed",
            )
            for index, (peak_frequency, peak_value) in enumerate(self.clicked_peaks, start=1):
                self.ax.annotate(str(index), (peak_frequency, peak_value), xytext=(4, 5), textcoords="offset points", fontsize=8)
        if isinstance(result, SingleFitResult):
            self.ax.plot(
                result.frequency_thz,
                result.fitted_transmittance,
                color="tab:red",
                linewidth=2.2,
                label=f"{result.model_name} fit",
            )
            self.ax.scatter(
                [result.frequency_at_tmax_thz], [result.tmax], color="tab:red", zorder=4, label="Tmax (fit)"
            )
        elif isinstance(result, CumulativeFitResult):
            colors = ["tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
            for peak, component in zip(result.peaks, result.component_transmittances):
                self.ax.plot(
                    result.frequency_thz,
                    result.background + component,
                    color=colors[(peak.index - 1) % len(colors)],
                    linestyle="--",
                    linewidth=1.25,
                    alpha=0.85,
                    label=f"Peak {peak.index} {result.model_name}",
                )
            self.ax.plot(
                result.frequency_thz,
                result.fitted_transmittance,
                color="tab:red",
                linewidth=2.35,
                label=f"Cumulative {result.model_name} fit",
            )
            self.ax.scatter(
                [result.frequency_at_tmax_thz], [result.tmax], color="tab:red", zorder=6, label="Tmax (fit)"
            )
        if result is not None:
            self.ax.scatter([result.frequency_at_tmin_thz], [result.tmin],
                            color="tab:blue", marker="v", zorder=6, label="Tmin (fit)")
        self.ax.set_title(f"Transmittance fitting: {self.source_name}")
        self.ax.set_xlabel("Frequency [THz]")
        self.ax.set_ylabel("Transmittance")
        self.ax.set_ylim(bottom=0)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="best")
        self.canvas.draw_idle()

    def _run_fit(self) -> None:
        try:
            band = self._read_band()
            if self.fit_mode.get() == "single":
                model_name = self._selected_model_name()
                if model_name not in FIT_MODELS:
                    messagebox.showinfo("Model unavailable", f"{model_name} is planned but is not implemented yet.")
                    return
                result = fit_single_transmittance(self.transmittance, model_name, *band, extremum=self.extremum.get())
                self._show_single_result(result)
                self.status.set(f"{result.model_name} single fit complete.")
            else:
                model_name = self._selected_model_name()
                if model_name not in FIT_MODELS:
                    messagebox.showinfo("Model unavailable", f"{model_name} is planned but is not implemented yet.")
                    return
                result = fit_cumulative_resonances(self.transmittance, model_name, self.clicked_peaks, *band, extremum=self.extremum.get())
                self._show_cumulative_result(result)
                self.status.set(f"Cumulative {result.model_name} fit complete: {len(result.peaks)} peak(s).")
        except Exception as exc:
            self.status.set(f"Fit failed: {exc}")
            messagebox.showerror("Fitting failed", str(exc))
            return
        self._draw_data(result, band)

    def _show_single_result(self, result: SingleFitResult) -> None:
        self._reset_fit_results()
        self.result_values["Tmin"].set(f"{result.tmin:.6g}")
        self.result_values["f @ Tmin"].set(f"{result.frequency_at_tmin_thz:.6g} THz")
        self.result_values["Tmax"].set(f"{result.tmax:.6g}")
        self.result_values["f @ Tmax"].set(f"{result.frequency_at_tmax_thz:.6g} THz")
        self.result_values["FWHM"].set(f"{result.fwhm_thz:.6g} THz")
        self.result_values["Q"].set(f"{result.q_factor:.6g}")
        self._set_quality_values(result.r_squared, result.adjusted_r_squared)

    def _show_cumulative_result(self, result: CumulativeFitResult) -> None:
        self._reset_fit_results()
        self.result_values["Tmin"].set(f"{result.tmin:.6g}")
        self.result_values["f @ Tmin"].set(f"{result.frequency_at_tmin_thz:.6g} THz")
        self.result_values["Tmax"].set(f"{result.tmax:.6g}")
        self.result_values["f @ Tmax"].set(f"{result.frequency_at_tmax_thz:.6g} THz")
        self.result_values["FWHM"].set("See peak table")
        self.result_values["Q"].set("See peak table")
        self._set_quality_values(result.r_squared, result.adjusted_r_squared)
        self._show_cumulative_peak_results(result)

    def _set_quality_values(self, r_squared: float, adjusted_r_squared: float) -> None:
        quality_keys = [key for key in self.result_values if "R" in key]
        self.result_values[quality_keys[0]].set(self._format_percent(r_squared))
        self.result_values[quality_keys[1]].set(self._format_percent(adjusted_r_squared))

    @staticmethod
    def _format_percent(value: float) -> str:
        return f"{100.0 * value:.3f}%" if np.isfinite(value) else "—"


def open_fitting_window(parent: tk.Misc, transmittance: pd.DataFrame, source_name: str) -> TransmittanceFittingWindow:
    """Open a fitting window for one transmittance result from the main GUI."""

    return TransmittanceFittingWindow(parent, transmittance, source_name)
