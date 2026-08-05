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


@dataclass(frozen=True)
class SingleFitResult:
    model_name: str
    parameters: np.ndarray
    frequency_thz: np.ndarray
    fitted_transmittance: np.ndarray
    tmax: float
    frequency_at_tmax_thz: float
    fwhm_thz: float
    q_factor: float


def fit_single_transmittance(
    transmittance: pd.DataFrame,
    model_name: str,
    frequency_min_thz: float,
    frequency_max_thz: float,
) -> SingleFitResult:
    """Fit one positive resonance in the selected transmittance band.

    The input DataFrame must have the same ``freq`` and ``mag`` columns used
    by ``AnalysisResult.transmittance`` in the main FFT application.
    """

    if model_name not in FIT_MODELS:
        raise ValueError(f"{model_name} is not available for fitting yet.")
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

    model = FIT_MODELS[model_name]
    peak_index = int(np.argmax(values))
    baseline_guess = float(np.percentile(values, 10))
    amplitude_guess = max(float(values[peak_index] - baseline_guess), max(abs(float(values[peak_index])) * 1e-3, 1e-6))
    center_guess = float(frequency[peak_index])
    minimum_width = max(float(np.median(np.diff(np.sort(frequency)))), 1e-6)
    width_guess = max((frequency_max_thz - frequency_min_thz) / 10.0, minimum_width * 2.0)

    lower_bounds = (-np.inf, 0.0, frequency_min_thz, minimum_width / 2.0)
    upper_bounds = (np.inf, np.inf, frequency_max_thz, frequency_max_thz - frequency_min_thz)
    parameters, _ = curve_fit(
        model.function,
        frequency,
        values,
        p0=(baseline_guess, amplitude_guess, center_guess, width_guess),
        bounds=(lower_bounds, upper_bounds),
        maxfev=20_000,
    )

    dense_frequency = np.linspace(frequency_min_thz, frequency_max_thz, 4_001)
    fitted_values = model.function(dense_frequency, *parameters)
    max_index = int(np.argmax(fitted_values))
    fwhm = float(model.width_to_fwhm * parameters[3])
    frequency_at_tmax = float(dense_frequency[max_index])
    return SingleFitResult(
        model_name=model.name,
        parameters=parameters,
        frequency_thz=dense_frequency,
        fitted_transmittance=fitted_values,
        tmax=float(fitted_values[max_index]),
        frequency_at_tmax_thz=frequency_at_tmax,
        fwhm_thz=fwhm,
        q_factor=frequency_at_tmax / fwhm if fwhm > 0 else float("nan"),
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
        self.fit_mode = tk.StringVar(value="single")
        self.status = tk.StringVar(value="Select a resonance model, then run a single fit.")
        self.result_values = {
            "Tmax": tk.StringVar(value="—"),
            "f @ Tmax": tk.StringVar(value="—"),
            "FWHM": tk.StringVar(value="—"),
            "Q": tk.StringVar(value="—"),
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

        self.run_button = ttk.Button(controls, text="Run single fit", command=self._run_fit)
        self.run_button.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(12, 6))
        ttk.Label(controls, textvariable=self.status, wraplength=300).grid(
            row=9, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )

        ttk.Label(controls, text="Fit result", font=("Segoe UI", 10, "bold")).grid(
            row=10, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        for row, (label, value) in enumerate(self.result_values.items(), start=11):
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(controls, textvariable=value).grid(row=row, column=1, sticky="w", pady=2)

        self.figure = Figure(figsize=(10.5, 7.5), dpi=100, constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _selected_model_name(self) -> str | None:
        selection = self.model_list.curselection()
        return self.model_list.get(selection[0]) if selection else None

    def _on_model_changed(self, _event=None) -> None:
        model_name = self._selected_model_name()
        if model_name not in FIT_MODELS:
            self.status.set(f"{model_name} is reserved for a future implementation.")
        else:
            self.status.set(f"{model_name} selected. Run single fit to calculate resonance metrics.")

    def _on_mode_changed(self) -> None:
        if self.fit_mode.get() == "cumulative":
            self.status.set("Cumulative fitting UI is reserved; click-based peak fitting will be added next.")
        else:
            self.status.set("Single mode fits one selected model across the chosen frequency band.")

    def _read_band(self) -> tuple[float, float]:
        try:
            minimum = float(self.fit_min.get())
            maximum = float(self.fit_max.get())
        except ValueError as exc:
            raise ValueError("Fit-band limits must be numbers.") from exc
        if minimum >= maximum:
            raise ValueError("Fit minimum frequency must be smaller than the maximum frequency.")
        return minimum, maximum

    def _draw_data(self, result: SingleFitResult | None = None, band: tuple[float, float] | None = None) -> None:
        self.ax.clear()
        frequency = self.transmittance["freq"].to_numpy(dtype=float)
        values = self.transmittance["mag"].to_numpy(dtype=float)
        valid = np.isfinite(frequency) & np.isfinite(values)
        self.ax.plot(frequency[valid], values[valid], color="0.65", linewidth=1.0, label="Measured transmittance")
        if band is not None:
            self.ax.axvspan(band[0], band[1], color="tab:blue", alpha=0.08, label="Fit band")
        if result is not None:
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
        self.ax.set_title(f"Transmittance fitting: {self.source_name}")
        self.ax.set_xlabel("Frequency [THz]")
        self.ax.set_ylabel("Transmittance")
        self.ax.set_ylim(bottom=0)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="best")
        self.canvas.draw_idle()

    def _run_fit(self) -> None:
        if self.fit_mode.get() != "single":
            messagebox.showinfo("Cumulative fitting", "Cumulative fitting will be added after click-based peak selection is implemented.")
            return
        model_name = self._selected_model_name()
        if model_name not in FIT_MODELS:
            messagebox.showinfo("Model unavailable", f"{model_name} is planned but is not implemented yet.")
            return
        try:
            band = self._read_band()
            result = fit_single_transmittance(self.transmittance, model_name, *band)
        except Exception as exc:
            self.status.set(f"Fit failed: {exc}")
            messagebox.showerror("Fitting failed", str(exc))
            return

        self.result_values["Tmax"].set(f"{result.tmax:.6g}")
        self.result_values["f @ Tmax"].set(f"{result.frequency_at_tmax_thz:.6g} THz")
        self.result_values["FWHM"].set(f"{result.fwhm_thz:.6g} THz")
        self.result_values["Q"].set(f"{result.q_factor:.6g}")
        self.status.set(f"{result.model_name} single fit complete.")
        self._draw_data(result, band)


def open_fitting_window(parent: tk.Misc, transmittance: pd.DataFrame, source_name: str) -> TransmittanceFittingWindow:
    """Open a fitting window for one transmittance result from the main GUI."""

    return TransmittanceFittingWindow(parent, transmittance, source_name)
