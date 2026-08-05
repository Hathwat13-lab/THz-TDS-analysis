from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from functions import AsymmetricTDSAnalyzer, find_transmittance_maximum


DEFAULT_REFERENCE = ""
DEFAULT_SAMPLE_FOLDER = ""
TMAX_FREQUENCY_MIN_THZ = 0.5
TMAX_FREQUENCY_MAX_THZ = 2.5


class FFTPlatformGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FFT Platform GUI — Tabbed Monitor")
        self.geometry("1800x1100")
        self.minsize(1500, 900)

        self.analyzer = AsymmetricTDSAnalyzer()

        self.sample_folder = tk.StringVar(value=DEFAULT_SAMPLE_FOLDER)
        self.reference_path = tk.StringVar(value=DEFAULT_REFERENCE)
        self.lower_bound = tk.StringVar(value="1")
        self.upper_bound = tk.StringVar(value="9")
        self.alpha_1 = tk.StringVar(value="0.0")
        self.alpha_2 = tk.StringVar(value="0.35")
        self.pad_mode = tk.StringVar(value="factor")
        self.pad_value = tk.StringVar(value="4")
        self.crop_min = tk.StringVar(value="0.2")
        self.crop_max = tk.StringVar(value="3.0")
        # The analyzer uses centimetres internally; expose the more convenient
        # micrometre unit in the GUI.
        self.thickness_um = tk.StringVar(value="460")
        self.thickness_mode = tk.StringVar(value="thick")
        self.echo_guideline_enabled = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Ready")
        self.active_sample_name = tk.StringVar(value="(none)")

        self.results: list[tuple[Path, object]] = []
        self.sample_checks: list[tuple[Path, tk.BooleanVar]] = []
        self.common_length: int | None = None
        self._active_result_index: int | None = None
        self.echo_guideline_cache: dict[Path, dict[str, object]] = {}
        self.tmax_metrics: list[tuple[Path, float, float]] = []

        self._build_layout()
        self._build_default_views()
        self._bind_events()

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)

        self.control_frame = ttk.Frame(root, padding=(0, 0, 10, 0))
        self.control_frame.grid(row=0, column=0, sticky="nsw")
        self.control_frame.columnconfigure(1, weight=1)

        self.plot_frame = ttk.Frame(root)
        self.plot_frame.grid(row=0, column=1, sticky="nsew")
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        self._build_controls()
        self._build_tabs()

    def _build_controls(self) -> None:
        row = 0

        ttk.Label(self.control_frame, text="Input & Status", font=("Segoe UI", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        row += 1

        row = self._path_row(row, "Reference", self.reference_path)
        row = self._folder_row(row)

        ttk.Label(self.control_frame, text="Samples to analyze", font=("Segoe UI", 10, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(8, 2)
        )
        row += 1

        sample_list_frame = ttk.Frame(self.control_frame)
        sample_list_frame.grid(row=row, column=0, columnspan=3, sticky="ew")
        sample_list_frame.columnconfigure(0, weight=1)
        self.sample_canvas = tk.Canvas(sample_list_frame, height=165, highlightthickness=0)
        sample_scrollbar = ttk.Scrollbar(sample_list_frame, orient="vertical", command=self.sample_canvas.yview)
        self.sample_checks_frame = ttk.Frame(self.sample_canvas)
        self.sample_checks_frame.bind(
            "<Configure>",
            lambda _event: self.sample_canvas.configure(scrollregion=self.sample_canvas.bbox("all")),
        )
        self.sample_canvas.create_window((0, 0), window=self.sample_checks_frame, anchor="nw")
        self.sample_canvas.configure(yscrollcommand=sample_scrollbar.set)
        self.sample_canvas.grid(row=0, column=0, sticky="ew")
        sample_scrollbar.grid(row=0, column=1, sticky="ns")
        row += 1

        selection_buttons = ttk.Frame(self.control_frame)
        selection_buttons.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(3, 2))
        ttk.Button(selection_buttons, text="Select all", command=lambda: self._set_all_samples(True)).pack(side="left")
        ttk.Button(selection_buttons, text="Clear all", command=lambda: self._set_all_samples(False)).pack(side="left", padx=(6, 0))
        row += 1

        ttk.Separator(self.control_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=10
        )
        row += 1

        row = self._entry_row(row, "l (ps)", self.lower_bound)
        row = self._entry_row(row, "u (ps)", self.upper_bound)
        row = self._entry_row(row, "alpha1 (-)", self.alpha_1)
        row = self._entry_row(row, "alpha2 (-)", self.alpha_2)
        row = self._entry_row(row, "Npad (× N)", self.pad_value)

        ttk.Label(self.control_frame, text="Pad mode").grid(row=row, column=0, sticky="w", pady=4)
        pad_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.pad_mode,
            values=("factor", "length"),
            width=10,
            state="readonly",
        )
        pad_combo.grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        row = self._entry_row(row, "crop_min (THz)", self.crop_min)
        row = self._entry_row(row, "crop_max (THz)", self.crop_max)
        row = self._entry_row(row, "thickness (µm)", self.thickness_um)

        ttk.Label(self.control_frame, text="Thickness mode").grid(row=row, column=0, sticky="w", pady=4)
        thickness_mode_frame = ttk.Frame(self.control_frame)
        thickness_mode_frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Radiobutton(thickness_mode_frame, text="thin (film)", value="thin", variable=self.thickness_mode).pack(side="left")
        ttk.Radiobutton(thickness_mode_frame, text="thick (pellet)", value="thick", variable=self.thickness_mode).pack(side="left", padx=(10, 0))
        row += 1

        echo_frame = ttk.Frame(self.control_frame)
        echo_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(2, 4))
        ttk.Checkbutton(
            echo_frame,
            text="Echo guideline",
            variable=self.echo_guideline_enabled,
            command=self._render_active_result,
        ).pack(side="left")
        ttk.Button(echo_frame, text="Calc echo guideline", command=self._compute_echo_guideline_for_active_sample).pack(
            side="left", padx=(10, 0)
        )
        row += 1

        ttk.Button(self.control_frame, text="Run Analysis", command=self.run_analysis).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(10, 6)
        )
        row += 1

        ttk.Button(self.control_frame, text="Open fitting window", command=self._open_fitting_window).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6)
        )
        row += 1

        ttk.Label(self.control_frame, text="Status").grid(row=row, column=0, sticky="w")
        ttk.Label(self.control_frame, textvariable=self.status, wraplength=280).grid(
            row=row, column=1, columnspan=2, sticky="w"
        )
        row += 1

        ttk.Label(self.control_frame, text="Active sample").grid(row=row, column=0, sticky="w")
        self.active_sample_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.active_sample_name,
            values=(),
            state="readonly",
            width=24,
        )
        self.active_sample_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        row += 1

        self.summary = tk.Text(self.control_frame, width=38, height=24, wrap="word")
        self.summary.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
        self.summary.insert("1.0", "Parameters and results will appear here.\n")
        self.summary.configure(state="disabled")

        for child in self.control_frame.winfo_children():
            if isinstance(child, ttk.Entry) or isinstance(child, ttk.Combobox):
                child.configure(width=12)

    def _path_row(self, row: int, label: str, variable: tk.StringVar) -> int:
        ttk.Label(self.control_frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(self.control_frame, textvariable=variable, width=35)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        browse_command = self._browse_reference_file if variable is self.reference_path else lambda var=variable: self._browse_file(var)
        ttk.Button(self.control_frame, text="Browse", command=browse_command).grid(
            row=row, column=2, sticky="ew", padx=(6, 0), pady=4
        )
        return row + 1

    def _folder_row(self, row: int) -> int:
        ttk.Label(self.control_frame, text="Sample folder").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(self.control_frame, textvariable=self.sample_folder, width=35).grid(
            row=row, column=1, sticky="ew", pady=4
        )
        ttk.Button(self.control_frame, text="Browse", command=self._browse_sample_folder).grid(
            row=row, column=2, sticky="ew", padx=(6, 0), pady=4
        )
        return row + 1

    def _entry_row(self, row: int, label: str, variable: tk.StringVar) -> int:
        ttk.Label(self.control_frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(self.control_frame, textvariable=variable, width=18).grid(row=row, column=1, sticky="ew", pady=4)
        return row + 1

    def _browse_file(self, variable: tk.StringVar) -> bool:
        path = filedialog.askopenfilename(
            title="Select THz-TDS text file",
            filetypes=[("THz data files", "*.txt *.dat"), ("Text files", "*.txt"), ("DAT files", "*.dat"), ("All files", "*.*")],
        )
        if path:
            variable.set(path)
            return True
        return False

    def _browse_reference_file(self) -> None:
        if self._browse_file(self.reference_path) and self.sample_folder.get().strip():
            self._load_sample_folder(self.sample_folder.get().strip())

    def _browse_sample_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select the folder containing sample text files")
        if folder:
            self.sample_folder.set(folder)
            self._load_sample_folder(folder)

    def _load_sample_folder(self, folder: str) -> None:
        for child in self.sample_checks_frame.winfo_children():
            child.destroy()
        self.sample_checks.clear()

        try:
            files = sorted(
                [*Path(folder).glob("*.txt"), *Path(folder).glob("*.dat")],
                key=lambda path: path.name.lower(),
            )
        except OSError as exc:
            self.status.set(f"Could not read sample folder: {exc}")
            return

        reference = Path(self.reference_path.get().strip()).resolve()
        files = [path for path in files if path.resolve() != reference]
        for path in files:
            selected = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.sample_checks_frame, text=path.name, variable=selected).pack(anchor="w")
            self.sample_checks.append((path, selected))

        self.sample_canvas.yview_moveto(0)
        if files:
            self.status.set(f"Loaded {len(files)} sample file(s); all selected.")
        else:
            self.status.set("No .txt or .dat sample files found in the selected folder.")

    def _set_all_samples(self, selected: bool) -> None:
        for _, variable in self.sample_checks:
            variable.set(selected)

    def _build_tabs(self) -> None:
        self.view_notebook = ttk.Notebook(self.plot_frame)
        self.view_notebook.grid(row=0, column=0, sticky="nsew")

        self.monitor_tab = ttk.Frame(self.view_notebook)
        self.optical_tab = ttk.Frame(self.view_notebook)
        self.tmax_tab = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.monitor_tab, text="T Monitor")
        self.view_notebook.add(self.optical_tab, text="n / k / alpha / phase")
        self.view_notebook.add(self.tmax_tab, text="Tmax trend")

        self.monitor_tab.rowconfigure(0, weight=1)
        self.monitor_tab.columnconfigure(0, weight=1)
        self.optical_tab.rowconfigure(0, weight=1)
        self.optical_tab.columnconfigure(0, weight=1)
        self.tmax_tab.rowconfigure(0, weight=1)
        self.tmax_tab.columnconfigure(0, weight=1)

        self.monitor_figure = Figure(figsize=(11.5, 9.0), dpi=100, constrained_layout=True)
        self.monitor_figure.set_constrained_layout_pads(w_pad=0.10, h_pad=0.12, wspace=0.10, hspace=0.12)
        self.monitor_gs = self.monitor_figure.add_gridspec(2, 2, height_ratios=(1.05, 0.85))
        self.ax_ref_td = self.monitor_figure.add_subplot(self.monitor_gs[0, 0])
        self.ax_sample_td = self.monitor_figure.add_subplot(self.monitor_gs[0, 1])
        self.ax_t = self.monitor_figure.add_subplot(self.monitor_gs[1, :])

        self.optical_figure = Figure(figsize=(11.5, 9.0), dpi=100, constrained_layout=True)
        self.optical_figure.set_constrained_layout_pads(w_pad=0.10, h_pad=0.12, wspace=0.10, hspace=0.12)
        self.optical_gs = self.optical_figure.add_gridspec(2, 2)
        self.ax_n = self.optical_figure.add_subplot(self.optical_gs[0, 0])
        self.ax_k = self.optical_figure.add_subplot(self.optical_gs[0, 1])
        self.ax_alpha = self.optical_figure.add_subplot(self.optical_gs[1, 0])
        self.ax_phase = self.optical_figure.add_subplot(self.optical_gs[1, 1])

        self.tmax_figure = Figure(figsize=(11.5, 9.0), dpi=100, constrained_layout=True)
        self.tmax_figure.set_constrained_layout_pads(w_pad=0.10, h_pad=0.12, wspace=0.10, hspace=0.16)
        self.tmax_gs = self.tmax_figure.add_gridspec(2, 1)
        self.ax_tmax_frequency = self.tmax_figure.add_subplot(self.tmax_gs[0, 0])
        self.ax_tmax_value = self.tmax_figure.add_subplot(self.tmax_gs[1, 0], sharex=self.ax_tmax_frequency)

        self.monitor_canvas = FigureCanvasTkAgg(self.monitor_figure, master=self.monitor_tab)
        self.monitor_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.optical_canvas = FigureCanvasTkAgg(self.optical_figure, master=self.optical_tab)
        self.optical_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.tmax_canvas = FigureCanvasTkAgg(self.tmax_figure, master=self.tmax_tab)
        self.tmax_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_default_views(self) -> None:
        self._style_monitor_axes()
        self._style_optical_axes()
        self._style_tmax_axes()
        self.monitor_canvas.draw_idle()
        self.optical_canvas.draw_idle()
        self.tmax_canvas.draw_idle()

    def _bind_events(self) -> None:
        self.active_sample_combo.bind("<<ComboboxSelected>>", self._on_active_sample_selected)
        self.view_notebook.bind("<<NotebookTabChanged>>", lambda _event: self._render_active_result())

    def _on_active_sample_selected(self, _event) -> None:
        selected_name = self.active_sample_name.get().strip()
        for index, (sample_path, _result) in enumerate(self.results):
            if sample_path.name == selected_name:
                self._active_result_index = index
                break
        self._render_active_result()

    def _compute_echo_guideline_for_active_sample(self) -> None:
        if self.thickness_mode.get().strip().lower() != "thick":
            messagebox.showinfo("Echo guideline", "Echo guideline is only used in thick mode.")
            return

        index, entry = self._selected_result()
        if entry is None:
            messagebox.showinfo("Echo guideline", "Run the analysis first and select a sample.")
            return

        sample_path, result = entry
        thickness_um = self._read_float(self.thickness_um, "thickness (µm)")
        guideline = self.analyzer.compute_echo_guideline(result, thickness_um=thickness_um)
        self.echo_guideline_cache[sample_path] = guideline
        self.status.set(
            f"Echo guideline ready for {sample_path.name}: n={guideline['n_mean']:.4f}, dt={guideline['echo_dt_ps']:.4f} ps"
        )
        self._render_active_result()

    def _open_fitting_window(self) -> None:
        """Launch the independent fitting UI for the currently active spectrum."""

        _index, entry = self._selected_result()
        if entry is None:
            messagebox.showinfo("Transmittance fitting", "Run the analysis first and select an active sample.")
            return

        sample_path, result = entry
        try:
            from fitting_gui import open_fitting_window

            open_fitting_window(self, result.transmittance, sample_path.name)
        except Exception as exc:
            self.status.set(f"Could not open fitting window: {exc}")
            messagebox.showerror("Transmittance fitting", str(exc))

    def _style_monitor_axes(self) -> None:
        for ax in [self.ax_ref_td, self.ax_sample_td, self.ax_t]:
            ax.clear()
            ax.grid(True, alpha=0.3)

        self.ax_ref_td.set_title("Reference E(TD)")
        self.ax_ref_td.set_xlabel("Time [ps]")
        self.ax_ref_td.set_ylabel("THz Amplitude")

        self.ax_sample_td.set_title("Sample E(TD)")
        self.ax_sample_td.set_xlabel("Time [ps]")
        self.ax_sample_td.set_ylabel("THz Amplitude")

        self.ax_t.set_title("Transmittance")
        self.ax_t.set_xlabel("Frequency [THz]")
        self.ax_t.set_ylabel("Transmittance")

    def _style_optical_axes(self) -> None:
        for ax in [self.ax_n, self.ax_k, self.ax_alpha, self.ax_phase]:
            ax.clear()
            ax.grid(True, alpha=0.3)

        self.ax_n.set_title("n")
        self.ax_n.set_xlabel("Frequency [THz]")
        self.ax_n.set_ylabel("Refractive index")

        self.ax_k.set_title("k")
        self.ax_k.set_xlabel("Frequency [THz]")
        self.ax_k.set_ylabel("Extinction coefficient")

        self.ax_alpha.set_title("alpha")
        self.ax_alpha.set_xlabel("Frequency [THz]")
        self.ax_alpha.set_ylabel("Absorption [cm^-1]")

        self.ax_phase.set_title("Phase")
        self.ax_phase.set_xlabel("Frequency [THz]")
        self.ax_phase.set_ylabel("Phase [rad]")

    def _style_tmax_axes(self) -> None:
        for ax in [self.ax_tmax_frequency, self.ax_tmax_value]:
            ax.clear()
            ax.grid(True, axis="y", alpha=0.3)

        self.ax_tmax_frequency.set_title(
            f"Frequency at Tmax ({TMAX_FREQUENCY_MIN_THZ:g}-{TMAX_FREQUENCY_MAX_THZ:g} THz)")
        self.ax_tmax_frequency.set_ylabel("Frequency [THz]")
        self.ax_tmax_value.set_title(
            f"Tmax ({TMAX_FREQUENCY_MIN_THZ:g}-{TMAX_FREQUENCY_MAX_THZ:g} THz)")
        self.ax_tmax_value.set_xlabel("Sample")
        self.ax_tmax_value.set_ylabel("Transmittance")

    def _clear_views(self) -> None:
        self._style_monitor_axes()
        self._style_optical_axes()
        self._style_tmax_axes()

    def _selected_result(self) -> tuple[int | None, tuple[Path, object] | None]:
        if not self.results:
            return None, None
        if self._active_result_index is None:
            self._active_result_index = 0
        self._active_result_index = max(0, min(self._active_result_index, len(self.results) - 1))
        return self._active_result_index, self.results[self._active_result_index]

    def _set_active_result_options(self) -> None:
        names = [path.name for path, _ in self.results]
        self.active_sample_combo["values"] = names
        if names:
            self._active_result_index = 0
            self.active_sample_name.set(names[0])
        else:
            self._active_result_index = None
            self.active_sample_name.set("(none)")

    def _render_active_result(self) -> None:
        index, entry = self._selected_result()
        self._clear_views()

        if entry is None:
            self.monitor_canvas.draw_idle()
            self.optical_canvas.draw_idle()
            self.tmax_canvas.draw_idle()
            return

        highlight_name = None if index is None else self.results[index][0].name
        sample_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        reference = self.results[0][1].reference
        reference_peak = np.max(np.abs(reference.amplitude))

        self.ax_ref_td.plot(reference.time, reference.amplitude, color="black", alpha=0.65, label="Reference")
        self.ax_ref_td.plot(
            reference.time,
            reference.window * reference_peak,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="Shared window",
        )

        representative_sample = None
        representative_result = None
        representative_peak = None
        for sample_path, result in self.results:
            sample_peak = np.max(np.abs(result.sample.amplitude))
            if representative_peak is None or sample_peak > representative_peak:
                representative_peak = sample_peak
                representative_sample = sample_path
                representative_result = result

        if representative_result is not None and representative_peak is not None:
            representative_sample_data = representative_result.sample
            self.ax_sample_td.plot(
                representative_sample_data.time,
                representative_sample_data.window * representative_peak,
                color="black",
                linestyle=":",
                linewidth=1.2,
                label=f"Shared window ({representative_sample.name})",
            )

        for sample_index, (sample_path, result) in enumerate(self.results):
            sample = result.sample
            color = sample_colors[sample_index % len(sample_colors)]
            is_active = sample_path.name == highlight_name
            line_width = 2.4 if is_active else 1.1
            alpha_value = 0.85 if is_active else 0.35
            phase_alpha = 0.95 if is_active else 0.55

            self.ax_sample_td.plot(
                sample.time,
                sample.amplitude,
                color=color,
                alpha=alpha_value,
                linewidth=line_width,
                label=sample_path.stem,
            )

            self.ax_t.plot(
                result.transmittance["freq"],
                result.transmittance["mag"],
                color=color,
                alpha=alpha_value,
                linewidth=line_width,
                label=sample_path.stem,
            )
            self.ax_n.plot(
                result.refractive_index["freq"],
                result.refractive_index["n"],
                color=color,
                alpha=alpha_value,
                linewidth=line_width,
                label=sample_path.stem,
            )
            self.ax_k.plot(
                result.extinction["freq"],
                result.extinction["k"],
                color=color,
                alpha=alpha_value,
                linewidth=line_width,
                label=sample_path.stem,
            )
            self.ax_alpha.plot(
                result.absorption["freq"],
                result.absorption["alpha"],
                color=color,
                alpha=alpha_value,
                linewidth=line_width,
                label=sample_path.stem,
            )
            self.ax_phase.plot(
                result.sample_spectrum_crop.frequency,
                np.unwrap(result.sample_spectrum_crop.phase),
                color=color,
                alpha=phase_alpha,
                linewidth=line_width,
                label=sample_path.stem,
            )

        self.ax_phase.plot(
            self.results[0][1].reference_spectrum_crop.frequency,
            np.unwrap(self.results[0][1].reference_spectrum_crop.phase),
            color="black",
            alpha=0.7,
            linewidth=1.4,
            label="Reference",
        )

        if self.echo_guideline_enabled.get() and highlight_name is not None:
            active_sample_path, active_result = self.results[index]
            guideline = self.echo_guideline_cache.get(active_sample_path)
            if guideline is not None:
                peak_time = float(guideline["peak_time_ps"])
                peak_amp = float(guideline["peak_amplitude"])
                echo_time = float(guideline["echo_time_ps"])
                echo_amp = float(guideline["echo_amplitude"])
                # Draw a two-layer guideline at a+dt so it remains visible even
                # when multiple sample traces overlap on the same axis.
                self.ax_sample_td.axvline(
                    echo_time,
                    color="white",
                    linewidth=4.0,
                    alpha=0.95,
                    zorder=5,
                )
                self.ax_sample_td.axvline(
                    echo_time,
                    color="red",
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.95,
                    zorder=6,
                )
                self.ax_sample_td.scatter(
                    [peak_time, echo_time],
                    [peak_amp, echo_amp],
                    s=68,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.5,
                    zorder=6,
                    label="Echo guideline",
                )

        self.ax_t.set_ylim(bottom=0)
        self.ax_ref_td.legend(loc="best", fontsize=8)
        self.ax_sample_td.legend(loc="best", fontsize=7)
        self.ax_t.legend(loc="best", fontsize=7)
        self.ax_n.legend(loc="best", fontsize=7)
        self.ax_k.legend(loc="best", fontsize=7)
        self.ax_alpha.legend(loc="best", fontsize=7)
        self.ax_phase.legend(loc="best", fontsize=7)

        self._render_tmax_trend()

        self.monitor_canvas.draw_idle()
        self.optical_canvas.draw_idle()
        self.tmax_canvas.draw_idle()

    def _update_tmax_metrics(self) -> list[str]:
        """Extract one in-band transmittance maximum per analyzed sample."""

        self.tmax_metrics = []
        unavailable = []
        for sample_path, result in self.results:
            try:
                maximum = find_transmittance_maximum(
                    result.transmittance,
                    frequency_min_thz=TMAX_FREQUENCY_MIN_THZ,
                    frequency_max_thz=TMAX_FREQUENCY_MAX_THZ,
                )
                self.tmax_metrics.append((sample_path, maximum.frequency_thz, maximum.transmittance))
            except ValueError as exc:
                unavailable.append(f"{sample_path.name}: {exc}")
        self.tmax_metrics.sort(key=lambda metric: metric[1])
        return unavailable

    def _render_tmax_trend(self) -> None:
        if not self.tmax_metrics:
            message = (
                f"No finite transmittance data is available in "
                f"{TMAX_FREQUENCY_MIN_THZ:g}-{TMAX_FREQUENCY_MAX_THZ:g} THz.\n"
                "Adjust the FFT crop range and run the analysis again."
            )
            for ax in [self.ax_tmax_frequency, self.ax_tmax_value]:
                ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
            return

        sample_labels = [f"{index}." for index in range(1, len(self.tmax_metrics) + 1)]
        frequencies = np.array([frequency for _, frequency, _ in self.tmax_metrics])
        transmittances = np.array([value for _, _, value in self.tmax_metrics])
        positions = np.arange(1, len(self.tmax_metrics) + 1)

        self.ax_tmax_frequency.plot(
            positions, frequencies, color="tab:blue", marker="o", linewidth=1.8, markersize=5.5
        )
        self.ax_tmax_value.plot(
            positions, transmittances, color="tab:orange", marker="o", linewidth=1.8, markersize=5.5
        )
        self.ax_tmax_frequency.set_ylim(
            max(0.0, TMAX_FREQUENCY_MIN_THZ - 0.1), TMAX_FREQUENCY_MAX_THZ + 0.1
        )
        self.ax_tmax_value.set_ylim(bottom=0, top=max(1.0, float(np.max(transmittances)) * 1.15))
        self.ax_tmax_value.set_xticks(positions, sample_labels, rotation=0, ha="center", fontsize=7)
        self.ax_tmax_value.set_xlabel("Sample (ascending frequency at Tmax)")
        self.ax_tmax_frequency.tick_params(axis="x", labelbottom=False)

        for position, frequency in zip(positions, frequencies):
            self.ax_tmax_frequency.annotate(
                f"{frequency:.3g}",
                (position, frequency),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8,
            )
        for position, value in zip(positions, transmittances):
            self.ax_tmax_value.annotate(
                f"{value:.3g}",
                (position, value),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8,
            )

    def _set_summary(self, text: str) -> None:
        self.summary.configure(state="normal")
        self.summary.delete("1.0", tk.END)
        self.summary.insert("1.0", text)
        self.summary.configure(state="disabled")

    def _read_float(self, variable: tk.StringVar, label: str) -> float:
        try:
            return float(variable.get())
        except ValueError as exc:
            raise ValueError(f"{label} must be a number.") from exc

    def _read_int(self, variable: tk.StringVar, label: str) -> int:
        try:
            return int(float(variable.get()))
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer.") from exc

    def run_analysis(self) -> None:
        try:
            reference_path = self.reference_path.get().strip()
            selected_paths = [path for path, selected in self.sample_checks if selected.get()]
            lower = self._read_float(self.lower_bound, "l")
            upper = self._read_float(self.upper_bound, "u")
            alpha_1 = self._read_float(self.alpha_1, "alpha1")
            alpha_2 = self._read_float(self.alpha_2, "alpha2")
            pad_mode = self.pad_mode.get().strip()
            pad_value = self._read_int(self.pad_value, "Npad")
            crop_min = self._read_float(self.crop_min, "crop_min")
            crop_max = self._read_float(self.crop_max, "crop_max")
            thickness_um = self._read_float(self.thickness_um, "thickness (µm)")
            thickness_mode = self.thickness_mode.get().strip().lower()
            thickness_cm = thickness_um * 1e-4

            if not reference_path:
                raise ValueError("A reference file must be selected.")
            if not selected_paths:
                raise ValueError("Select at least one sample file from the sample folder.")
            if thickness_um <= 0:
                raise ValueError("thickness (µm) must be greater than zero.")
            if thickness_mode not in {"thin", "thick"}:
                raise ValueError("Thickness mode must be either thin or thick.")

            reference_df = self.analyzer.load_signal(reference_path)
            loaded_samples = []
            failures = []
            for sample_path in selected_paths:
                try:
                    loaded_samples.append((sample_path, self.analyzer.load_signal(sample_path)))
                except Exception as exc:
                    failures.append(f"{sample_path.name}: {exc}")
            if not loaded_samples:
                raise ValueError("No selected sample could be loaded.\n" + "\n".join(failures))

            common_length = min(len(reference_df), *(len(sample_df) for _, sample_df in loaded_samples))
            if common_length < 2:
                raise ValueError("At least two common time-domain data points are required.")
            self.common_length = common_length
            reference_df = reference_df.iloc[:common_length].reset_index(drop=True)
            results = []
            for sample_path, sample_df in loaded_samples:
                try:
                    sample_df = sample_df.iloc[:common_length].reset_index(drop=True)
                    result = self.analyzer.analyze_pair(
                        sample_df, reference_df, lower_bound=lower, upper_bound=upper,
                        alpha_1=alpha_1, alpha_2=alpha_2, pad_mode=pad_mode,
                        pad_value=pad_value, crop_min=crop_min, crop_max=crop_max,
                        thickness_cm=thickness_cm, thickness_mode=thickness_mode,
                    )
                    results.append((sample_path, result))
                except Exception as exc:
                    failures.append(f"{sample_path.name}: {exc}")
            if not results:
                raise ValueError("No selected sample could be analyzed.\n" + "\n".join(failures))
            self.results = results
            self.echo_guideline_cache.clear()
            tmax_unavailable = self._update_tmax_metrics()
            self._set_active_result_options()
            self._render_active_result()
            self.status.set(f"Analysis complete: {len(results)}/{len(selected_paths)} sample(s)")
            self._set_summary(
                self._format_multi_summary(
                    results, reference_path, thickness_um, thickness_mode, failures, tmax_unavailable
                )
            )
        except Exception as exc:
            self.status.set(f"Error: {exc}")
            messagebox.showerror("Analysis failed", str(exc))

    def _format_multi_summary(
        self,
        results,
        reference_path: str,
        thickness_um: float,
        thickness_mode: str,
        failures: list[str],
        tmax_unavailable: list[str],
    ) -> str:
        lines = [
            f"Reference file: {reference_path}",
            f"Thickness: {thickness_um:g} um",
            f"Thickness mode: {thickness_mode}",
            "",
            f"All selected samples and the reference were tail-trimmed to one common length: {self.common_length}.",
            "One shared analysis window is applied before every FFT.",
            "",
            f"Analyzed samples ({len(results)}):",
        ]
        for sample_path, result in results:
            sample_info = result.sample
            lines.extend([
                f"- {sample_path.name}",
                f"  Pair N: {len(sample_info.time)} | dt: {sample_info.dt:.8f} ps | pad: {sample_info.pad_length}",
                f"  Window: start {sample_info.start_idx}, width {sample_info.width} | output rows: {len(result.transmittance)}",
            ])
        lines.extend([
            "",
            f"Tmax trend band: {TMAX_FREQUENCY_MIN_THZ:g}-{TMAX_FREQUENCY_MAX_THZ:g} THz",
            "Sorted sample order (ascending frequency at Tmax):",
        ])
        for index, (sample_path, frequency, value) in enumerate(self.tmax_metrics, start=1):
            lines.append(f"{index}. {sample_path.name}: Tmax={value:.6g} at {frequency:.6g} THz")
        if tmax_unavailable:
            lines.extend(["Tmax unavailable:", *[f"- {message}" for message in tmax_unavailable]])
        if failures:
            lines.extend(["", "Skipped files (error):", *[f"- {failure}" for failure in failures]])
        return "\n".join(lines) + "\n"

    def _format_summary(self, result, sample_path: str, reference_path: str, thickness_um: float) -> str:
        sample_info = result.sample
        reference_info = result.reference
        return (
            f"Sample file: {sample_path}\n"
            f"Reference file: {reference_path}\n\n"
            f"dt: {sample_info.dt:.8f} ps\n"
            f"N: {len(sample_info.time)}\n"
            f"Window start_idx: {sample_info.start_idx}\n"
            f"Window width: {sample_info.width}\n"
            f"Pad length: {sample_info.pad_length}\n"
            f"Total length (N + Npad): {len(sample_info.time) + sample_info.pad_length}\n"
            f"Reference dt: {reference_info.dt:.8f} ps\n"
            f"Reference N: {len(reference_info.time)}\n\n"
            f"Thickness: {thickness_um:g} µm\n\n"
            f"Transmittance rows: {len(result.transmittance)}\n"
            f"Alpha rows: {len(result.absorption)}\n"
            f"k rows: {len(result.extinction)}\n"
            f"n rows: {len(result.refractive_index)}\n"
        )

def main() -> None:
    app = FFTPlatformGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
