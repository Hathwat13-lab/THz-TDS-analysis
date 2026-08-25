from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from functions import (
    AsymmetricTDSAnalyzer,
    TransmittanceLocalMaximum,
    find_transmittance_local_maxima,
    find_transmittance_local_minima,
    find_transmittance_maximum,
    find_transmittance_minimum,
)


DEFAULT_REFERENCE = ""
DEFAULT_SAMPLE_FOLDER = ""
TMAX_FREQUENCY_MIN_THZ = 0.5
TMAX_FREQUENCY_MAX_THZ = 2.5


class LocalMaximaWindow(tk.Toplevel):
    """Display local transmittance maxima or minima for one or more analyzed samples."""

    def __init__(self, parent: tk.Misc, spectra: list[tuple[Path, pd.DataFrame]]) -> None:
        super().__init__(parent)
        self.geometry("1450x900")
        self.minsize(1000, 650)
        self.spectra = spectra
        self.minima_mode = tk.BooleanVar(value=False)
        self.extrema_by_sample: list[list[TransmittanceLocalMaximum]] = []
        self.trees: list[ttk.Treeview] = []

        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        # Long source filenames in the table notebook must not collapse the
        # plot pane to zero width.
        root.columnconfigure(0, weight=3, minsize=520)
        root.columnconfigure(1, weight=2, minsize=360)
        root.rowconfigure(0, weight=1)

        plot_frame = ttk.Frame(root)
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        table_frame = ttk.Frame(root)
        table_frame.grid(row=0, column=1, sticky="nsew")
        table_frame.rowconfigure(2, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.header_label = ttk.Label(table_frame, text="", font=("Segoe UI", 11, "bold"))
        self.header_label.grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Checkbutton(
            table_frame, text="Minima mode", variable=self.minima_mode, command=self._on_mode_toggled
        ).grid(row=1, column=0, sticky="w", pady=(0, 6))
        self.tables = ttk.Notebook(table_frame)
        self.tables.grid(row=2, column=0, sticky="nsew")
        self.export_button = ttk.Button(table_frame, text="", command=self._export_extrema)
        self.export_button.grid(row=3, column=0, sticky="e", pady=(6, 0))

        self.figure = Figure(figsize=(9, 7), dpi=100, constrained_layout=True)
        self.axis = self.figure.add_subplot(111)

        for index, (sample_path, _transmittance) in enumerate(spectra):
            tab = ttk.Frame(self.tables, padding=6)
            # A notebook computes its requested width from every tab label.
            # Keep labels short and show the complete source name inside its
            # table instead, so multiple selections leave room for the plot.
            self.tables.add(tab, text=f"#{index + 1}")
            ttk.Label(tab, text=sample_path.name, wraplength=330).grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 5)
            )
            tab.rowconfigure(1, weight=1)
            tab.columnconfigure(0, weight=1)
            table = ttk.Treeview(tab, columns=("number", "frequency", "transmittance"), show="headings")
            table.heading("number", text="#")
            table.heading("frequency", text="Frequency [THz]")
            table.heading("transmittance", text="Transmittance")
            table.column("number", width=45, anchor="center", stretch=False)
            table.column("frequency", width=135, anchor="e")
            table.column("transmittance", width=135, anchor="e")
            scrollbar = ttk.Scrollbar(tab, orient="vertical", command=table.yview)
            table.configure(yscrollcommand=scrollbar.set)
            table.grid(row=1, column=0, sticky="nsew")
            scrollbar.grid(row=1, column=1, sticky="ns")
            self.trees.append(table)

        # Keep Python references for the lifetime of the Toplevel.  Without
        # them Tk can retain an empty widget after Matplotlib's canvas object
        # has been garbage-collected.
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.tables.bind("<<NotebookTabChanged>>", self._on_table_changed)
        self._recompute_extrema()
        self.update_idletasks()
        self.canvas.draw()

    def _extremum_label(self) -> str:
        return "minima" if self.minima_mode.get() else "maxima"

    def _current_tab_index(self) -> int:
        selection = self.tables.select()
        return self.tables.index(selection) if selection else 0

    def _on_mode_toggled(self) -> None:
        self._recompute_extrema()

    def _recompute_extrema(self) -> None:
        finder = find_transmittance_local_minima if self.minima_mode.get() else find_transmittance_local_maxima
        self.extrema_by_sample = [finder(frame) for _, frame in self.spectra]
        label = self._extremum_label()
        singular = "minimum" if self.minima_mode.get() else "maximum"

        self.title(f"Transmittance local {label}")
        self.header_label.configure(text=f"Local-{singular} coordinates")
        self.export_button.configure(text=f"Export local {label}...")

        for tree, extrema in zip(self.trees, self.extrema_by_sample):
            tree.delete(*tree.get_children())
            for number, point in enumerate(extrema, start=1):
                tree.insert("", "end", values=(number, f"{point.frequency_thz:.8g}", f"{point.transmittance:.8g}"))
            if not extrema:
                tree.insert("", "end", values=("-", f"No local {label}", ""))

        self._draw_selected_spectrum(self._current_tab_index())
        self.canvas.draw_idle()

    def _on_table_changed(self, _event=None) -> None:
        selected_index = self.tables.index(self.tables.select())
        self._draw_selected_spectrum(selected_index)
        self.canvas.draw_idle()

    def _draw_selected_spectrum(self, index: int) -> None:
        """Draw only the sample selected in the coordinate-table tabs."""
        sample_path, transmittance = self.spectra[index]
        extrema = self.extrema_by_sample[index]
        label = self._extremum_label()
        frequency = transmittance["freq"].to_numpy(dtype=float)
        values = transmittance["mag"].to_numpy(dtype=float)
        valid = np.isfinite(frequency) & np.isfinite(values)

        self.axis.clear()
        self.axis.plot(frequency[valid], values[valid], color="tab:blue", linewidth=1.45, label=sample_path.stem)
        if extrema:
            self.axis.scatter(
                [point.frequency_thz for point in extrema],
                [point.transmittance for point in extrema],
                marker="x", color="crimson", s=62, linewidths=1.9, zorder=5, label=f"Local {label}",
            )
        self.axis.set_title(f"Local {label} — #{index + 1}: {sample_path.name}")
        self.axis.set_xlabel("Frequency [THz]")
        self.axis.set_ylabel("Transmittance")
        self.axis.grid(True, alpha=0.3)
        self.axis.legend(loc="best", fontsize=8)

    def _export_extrema(self) -> None:
        label = self._extremum_label()
        if not any(self.extrema_by_sample):
            messagebox.showinfo(f"Export local {label}", f"No local {label} were found for the loaded samples.")
            return

        output_path = filedialog.asksaveasfilename(
            title=f"Export local {label}",
            defaultextension=".xlsx",
            initialfile=f"local_{label}.xlsx",
            filetypes=[
                ("Excel workbook", "*.xlsx"),
                ("CSV file", "*.csv"),
                ("Text file", "*.txt"),
            ],
        )
        if not output_path:
            return
        try:
            table = export_local_maxima_table(self.spectra, self.extrema_by_sample, output_path, kind=label)
        except Exception as exc:
            messagebox.showerror(f"Export local {label}", str(exc))
            return

        messagebox.showinfo(f"Export local {label}", f"Saved {len(table)} row(s) to:\n{output_path}")


def export_local_maxima_table(
    spectra: list[tuple[Path, pd.DataFrame]],
    maxima_by_sample: list[list[TransmittanceLocalMaximum]],
    output_path: str | Path,
    kind: str = "maxima",
) -> pd.DataFrame:
    """Save every sample's local transmittance maxima or minima as one combined table.

    The output format is chosen from output_path's extension: .xlsx gets the
    same formatted-worksheet treatment as the Tmax export, .txt is
    tab-separated, and anything else (typically .csv) is comma-separated.
    """

    rows = [
        {
            "sample": sample_path.name,
            "no.": number,
            "frequency [THz]": point.frequency_thz,
            "transmittance": point.transmittance,
        }
        for (sample_path, _), maxima in zip(spectra, maxima_by_sample)
        for number, point in enumerate(maxima, start=1)
    ]
    table = pd.DataFrame(rows, columns=["sample", "no.", "frequency [THz]", "transmittance"])

    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    sheet_name = f"Local {kind}"
    if suffix == ".xlsx":
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            table.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
            for column, width in {"A": 48, "B": 8, "C": 18, "D": 18}.items():
                worksheet.column_dimensions[column].width = width
            for row in worksheet.iter_rows(min_row=2, min_col=3, max_col=4):
                for cell in row:
                    cell.number_format = "0.000000"
    elif suffix == ".txt":
        table.to_csv(output_path, sep="\t", index=False)
    else:
        table.to_csv(output_path, index=False)
    return table


def export_tmax_metrics_to_excel(
    metrics: list[tuple[Path, float, float]], output_path: str | Path, label: str = "Tmax"
) -> pd.DataFrame:
    """Save the already-sorted Tmax/Tmin metrics as a formatted Excel worksheet."""

    table = pd.DataFrame(
        {
            "no.": range(1, len(metrics) + 1),
            "name": [sample_path.name for sample_path, _, _ in metrics],
            f"f@{label} [THz]": [frequency for _, frequency, _ in metrics],
            label: [value for _, _, value in metrics],
        }
    )
    sheet_name = f"{label} trend"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.font = cell.font.copy(bold=True)
        for column, width in {"A": 8, "B": 48, "C": 18, "D": 18}.items():
            worksheet.column_dimensions[column].width = width
        for row in worksheet.iter_rows(min_row=2, min_col=3, max_col=4):
            for cell in row:
                cell.number_format = "0.000000"
    return table


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
        self.transmittance_y_mode = tk.StringVar(value="auto")
        self.transmittance_y_min = tk.StringVar(value="0.0")
        self.transmittance_y_max = tk.StringVar(value="1.0")
        # The analyzer uses centimetres internally; expose the more convenient
        # micrometre unit in the GUI.
        self.thickness_um = tk.StringVar(value="460")
        self.thickness_mode = tk.StringVar(value="thick")
        self.fp_removal_method = tk.StringVar(value="none")
        self.film_n_guess = tk.StringVar(value="1.5")
        self.film_k_guess = tk.StringVar(value="0.0")
        # Default matches high-resistivity Si, a common THz-TDS substrate;
        # override for SiO2 (~1.95-2.1) or whatever the actual substrate is.
        self.substrate_n_guess = tk.StringVar(value="3.42")
        self.substrate_k_guess = tk.StringVar(value="0.0")
        self.fit_film_n_k = tk.BooleanVar(value=False)
        self.echo_guideline_enabled = tk.BooleanVar(value=False)
        self.tmax_minima_mode = tk.BooleanVar(value=False)
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

        row = self._paired_entry_row(row, "l (ps)", self.lower_bound, "u (ps)", self.upper_bound)
        row = self._paired_entry_row(row, "alpha1 (-)", self.alpha_1, "alpha2 (-)", self.alpha_2)
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

        row = self._paired_entry_row(row, "crop_min (THz)", self.crop_min, "crop_max (THz)", self.crop_max)
        ttk.Label(self.control_frame, text="T y-axis").grid(row=row, column=0, sticky="w", pady=4)
        y_axis_frame = ttk.Frame(self.control_frame)
        y_axis_frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.transmittance_y_mode_combo = ttk.Combobox(
            y_axis_frame, textvariable=self.transmittance_y_mode,
            values=("auto", "manual"), width=8, state="readonly",
        )
        self.transmittance_y_mode_combo.pack(side="left")
        ttk.Label(y_axis_frame, text=" min").pack(side="left")
        self.transmittance_y_min_entry = ttk.Entry(y_axis_frame, textvariable=self.transmittance_y_min, width=7)
        self.transmittance_y_min_entry.pack(side="left", padx=(2, 0))
        ttk.Label(y_axis_frame, text=" max").pack(side="left")
        self.transmittance_y_max_entry = ttk.Entry(y_axis_frame, textvariable=self.transmittance_y_max, width=7)
        self.transmittance_y_max_entry.pack(side="left", padx=(2, 0))
        row += 1
        row = self._entry_row(row, "thickness (µm)", self.thickness_um)

        ttk.Label(self.control_frame, text="Thickness mode").grid(row=row, column=0, sticky="w", pady=4)
        thickness_mode_frame = ttk.Frame(self.control_frame)
        thickness_mode_frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Radiobutton(thickness_mode_frame, text="thin (film)", value="thin", variable=self.thickness_mode).pack(side="left")
        ttk.Radiobutton(thickness_mode_frame, text="thick (pellet)", value="thick", variable=self.thickness_mode).pack(side="left", padx=(10, 0))
        row += 1

        ttk.Label(self.control_frame, text="FP removal (thin only)").grid(row=row, column=0, sticky="w", pady=4)
        self.fp_removal_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.fp_removal_method,
            values=("none", "spectral_notch", "known_film", "auto_calibrate"),
            width=15,
            state="readonly",
        )
        self.fp_removal_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        row += 1

        row = self._paired_entry_row(row, "film n (guess)", self.film_n_guess, "film k (guess)", self.film_k_guess)
        row = self._paired_entry_row(row, "substrate n", self.substrate_n_guess, "substrate k", self.substrate_k_guess)
        fit_nk_frame = ttk.Frame(self.control_frame)
        fit_nk_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        ttk.Checkbutton(
            fit_nk_frame,
            text="auto_calibrate also fits film n/k (not just thickness)",
            variable=self.fit_film_n_k,
        ).pack(side="left")
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

        ttk.Button(self.control_frame, text="Show local extrema", command=self._open_local_maxima_window).grid(
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

    def _paired_entry_row(
        self, row: int, left_label: str, left_variable: tk.StringVar, right_label: str, right_variable: tk.StringVar
    ) -> int:
        """Place two compact labelled inputs on one control-panel row."""
        frame = ttk.Frame(self.control_frame)
        frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(frame, text=left_label).grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=left_variable, width=9).grid(row=0, column=1, sticky="w", padx=(3, 10))
        ttk.Label(frame, text=right_label).grid(row=0, column=2, sticky="w")
        ttk.Entry(frame, textvariable=right_variable, width=9).grid(row=0, column=3, sticky="w", padx=(3, 0))
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
        self.tmax_tab.rowconfigure(1, weight=1)
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
        self.monitor_canvas.mpl_connect("motion_notify_event", self._show_transmittance_cursor_coordinates)

        self.optical_canvas = FigureCanvasTkAgg(self.optical_figure, master=self.optical_tab)
        self.optical_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.tmax_sort_label = tk.StringVar(value="Sorted by f@Tmax (ascending)")
        tmax_toolbar = ttk.Frame(self.tmax_tab, padding=(0, 0, 0, 6))
        tmax_toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Label(tmax_toolbar, textvariable=self.tmax_sort_label).pack(side="left")
        ttk.Checkbutton(
            tmax_toolbar, text="Minima mode (Tmin)", variable=self.tmax_minima_mode, command=self._on_tmax_mode_toggled
        ).pack(side="left", padx=(12, 0))
        self.tmax_export_button = ttk.Button(tmax_toolbar, text="Export Tmax to Excel", command=self._export_tmax_to_excel)
        self.tmax_export_button.pack(side="right")

        self.tmax_canvas = FigureCanvasTkAgg(self.tmax_figure, master=self.tmax_tab)
        self.tmax_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

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
        self.transmittance_y_mode_combo.bind("<<ComboboxSelected>>", self._on_transmittance_y_mode_changed)
        for entry in (self.transmittance_y_min_entry, self.transmittance_y_max_entry):
            entry.bind("<Return>", self._on_transmittance_y_limits_changed)
            entry.bind("<FocusOut>", self._on_transmittance_y_limits_changed)
        self._update_transmittance_y_controls()

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

    def _open_local_maxima_window(self) -> None:
        """Open a marked plot and coordinate table for the checked samples."""
        selected_names = {path.name for path, selected in self.sample_checks if selected.get()}
        spectra = [
            (sample_path, result.transmittance)
            for sample_path, result in self.results
            if sample_path.name in selected_names
        ]
        if not spectra:
            messagebox.showinfo("Local maxima", "Run the analysis and select at least one analyzed sample.")
            return
        try:
            LocalMaximaWindow(self, spectra)
        except Exception as exc:
            self.status.set(f"Could not show local maxima: {exc}")
            messagebox.showerror("Local maxima", str(exc))

    def _on_transmittance_y_mode_changed(self, _event=None) -> None:
        self._update_transmittance_y_controls()
        self._render_active_result()

    def _on_transmittance_y_limits_changed(self, _event=None) -> None:
        if self.transmittance_y_mode.get() == "manual":
            self._render_active_result()

    def _update_transmittance_y_controls(self) -> None:
        state = "!disabled" if self.transmittance_y_mode.get() == "manual" else "disabled"
        self.transmittance_y_min_entry.state([state])
        self.transmittance_y_max_entry.state([state])

    def _apply_transmittance_y_limits(self) -> None:
        if self.transmittance_y_mode.get() != "manual":
            self.ax_t.set_ylim(bottom=0)
            return
        try:
            lower = float(self.transmittance_y_min.get())
            upper = float(self.transmittance_y_max.get())
        except ValueError:
            self.status.set("Manual T y-axis limits must be numbers.")
            return
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            self.status.set("Manual T y-axis requires finite minimum < maximum.")
            return
        self.ax_t.set_ylim(lower, upper)

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
        self._transmittance_hover_text = self.ax_t.text(
            0.012, 0.98, "", transform=self.ax_t.transAxes, ha="left", va="top",
            fontsize=9, bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.9},
            visible=False, zorder=10,
        )

    def _show_transmittance_cursor_coordinates(self, event) -> None:
        """Show the cursor's data-space coordinates while hovering over T(f)."""
        label = getattr(self, "_transmittance_hover_text", None)
        if label is None:
            return
        if event.inaxes is self.ax_t and event.xdata is not None and event.ydata is not None:
            label.set_text(f"x = {event.xdata:.6g} THz\ny = {event.ydata:.6g}")
            label.set_visible(True)
        elif label.get_visible():
            label.set_visible(False)
        else:
            return
        self.monitor_canvas.draw_idle()

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

    def _tmax_label(self) -> str:
        return "Tmin" if self.tmax_minima_mode.get() else "Tmax"

    def _style_tmax_axes(self) -> None:
        label = self._tmax_label()
        for ax in [self.ax_tmax_frequency, self.ax_tmax_value]:
            ax.clear()
            ax.grid(True, axis="y", alpha=0.3)

        self.ax_tmax_frequency.set_title(
            f"Frequency at {label} ({TMAX_FREQUENCY_MIN_THZ:g}-{TMAX_FREQUENCY_MAX_THZ:g} THz)")
        self.ax_tmax_frequency.set_ylabel("Frequency [THz]")
        self.ax_tmax_value.set_title(
            f"{label} ({TMAX_FREQUENCY_MIN_THZ:g}-{TMAX_FREQUENCY_MAX_THZ:g} THz)")
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

        self._apply_transmittance_y_limits()
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

    def _on_tmax_mode_toggled(self) -> None:
        label = self._tmax_label()
        self.tmax_sort_label.set(f"Sorted by f@{label} (ascending)")
        self.tmax_export_button.configure(text=f"Export {label} to Excel")
        self._update_tmax_metrics()
        self._render_active_result()

    def _update_tmax_metrics(self) -> list[str]:
        """Extract one in-band transmittance maximum or minimum per analyzed sample."""

        finder = find_transmittance_minimum if self.tmax_minima_mode.get() else find_transmittance_maximum
        self.tmax_metrics = []
        unavailable = []
        for sample_path, result in self.results:
            try:
                extremum = finder(
                    result.transmittance,
                    frequency_min_thz=TMAX_FREQUENCY_MIN_THZ,
                    frequency_max_thz=TMAX_FREQUENCY_MAX_THZ,
                )
                self.tmax_metrics.append((sample_path, extremum.frequency_thz, extremum.transmittance))
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

    def _export_tmax_to_excel(self) -> None:
        label = self._tmax_label()
        if not self.tmax_metrics:
            messagebox.showinfo(f"Export {label}", f"Run the analysis first so {label} metrics are available.")
            return

        output_path = filedialog.asksaveasfilename(
            title=f"Export {label} trend to Excel",
            defaultextension=".xlsx",
            initialfile=f"{label.lower()}_trend.xlsx",
            filetypes=[("Excel workbook", "*.xlsx")],
        )
        if not output_path:
            return
        try:
            table = export_tmax_metrics_to_excel(self.tmax_metrics, output_path, label=label)
        except Exception as exc:
            self.status.set(f"Excel export failed: {exc}")
            messagebox.showerror(f"Export {label}", str(exc))
            return

        self.status.set(f"Exported {len(table)} {label} row(s) to {Path(output_path).name}")
        messagebox.showinfo(f"Export {label}", f"Saved {len(table)} row(s) to:\n{output_path}")

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
            fp_removal_method = self.fp_removal_method.get().strip().lower()
            film_n_guess = self._read_float(self.film_n_guess, "film n (guess)")
            film_k_guess = self._read_float(self.film_k_guess, "film k (guess)")
            substrate_n_guess = self._read_float(self.substrate_n_guess, "substrate n")
            substrate_k_guess = self._read_float(self.substrate_k_guess, "substrate k")
            fit_film_n_k = self.fit_film_n_k.get()

            if not reference_path:
                raise ValueError("A reference file must be selected.")
            if not selected_paths:
                raise ValueError("Select at least one sample file from the sample folder.")
            if thickness_um <= 0:
                raise ValueError("thickness (µm) must be greater than zero.")
            if thickness_mode not in {"thin", "thick"}:
                raise ValueError("Thickness mode must be either thin or thick.")
            if fp_removal_method != "none" and thickness_mode != "thin":
                raise ValueError("FP removal is only used in thin (film) mode. Set 'FP removal' to 'none' or switch to thin mode.")

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
                        fp_removal_method=fp_removal_method, film_n_guess=film_n_guess,
                        film_k_guess=film_k_guess, substrate_n=substrate_n_guess,
                        substrate_k=substrate_k_guess, fit_film_n_k=fit_film_n_k,
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
            if result.fp_removal_info:
                info = result.fp_removal_info
                detail = ", ".join(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}" for key, value in info.items() if key != "method")
                lines.append(f"  FP removal: {info['method']} ({detail})")
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
