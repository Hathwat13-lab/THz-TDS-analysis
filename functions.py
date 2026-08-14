"""Reusable FFT analysis helpers for THz-TDS data.

This module turns the previous notebook-like script into a callable pipeline
that can later be reused from a GUI, a CLI, or tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TIME_COLUMN = "Time [ps]"
SIGNAL_COLUMN = "THz Amp (TD)"
SPEED_OF_LIGHT_CM_THZ = 0.0299792458
SPEED_OF_LIGHT_UM_PS = 299.792458


def _normalize_column_name(name: object) -> str:
	text = str(name).strip()
	text = text.lstrip("#").strip()
	text = re.sub(r"\s+", " ", text)
	return text


def _find_header_row(file_path: str | Path) -> int:
	with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as handle:
		for line_number, line in enumerate(handle):
			normalized = line.strip().lower()
			if not normalized:
				continue
			if "time" in normalized and ("amp" in normalized or "amplitude" in normalized or "thz" in normalized):
				return line_number
	raise ValueError(f"Could not detect a tabular header row in {file_path}.")


def _standardize_time_domain_frame(df: pd.DataFrame) -> pd.DataFrame:
	normalized_columns = {column: _normalize_column_name(column) for column in df.columns}
	time_column = next((column for column, normalized in normalized_columns.items() if "time" in normalized.lower()), None)
	if time_column is None:
		raise ValueError("No time column could be found in the loaded file.")

	signal_candidates = [
		column
		for column, normalized in normalized_columns.items()
		if column != time_column and any(keyword in normalized.lower() for keyword in ("amp", "amplitude", "thz", "signal"))
	]
	if not signal_candidates:
		signal_candidates = [column for column in df.columns if column != time_column]
	if not signal_candidates:
		raise ValueError("No signal column could be found in the loaded file.")

	preferred_signal = next(
		(column for column in signal_candidates if "raw" in normalized_columns[column].lower()),
		signal_candidates[0],
	)
	return df.rename(columns={time_column: TIME_COLUMN, preferred_signal: SIGNAL_COLUMN})[[TIME_COLUMN, SIGNAL_COLUMN]]


@dataclass(frozen=True)
class WindowSpec:
	lower_bound: float
	upper_bound: float
	alpha_1: float = 0.0
	alpha_2: float = 0.35


@dataclass(frozen=True)
class PreprocessedSignal:
	time: np.ndarray
	amplitude: np.ndarray
	window: np.ndarray
	start_idx: int
	width: int
	windowed: pd.DataFrame
	padded: pd.DataFrame
	dt: float
	pad_length: int


@dataclass(frozen=True)
class SpectrumFrame:
	frequency: np.ndarray
	complex_values: np.ndarray
	magnitude: np.ndarray
	phase: np.ndarray
	df: pd.DataFrame


@dataclass(frozen=True)
class AnalysisResult:
	sample: PreprocessedSignal
	reference: PreprocessedSignal
	sample_spectrum: SpectrumFrame
	reference_spectrum: SpectrumFrame
	sample_spectrum_crop: SpectrumFrame
	reference_spectrum_crop: SpectrumFrame
	transmittance: pd.DataFrame
	absorption: pd.DataFrame
	extinction: pd.DataFrame
	refractive_index: pd.DataFrame
	echo_guideline: dict[str, object] | None = None
	fp_removal_info: dict[str, object] | None = None


@dataclass(frozen=True)
class TransmittanceMaximum:
	"""Maximum transmittance and its frequency within a trusted band."""

	frequency_thz: float
	transmittance: float


@dataclass(frozen=True)
class TransmittanceLocalMaximum:
	"""One local maximum in a transmittance spectrum."""

	frequency_thz: float
	transmittance: float


def load_time_domain_data(
	file_path: str | Path,
	skiprows: int | None = None,
	sep: str | None = None,
	**read_csv_kwargs: object,
) -> pd.DataFrame:
	"""Load a time-domain THz file into a normalized DataFrame.

	The loader auto-detects the header row and standardizes the time/signal
	columns so both legacy TXT exports and newer DAT exports can share the same
	analysis pipeline.
	"""

	path = Path(file_path)
	try:
		header_row = _find_header_row(path)
		frame = pd.read_csv(
			path,
			skiprows=header_row,
			header=0,
			sep=sep or None,
			engine="python",
			**read_csv_kwargs,
		)
		return _standardize_time_domain_frame(frame)
	except Exception:
		if skiprows is None:
			raise
		frame = pd.read_csv(
			path,
			sep=sep or "\t",
			skiprows=skiprows,
			**read_csv_kwargs,
		)
		return _standardize_time_domain_frame(frame)


def extract_time_and_signal(
	df: pd.DataFrame,
	time_column: str = TIME_COLUMN,
	signal_column: str = SIGNAL_COLUMN,
) -> tuple[np.ndarray, np.ndarray]:
	"""Extract time and signal arrays from a DataFrame."""

	return df[time_column].to_numpy(), df[signal_column].to_numpy()


def align_to_min_length(
	first: pd.DataFrame,
	second: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Trim two frames to the same minimum length."""

	min_len = min(len(first), len(second))
	return (
		first.iloc[:min_len].reset_index(drop=True),
		second.iloc[:min_len].reset_index(drop=True),
	)


def ensure_uniform_sampling(time: np.ndarray) -> float:
	"""Estimate the time step from a nearly uniform time axis and return dt."""

	if len(time) < 2:
		raise ValueError("At least two time samples are required to infer dt.")
	deltas = np.diff(time)
	return float(np.median(deltas))


def detrend_signal(signal_values: np.ndarray, mode: str = "constant") -> np.ndarray:
	"""Remove DC offset or linear trend from a signal."""

	if mode == "constant":
		return signal_values - np.mean(signal_values)
	if mode == "linear":
		indices = np.arange(len(signal_values), dtype=float)
		coefficients = np.polyfit(indices, signal_values, 1)
		trend = np.polyval(coefficients, indices)
		return signal_values - trend
	raise ValueError(f"Unsupported detrend mode: {mode}")


def make_tukey_window(length: int, alpha: float = 0.5) -> np.ndarray:
	"""Create a standard symmetric Tukey window."""

	if length <= 0:
		return np.zeros(0, dtype=float)
	if alpha <= 0:
		return np.ones(length, dtype=float)
	if alpha >= 1:
		return np.hanning(length)
	indices = np.arange(length, dtype=float)
	window = np.ones(length, dtype=float)
	edge = alpha * (length - 1) / 2.0
	left = indices < edge
	right = indices >= (length - 1) * (1 - alpha / 2.0)
	window[left] = 0.5 * (1 + np.cos(np.pi * ((2 * indices[left]) / (alpha * (length - 1)) - 1)))
	window[right] = 0.5 * (1 + np.cos(np.pi * ((2 * indices[right]) / (alpha * (length - 1)) - (2 / alpha) + 1)))
	return window


def make_asymmetric_tukey_window(
	length: int,
	alpha_1: float,
	alpha_2: float,
	start_idx: int,
	width: int,
) -> np.ndarray:
	"""Create an asymmetric Tukey-like window anchored at start_idx."""

	window = np.zeros(length, dtype=float)
	if length <= 0 or width <= 0:
		return window

	indices = np.arange(length, dtype=float)
	active = (indices >= start_idx) & (indices < start_idx + width)
	left_end = start_idx + alpha_1 * width / 2.0
	right_start = start_idx + width - alpha_2 * width / 2.0
	right_end = start_idx + width

	if alpha_1 > 0:
		left_mask = active & (indices < left_end)
		window[left_mask] = 0.5 * (
			1 - np.cos(2 * np.pi * (indices[left_mask] - start_idx) / (alpha_1 * width))
		)

	plateau_mask = active & (indices >= left_end) & (indices < right_start)
	window[plateau_mask] = 1.0

	if alpha_2 > 0:
		right_mask = active & (indices >= right_start) & (indices < right_end)
		window[right_mask] = 0.5 * (
			1 - np.cos(2 * np.pi * (right_end - indices[right_mask]) / (alpha_2 * width))
		)

	return window


def build_window_from_bounds(
	time: np.ndarray,
	lower_bound: float,
	upper_bound: float,
	alpha_1: float,
	alpha_2: float,
) -> tuple[np.ndarray, int, int]:
	"""Convert time bounds into a full-length asymmetric window."""

	start_idx = int(np.searchsorted(time, lower_bound, side="left"))
	end_idx = int(np.searchsorted(time, upper_bound, side="right"))
	width = max(0, end_idx - start_idx)
	window = make_asymmetric_tukey_window(len(time), alpha_1, alpha_2, start_idx, width)
	return window, start_idx, width


def apply_window(signal_values: np.ndarray, window: np.ndarray) -> np.ndarray:
	"""Multiply a signal by a window."""

	if len(signal_values) != len(window):
		raise ValueError("Signal and window must have the same length.")
	return signal_values * window


def apply_window_to_frame(
	df: pd.DataFrame,
	window: np.ndarray,
	time_column: str = TIME_COLUMN,
	signal_column: str = SIGNAL_COLUMN,
) -> pd.DataFrame:
	"""Apply a full-length window to a DataFrame signal column."""

	windowed = df.copy()
	windowed[signal_column] = windowed[signal_column].to_numpy() * window
	return windowed


def zero_pad_signal(
	signal_values: np.ndarray,
	target_length: int | None = None,
	pad_factor: int = 1,
) -> np.ndarray:
	"""Pad a signal with zeros to a target length or by factor."""

	if target_length is not None:
		pad_length = max(0, int(target_length) - len(signal_values))
	else:
		pad_length = max(0, int(pad_factor) * len(signal_values))
	return np.concatenate([signal_values, np.zeros(pad_length, dtype=float)])


def zero_pad_frame(
	df: pd.DataFrame,
	pad_length: int,
	time_column: str = TIME_COLUMN,
	signal_column: str = SIGNAL_COLUMN,
) -> pd.DataFrame:
	"""Append a zero-valued tail to a time-domain frame."""

	if pad_length <= 0:
		return df.copy()
	time = df[time_column].to_numpy()
	if len(time) < 2:
		raise ValueError("At least two time points are required for zero padding.")
	dt = ensure_uniform_sampling(time)
	pad_time = time[-1] + dt * np.arange(1, pad_length + 1)
	pad_signal = np.zeros(pad_length, dtype=float)
	pad_df = pd.DataFrame({time_column: pad_time, signal_column: pad_signal})
	return pd.concat([df, pad_df], ignore_index=True)


def compute_fft(signal_values: np.ndarray, dt: float, normalization_length: int | None = None) -> tuple[np.ndarray, np.ndarray]:
	"""Return frequency axis and complex FFT values."""

	fft_values = np.fft.fft(signal_values)
	frequency = np.fft.fftfreq(len(signal_values), dt)
	return frequency, fft_values


def compute_fft_frame(
	signal_values: np.ndarray,
	dt: float,
	normalization_length: int | None = None,
) -> SpectrumFrame:
	"""Compute FFT and package frequency, magnitude, and phase into a frame."""

	frequency, fft_values = compute_fft(signal_values, dt, normalization_length=normalization_length)
	if normalization_length is None:
		normalization_length = len(signal_values)
	magnitude = 2.0 / normalization_length * np.abs(fft_values)
	phase = np.angle(fft_values)
	df = pd.DataFrame({"freq": frequency, "mag": magnitude, "phase": phase})
	return SpectrumFrame(
		frequency=frequency,
		complex_values=fft_values,
		magnitude=magnitude,
		phase=phase,
		df=df,
	)


def compute_rfft(signal_values: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
	"""Return the one-sided FFT and frequency axis."""

	fft_values = np.fft.rfft(signal_values)
	frequency = np.fft.rfftfreq(len(signal_values), dt)
	return frequency, fft_values


def compute_amplitude_spectrum(fft_values: np.ndarray, normalize: bool = True) -> np.ndarray:
	"""Convert complex FFT values to amplitudes."""

	amplitude = np.abs(fft_values)
	return amplitude / len(fft_values) if normalize and len(fft_values) else amplitude


def compute_phase_spectrum(fft_values: np.ndarray) -> np.ndarray:
	"""Compute phase values from complex FFT results."""

	return np.angle(fft_values)


def convert_to_db(amplitude: np.ndarray, reference: float = 1.0, floor: float = 1e-12) -> np.ndarray:
	"""Convert amplitude to dB scale."""

	return 20 * np.log10(np.maximum(amplitude / reference, floor))


def normalize_spectrum(amplitude: np.ndarray, mode: str = "max") -> np.ndarray:
	"""Normalize a spectrum by max, area, or not at all."""

	if mode == "none":
		return amplitude
	if mode == "max":
		peak = np.max(np.abs(amplitude))
		return amplitude if peak == 0 else amplitude / peak
	if mode == "area":
		area = np.trapz(np.abs(amplitude))
		return amplitude if area == 0 else amplitude / area
	raise ValueError(f"Unsupported normalize mode: {mode}")


def crop_frequency_range(
	frequency: np.ndarray,
	spectrum: np.ndarray,
	f_min: float | None = None,
	f_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Crop a frequency spectrum to the requested interval."""

	mask = np.ones_like(frequency, dtype=bool)
	if f_min is not None:
		mask &= frequency >= f_min
	if f_max is not None:
		mask &= frequency <= f_max
	return frequency[mask], spectrum[mask]


def crop_spectrum_frame(frame: SpectrumFrame, f_min: float | None = None, f_max: float | None = None) -> SpectrumFrame:
	"""Crop a SpectrumFrame to a frequency interval."""

	frequency, magnitude = crop_frequency_range(frame.frequency, frame.magnitude, f_min, f_max)
	_, phase = crop_frequency_range(frame.frequency, frame.phase, f_min, f_max)
	complex_values = frame.complex_values[(frame.frequency >= (f_min if f_min is not None else -np.inf)) & (frame.frequency <= (f_max if f_max is not None else np.inf))]
	df = pd.DataFrame({"freq": frequency, "mag": magnitude, "phase": phase})
	return SpectrumFrame(frequency=frequency, complex_values=complex_values, magnitude=magnitude, phase=phase, df=df)


def plot_time_domain(
	time: np.ndarray,
	signal_values: np.ndarray,
	ax: plt.Axes | None = None,
	title: str = "Time Domain",
) -> plt.Axes:
	"""Plot a time-domain signal."""

	if ax is None:
		_, ax = plt.subplots()
	ax.plot(time, signal_values)
	ax.set_title(title)
	ax.set_xlabel("Time [ps]")
	ax.set_ylabel("THz Amplitude (TD)")
	ax.grid(True)
	return ax


def plot_frequency_domain(
	frequency: np.ndarray,
	amplitude: np.ndarray,
	ax: plt.Axes | None = None,
	title: str = "Frequency Domain",
) -> plt.Axes:
	"""Plot a frequency-domain spectrum."""

	if ax is None:
		_, ax = plt.subplots()
	ax.plot(frequency, amplitude)
	ax.set_title(title)
	ax.set_xlabel("Frequency [THz]")
	ax.set_ylabel("Amplitude")
	ax.grid(True)
	return ax


def find_transmittance_maximum(
	transmittance: pd.DataFrame,
	frequency_min_thz: float = 0.5,
	frequency_max_thz: float = 2.5,
) -> TransmittanceMaximum:
	"""Return the largest finite transmittance in the inclusive frequency band.

	The default band is the high-SNR range used for sample-to-sample resonance
	comparison.  The input is the ``AnalysisResult.transmittance`` DataFrame.
	"""
	if frequency_min_thz > frequency_max_thz:
		raise ValueError("frequency_min_thz must not exceed frequency_max_thz.")
	if not {"freq", "mag"}.issubset(transmittance.columns):
		raise ValueError("transmittance must contain 'freq' and 'mag' columns.")

	frequency = transmittance["freq"].to_numpy(dtype=float)
	magnitude = transmittance["mag"].to_numpy(dtype=float)
	valid = (
		np.isfinite(frequency)
		& np.isfinite(magnitude)
		& (frequency >= frequency_min_thz)
		& (frequency <= frequency_max_thz)
	)
	if not np.any(valid):
		raise ValueError(
			f"No finite transmittance data is available in "
			f"{frequency_min_thz:g}-{frequency_max_thz:g} THz."
		)

	indices = np.flatnonzero(valid)
	peak_index = indices[int(np.argmax(magnitude[indices]))]
	return TransmittanceMaximum(
		frequency_thz=float(frequency[peak_index]),
		transmittance=float(magnitude[peak_index]),
	)


def find_transmittance_local_maxima(transmittance: pd.DataFrame) -> list[TransmittanceLocalMaximum]:
	"""Return finite local maxima from a frequency-sorted transmittance table.

	A maximum is higher than both adjacent finite points. The first and last
	points are excluded, and no arbitrary smoothing threshold is imposed.
	"""
	if not {"freq", "mag"}.issubset(transmittance.columns):
		raise ValueError("transmittance must contain 'freq' and 'mag' columns.")

	frequency = transmittance["freq"].to_numpy(dtype=float)
	magnitude = transmittance["mag"].to_numpy(dtype=float)
	valid = np.isfinite(frequency) & np.isfinite(magnitude)
	frequency = frequency[valid]
	magnitude = magnitude[valid]
	if len(frequency) < 3:
		return []

	order = np.argsort(frequency, kind="stable")
	frequency = frequency[order]
	magnitude = magnitude[order]
	maximum_indices = np.flatnonzero(
		(magnitude[1:-1] > magnitude[:-2]) & (magnitude[1:-1] > magnitude[2:])
	) + 1
	return [
		TransmittanceLocalMaximum(float(frequency[index]), float(magnitude[index]))
		for index in maximum_indices
	]


def build_frequency_axis(sample_rate: float, n_points: int) -> np.ndarray:
	"""Build a frequency axis from sample rate and point count."""

	return np.fft.fftfreq(n_points, d=1.0 / sample_rate)


def summarize_fft_parameters(dt: float, n_points: int, pad_factor: int, alpha: float) -> dict[str, float]:
	"""Summarize key FFT parameters for display or logging."""

	return {
		"dt": float(dt),
		"n_points": float(n_points),
		"pad_factor": float(pad_factor),
		"alpha": float(alpha),
	}


def compute_optical_constants(
	frequency: np.ndarray,
	transmittance: np.ndarray,
	phase_difference: np.ndarray,
	thickness_cm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Compute absorption coefficient, extinction coefficient, and refractive index."""

	alpha = -np.log(transmittance) / thickness_cm
	k = alpha * SPEED_OF_LIGHT_CM_THZ / (4 * np.pi * frequency)
	n = 1 - phase_difference * SPEED_OF_LIGHT_CM_THZ / (2 * thickness_cm * np.pi * frequency)
	return alpha, k, n


def film_fp_factor(
	frequency_thz: np.ndarray,
	n_film: float,
	k_film: float,
	thickness_um: float,
	n_substrate: float = 1.0,
	k_substrate: float = 0.0,
) -> np.ndarray:
	"""Complex Fabry-Perot ringing factor of a thin film on a substrate.

	Standard single-layer TMM result for incidence from air (n=1) through a
	film (index ``n_film - i k_film``, thickness ``thickness_um``) into a
	semi-infinite substrate (index ``n_substrate - i k_substrate``, defaults
	to air i.e. a free-standing film). This isolates just the film's own
	multiple-internal-reflection contribution: dividing a measured complex
	field-transmission spectrum by this factor removes the film's own etalon
	ringing while leaving everything else (e.g. a metasurface resonance in
	the substrate) untouched, since that resonance is not part of this
	factor. The substrate index matters here — it sets the film/substrate
	interface reflection ``r12``, which is generally different from the
	air/film interface ``r01`` unless the substrate happens to be air too.
	"""

	thickness_cm = thickness_um * 1e-4
	n_tilde = n_film - 1j * k_film
	n_sub_tilde = n_substrate - 1j * k_substrate
	r01 = (1 - n_tilde) / (1 + n_tilde)
	r12 = (n_tilde - n_sub_tilde) / (n_tilde + n_sub_tilde)
	delta = 2 * np.pi * frequency_thz * n_tilde * thickness_cm / SPEED_OF_LIGHT_CM_THZ
	return 1.0 / (1.0 + r01 * r12 * np.exp(2j * delta))


def _polynomial_baseline(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
	"""Fit a low-order polynomial baseline to (x, y) and evaluate it on x."""

	valid = np.isfinite(x) & np.isfinite(y)
	if np.count_nonzero(valid) <= degree:
		return np.zeros_like(y)
	coefficients = np.polyfit(x[valid], y[valid], degree)
	return np.polyval(coefficients, x)


def _notch_periodic_residual(
	freq: np.ndarray,
	residual: np.ndarray,
	target_period_thz: float,
	n_harmonics: int = 3,
	relative_bandwidth: float = 0.25,
) -> np.ndarray:
	"""Remove the periodic component (and a few harmonics) near ``target_period_thz``.

	``residual`` is assumed to already have any smooth (non-periodic) trend
	removed and to live on a near-uniformly spaced ``freq`` axis. The Fourier
	conjugate variable of frequency here is a time-like "quefrency" that, for
	a true Fabry-Perot ripple, equals the film's round-trip time.
	"""

	n_points = len(freq)
	if n_points < 8 or target_period_thz <= 0:
		return residual
	spacing = float(np.median(np.diff(freq)))
	spectrum = np.fft.rfft(residual)
	quefrency = np.fft.rfftfreq(n_points, d=spacing)
	target_quefrency = 1.0 / target_period_thz
	mask = np.zeros_like(quefrency, dtype=bool)
	for harmonic in range(1, n_harmonics + 1):
		center = harmonic * target_quefrency
		mask |= np.abs(quefrency - center) <= relative_bandwidth * target_quefrency
	spectrum[mask] = 0.0
	return np.fft.irfft(spectrum, n_points)


def remove_fp_ringing_spectral_notch(
	transmittance: pd.DataFrame,
	thickness_um: float,
	n_film_guess: float,
	baseline_degree: int = 3,
	n_harmonics: int = 2,
	relative_bandwidth: float = 0.12,
) -> pd.DataFrame:
	"""Tier 1: model-free removal of the film's periodic FP ripple.

	Fits a smooth polynomial baseline to log-magnitude and (unwrapped) phase,
	then notches out the periodic residual expected at the film's FP
	round-trip frequency ``c / (2 * n_film_guess * thickness_um)``. Does not
	require the film's true optical constants beyond a rough thickness/index
	estimate, so it cannot distort a resonance that is far from that ripple
	frequency, but a too-wide notch can smear closely spaced peaks.
	"""

	if thickness_um <= 0:
		raise ValueError("thickness_um must be greater than zero.")
	if not {"freq", "mag", "phase"}.issubset(transmittance.columns):
		raise ValueError("transmittance must contain 'freq', 'mag', and 'phase' columns.")

	freq = transmittance["freq"].to_numpy(dtype=float)
	mag = transmittance["mag"].to_numpy(dtype=float)
	phase = transmittance["phase"].to_numpy(dtype=float)

	thickness_cm = thickness_um * 1e-4
	target_period_thz = SPEED_OF_LIGHT_CM_THZ / (2 * n_film_guess * thickness_cm)

	ln_mag = np.log(np.clip(mag, 1e-12, None))
	baseline_ln_mag = _polynomial_baseline(freq, ln_mag, baseline_degree)
	residual_ln_mag = ln_mag - baseline_ln_mag
	filtered_residual_ln_mag = _notch_periodic_residual(
		freq, residual_ln_mag, target_period_thz, n_harmonics, relative_bandwidth
	)
	corrected_mag = np.exp(baseline_ln_mag + filtered_residual_ln_mag)

	baseline_phase = _polynomial_baseline(freq, phase, baseline_degree)
	residual_phase = phase - baseline_phase
	filtered_residual_phase = _notch_periodic_residual(
		freq, residual_phase, target_period_thz, n_harmonics, relative_bandwidth
	)
	corrected_phase = baseline_phase + filtered_residual_phase

	return pd.DataFrame({"freq": freq, "mag": corrected_mag, "phase": corrected_phase})


def correct_fp_ringing_known_film(
	transmittance: pd.DataFrame,
	n_film: float,
	k_film: float,
	thickness_um: float,
	n_substrate: float = 1.0,
	k_substrate: float = 0.0,
) -> pd.DataFrame:
	"""Tier 2: divide out a known film's own FP ringing factor.

	Reconstructs the measured complex field transmission from the stored
	power transmittance (``mag``) and unwrapped field phase (``phase``),
	divides it by :func:`film_fp_factor` for the supplied film parameters
	(and substrate index, since the film's bottom interface reflects off the
	substrate, not air), and returns the corrected power transmittance /
	phase in the same schema. ``n_film``/``k_film`` are expected to be
	roughly known ahead of time (literature value, ellipsometry, or a
	separate measurement) rather than solved for here.
	"""

	if thickness_um <= 0:
		raise ValueError("thickness_um must be greater than zero.")
	if not {"freq", "mag", "phase"}.issubset(transmittance.columns):
		raise ValueError("transmittance must contain 'freq', 'mag', and 'phase' columns.")

	freq = transmittance["freq"].to_numpy(dtype=float)
	mag = transmittance["mag"].to_numpy(dtype=float)
	phase = transmittance["phase"].to_numpy(dtype=float)

	fp_factor = film_fp_factor(freq, n_film, k_film, thickness_um, n_substrate, k_substrate)
	fp_mag = np.abs(fp_factor)
	fp_phase = np.unwrap(np.angle(fp_factor))

	corrected_mag = mag / np.clip(fp_mag**2, 1e-12, None)
	corrected_phase = phase - fp_phase
	return pd.DataFrame({"freq": freq, "mag": corrected_mag, "phase": corrected_phase})


def calibrate_fp_ringing(
	transmittance: pd.DataFrame,
	thickness_um: float,
	n_film_init: float,
	k_film_init: float = 0.0,
	n_substrate: float = 1.0,
	k_substrate: float = 0.0,
	fit_n_k: bool = False,
	baseline_degree: int = 3,
) -> tuple[pd.DataFrame, dict[str, float]]:
	"""Tier 3: fit film FP parameters to minimize residual ringing power.

	Starts from :func:`correct_fp_ringing_known_film` and uses
	``scipy.optimize.least_squares`` to adjust the film thickness (and
	optionally ``n_film``/``k_film``) so that the corrected log-magnitude has
	as little leftover smooth-baseline residual as possible. This is a
	calibration against the ringing itself, not a fit to any reference
	curve, so it cannot invent or erase a real resonance far from the film's
	own FP period.
	"""

	from scipy.optimize import least_squares

	if thickness_um <= 0:
		raise ValueError("thickness_um must be greater than zero.")

	freq = transmittance["freq"].to_numpy(dtype=float)
	spacing = float(np.median(np.diff(freq)))
	n_points = len(freq)

	def residual_for(params: np.ndarray) -> np.ndarray:
		if fit_n_k:
			d_um, n_film, k_film = params
		else:
			(d_um,) = params
			n_film, k_film = n_film_init, k_film_init
		corrected = correct_fp_ringing_known_film(transmittance, n_film, k_film, d_um, n_substrate, k_substrate)
		ln_mag = np.log(np.clip(corrected["mag"].to_numpy(dtype=float), 1e-12, None))
		baseline = _polynomial_baseline(freq, ln_mag, baseline_degree)
		full_residual = ln_mag - baseline

		# Only penalize the leftover power AT this candidate's own FP
		# quefrency (± a narrow band), not the whole residual — the
		# injected/real resonance is not periodic in frequency and must not
		# be flattened away by this objective.
		thickness_cm = d_um * 1e-4
		target_period_thz = SPEED_OF_LIGHT_CM_THZ / (2 * n_film * thickness_cm)
		target_quefrency = 1.0 / target_period_thz
		spectrum = np.fft.rfft(full_residual)
		quefrency = np.fft.rfftfreq(n_points, d=spacing)
		isolated = np.zeros_like(spectrum)
		band = np.abs(quefrency - target_quefrency) <= 0.15 * target_quefrency
		isolated[band] = spectrum[band]
		return np.fft.irfft(isolated, n_points)

	# A candidate thickness whose FP period is comparable to (or wider than)
	# the analyzed bandwidth is unresolvable: its "ringing" band in the
	# quefrency-domain objective collides with the near-DC content the
	# polynomial baseline already absorbs, creating a spurious low-cost
	# attractor at unrealistically small thickness. Floor the search range
	# so at least a few fringes are always resolvable.
	bandwidth_thz = float(freq.max() - freq.min())
	min_resolvable_um = 3 * SPEED_OF_LIGHT_CM_THZ / (2 * n_film_init * bandwidth_thz) * 1e4
	d_lower = max(thickness_um * 0.5, min_resolvable_um)
	d_upper = max(thickness_um * 1.5, d_lower * 1.05)

	if fit_n_k:
		initial = np.array([thickness_um, n_film_init, k_film_init], dtype=float)
		lower = np.array([d_lower, 1.0, 0.0])
		upper = np.array([d_upper, 10.0, 5.0])
	else:
		initial = np.array([thickness_um], dtype=float)
		lower = np.array([d_lower])
		upper = np.array([d_upper])
	initial = np.clip(initial, lower, upper)

	# The isolated-band objective is periodic/multimodal in thickness (many
	# local minima from aliasing between the candidate and true FP period),
	# so a gradient-based local optimizer easily gets stuck unless it starts
	# near the true thickness. Do a coarse grid search over d first (holding
	# n_film/k_film at their initial guess) and seed the local refinement
	# from the best grid point instead of the raw user guess.
	grid_d = np.linspace(d_lower, d_upper, 61)
	grid_cost = [
		float(np.sum(residual_for(np.array([d] if not fit_n_k else [d, n_film_init, k_film_init])) ** 2))
		for d in grid_d
	]
	initial[0] = grid_d[int(np.argmin(grid_cost))]

	fit = least_squares(residual_for, initial, bounds=(lower, upper))
	if fit_n_k:
		fitted_d_um, fitted_n, fitted_k = fit.x
	else:
		fitted_d_um = float(fit.x[0])
		fitted_n, fitted_k = n_film_init, k_film_init

	corrected = correct_fp_ringing_known_film(transmittance, fitted_n, fitted_k, fitted_d_um, n_substrate, k_substrate)
	info = {
		"thickness_um": float(fitted_d_um),
		"n_film": float(fitted_n),
		"k_film": float(fitted_k),
		"n_substrate": float(n_substrate),
		"k_substrate": float(k_substrate),
		"fit_n_k": bool(fit_n_k),
		"cost": float(fit.cost),
	}
	return corrected, info


def find_absolute_peak(time: np.ndarray, amplitude: np.ndarray) -> tuple[int, float, float]:
	"""Return the index, time, and amplitude of the largest absolute peak."""

	if len(time) == 0 or len(amplitude) == 0:
		raise ValueError("Peak detection requires a non-empty time-domain signal.")
	peak_index = 0
	peak_amplitude = amplitude[0]
	peak_magnitude = abs(peak_amplitude)
	for index, value in enumerate(amplitude):
		current_magnitude = abs(value)
		if current_magnitude > peak_magnitude:
			peak_index = index
			peak_amplitude = value
			peak_magnitude = current_magnitude
	return peak_index, float(time[peak_index]), float(peak_amplitude)


def mean_refractive_index_in_band(
	frequency: np.ndarray,
	refractive_index: np.ndarray,
	f_min: float = 0.5,
	f_max: float = 2.5,
) -> float:
	"""Compute the mean refractive index inside the requested band."""

	mask = (frequency >= f_min) & (frequency <= f_max)
	if not np.any(mask):
		raise ValueError(f"No refractive-index points found in the {f_min:g}-{f_max:g} THz band.")
	return float(np.mean(refractive_index[mask]))


class AsymmetricTDSAnalyzer:
	"""Reusable pipeline for asymmetric-window THz-TDS analysis."""

	def __init__(
		self,
		time_column: str = TIME_COLUMN,
		signal_column: str = SIGNAL_COLUMN,
		skiprows: int = 19,
		sep: str = "\t",
	) -> None:
		self.time_column = time_column
		self.signal_column = signal_column
		self.skiprows = skiprows
		self.sep = sep

	def load_signal(self, file_path: str | Path) -> pd.DataFrame:
		return load_time_domain_data(file_path, sep=self.sep, skiprows=self.skiprows)

	def extract(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
		return extract_time_and_signal(df, self.time_column, self.signal_column)

	def align_pair(self, sample: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
		return align_to_min_length(sample, reference)

	def _preprocess_common(
		self,
		df: pd.DataFrame,
		lower_bound: float,
		upper_bound: float,
		alpha_1: float,
		alpha_2: float,
		pad_mode: str = "factor",
		pad_value: int = 4,
	) -> PreprocessedSignal:
		time, amplitude = self.extract(df)
		dt = ensure_uniform_sampling(time)
		window, start_idx, width = build_window_from_bounds(time, lower_bound, upper_bound, alpha_1, alpha_2)
		windowed_amplitude = apply_window(amplitude, window)
		windowed = pd.DataFrame({self.time_column: time, self.signal_column: windowed_amplitude})

		if pad_mode == "factor":
			pad_length = int(pad_value) * len(windowed)
		elif pad_mode == "length":
			pad_length = int(pad_value)
		else:
			raise ValueError("pad_mode must be either 'factor' or 'length'.")

		padded = zero_pad_frame(windowed, pad_length, self.time_column, self.signal_column)
		return PreprocessedSignal(
			time=time,
			amplitude=amplitude,
			window=window,
			start_idx=start_idx,
			width=width,
			windowed=windowed,
			padded=padded,
			dt=dt,
			pad_length=pad_length,
		)

	def preprocess_thin(
		self,
		df: pd.DataFrame,
		lower_bound: float,
		upper_bound: float,
		alpha_1: float,
		alpha_2: float,
		pad_mode: str = "factor",
		pad_value: int = 4,
	) -> PreprocessedSignal:
		return self._preprocess_common(df, lower_bound, upper_bound, alpha_1, alpha_2, pad_mode, pad_value)

	def preprocess_thick(
		self,
		df: pd.DataFrame,
		lower_bound: float,
		upper_bound: float,
		alpha_1: float,
		alpha_2: float,
		pad_mode: str = "factor",
		pad_value: int = 4,
	) -> PreprocessedSignal:
		return self._preprocess_common(df, lower_bound, upper_bound, alpha_1, alpha_2, pad_mode, pad_value)

	def build_echo_guideline(
		self,
		sample_df: pd.DataFrame,
		reference_df: pd.DataFrame,
		lower_bound: float,
		upper_bound: float,
	) -> dict[str, object]:
		"""Placeholder metadata hook for future thick-sample echo handling."""

		sample_time, _ = self.extract(sample_df)
		reference_time, _ = self.extract(reference_df)
		return {
			"mode": "thick",
			"sample_time_range": (float(sample_time[0]), float(sample_time[-1])) if len(sample_time) else None,
			"reference_time_range": (float(reference_time[0]), float(reference_time[-1])) if len(reference_time) else None,
			"window_bounds": (float(lower_bound), float(upper_bound)),
			"note": "Placeholder echo guideline. Replace with TMM-based logic later.",
		}

	def compute_echo_guideline(
		self,
		result: AnalysisResult,
		thickness_um: float,
		f_min: float = 0.5,
		f_max: float = 2.5,
	) -> dict[str, object]:
		"""Estimate FP echo timing for a thick sample from the current analysis result."""

		if thickness_um <= 0:
			raise ValueError("thickness_um must be greater than zero.")

		n_mean = mean_refractive_index_in_band(
			result.refractive_index["freq"].to_numpy(),
			result.refractive_index["n"].to_numpy(),
			f_min=f_min,
			f_max=f_max,
		)
		peak_index, peak_time_ps, peak_amplitude = find_absolute_peak(
			result.sample.time,
			result.sample.amplitude,
		)
		echo_dt_ps = float(2.0 * thickness_um * n_mean / SPEED_OF_LIGHT_UM_PS)
		echo_time_ps = float(peak_time_ps + echo_dt_ps)
		return {
			"mode": "thick",
			"n_mean": n_mean,
			"band_thz": (float(f_min), float(f_max)),
			"peak_index": int(peak_index),
			"peak_time_ps": peak_time_ps,
			"peak_amplitude": peak_amplitude,
			"echo_dt_ps": echo_dt_ps,
			"echo_time_ps": echo_time_ps,
			"echo_amplitude": peak_amplitude,
			"thickness_um": float(thickness_um),
		}

	def spectrum(self, preprocessed: PreprocessedSignal, crop_min: float = 0.2, crop_max: float = 3.0) -> tuple[SpectrumFrame, SpectrumFrame]:
		full = compute_fft_frame(
			preprocessed.padded[self.signal_column].to_numpy(),
			preprocessed.dt,
			normalization_length=len(preprocessed.amplitude),
		)
		cropped = crop_spectrum_frame(full, crop_min, crop_max)
		return full, cropped

	def analyze_pair(
		self,
		sample_df: pd.DataFrame,
		reference_df: pd.DataFrame,
		lower_bound: float,
		upper_bound: float,
		alpha_1: float = 0.0,
		alpha_2: float = 0.35,
		pad_mode: str = "factor",
		pad_value: int = 4,
		crop_min: float = 0.2,
		crop_max: float = 3.0,
		thickness_cm: float = 0.0460,
		thickness_mode: str = "thick",
		fp_removal_method: str = "none",
		film_n_guess: float = 1.5,
		film_k_guess: float = 0.0,
		substrate_n: float = 1.0,
		substrate_k: float = 0.0,
		fit_film_n_k: bool = False,
	) -> AnalysisResult:
		sample_aligned, reference_aligned = self.align_pair(sample_df, reference_df)
		mode = thickness_mode.strip().lower()
		if mode == "thin":
			sample = self.preprocess_thin(sample_aligned, lower_bound, upper_bound, alpha_1, alpha_2, pad_mode, pad_value)
			reference = self.preprocess_thin(reference_aligned, lower_bound, upper_bound, alpha_1, alpha_2, pad_mode, pad_value)
			echo_guideline = None
		elif mode == "thick":
			sample = self.preprocess_thick(sample_aligned, lower_bound, upper_bound, alpha_1, alpha_2, pad_mode, pad_value)
			reference = self.preprocess_thick(reference_aligned, lower_bound, upper_bound, alpha_1, alpha_2, pad_mode, pad_value)
			echo_guideline = self.build_echo_guideline(sample_aligned, reference_aligned, lower_bound, upper_bound)
		else:
			raise ValueError("thickness_mode must be either 'thin' or 'thick'.")
		sample_full, sample_crop = self.spectrum(sample, crop_min, crop_max)
		reference_full, reference_crop = self.spectrum(reference, crop_min, crop_max)

		transmittance_mag = (sample_crop.magnitude / reference_crop.magnitude) ** 2
		phase_sample = np.unwrap(sample_crop.phase)
		phase_reference = np.unwrap(reference_crop.phase)
		phase_difference = phase_sample - phase_reference
		transmittance = pd.DataFrame(
			{
				"freq": sample_crop.frequency,
				"mag": transmittance_mag,
				"phase": phase_difference,
			}
		)

		fp_removal_info: dict[str, object] | None = None
		fp_method = fp_removal_method.strip().lower()
		if mode == "thin" and fp_method != "none":
			thickness_um = thickness_cm * 1e4
			if fp_method == "spectral_notch":
				transmittance = remove_fp_ringing_spectral_notch(
					transmittance, thickness_um, film_n_guess
				)
				fp_removal_info = {"method": fp_method, "thickness_um": thickness_um, "n_film_guess": film_n_guess}
			elif fp_method == "known_film":
				transmittance = correct_fp_ringing_known_film(
					transmittance, film_n_guess, film_k_guess, thickness_um,
					n_substrate=substrate_n, k_substrate=substrate_k,
				)
				fp_removal_info = {
					"method": fp_method,
					"thickness_um": thickness_um,
					"n_film": film_n_guess,
					"k_film": film_k_guess,
					"n_substrate": substrate_n,
					"k_substrate": substrate_k,
				}
			elif fp_method == "auto_calibrate":
				transmittance, calibration = calibrate_fp_ringing(
					transmittance, thickness_um, film_n_guess, film_k_guess,
					n_substrate=substrate_n, k_substrate=substrate_k, fit_n_k=fit_film_n_k,
				)
				fp_removal_info = {"method": fp_method, **calibration}
			else:
				raise ValueError(
					"fp_removal_method must be one of 'none', 'spectral_notch', 'known_film', 'auto_calibrate'."
				)
			transmittance_mag = transmittance["mag"].to_numpy()
			phase_difference = transmittance["phase"].to_numpy()

		alpha, k, n = compute_optical_constants(sample_crop.frequency, transmittance_mag, phase_difference, thickness_cm)
		absorption = pd.DataFrame({"freq": sample_crop.frequency, "alpha": alpha})
		extinction = pd.DataFrame({"freq": sample_crop.frequency, "k": k})
		refractive_index = pd.DataFrame({"freq": sample_crop.frequency, "n": n})

		return AnalysisResult(
			sample=sample,
			reference=reference,
			sample_spectrum=sample_full,
			reference_spectrum=reference_full,
			sample_spectrum_crop=sample_crop,
			reference_spectrum_crop=reference_crop,
			transmittance=transmittance,
			absorption=absorption,
			extinction=extinction,
			refractive_index=refractive_index,
			echo_guideline=echo_guideline,
			fp_removal_info=fp_removal_info,
		)
