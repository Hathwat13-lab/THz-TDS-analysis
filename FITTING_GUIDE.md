# Transmittance fitting 가이드

## 목적과 범위

`fitting_gui.py`는 **한 번에 하나의 transmittance spectrum**을 피팅합니다. 메인 FFT
창은 활성 샘플에서 이미 계산된 `transmittance` 데이터(`freq`, `mag`)만 별도 fitting
창으로 전달합니다. 따라서 time-domain 처리나 FFT를 다시 수행하지 않습니다.

The fitter supports Lorentzian and Gaussian peaks and dips in Single and Cumulative modes.
Fano and Drude-Smith remain planned.

## 사용 순서

1. 메인 창에서 일반 FFT 분석을 실행합니다.
2. 활성 샘플 하나를 선택하고 **Open fitting window**를 누릅니다.
3. `Single` 모드와 모델을 선택합니다.
4. fitting 주파수 범위를 지정합니다. 기본값은 0.5–2.5 THz이며 수정할 수 있습니다.
5. **Run single fit**을 누릅니다. 그래프에는 측정 spectrum, fit 대역, fitted curve,
   그리고 fitted maximum이 표시됩니다.

선택한 주파수 범위에 포함되는 finite data point만 사용합니다. 이 조건을 만족하는
data point가 최소 8개 필요합니다.

## 모델과 fitted parameter

### Lorentzian

$$
T(f) = b + A\frac{\Gamma^2}{(f-f_0)^2+\Gamma^2}
$$

| Parameter | Meaning |
| --- | --- |
| $b$ | Frequency-independent transmittance background. |
| $A$ | Signed amplitude: non-negative for Maximum, non-positive for Minimum. |
| $f_0$ | Resonance centre frequency in THz. |
| $\Gamma$ | Half width at half maximum (HWHM) in THz. |

이 식에서는,

$$
\mathrm{FWHM}=2\Gamma.
$$

### Gaussian

$$
T(f)=b+A\exp\left[-\frac{(f-f_0)^2}{2\sigma^2}\right]
$$

| Parameter | Meaning |
| --- | --- |
| $b$ | Frequency-independent transmittance background. |
| $A$ | Signed amplitude: non-negative for Maximum, non-positive for Minimum. |
| $f_0$ | Peak centre frequency in THz. |
| $\sigma$ | Standard deviation of the Gaussian in THz. |

이 식에서는,

$$
\mathrm{FWHM}=2\sqrt{2\ln2}\,\sigma.
$$

## fitting 계산 방식

fitter는 SciPy의 bounded nonlinear least-squares optimizer
(`scipy.optimize.curve_fit`)를 사용합니다. 측정 transmittance와 선택한 모델 사이의
unweighted squared residual 합을 최소화합니다.

$$
\min_\theta \sum_i\left[T_\text{measured}(f_i)-T_\text{model}(f_i;\theta)\right]^2.
$$

초기값은 다음 방식으로 자동 생성됩니다.

- 선택 data의 10th percentile로 $b$를 추정합니다.
- 가장 높은 측정 point로 $f_0$를 추정하고 $A$의 초기값을 만듭니다.
- 선택한 주파수 폭의 약 1/10을 $\Gamma$ 또는 $\sigma$ 초기 width로 사용합니다.

amplitude는 0 이상으로 제한하고, centre는 선택 fitting 대역 안에 머물며, width는
항상 양수가 되도록 제한합니다. 이 제약은 single-peak 모델이 물리적으로 해석 가능한
범위에 머물도록 돕습니다.

## 표시되는 값

fitting이 끝나면 선택한 모델을 fit 대역 전체의 4,001개 균일 간격 주파수에서
평가합니다.

| Output | Definition in the current implementation |
| --- | --- |
| `Tmax` | Largest value of the **fitted model** on that dense frequency grid. |
| `f @ Tmax` | Frequency at which that fitted maximum occurs. For a well-behaved positive Lorentzian or Gaussian, this is essentially $f_0$. |
| `FWHM` | Calculated analytically from $\Gamma$ (Lorentzian) or $\sigma$ (Gaussian), using the equations above. |
| `Q` | $Q = \frac{f_{T_{\max}}}{\mathrm{FWHM}}$. This is dimensionless because both terms use THz. |
| `R²` | 선택 fitting 대역의 원본 data point에서 계산한 coefficient of determination입니다. GUI에서는 percent로 표시합니다. |
| `Adjusted R²` | parameter 수를 반영해 보정한 R²입니다. 현재 single model은 $p=4$개의 parameter를 사용하며, GUI에서는 percent로 표시합니다. |

따라서 `Tmax`는 단순히 noise가 포함된 raw sample point 중 최댓값이 아니라
**model-based** 값입니다. 샘플 사이를 매끄럽고 일관된 기준으로 비교할 때 유용합니다.

현재 구현의 fit-quality 계산은 다음과 같습니다. $y_i$는 fitting에 사용된 원본
transmittance, $\hat{y}_i$는 해당 원본 주파수에서의 fitted value, $n$은 사용한 data
point 수입니다.

$$
R^2 = 1 - \frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}
$$

$$
\mathrm{Adjusted}\ R^2 = 1-(1-R^2)\frac{n-1}{n-p-1}
$$

Adjusted R²는 parameter를 추가해 무조건 좋아 보이는 현상을 어느 정도 보정합니다.
현재는 $n > p+1$일 때만 계산하며, 그렇지 않으면 `—`를 표시합니다. R²가 높더라도
residual의 모양과 모델의 물리적 타당성은 별도로 확인해야 합니다.

## Lorentzian과 Gaussian 선택

- 공명의 tail이 비교적 길고 damped-resonator와 유사한 line shape라면
  **Lorentzian**을 사용합니다.
- resonant frequency 분포 등으로 인해 broadening이 bell shape에 가깝다면
  **Gaussian**을 사용합니다.

두 모델 중 하나를 선택했다는 사실만으로 microscopic mechanism이 증명되지는 않습니다.
plot된 residual behavior를 확인하고, 타당한 주파수 범위에서 반복해 보며, 모델을
비교한 뒤 물리적 결론을 내려야 합니다.


## Maximum / Minimum and cumulative fitting

Select **Maximum** for upward peaks or **Minimum** for downward dips, then run
the fit. In Cumulative mode, click each peak/dip first. Changing direction clears
previous clicks and results. All cumulative components use the same direction;
mixed peaks and dips are not supported.

Minimum mode initializes the background at the 90th percentile and the single
center at the lowest measured point, with amplitudes constrained to be non-positive.
This replaces the positive-amplitude initialization described above for Maximum.

Both `Tmax` / `f @ Tmax` and `Tmin` / `f @ Tmin` are displayed and marked on the
fitted curve, using the same 4,001-point grid. These are fitted values, not raw
sample extrema. Single-fit Q uses `f @ Tmin` in Minimum mode and `f @ Tmax` in
Maximum mode. Dip FWHM is the full width at half depth relative to the background.

Cumulative fits optimize a shared background and all clicked components together.
The table reports each component center, FWHM and Q (center / FWHM); R-squared
and adjusted R-squared describe the overall fit. Overlapping components can depend
strongly on initial clicks. Asymmetric Fano shapes remain unsupported.
