# Thin-film FP ringing 제거 — 검증 가이드

이 문서의 목적은 "동작하는 것 같다"가 아니라 **당신이 직접, 코드와 물리식을
대조해가며 맞는지 틀렸는지 판단할 수 있게** 하는 것이다. 그래서 순서를 이렇게
잡았다: (1) 지금 상태 요약, (2) TMM 공식이 어디서 왔는지 처음부터 유도, (3)
`functions.py`에서 바뀐 것 전부, (4) 세 가지 방법의 코드, (5) 직접 재현 가능한
검증 스크립트와 체크리스트.

## 1. 지금 상태 (요약)

리뷰 과정에서 **버그를 두 개 찾았고 둘 다 고쳤다.** 하나는 이미 커밋(`0db56d8`)에
들어가 있고, 하나는 아직 **커밋 전 working tree 상태**로 남아 있다 — 당신이
직접 diff를 보고 판단한 뒤 커밋 여부를 정하라고 일부러 남겨뒀다.

```
git log --oneline -1                 # 0db56d8 Add thin-film Fabry-Perot ringing removal (3 tiers)
git diff -- functions.py             # 커밋 안 된 부분: 부호 버그 수정 (아래 1.2)
```

### 1.1. (커밋됨) substrate를 공기로 취급한 버그

필름이 실제로는 substrate(Si/SiO2) 위에 있는데, 처음 구현은 필름을 "공기 중에
뜬 필름"으로 취급하는 특수식을 썼다. `film_fp_factor`에 `n_substrate`/`k_substrate`
파라미터를 추가하고 공식을 일반화해서 고쳤다 (§2, §3.2 참고).

### 1.2. (아직 커밋 안 됨) 흡수 부호가 반대였던 버그

이건 이번에 검증 가이드를 쓰다가 "TMM이 진짜 맞나" 다시 손으로 유도해보면서
새로 찾은 것이다. 물리적으로 당연한 성질 하나를 코드가 만족하는지 직접 찍어봤다:

> **필름의 흡수(`k_film`)가 커질수록, 필름 내부에서 빛이 여러 번 왕복하는 게
> 물리적으로 점점 더 불가능해지므로, ringing factor의 크기는 1로 수렴해야
> 한다 (즉 ringing이 사라져야 한다).**

```python
# 실제로 돌려본 것
for k_film in [0.0, 0.2, 0.5, 1.0, 3.0]:
    fp = film_fp_factor(freq, 1.6, k_film, 150.0, n_substrate=3.42, k_substrate=0.0)
    print(np.min(np.abs(fp)), np.max(np.abs(fp)))
```

| k_film | 수정 전 `\|fp\|` 범위 | 수정 후 `\|fp\|` 범위 |
| --- | --- | --- |
| 0.0 | [0.923, 1.072] | [0.923, 1.072] (동일) |
| 0.2 | [0.328, 1.477] | [0.981, 1.019] |
| 0.5 | [0.004, 0.824] | [0.990, 1.003] |
| 1.0 | [0.0000, 0.228] | [0.998, 1.000] |
| 3.0 | [0.0000, 0.0002] | [1.000, 1.000] |

수정 전 코드는 흡수가 커질수록 **오히려 분모가 0에 가까워져서**(어떤
주파수에서는 correction factor가 거의 무한대가 됨 — 나누면 결과가 폭발함)
물리적으로 거꾸로 갔다. 원인은 전파 위상의 부호: 이 파일 전체(기존
`compute_optical_constants`가 암묵적으로 쓰는 관례)는

```
H(ω) ∝ exp(-i(ñ-1)ωd/c),  ñ = n - ik  (k≥0가 흡수)
```

를 쓰는데, `film_fp_factor`의 왕복 위상 항은 `exp(+2iδ)`(플러스 부호)로 짜여
있었다 — 부호가 반대라 `k_film`이 커질수록 감쇠 대신 증폭이 걸렸다. 고친 diff는
이 한 줄뿐이다:

```diff
- return 1.0 / (1.0 + r01 * r12 * np.exp(2j * delta))
+ return 1.0 / (1.0 + r01 * r12 * np.exp(-2j * delta))
```

**중요**: 이 부호는 `k_film`이 작을 때는(예: 검증에 쓴 0.02) 오차가 작아서
합성 데이터 검증 결과에 큰 영향을 안 줬다 — 그래서 처음엔 못 잡았다. 흡수가 큰
필름(k가 0.2 이상)에 `known_film`/`auto_calibrate`를 그대로 썼다면 결과가 크게
틀어졌을 것이다.

## 2. TMM 공식은 어디서 왔는가 (직접 유도, 외부 라이브러리 없음)

**이 프로젝트는 어떤 TMM/MTMM 오픈소스 코드도 참조하거나 복사하지 않았다.**
아래는 표준 광학 교과서(Hecht, *Optics*; Born & Wolf, *Principles of Optics*;
Macleod, *Thin-Film Optical Filters*)에 나오는 단일층 박막 간섭(Airy 공식)을
그대로 손으로 다시 유도한 것이다 — 아무 교과서나 펴서 대조해볼 수 있다.

**설정**: 평면파가 수직 입사. z<0: 공기(n0=1). 0<z<d: 필름(ñ1 = n_film - i·k_film).
z>d: substrate(ñ2 = n_substrate - i·k_substrate).

**계면에서의 Fresnel 진폭 계수** (수직 입사):

```
r_ab = (n_a - n_b) / (n_a + n_b)      t_ab = 2 n_a / (n_a + n_b)
```

**유도**: z=0에서 투과한 진폭은 `t01`. 필름을 한 번 지나면서 위상
`exp(iδ)`(δ = ñ1·ω·d/c)을 얻는다. z=d에서 일부는 그대로 투과(`t12`)해서 나가고
(0차 항: `t01·t12·exp(iδ)`), 나머지는 반사(`r12`)해서 되돌아가 필름을 한 번 더
지나고(`exp(iδ)`), z=0 안쪽 면에서 반사(`r10 = -r01`, Stokes relation)한 뒤 다시
필름을 지나(`exp(iδ)`) z=d에서 또 일부가 투과한다. 왕복 한 번마다
`r10·r12·exp(2iδ)` 배가 곱해지는 등비급수이므로:

```
t_total = t01·t12·exp(iδ) · Σ_{m=0}^∞ [r10·r12·exp(2iδ)]^m
        = t01·t12·exp(iδ) / (1 - r10·r12·exp(2iδ))
        = t01·t12·exp(iδ) / (1 + r01·r12·exp(2iδ))      (∵ r10 = -r01)
```

이건 어느 광학 교과서에도 나오는 표준 결과다(예: Hecht *Optics* 5판, "Multiple
Reflections in a Film"). 분자 `t01·t12·exp(iδ)`는 주파수에 따라 완만하게만
변하는(주기적 진동이 없는) 항이고, **주기적 ringing은 전부 분모
`(1 + r01·r12·exp(2iδ))⁻¹`에서 나온다** — 그래서 `film_fp_factor`는 분자를
버리고 분모(의 역수)만 계산한다. 즉 "필름이 얹혀서 생긴 추가적인 등비급수
간섭 성분만" 분리해내는 것이지, 전체 투과율 자체를 계산하는 게 아니다.

**부호 관례**: 이 프로젝트는 `exp(+iδ)`가 아니라 `exp(-iδ)`를 순방향 전파
위상으로 쓴다(§1.2). 이건 순전히 시간 관례(`exp(-iωt)` vs `exp(+iωt)`) 선택의
문제라 어느 쪽이든 자기 일관적이면 되는데, **기존에 이미 있던
`compute_optical_constants`(내가 건드리지 않은 코드)가 이미 `exp(-i(ñ-1)ωd/c)`
관례를 쓰고 있었으므로 거기에 맞춘 것**이다. 이 부호를 반대로 짜면 위 표처럼
흡수가 감쇠가 아니라 증폭으로 나온다.

### 2.1. 손으로/코드로 바로 확인할 수 있는 성질들

| 성질 | 검증 방법 |
| --- | --- |
| `n_substrate == n_film` → `r12=0` → `film_fp_factor ≡ 1` (ringing 없음) | 아래 스크립트 `assert` 참고. 계면이 없으면 반사도 없다는 당연한 성질. |
| `n_substrate = 1, k_substrate = 0` → `r12 = -r01`이 되어 `factor = 1/(1 + r01·(-r01)·exp(-2iδ)) = 1/(1 - r01²·exp(-2iδ))` | free-standing film의 특수해로 정확히 환원되는지 직접 대수로 계산해볼 수 있다. |
| `k_film` 증가 → `\|film_fp_factor\|` → 1로 수렴 (진동 폭이 줄어듦) | §1.2 표. |
| ringing의 주파수 주기는 `Δf = c / (2·n_film·d)` (substrate와 무관) | `exp(∓2iδ)`의 실수부만 주기성을 결정하고, 그 계수가 `n_film`에만 의존하기 때문. `remove_fp_ringing_spectral_notch`/`calibrate_fp_ringing`이 이 식을 그대로 쓴다. |

## 3. `functions.py`에서 바뀐 것 전부

`git diff 369702d2 HEAD -- functions.py`(이번 기능 커밋 직전 대비)를 그대로
확인했고, **삭제되거나 수정된 기존 줄은 단 하나도 없다 — 전부 순수 추가(append)다.**
직접 확인하려면:

```
git diff 369702d2 HEAD -- functions.py | grep '^-'
# 출력이 "--- a/functions.py" 한 줄뿐이면 기존 코드가 안 건드려졌다는 뜻
```

### 3.1. `AnalysisResult` (dataclass, 기존 코드 근처)

- `fp_removal_info: dict[str, object] | None = None` 필드 **추가만** 됨. 기존
  필드(`transmittance`, `refractive_index` 등) 순서/이름 변화 없음.

### 3.2. 새 함수 (전부 신규, 기존 함수 옆에 추가됨)

| 함수 | 역할 | 줄 수 |
| --- | --- | --- |
| `film_fp_factor(freq, n_film, k_film, d_um, n_substrate=1.0, k_substrate=0.0)` | §2의 TMM 공식. 모든 티어가 공유하는 유일한 물리 primitive. | ~20줄 |
| `_polynomial_baseline(x, y, degree)` | `np.polyfit`/`np.polyval` 감싼 헬퍼. "완만한 배경"과 "주기적 ringing"을 분리하는 데 공통으로 씀. | ~7줄 |
| `_notch_periodic_residual(freq, residual, target_period_thz, ...)` | residual을 FFT해서 목표 주기(및 배음) 근처 성분만 0으로 지우고 역변환. Tier 1에서만 씀 (Tier 3은 비슷한 로직을 목적함수 안에 별도로 인라인했다 — 완전히 재사용은 안 됨, 아래 참고). | ~15줄 |
| `remove_fp_ringing_spectral_notch(...)` | **Tier 1**. TMM 아님, 순수 신호처리. | ~25줄 |
| `correct_fp_ringing_known_film(...)` | **Tier 2**. `film_fp_factor`로 대수적 나눗셈. | ~20줄 |
| `calibrate_fp_ringing(...)` | **Tier 3**. Tier 2 + `scipy.optimize.least_squares`로 두께(옵션 n,k) 자동 보정. | ~65줄 |

### 3.3. `AsymmetricTDSAnalyzer.analyze_pair()` (기존 메서드, 확장만 됨)

- 시그니처에 키워드 인자 6개 추가: `fp_removal_method="none"`, `film_n_guess=1.5`,
  `film_k_guess=0.0`, `substrate_n=1.0`, `substrate_k=0.0`, `fit_film_n_k=False`.
  전부 default가 있어서 **기존 호출 코드(`history/FFT(thick).py` 포함)는 전혀
  안 바뀐 것처럼 동작한다** (기본값이 전부 "아무것도 안 함"에 해당).
- 기존의 transmittance 계산(`transmittance_mag`, `phase_difference` 산출부)과
  `compute_optical_constants` 호출 사이에 28줄짜리 블록이 새로 끼어들었다:
  `thickness_mode == "thin"`이고 `fp_removal_method != "none"`일 때만 위 세
  함수 중 하나를 호출해서 `transmittance`를 덮어쓴다. **`thick` 모드거나
  `fp_removal_method="none"`이면 이 블록은 아예 실행되지 않는다** (§4.2에서
  직접 실행해서 확인 가능).
- 반환값 `AnalysisResult(...)`에 `fp_removal_info=fp_removal_info` 인자 추가.

### 3.4. 건드리지 않은 것 (functions.py 안에서)

`compute_optical_constants`, `preprocess_thin`/`preprocess_thick`(여전히 서로
동일한 windowing 로직 — 이번 변경은 windowing을 안 바꿨다), `build_echo_guideline`,
`compute_echo_guideline`, FFT/윈도우/스펙트럼 관련 모든 함수, `AsymmetricTDSAnalyzer`의
나머지 메서드. 전부 원본 그대로.

### 3.5. `FFT(multi).py`에서 바뀐 것 (요약, `functions.py`가 아니라 GUI 쪽)

- `self.fp_removal_method`, `self.film_n_guess`, `self.film_k_guess`,
  `self.substrate_n_guess`(기본 3.42=Si), `self.substrate_k_guess`(기본 0.0),
  `self.fit_film_n_k` — `StringVar`/`BooleanVar` 추가.
- 컨트롤 패널에 콤보박스 1개 + `_paired_entry_row` 2줄 + 체크박스 1개 추가.
- `run_analysis()`에서 이 값들을 읽어서 `analyze_pair()`로 키워드 전달.
  `thick` 모드에서 FP removal이 `none`이 아니면 에러 발생시키는 유효성 검사 추가.
- `_format_multi_summary`에 `fp_removal_info` 출력 줄 추가.
- 그 외 플로팅 코드(`_render_active_result` 등)는 안 건드림 — `result.transmittance`를
  그대로 그리는 기존 구조라 자동으로 보정된 곡선을 받는다.

## 4. 세 가지 방법의 실제 코드 (최신, 방금 확인한 버전과 동일)

공통 물리 primitive:

```python
def film_fp_factor(frequency_thz, n_film, k_film, thickness_um, n_substrate=1.0, k_substrate=0.0):
    thickness_cm = thickness_um * 1e-4
    n_tilde = n_film - 1j * k_film
    n_sub_tilde = n_substrate - 1j * k_substrate
    r01 = (1 - n_tilde) / (1 + n_tilde)
    r12 = (n_tilde - n_sub_tilde) / (n_tilde + n_sub_tilde)
    delta = 2 * np.pi * frequency_thz * n_tilde * thickness_cm / SPEED_OF_LIGHT_CM_THZ
    return 1.0 / (1.0 + r01 * r12 * np.exp(-2j * delta))   # 부호 수정 반영(§1.2)
```

### Tier 1 — `spectral_notch` (TMM 아님, 순수 신호처리)

이미 계산된 transmittance의 `ln(mag)`, `phase`에 저차 다항식 baseline을 맞추고,
남은 residual을 FFT해서 필름의 예상 round-trip 주기(`Δf ≈ c/(2·n·d)`) 근처만
지운다. substrate 정보 불필요 (주기가 substrate와 무관하므로).

```python
def _notch_periodic_residual(freq, residual, target_period_thz, n_harmonics=3, relative_bandwidth=0.25):
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

def remove_fp_ringing_spectral_notch(transmittance, thickness_um, n_film_guess,
                                      baseline_degree=3, n_harmonics=2, relative_bandwidth=0.12):
    freq, mag, phase = transmittance["freq"].to_numpy(), transmittance["mag"].to_numpy(), transmittance["phase"].to_numpy()
    thickness_cm = thickness_um * 1e-4
    target_period_thz = SPEED_OF_LIGHT_CM_THZ / (2 * n_film_guess * thickness_cm)

    ln_mag = np.log(np.clip(mag, 1e-12, None))
    baseline_ln_mag = _polynomial_baseline(freq, ln_mag, baseline_degree)
    residual_ln_mag = ln_mag - baseline_ln_mag
    filtered = _notch_periodic_residual(freq, residual_ln_mag, target_period_thz, n_harmonics, relative_bandwidth)
    corrected_mag = np.exp(baseline_ln_mag + filtered)

    baseline_phase = _polynomial_baseline(freq, phase, baseline_degree)
    residual_phase = phase - baseline_phase
    filtered_phase = _notch_periodic_residual(freq, residual_phase, target_period_thz, n_harmonics, relative_bandwidth)
    corrected_phase = baseline_phase + filtered_phase
    return pd.DataFrame({"freq": freq, "mag": corrected_mag, "phase": corrected_phase})
```

### Tier 2 — `known_film` (TMM, 대수적 나눗셈)

`mag`는 파워 투과율(`|t|²`)이라 factor 크기를 제곱해서 나누고, `phase`는 unwrap된
field 위상이라 factor의 unwrap 위상을 그대로 뺀다.

```python
def correct_fp_ringing_known_film(transmittance, n_film, k_film, thickness_um, n_substrate=1.0, k_substrate=0.0):
    freq = transmittance["freq"].to_numpy(dtype=float)
    mag = transmittance["mag"].to_numpy(dtype=float)
    phase = transmittance["phase"].to_numpy(dtype=float)

    fp_factor = film_fp_factor(freq, n_film, k_film, thickness_um, n_substrate, k_substrate)
    fp_mag = np.abs(fp_factor)
    fp_phase = np.unwrap(np.angle(fp_factor))

    corrected_mag = mag / np.clip(fp_mag**2, 1e-12, None)
    corrected_phase = phase - fp_phase
    return pd.DataFrame({"freq": freq, "mag": corrected_mag, "phase": corrected_phase})
```

### Tier 3 — `auto_calibrate` (TMM + ringing 최소화 피팅)

두께(옵션 n,k)를 `scipy.optimize.least_squares`로 조정해서 **보정 후 남은 ringing
전력만** 최소화한다 (전체 residual을 쓰면 진짜 공명까지 눌러버리는 걸 확인하고
고쳤음 — depth 오차가 135~216%까지 치솟는 걸 봤다). 목적함수는 두께에 대해
multi-modal이라 coarse grid로 먼저 시작점을 찾는다.

```python
def residual_for(params):
    if fit_n_k:
        d_um, n_film, k_film = params
    else:
        (d_um,) = params
        n_film, k_film = n_film_init, k_film_init
    corrected = correct_fp_ringing_known_film(transmittance, n_film, k_film, d_um, n_substrate, k_substrate)
    ln_mag = np.log(np.clip(corrected["mag"].to_numpy(), 1e-12, None))
    baseline = _polynomial_baseline(freq, ln_mag, baseline_degree)
    full_residual = ln_mag - baseline

    thickness_cm = d_um * 1e-4
    target_period_thz = SPEED_OF_LIGHT_CM_THZ / (2 * n_film * thickness_cm)
    target_quefrency = 1.0 / target_period_thz
    spectrum = np.fft.rfft(full_residual)
    quefrency = np.fft.rfftfreq(n_points, d=spacing)
    isolated = np.zeros_like(spectrum)
    band = np.abs(quefrency - target_quefrency) <= 0.15 * target_quefrency
    isolated[band] = spectrum[band]
    return np.fft.irfft(isolated, n_points)

# 두께 탐색 하한을 "몇 개 fringe가 실제로 보일 만큼"으로 강제 + coarse grid presearch
bandwidth_thz = float(freq.max() - freq.min())
min_resolvable_um = 3 * SPEED_OF_LIGHT_CM_THZ / (2 * n_film_init * bandwidth_thz) * 1e4
d_lower = max(thickness_um * 0.5, min_resolvable_um)
d_upper = max(thickness_um * 1.5, d_lower * 1.05)
grid_d = np.linspace(d_lower, d_upper, 61)
grid_cost = [float(np.sum(residual_for(...) ** 2)) for d in grid_d]
initial[0] = grid_d[int(np.argmin(grid_cost))]
fit = least_squares(residual_for, initial, bounds=(lower, upper))
```

`fit_n_k=True`는 두께-굴절률 degeneracy 때문에 부정확해질 수 있어(검증:
`d=150,n=1.6` 참값 → `d=132,n=1.41` 복원) 기본값은 `False`.

## 5. 직접 재현하는 검증 스크립트

아래 전체를 `verify_thin_film.py`로 저장하고 프로젝트 루트에서
`"./.venv/Scripts/python.exe" verify_thin_film.py`로 실행하면 이 문서의 모든
수치를 스스로 재현할 수 있다. (이 파일은 검증용이라 커밋 대상 아님 — 확인 후
지워도 됨.)

```python
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import functions as fn

freq = np.linspace(0.5, 2.5, 5)

# --- 성질 1: 계면이 없으면(n_substrate == n_film) ringing이 없어야 한다 ---
fp_no_interface = fn.film_fp_factor(freq, 1.6, 0.02, 150.0, n_substrate=1.6, k_substrate=0.02)
assert np.allclose(fp_no_interface, 1.0), "r12=0인데 ringing이 생기면 버그"
print("[OK] substrate == film 이면 factor == 1")

# --- 성질 2: n_substrate=1 이면 free-standing 특수식(1/(1-r01^2 e^{-2i delta}))과 일치해야 한다 ---
n_film, k_film, d_um = 1.6, 0.02, 150.0
n_tilde = n_film - 1j * k_film
r01 = (1 - n_tilde) / (1 + n_tilde)
delta = 2 * np.pi * freq * n_tilde * (d_um * 1e-4) / fn.SPEED_OF_LIGHT_CM_THZ
expected_free_standing = 1.0 / (1.0 - r01**2 * np.exp(-2j * delta))
actual = fn.film_fp_factor(freq, n_film, k_film, d_um, n_substrate=1.0, k_substrate=0.0)
assert np.allclose(actual, expected_free_standing), "n_substrate=1 특수해와 안 맞으면 버그"
print("[OK] n_substrate=1 -> free-standing 특수식과 일치")

# --- 성질 3: 흡수가 커지면 |factor| -> 1로 수렴해야 한다 (§1.2) ---
prev_spread = None
for k in [0.0, 0.5, 1.0, 3.0]:
    fp = fn.film_fp_factor(freq, n_film, k, d_um, n_substrate=3.42, k_substrate=0.0)
    spread = float(np.max(np.abs(fp)) - np.min(np.abs(fp)))
    assert np.max(np.abs(fp)) < 5, f"k_film={k}에서 발산하면 부호 버그"
    if prev_spread is not None:
        assert spread <= prev_spread + 1e-9, f"흡수가 늘었는데 ringing이 커지면 버그 (k={k})"
    prev_spread = spread
fp_high_k = np.abs(fn.film_fp_factor(freq, n_film, 3.0, d_um, n_substrate=3.42, k_substrate=0.0))
assert np.allclose(fp_high_k, 1.0, atol=1e-3), "흡수 큰데 factor가 1로 안 가면 버그"
print("[OK] k_film 증가 -> ringing 진폭 단조 감소, |factor| -> 1")


def polynomial_residual_rms(freq, values, degree=3):
    baseline = fn._polynomial_baseline(freq, values, degree)
    return float(np.std(values - baseline))


def main():
    freq = np.linspace(0.2, 3.0, 2000)
    f0, resonance_width, resonance_depth = 1.5, 0.03, 0.05
    background_mag = 0.55 + 0.05 * np.sin(2 * np.pi * freq / 6.0)
    true_mag = background_mag - resonance_depth * np.exp(-0.5 * ((freq - f0) / resonance_width) ** 2)
    true_phase = 0.4 * freq + 0.03 * np.sin(2 * np.pi * freq / 4.0)
    t_true = np.sqrt(true_mag) * np.exp(1j * true_phase)

    n_film, k_film, d_film_um = 1.6, 0.02, 150.0
    n_sub, k_sub = 3.42, 0.0
    fp = fn.film_fp_factor(freq, n_film, k_film, d_film_um, n_sub, k_sub)
    t_meas = t_true * fp
    measured_mag = np.abs(t_meas) ** 2
    measured_phase = np.unwrap(np.angle(t_meas))
    transmittance = pd.DataFrame({"freq": freq, "mag": measured_mag, "phase": measured_phase})

    true_rms = polynomial_residual_rms(freq, np.log(true_mag))
    measured_rms = polynomial_residual_rms(freq, np.log(measured_mag))

    def depth_at(mag):
        baseline = fn._polynomial_baseline(freq, mag, 3)
        window = np.abs(freq - f0) < resonance_width
        return float(np.max(baseline[window] - mag[window]))

    true_depth = depth_at(true_mag)

    results = {
        "Tier 1 spectral_notch": fn.remove_fp_ringing_spectral_notch(transmittance, d_film_um, n_film),
        "Tier 2 known_film (WRONG: air substrate)": fn.correct_fp_ringing_known_film(transmittance, n_film, k_film, d_film_um),
        "Tier 2 known_film (correct substrate)": fn.correct_fp_ringing_known_film(transmittance, n_film, k_film, d_film_um, n_sub, k_sub),
    }
    corrected3, info3 = fn.calibrate_fp_ringing(
        transmittance, thickness_um=100.0, n_film_init=n_film, k_film_init=k_film,
        n_substrate=n_sub, k_substrate=k_sub, fit_n_k=False,
    )
    results["Tier 3 auto_calibrate (wrong initial d)"] = corrected3

    print(f"\ntrue ringing RMS={true_rms:.5f}  measured ringing RMS={measured_rms:.5f}  true depth={true_depth:.5f}")
    print(f"Tier 3 recovered thickness: {info3['thickness_um']:.3f} um (true {d_film_um})")
    print(f"\n{'method':40s}{'ringing RMS':>12s}{'ringing cut':>12s}{'depth':>10s}{'depth err':>12s}")
    for name, corrected in results.items():
        mag = corrected["mag"].to_numpy()
        rms = polynomial_residual_rms(freq, np.log(np.clip(mag, 1e-12, None)))
        depth = depth_at(mag)
        print(f"{name:40s}{rms:12.5f}{1 - rms / measured_rms:11.1%}{depth:10.5f}{abs(depth - true_depth) / true_depth:11.1%}")


if __name__ == "__main__":
    main()
```

기대 출력(참고용 — 정확히 같은 소수점까지 나올 필요는 없고, **부호/방향**이
같으면 됨): Tier 2(correct substrate)는 ringing cut 약 88%, depth err 0%에 가까워야
하고, Tier 2(WRONG: air substrate)는 ringing cut이 **음수**(즉 ringing이 더
심해짐)로 나와야 한다 — 이게 §1.1에서 고친 버그가 진짜 위험했다는 재현.

## 6. GUI 사용법 (참고, `functions.py`와 무관한 부분)

`thickness_mode`를 `thin (film)`으로, **FP removal (thin only)** 콤보박스에서
`none`/`spectral_notch`/`known_film`/`auto_calibrate` 선택. `film n/k (guess)`에
필름 굴절률, **`substrate n/k`에 실제 substrate 재질의 굴절률**(기본값 Si
3.42/0.0, SiO2면 ~1.95–2.1로 변경) 입력. `auto_calibrate` 체크박스는 두께 외
n,k도 같이 피팅할지 여부(기본 꺼짐). `thick` 모드에서 FP removal을 켜면 에러.

## 7. 아직 확인 못 한 것 / 다음 단계

- **실제 substrate+Au(+film) 데이터로 GUI를 열어서 눈으로 ringing이 줄어드는지
  확인 안 했다.** 지금까지는 전부 합성 데이터 + 코드 직접 호출 검증뿐이다.
- §1.2의 부호 수정은 아직 커밋 안 됨 (`git diff -- functions.py`로 확인 가능).
- `fitting_gui.py`에서 raw vs FP-corrected 비교 기능, substrate 광학상수 자동
  로드, Tier 1 notch 폭 GUI 노출 — 전부 이번 범위 밖.
