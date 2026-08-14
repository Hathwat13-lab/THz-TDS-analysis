# Thin-film FP ringing 제거 가이드

## 구현 범위

이번 작업은 **`thickness_mode = "thin (film)"`일 때 transmittance에 남는 Fabry–Pérot
(FP) 링잉을 제거하는 기능**만 구현한다. 그 외의 것은 이번 변경에 포함되지 않는다:

- 메타표면(패턴된 Au)의 공명 응답 자체를 TMM/RCWA로 시뮬레이션하는 기능은 없다.
  기존 워크플로처럼 "substrate+Au"를 reference로, "substrate+Au+film"을 sample로
  측정한 비율을 그대로 쓴다 — 메타표면 응답은 측정으로 이미 반영되어 있으므로
  모델링 대상이 아니다.
- substrate(SiO2, Si)의 광학상수를 새로 추출하는 기능은 없다. **다만 substrate의
  n, k는 입력값으로 반드시 필요하다** (아래 버그/수정 참고) — 대략적인 문헌값을
  직접 입력해야 한다.
- `n(ω), k(ω)`를 자유형(free-form)으로 역산하는 기능은 **의도적으로** 넣지 않았다
  (아래 "왜 자유형 n,k 역산을 안 했는가" 참고).

즉 이번 기능은 "필름 자신의 내부 반사가 만드는 주기적 ripple을 transmittance에서
제거"하는 것이 전부다.

## 발견하고 고친 버그: substrate를 공기로 취급했었음

**Tier 2/3(`known_film`, `auto_calibrate`)가 TMM에 기반한 건 맞지만, 처음 구현에서
필름을 "공기 중에 떠 있는 필름"으로 취급하는 실수를 했다.** 실제 샘플은 필름이
공기 위가 아니라 substrate(Si, SiO2) 위에 있으므로 이건 잘못된 경계조건이다.

표준 단일층 TMM 공식(medium 0=공기, 1=필름, 2=substrate)은

```
t = t01 * t12 * exp(iδ) / (1 + r01*r12*exp(2iδ))
r01 = (n0-ñ1)/(n0+ñ1),  r12 = (ñ1-n2)/(ñ1+n2)
```

인데, 처음 구현에서는 `r12 = -r01`(즉 n2=n0=공기인 경우에만 성립하는 특수해)로
암묵적으로 가정해서 `1/(1 - r01²·e^{2iδ})`를 썼다. `n2`(substrate)가 공기가 아니면
`r12 ≠ -r01`이라 이 식 자체가 틀린다. GUI에도 substrate n/k를 입력할 곳이 아예
없었다 — 물리 파라미터가 통째로 빠진 상태였다.

**고친 후 (`functions.py`, 실제 코드):**

```python
def film_fp_factor(frequency_thz, n_film, k_film, thickness_um, n_substrate=1.0, k_substrate=0.0):
    thickness_cm = thickness_um * 1e-4
    n_tilde = n_film - 1j * k_film
    n_sub_tilde = n_substrate - 1j * k_substrate
    r01 = (1 - n_tilde) / (1 + n_tilde)
    r12 = (n_tilde - n_sub_tilde) / (n_tilde + n_sub_tilde)
    delta = 2 * np.pi * frequency_thz * n_tilde * thickness_cm / SPEED_OF_LIGHT_CM_THZ
    return 1.0 / (1.0 + r01 * r12 * np.exp(2j * delta))
```

`n_substrate=1.0`(공기)이면 예전 공식으로 정확히 환원되니, 자유필름(free-standing)
케이스는 그대로 특수해로 남아있다. `correct_fp_ringing_known_film`,
`calibrate_fp_ringing`, `analyze_pair()`, GUI까지 전부 `n_substrate`/`k_substrate`를
받아서 여기로 흘려보내도록 고쳤다. GUI에는 **substrate n / substrate k** 입력칸이
새로 생겼고 기본값은 Si(3.42, 0.0) — SiO2를 쓰면 ~1.95–2.1로 바꿔줘야 한다.

**이 버그가 실제로 얼마나 위험했는지 (합성 데이터로 재현)**: 필름을 Si 기판(n=3.42)
위에 올려놓고 ringing을 주입한 뒤, (a) 예전처럼 공기로 잘못 가정하고 보정 vs
(b) 올바른 substrate 인덱스로 보정을 비교:

```
method                                          ringing RMS  ringing cut   res. depth  depth err
Tier 2 known_film (WRONG: air substrate)            0.23032      -64.0%     -0.09606     308.0%
Tier 2 known_film (correct substrate)               0.01120       92.0%      0.04618       0.0%
```

잘못된 가정으로 보정하면 ringing이 **줄어드는 게 아니라 64% 더 커지고**, 공명
depth 오차는 308%(부호까지 반전)로 완전히 망가진다 — "근본적으로 위험하다"는
지적이 정확했다. Substrate 인덱스를 올바르게 주면 ringing 92% 감소, depth 오차
0%로 정상 동작한다.

## 왜 자유형 n,k 역산을 안 했는가

패턴된 Au 공진기의 공명(Rabi splitting)은 필름이 그 위에 올라가면서 생기는
**진짜 신호**이지, 제거해야 할 노이즈가 아니다. 만약 각 주파수에서 자유롭게
`n(ω), k(ω)`를 풀어버리면, 그 역산 과정이 공진기의 공명 자체를 "필름의 굴절률
구조"로 오인해서 흡수해버릴 위험이 크다. 그래서 세 가지 방법 모두
**필름 자신의 FP factor만**을 대상으로 하고, 그 외의 스펙트럼 구조(공명 포함)는
건드리지 않도록 설계했다.

## 세 가지 방법 (rigor 순서)

공통 물리 primitive는 위 `functions.film_fp_factor`. 세 방법 모두 이 factor를
어떻게 얻고 어떻게 나눠주는지에서만 차이가 난다.

### Tier 1 — `spectral_notch` (모델 불필요, TMM 아님)

**이 방법은 TMM과 무관하다** — 순수 신호처리다. 이미 계산된 transmittance의
`ln(mag)`와 `phase`에 저차 다항식 baseline을 맞추고, 남은 residual을 주파수축
FFT("quefrency" 도메인)에서 봤을 때 필름의 예상 round-trip 주파수
(`Δf ≈ c / (2 n d)`) 근처만 notch로 제거한다. substrate 정보 없이 두께 어림값만
있으면 된다 (ringing의 "주기"는 substrate와 무관하게 필름 자신의 왕복 광로만으로
정해지기 때문). 가장 가볍지만 무딘 도구 — notch 폭이 넓으면 진짜 공명도 같이
깎여나간다 (검증 스크립트 기준 약 19~36% 정도 depth 손실).

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

### Tier 2 — `known_film` (TMM, 알려진 필름+substrate 파라미터로 대수적 보정)

필름의 n, k, 두께, 그리고 **substrate의 n, k**가 어느 정도 알려져 있다는
전제(문헌값, ellipsometry, 또는 별도 측정)로 `film_fp_factor`를 계산해서 측정된
복소 transmission에서 바로 나눠버린다. `mag`는 파워 투과율(`|t|^2`)이라 factor의
크기는 제곱해서 나누고, `phase`는 unwrap된 field 위상이라 factor의 unwrap된
위상을 그대로 빼준다. 피팅이 아니라 대수적 나눗셈이라 빠르고 결정론적이다.

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

파라미터(필름과 substrate 둘 다)가 정확하면 ringing을 거의 완벽하게 제거한다
(검증 결과 참고). substrate를 틀리면 위 "버그" 절에서 보듯 결과가 오히려 나빠질
수 있으므로 **아무 값이나 넣지 말고 실제 substrate 재질에 맞는 값을 넣어야 한다.**

### Tier 3 — `auto_calibrate` (TMM + ringing 최소화로 두께 자동 보정)

Tier 2와 같은 공식을 쓰되(substrate n,k는 여전히 입력값으로 고정), 필름 두께
(옵션으로 n,k도)를 `scipy.optimize.least_squares`로 조정해서 **보정 후 남은
ringing 전력을 최소화**한다. 어떤 "정답" 곡선에 맞추는 게 아니라 ringing 자체를
줄이는 방향으로 캘리브레이션하는 것이므로, 필름 FP 주기와 무관한 진짜 공명을
지우거나 만들어내지 않는다.

구현상 중요한 디테일 두 가지 (둘 다 검증 스크립트 실행 중 실제로 실패를 보고
고친 것):

1. **목적함수는 residual 전체가 아니라 후보 두께가 만드는 FP quefrency 근처
   대역의 전력만** 최소화한다 (`band = |quefrency - target_quefrency| <= 0.15 *
   target_quefrency`). 처음엔 residual 전체(`ln_mag - baseline`)를 그대로
   least_squares에 넘겼는데, 그러면 옵티마이저가 진짜 공명(주기적이지 않은
   feature)까지 납작하게 눌러서 depth 오차가 135~216%까지 치솟는 걸 확인했다.
2. **두께 탐색 범위 하한을 "몇 개 fringe가 실제로 보일 만큼" 강제**하고
   (`min_resolvable_um`), 그 범위 안에서 **coarse grid로 먼저 최적 시작점을
   찾은 뒤** local least_squares를 돌린다. 이 목적함수는 두께에 대해
   multi-modal(여러 local minimum)이라 grid presearch 없이 gradient 기반
   local optimizer만 돌리면 엉뚱한 두께에 갇히는 것도 확인했다.

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

# resolvability floor + coarse grid presearch before least_squares:
bandwidth_thz = float(freq.max() - freq.min())
min_resolvable_um = 3 * SPEED_OF_LIGHT_CM_THZ / (2 * n_film_init * bandwidth_thz) * 1e4
d_lower = max(thickness_um * 0.5, min_resolvable_um)
d_upper = max(thickness_um * 1.5, d_lower * 1.05)
grid_d = np.linspace(d_lower, d_upper, 61)
grid_cost = [float(np.sum(residual_for(...) ** 2)) for d in grid_d]
initial[0] = grid_d[int(np.argmin(grid_cost))]
fit = least_squares(residual_for, initial, bounds=(lower, upper))
```

`fit_n_k=True`로 n,k까지 같이 풀면 두께-굴절률 사이에 degeneracy가 있어 정확도가
떨어진다 (검증: 참값 `d=150, n=1.6` → 복원값 `d=132, n=1.41`). 기본값은
`False`(두께만 보정)이고, 이쪽이 더 안정적이다.

### 파이프라인 연결부 — `AsymmetricTDSAnalyzer.analyze_pair()`

기존 naive transmittance 계산(`transmittance_mag`, `phase_difference`)은 그대로
두고, `thin` 모드 + `fp_removal_method != "none"`일 때만 위 세 함수 중 하나로
`transmittance`를 덮어쓴다. `thick` 모드는 이 블록에 아예 들어가지 않는다:

```python
fp_removal_info = None
fp_method = fp_removal_method.strip().lower()
if mode == "thin" and fp_method != "none":
    thickness_um = thickness_cm * 1e4
    if fp_method == "spectral_notch":
        transmittance = remove_fp_ringing_spectral_notch(transmittance, thickness_um, film_n_guess)
        fp_removal_info = {"method": fp_method, "thickness_um": thickness_um, "n_film_guess": film_n_guess}
    elif fp_method == "known_film":
        transmittance = correct_fp_ringing_known_film(
            transmittance, film_n_guess, film_k_guess, thickness_um,
            n_substrate=substrate_n, k_substrate=substrate_k,
        )
        fp_removal_info = {"method": fp_method, "thickness_um": thickness_um, "n_film": film_n_guess,
                            "k_film": film_k_guess, "n_substrate": substrate_n, "k_substrate": substrate_k}
    elif fp_method == "auto_calibrate":
        transmittance, calibration = calibrate_fp_ringing(
            transmittance, thickness_um, film_n_guess, film_k_guess,
            n_substrate=substrate_n, k_substrate=substrate_k, fit_n_k=fit_film_n_k,
        )
        fp_removal_info = {"method": fp_method, **calibration}
    else:
        raise ValueError("fp_removal_method must be one of 'none', 'spectral_notch', 'known_film', 'auto_calibrate'.")
    transmittance_mag = transmittance["mag"].to_numpy()
    phase_difference = transmittance["phase"].to_numpy()

alpha, k, n = compute_optical_constants(sample_crop.frequency, transmittance_mag, phase_difference, thickness_cm)
```

`thick` 모드에서는 `mode == "thin"`이 거짓이라 이 블록이 전혀 실행되지 않으므로
기존 pellet 경로는 코드 레벨에서 그대로 보존된다. 모든 tier 함수 호출은 **키워드
인자로만** 넘긴다 — 뒤에 substrate 파라미터를 끼워 넣으면서 위치 인자로 부르면
엉뚱한 파라미터에 값이 바인딩되는 사고가 실제로 날 뻔했다 (`calibrate_fp_ringing`
호출부에서 한 번 발견하고 고쳤다).

## GUI 사용법 (FFT(multi).py)

`thickness_mode`를 `thin (film)`으로 두고, **FP removal (thin only)** 콤보박스에서
`none` / `spectral_notch` / `known_film` / `auto_calibrate` 중 선택한다.
`film n (guess)` / `film k (guess)`에 필름의 대략적인 굴절률을, **`substrate n` /
`substrate k`에 실제 substrate 재질의 굴절률**을 입력한다 (기본값은 Si 기준
3.42/0.0 — SiO2면 ~1.95–2.1로 바꿔야 함). `spectral_notch`는 substrate 값을 쓰지
않는다. `auto_calibrate` 아래 체크박스를 켜면 두께뿐 아니라 n,k도 같이 피팅한다
(기본은 꺼짐, degeneracy 때문). `thick (pellet)` 모드에서 FP removal을 `none`이
아닌 값으로 두면 Run Analysis가 에러를 낸다 — thin 모드 전용 기능이기 때문. 결과
요약 텍스트에 어떤 방법과 파라미터가 쓰였는지(`FP removal: ...`) 표시된다.
`fitting_gui.py`와 local-maxima 창은 수정하지 않았다 — 둘 다 `result.transmittance`를
그대로 받아쓰는 구조라서, FP 보정이 적용되면 그 보정된 곡선을 그대로 넘겨받는다.

## 검증

자동화된 테스트 스위트가 프로젝트에 없어서, 아래 세 가지로 확인했다 (스크립트
자체는 프로젝트 밖 scratchpad에 있어 커밋에는 포함되지 않음 — 재현하려면 같은
구조의 스크립트를 직접 짜서 돌리면 됨. `functions.py`/`FFT(multi).py`만 커밋 대상).

### 1. 합성 데이터 검증 (물리 로직이 맞는지, substrate 버그 수정 반영)

필름이 Si 기판(n=3.42) 위에 있다고 가정하고, 알려진 FP factor를 주입한 합성
transmittance(부드러운 배경 + 좁은 "공명" 하나 + FP ringing)에 세 방법을 적용해서
(a) ringing이 실제로 줄어드는지 (b) 공명이 smear되지 않는지 확인한다. 최종 실행
결과:

```
true (no-ringing) ln-mag residual RMS  : 0.01120
measured (ringing) ln-mag residual RMS : 0.14048
injected resonance depth (ground truth): 0.04618

method                                            ringing RMS  ringing cut   res. depth  depth err
Tier 1 spectral_notch                                 0.05612       60.1%      0.05495     19.0%
Tier 2 known_film (WRONG: air substrate)              0.23032      -64.0%     -0.09606    308.0%
Tier 2 known_film (correct substrate)                 0.01120       92.0%      0.04618      0.0%
Tier 3 auto_calibrate (wrong initial d, correct sub)  0.01109       92.1%      0.04528      2.0%
```

Tier 2(정확한 substrate 포함 파라미터)는 ringing을 92% 줄이고 공명 depth 오차
0% — 물리 공식 자체가 맞다는 증거. substrate를 공기로 잘못 가정하면(고치기 전
버전) 오히려 ringing이 64% 늘고 depth가 308% 어긋난다 — 실제로 위험했다는 걸
숫자로 확인. Tier 3은 두께를 33% 틀리게 준 초기값(100 µm, 참값 150 µm)에서도
149.9 µm로 수렴해서 Tier 2와 거의 동일한 결과에 도달. Tier 1은 ringing을 60% 줄이지만
공명 depth를 19% 깎아먹는다 — 버그가 아니라 "모델 없이 순수 주기성만으로 구분"하는
방식의 근본적 트레이드오프. `fit_n_k=True`(두께+n+k 동시 피팅)는 참값
`d=150,n=1.6`에서 `d=132,n=1.41`로 부정확하게 수렴 — degeneracy 때문에 기본값을
`False`로 둔 이유. 1% 노이즈를 섞어도 세 방법 모두 합리적으로 동작.

### 2. 회귀 확인 (기존 thick pellet 경로가 안 깨졌는지)

```python
result_thick = analyzer.analyze_pair(sample_df, reference_df, ..., thickness_mode='thick')
assert result_thick.fp_removal_info is None
assert result_thick.echo_guideline is not None   # 기존 로직 그대로 동작

result_thin_none = analyzer.analyze_pair(sample_df, reference_df, ..., thickness_mode='thin', fp_removal_method='none')
assert result_thin_none.fp_removal_info is None   # 새 코드 경로 자체가 실행 안 됨
```

실제 실행 결과: `thick-mode regression: OK`, `thin/none regression: OK`. substrate
파라미터를 포함한 세 가지 새 방법도 `analyze_pair()`를 통해 end-to-end로 호출해서
예외 없이 finite한 값을 반환하고 `fp_removal_info`에 `n_substrate`/`k_substrate`가
정확히 기록되는지 확인했다. 잘못된 `fp_removal_method` 문자열을 주면 `ValueError`로
명확히 실패하는 것도 확인했다.

### 3. 정적/구성 검증

```
"./.venv/Scripts/python.exe" -m py_compile "FFT(multi).py" functions.py fitting_gui.py "history/FFT(thick).py"
```

전부 컴파일 통과. `history/FFT(thick).py`도 `functions.py`를 import하지만
`analyze_pair()`를 전부 keyword 인자로 호출하므로 이번에 뒤쪽에 default 인자로만
추가된 파라미터들과 충돌하지 않는다 (직접 확인함). `FFT(multi).py`는 headless로
`FFTPlatformGUI()`를 생성/파괴해서 새 콤보박스·substrate 입력칸·StringVar 초기화가
예외 없이 동작하는 것까지 확인했다. **실제 tkinter 창을 띄워 클릭해보는 대화형
테스트, 그리고 실제 substrate+Au(+film) 데이터로 ringing이 눈으로 줄어드는지는
아직 확인 못 했다** — 다음 단계로 남아 있음.

## 다음에 고려할 만한 것 (이번 범위 밖)

- `fitting_gui.py`에서 raw vs FP-corrected 곡선을 겹쳐서 비교하는 기능.
- substrate 광학상수를 별도 측정에서 자동으로 불러오는 기능 (지금은 수동 입력).
- Tier 1의 notch 폭을 GUI에서 조절 가능하게 노출.
