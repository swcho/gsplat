# 3DGS 색 공식의 `+0.5` 오프셋은 어떤 역할을 하는가?

> **답**: 계수가 전부 0일 때 색이 검정이 아니라 **중간 회색(0.5)**이 되게 한다. 덕분에 계수의 부호가 양·음 대칭으로 움직일 수 있어 학습이 안정된다.

3DGS의 Gaussian $i$ 색은

$$
\mathbf c_i(\mathbf d) = \max\!\Big(0,\ \underbrace{\sum_{k=0}^{15} \mathbf c_{i,k}\,Y_k(\mathbf d)}_{\text{SH 합}} + 0.5\Big)
$$

로 계산된다(노트북 2절 "3DGS에서의 역할"). 이 중 **`+0.5`**는 SH 이론에서 나온 항이 아니라 순전히 *학습을 위한 파라미터화 장치*다. 아래 다섯 가지 관점에서 정리한다.

---

## 1. 오프셋이 없다면 — "0 초기화 = 검정"이라는 문제

계수를 학습할 때 가장 자연스러운 초기값은 0이다(고차 계수 `shN`은 실제로 0으로 시작한다). 오프셋이 없으면

$$
\mathbf c(\mathbf d) = \max(0,\ \textstyle\sum_k \mathbf c_k Y_k(\mathbf d)) \xrightarrow{\ \mathbf c_k = 0\ } \max(0, 0) = 0 \quad (\text{검정})
$$

즉 **"아무것도 모르는 상태"가 곧 가장 어두운 색**이다. 이것은 두 가지 문제를 낳는다.

- **초기 상태가 [0,1] 색 범위의 한쪽 끝에 치우쳐 있다.** 어떤 색이든 밝게 만들려면 계수를 양의 방향으로만 키워야 하고, 어둡게 만드는 방향으로는 갈 곳이 없다.
- **`clamp(0)`의 dead zone에 바로 걸쳐 있다.** $\max(0, x)$는 $x<0$에서 기울기가 0이다. 초기값 $x=0$ 근방에서 SH 합이 살짝 음수가 되면(고차 항의 ringing만으로도 쉽게 발생) 그 방향에서는 gradient가 0이 되어 **색이 영원히 검정에 고정**될 수 있다. ReLU 신경망의 "dying ReLU"와 같은 현상이다.

## 2. `+0.5`를 두면 — 파라미터 공간이 색 범위의 중앙에 놓인다

오프셋을 두면 "계수 전부 0"이 **중간 회색 0.5**에 대응한다. 그러면

- 색 범위 [0,1]의 **정중앙**에서 출발하므로 밝아지는 방향(+)과 어두워지는 방향(−)이 **대등**하다. 계수의 부호가 자연스럽게 양·음 대칭으로 분포한다.
- 초기값이 `clamp(0)` 경계에서 0.5만큼 떨어져 있어 dead zone에 빠질 여유가 생긴다.
- Adam과 같이 스텝 크기가 gradient 부호에 의해 결정되는 옵티마이저는 파라미터가 0을 중심으로 대칭일 때 가장 잘 동작한다.

이것은 신경망에서 **입력을 zero-centering(평균 0)** 하거나 출력에 bias를 두어 활성 함수의 "좋은 영역"에서 시작하게 하는 것과 정확히 같은 발상이다. 색이 $[0,1]$이므로 그 중앙 0.5를 기준점(bias)으로 삼은 것뿐이다.

### 초기화 식 `sh0 = (rgb − 0.5) / C0`가 여기서 나온다

학습 시작 시 고차 계수를 0으로 두면 색은 방향과 무관하게 $c_0 Y_0^0 + 0.5$다. 이것이 SfM 포인트 색 $\mathbf{rgb}$와 같아야 하므로

$$
c_0 = \frac{\mathbf{rgb} - 0.5}{Y_0^0} = \frac{\mathbf{rgb} - 0.5}{C_0},\qquad C_0 = \tfrac{1}{2\sqrt\pi} \approx 0.2821
$$

노트북 3절의 셀이 이를 그대로 확인한다.

```python
rgb = torch.tensor([[0.8, 0.3, 0.1], [0.1, 0.6, 0.9]], device=DEVICE)      # SfM 색 2개
sh0 = (rgb - 0.5) / C0                                                     # [N,3]  DC 계수
shN = torch.zeros(2, 15, 3, device=DEVICE)                                 # 고차 계수는 0으로 시작
coeffs = torch.cat([sh0[:, None], shN], dim=1)                             # [N,16,3]

some_dir = F.normalize(torch.randn(2, 3, device=DEVICE), dim=-1)           # 아무 방향
color = torch.clamp_min(torch.einsum("nk,nkc->nc", sh_bases(some_dir, 3), coeffs) + 0.5, 0.0)
# 복원된 색 == rgb (방향 무관)
```

`rgb − 0.5`를 빼는 이유가 바로 "$+0.5$가 나중에 더해질 것을 미리 상쇄"하기 위함이다. 회색(0.5)인 점은 `sh0 = 0`으로 시작하고, 밝은 점은 양의 DC, 어두운 점은 음의 DC를 갖게 된다 — 부호가 대칭으로 쓰이는 것이 초기화 시점부터 드러난다.

## 3. 수학적으로는 표현력이 같다 — 파라미터화(reparameterization)의 차이일 뿐

$Y_0^0 = C_0$는 방향과 무관한 상수이므로, 상수 0.5는 언제든 DC 계수에 흡수할 수 있다.

$$
0.5 = \frac{0.5}{C_0}\,Y_0^0
\quad\Longrightarrow\quad
\sum_k c_k Y_k + 0.5 = \Big(c_0 + \frac{0.5}{C_0}\Big) Y_0^0 + \sum_{k\ge1} c_k Y_k
$$

즉 "오프셋 있는 모델"과 "오프셋 없는 모델"은 **같은 함수 집합**을 표현한다. 차이는 오직 *같은 색을 어떤 파라미터 값으로 표현하는가*(좌표계의 원점이 어디인가)이고, 그것이 최적화의 동역학(초기값 위치, dead zone 거리, 부호 대칭)만 바꾼다.

노트북 4.2절의 Adam 학습 셀은 이 등가성을 실제로 사용한다. `+0.5`가 포함된 식으로 학습한 뒤, 오프셋 없는 최소제곱 해와 비교하기 위해 상수를 DC로 옮긴다.

```python
pred = torch.clamp_min(A_obs[:, :Kd] @ coeffs[:Kd] + 0.5, 0.0)   # 학습식 (3DGS와 동일)
...
coeffs_gd = torch.cat([sh0, shN], dim=0).detach().clone()
coeffs_gd[0] += 0.5 / C0   # 학습식의 +0.5 오프셋을 DC 계수로 흡수해 f*와 직접 비교 (0.5 = (0.5/C0)·Y₀⁰)
coeffs_ls = fit_lstsq(obs_d, obs_c, 3)
```

이 한 줄(`coeffs_gd[0] += 0.5 / C0`)이 "오프셋은 DC 계수의 재파라미터화일 뿐"이라는 사실의 실행 가능한 증명이다.

## 4. 코드에서 오프셋이 적용되는 정확한 위치와 순서

순서는 항상 **SH 합 → `+0.5` → `clamp_min(0)`** 이다. 오프셋을 clamp *뒤에* 더하면 음수 색 방지와 dead zone 회피 효과가 사라지므로 순서가 중요하다.

**원본 3DGS (Inria, `cuda_rasterizer/forward.cu`의 `computeColorFromSH`)**

```cpp
glm::vec3 result = SH_C0 * sh[0];             // SH_C0 = 0.28209479177387814
if (deg > 0) { ... result += ...; }           // 1~3차 항 누적
result += 0.5f;                               // (1) 오프셋
clamped[...] = (result.x < 0); ...            // (2) 어느 채널이 잘렸는지 기록 → backward에서 gradient 0
return glm::max(result, 0.0f);                // (3) clamp
```

**gsplat (`gsplat/rendering.py`, `_maybe_evaluate_sh`)**

```python
features = spherical_harmonics(sh_degree, means, viewmats, features, masks=masks)  # [..., C, N, D]  SH 합
if clamp:
    # make it apple-to-apple with Inria's CUDA Backend.
    features = torch.clamp_min(features + 0.5, 0.0)   # +0.5 → clamp_min(0)
else:
    features = features + 0.5                          # +0.5 만
```

호출부(`_rasterization`)에서 **색(`colors`)은 `clamp=True`**, **`extra_signals`(깊이·법선 등 부가 신호)는 `clamp=False`**로 넘긴다. 부가 신호는 음수가 의미 있을 수 있으니 오프셋만 더하고 자르지 않는다. 같은 파일 상단의 CUDA 융합 경로 코드표(`_POST_CODE = {"none": 0, "shift": 1, "shift_relu": 2}`)도 동일한 세 단계를 `none / +0.5 / max(x+0.5, 0)`으로 이름 붙이고 있다.

참고로 gsplat은 파라미터를 `sh0`(DC, [N,1,3])와 `shN`(나머지 15개, [N,15,3])으로 나누고 `shN`의 학습률을 1/20로 둔다. 초기화 `sh0 = (rgb − 0.5)/C0`, `shN = 0`은 위 2절의 식 그대로다.

## 5. `clamp_min(0)`와의 상호작용

`+0.5`와 `clamp_min(0)`는 짝으로 동작한다.

- **음수 색 방지**: SH는 유한 차수의 다항식 근사이므로 밝은 봉우리(하이라이트) 주변에 **음의 물결(ringing)**이 생긴다(노트북 1절 하늘+태양 예제). 색이 음수면 알파 블렌딩에서 물리적으로 말이 안 되므로 0에서 잘라야 한다.
- **링잉이 잘리는 효과**: 오프셋 덕분에 기준선이 0.5에 있으므로, 진폭 0.5 이하의 ringing은 clamp에 걸리지 않고 그대로 표현된다. 반대로 매우 어두운 영역(색 ≈ 0)의 하이라이트 주변 음의 물결은 0에서 잘려 **어두운 배경 위 하이라이트가 더 깨끗하게** 보인다. 이것은 부수 효과이며, 잘린 채널에서는 gradient가 0이 되므로(Inria 코드의 `clamped` 플래그, PyTorch에서는 autograd가 자동 처리) 그 방향의 학습은 멈춘다.
- **dead zone과의 거리**: 1절에서 본 것처럼 오프셋이 없으면 초기 상태가 clamp 경계 위에 놓인다. `+0.5`는 초기 상태를 경계에서 0.5만큼 밀어내어, 학습 초기에 "어두운 채널이 clamp에 걸려 영원히 검정"이 되는 사고를 크게 줄인다.

---

## 한 줄 정리

`+0.5`는 SH 수학과 무관한 **학습용 bias**다. (a) "계수 0 = 회색"이 되어 파라미터가 [0,1] 중앙에서 양·음 대칭으로 움직이고, (b) `clamp_min(0)`의 dead zone에서 안전 거리를 확보하며, (c) DC 계수에 흡수 가능하므로 표현력은 전혀 잃지 않는다. 적용 순서는 코드 어디서나 **SH 합 → +0.5 → clamp_min(0)** 이고, 초기화 `sh0 = (rgb − 0.5)/C0`는 이 오프셋을 미리 상쇄하기 위한 식이다.

## 참고

- 노트북 `sh_walkthrough.py` 2절(3DGS 색 공식), 3절(DC 계수, `(rgb − 0.5)/C0`), 4.2절(`coeffs_gd[0] += 0.5 / C0`)
- `gsplat/rendering.py` — `_maybe_evaluate_sh`, `_POST_CODE`, `_rasterization`의 `colors`/`extra_signals` 호출부
- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH 2023) 공개 코드 `cuda_rasterizer/forward.cu` `computeColorFromSH`
