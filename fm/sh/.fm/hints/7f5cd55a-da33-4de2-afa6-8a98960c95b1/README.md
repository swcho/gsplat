# "SH 평가(SH evaluation)"란 무엇인가?

> **한 줄 답**: 계수 $\mathbf c_k$와 방향 $\mathbf d$가 주어졌을 때 **함수값을 계산하는 것**, 즉 급수
> $\mathbf c(\mathbf d)=\sum_k \mathbf c_k\,Y_k(\mathbf d)$를 실제로 더하는 일이다.
> 함수에서 계수를 뽑아내는 **사영(projection)** 의 역방향 연산이다.

---

## 1. 사영(분석) vs 평가(합성) — 쌍으로 이해하기

SH를 다룰 때는 항상 서로 반대 방향인 두 연산이 짝을 이룬다.

| | 사영 (projection, **analysis**) | 평가 (evaluation, **synthesis**) |
|---|---|---|
| 입력 | 구면 위 함수 $f(\mathbf d)$ 전체 | 계수 $\{\mathbf c_k\}$ + 방향 $\mathbf d$ 하나 |
| 출력 | 계수 $\mathbf c_k$ (K개) | 함수값 $\mathbf c(\mathbf d)$ (한 점) |
| 식 | $\mathbf c_k=\displaystyle\int_{S^2} f(\mathbf d)\,Y_k(\mathbf d)\,d\Omega$ | $\mathbf c(\mathbf d)=\displaystyle\sum_{k=0}^{K-1}\mathbf c_k\,Y_k(\mathbf d)$ |
| 연산 성격 | 적분(구면 전체를 훑음) — 비싸다 | 유한합(K항) — 싸다 |
| 노트북 대응 | `project_to_sh(f, B, w)` | `reconstruct(...)`, `sh_eval(coeffs, dirs, degree)` |

**푸리에 변환 비유**가 가장 직관적이다. 1차원 주기 신호에서

- 푸리에 **정변환**(분석): 신호 $x(t)$ → 계수 $X_n=\frac1T\int x(t)e^{-i n\omega t}dt$
- 푸리에 **역변환**(합성): 계수 $X_n$ → 신호 $x(t)=\sum_n X_n e^{i n\omega t}$

SH는 "구면 위의 푸리에 급수"이므로, **SH 사영 = 정변환, SH 평가 = 역변환**에 해당한다.
기저가 정규직교($\int Y_kY_{k'}d\Omega=\delta_{kk'}$)하기 때문에 사영 → 평가를 연달아 하면
(차수가 충분할 때) 원래 함수로 돌아온다. 노트북 1.3절에서 환경 조명을 사영한 뒤 L=0..3으로 복원한 그림이
정확히 "사영 후 평가"를 시각화한 것이다.

주의할 점 하나: 3DGS는 **사영을 아예 하지 않는다**. 진짜 $f$(각 Gaussian이 방향별로 내는 색)를 모르기 때문이다.
계수는 적분 대신 **경사하강으로 학습**되고, 렌더링 파이프라인 안에서 실제로 실행되는 SH 연산은
**평가 하나뿐**이다. 그래서 "SH 평가"라는 용어가 3DGS 문헌에서 특히 자주 등장한다.

---

## 2. 평가의 두 단계와 연산량

$$
\mathbf c(\mathbf d) =
\underbrace{\big[\,Y_0(\mathbf d)\ \cdots\ Y_{K-1}(\mathbf d)\,\big]}_{1\times K}
\underbrace{\begin{bmatrix}\mathbf c_0^\top\\ \vdots\\ \mathbf c_{K-1}^\top\end{bmatrix}}_{K\times 3}
$$

### 단계 1 — 기저값 계산 $Y_k(\mathbf d)$

방향을 단위 벡터로 정규화한 뒤 $(x,y,z)$의 다항식을 계산한다. 3차(K=16)까지 필요한 것은

- $\ell=0$: 상수 $0.2821$
- $\ell=1$: $-0.4886\,y,\ 0.4886\,z,\ -0.4886\,x$
- $\ell=2$: $xy,\ yz,\ 3z^2-1,\ xz,\ x^2-y^2$ 에 상수 곱
- $\ell=3$: $y(3x^2-y^2),\ xyz,\ y(5z^2-1),\ z(5z^2-3),\ x(5z^2-1),\ z(x^2-y^2),\ x(x^2-3y^2)$ 에 상수 곱

노트북의 `sh_bases(dirs, degree)`가 이것을 $(\ell,m)$별로 풀어 쓴 것이고,
gsplat의 `_eval_sh_bases_fast`는 Sloan(2013, *Efficient Spherical Harmonic Evaluation*)의 점화식으로
같은 값을 더 적은 곱셈으로 얻는다(`fC1 = x²−y²`, `fS1 = 2xy`, `fC2 = x·fC1 − y·fS1`, … 처럼
$\cos m\varphi,\ \sin m\varphi$ 항을 점화식으로 누적). 방향과만 관계있고 계수와는 무관하다.

### 단계 2 — 계수와 내적

채널(RGB) 하나마다 $\sum_k c_k Y_k$ 를 더한다. **채널당 K회의 곱셈-덧셈(FMA)**, RGB 3채널이면
K=16일 때 48회 FMA. 뉴럴 네트워크의 MLP 한 층보다 훨씬 작은 연산이다.

### 3DGS에서의 방향과 실행 횟수

- 방향 $\mathbf d$ = **카메라 위치에서 Gaussian 중심을 보는 방향**
  $\mathbf d=\boldsymbol\mu_i-\mathbf o_{\text{cam}}$, 카메라 위치는 world→camera 행렬 $[R\,|\,\mathbf t]$에서
  $\mathbf o_{\text{cam}}=-R^\top\mathbf t$ (노트북 `view_dirs`, gsplat `spherical_harmonics` docstring의 "-R^T t").
- 카메라(이미지) × Gaussian 쌍마다 한 번씩 평가된다. 수백만 Gaussian × 카메라 1대 → 수백만 번의 48-FMA.
  이것이 GPU에서 병렬로 잘 돌아가는 이유이자 3DGS가 MLP 없이 실시간 렌더링이 가능한 이유다.
- 결과는 SH 값 그대로가 아니라 `+0.5` 오프셋 후 `clamp_min(0)`을 거쳐 색이 된다.
  (오프셋의 의미는 DC 계수 카드 참조: 계수가 전부 0이면 중간 회색.)

---

## 3. 역전파가 단순한 이유 — 평가는 계수에 대해 선형

평가식 $\mathbf c(\mathbf d)=\sum_k \mathbf c_k Y_k(\mathbf d)$는 **계수 $\mathbf c_k$에 대해 선형**이다.
따라서 손실 $\mathcal L$이 색 $\mathbf c$에 대해 기울기 $\partial\mathcal L/\partial\mathbf c$ 를 넘겨주면

$$
\frac{\partial \mathbf c}{\partial \mathbf c_k} = Y_k(\mathbf d)
\qquad\Longrightarrow\qquad
\frac{\partial \mathcal L}{\partial \mathbf c_k} = Y_k(\mathbf d)\;\frac{\partial\mathcal L}{\partial \mathbf c}
$$

즉 **계수의 기울기 = (방향에서의 기저값) × (색의 기울기)**. 순전파에서 계산한 $Y_k(\mathbf d)$를 그대로 다시 쓰면 끝이라,
역전파도 채널당 K회 FMA로 끝난다. 곱해 주는 것 외에 비선형 함수의 도함수도, 행렬 역산도 없다.
3DGS 학습 한 스텝이 빠른 데는 래스터라이저의 효율도 있지만, **색 모델(SH)이 선형이라 색 파라미터의 기울기가 거의 공짜**라는 점도 한몫한다.

노트북 4.2절이 이 사실을 두 방식으로 보여 준다.
- 방법 A(최소제곱): 선형이므로 $\min_{\mathbf c}\sum_j\|Y(\mathbf d_j)\mathbf c-f^\star(\mathbf d_j)\|^2$는 **닫힌 해**가 있다 (`torch.linalg.lstsq`).
- 방법 B(Adam): 실제 학습처럼 `pred = clamp_min(A_obs @ coeffs + 0.5, 0)`을 L1 손실로 역전파. 선형 계층 하나를 학습하는 것과 같다.

방향에 대한 기울기 $\partial\mathbf c/\partial\mathbf d$는 **선택 사항**이다. 방향은 Gaussian 중심 $\boldsymbol\mu$와 카메라 위치에서 나오므로,
이를 계산하면 색 오차가 Gaussian **위치**(및 카메라 파라미터)까지 흘러간다. 이 경로는 기저 다항식을 $x,y,z$로 미분해야 하고,
정규화 $\mathbf d/\|\mathbf d\|$의 야코비안(접평면으로의 사영 $(\mathbf v-\langle\mathbf v,\hat{\mathbf d}\rangle\hat{\mathbf d})/\|\mathbf d\|$)도 곱해야 해서 계수 기울기보다 훨씬 비싸다.
gsplat은 `means`에 기울기가 필요할 때만 이 경로를 켠다.

---

## 4. gsplat 구현과의 대응

### 4.1 PyTorch 참조 구현 — `gsplat/cuda/_torch_impl.py`

```python
def _spherical_harmonics(degrees_to_use, dirs, coeffs):     # dirs [..., N, 3], coeffs [N, K, D]
    K = coeffs.shape[1]
    dirs = F.normalize(dirs, p=2, dim=-1)                   # 단계 0: 정규화
    num_bases = (degrees_to_use + 1) ** 2
    bases = coeffs.new_zeros(dirs.shape[:-1] + (K,))
    bases[..., :num_bases] = _eval_sh_bases_fast(num_bases, dirs)   # 단계 1: 기저값
    return (bases[..., None] * coeffs).sum(dim=-2)          # 단계 2: 계수와 내적
```

- `degrees_to_use`(= `sh_degree`)로 **활성 차수를 제한**한다. 배열에는 K=16개 계수가 모두 있어도
  `num_bases` 이후의 기저값을 0으로 두어 고차 항을 꺼 버린다. 그래서 `simple_trainer.py`의
  `sh_degree_to_use = min(step // sh_degree_interval, sh_degree)`처럼 1000스텝마다 한 차수씩 켤 수 있다
  (노트북 4.1절 그림: 같은 계수를 `sh_degree=0..3`으로 평가).
- `_eval_sh_bases_fast`는 4차(25개)까지 지원하며, 노트북 5절의 교차 검증 셀은
  `sh_bases(d,3)` ↔ `_eval_sh_bases_fast(16,d)`, `sh_eval(c,d,3)` ↔ `_spherical_harmonics(3,d,c)` 가 일치함을 확인한다.

노트북 `sh_eval`은 이 함수를 그대로 옮긴 것이다.

```python
def sh_eval(coeffs, dirs, degree):                          # coeffs[N,K,3], dirs[N,3]
    K = (degree + 1) ** 2
    d = F.normalize(dirs, dim=-1)
    return torch.einsum("nk,nkc->nc", sh_bases(d, degree), coeffs[:, :K])
```

| 개념 | 노트북 | gsplat |
|---|---|---|
| 기저값 계산 | `sh_bases(d, degree)` | `_eval_sh_bases_fast(num_bases, dirs)` / CUDA `sh_coeffs_to_color_fast` 내부 |
| 평가(내적) | `sh_eval(coeffs, dirs, degree)` | `_spherical_harmonics(degrees_to_use, dirs, coeffs)` / `spherical_harmonics(...)` CUDA |
| 시점 방향 | `view_dirs(means, viewmat)` (−Rᵀt) | 커널 내부 `view_direction_from_camera_data` |
| 후처리 | `clamp_min(... + 0.5, 0)` | `rendering.py` `_maybe_evaluate_sh` |

### 4.2 CUDA 커널 — `gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu`

**Forward** (`spherical_harmonics_fwd_kernel`)
- 스레드 하나가 (이미지, Gaussian, 채널) 하나를 맡는다. `idx / D`로 요소, `idx % D`로 채널을 뽑고,
  `means`와 `viewmats`에서 방향을 계산한 뒤 `sh_coeffs_to_color_fast(degree, D, c, dir, coeffs, colors)`를 호출한다.
- `sh_coeffs_to_color_fast`는 기저 계산과 내적을 **한 함수 안에서 섞어서** 한다:
  `result = 0.2821·coeffs[0]` → `if(degree>=1)` 정규화 후 1차 항 누적 → `if(degree>=2)` … 식으로
  **`degree`에 따라 고차 블록을 통째로 건너뛴다**. 즉 활성 차수 제한이 분기 하나로 구현된다.
- K=16, D=3(RGB)인 핫패스에는 별도 커널 `spherical_harmonics_fwd_kernel_k16_3channel`이 있어
  스레드 하나가 Gaussian 하나의 3채널을 모두 처리하고 계수를 `uint4` 와이드 로드로 읽는다. `DEGREE`는 템플릿 인자.
- **컬링된 Gaussian 마스크**: `masks[output_id]`가 false면 즉시 `return`. `_maybe_evaluate_sh`가
  `masks = (radii > 0).all(dim=-1)`을 넘기므로, 투영 단계에서 화면 밖·너무 작아 잘린 Gaussian은 SH 평가 자체를 건너뛴다.

**Backward** (`spherical_harmonics_bwd_kernel` + `sh_coeffs_to_color_fast_vjp`)
- 스레드 하나가 (계수 행, 채널) 하나를 맡고, 그 Gaussian을 보는 **모든 이미지에 대해 루프**를 돌며 레지스터 누적기 `acc[k]`에 더한다. 마지막에 한 번만 `v_coeffs`에 쓴다(전역 atomic 불필요).
- 핵심 식은 위 3절 그대로다.
  ```cpp
  acc[0] += 0.2820947917738781f * v_colors_local;                 // ∂L/∂c₀ = Y₀ · ∂L/∂color
  acc[1] += -0.48860251190292f * y * v_colors_local;               // Y₁⁻¹ = −0.4886 y
  acc[2] +=  0.48860251190292f * z * v_colors_local;
  acc[3] += -0.48860251190292f * x * v_colors_local;
  ```
  기저값에 들어온 색 기울기를 곱하는 것이 전부다.
- 방향 기울기는 `v_dir != nullptr`일 때만 계산한다(`v_x += -0.4886·coeffs[3]·v_color` 같은 식으로 계수를 곱한 뒤
  `v_d = (v_dir_n − dot(v_dir_n, dir_n)·dir_n)·inorm` 으로 정규화의 야코비안을 적용). 결과는 `v_means`에 `gpuAtomicAdd`로 모인다.
  Python 쪽 `RegisterSphericalHarmonics.backward`가 `ctx.needs_input_grad[1]`(means)을 커널에 넘겨 이 경로를 켜고 끈다.
- 마스크된 이미지는 `continue`로 건너뛰고, 활성 차수보다 높은 계수 자리는 0으로 채운다(기울기 0 = 그 계수는 이번 스텝에 학습되지 않음).

### 4.3 파이프라인 위치 — `gsplat/rendering.py`

```python
masks = (radii > 0).all(dim=-1)                          # 투영에서 살아남은 Gaussian만
features = spherical_harmonics(sh_degree, means, viewmats, features, masks=masks)
features = torch.clamp_min(features + 0.5, 0.0)          # 색 채널이면 clamp, extra_signals면 +0.5만
```

`rasterization()`은 `sh_degree=None`이면 `colors`를 이미 계산된 색으로 보고 SH 평가를 건너뛰며,
정수를 주면 `colors`를 `[N, K, 3]` 계수로 해석해 위 경로를 탄다.

---

## 5. 요약

- **SH 평가** = 계수 × 기저값의 유한합. 사영(적분으로 계수 얻기)의 역연산이며, 푸리에 역변환에 해당한다.
- 두 단계: (1) 방향에서 기저값 $Y_k(\mathbf d)$ 계산, (2) 계수와 내적 — 채널당 K회 FMA. 3DGS(K=16, RGB)는 48회.
- 계수에 대해 **선형**이므로 $\partial\mathbf c/\partial\mathbf c_k=Y_k(\mathbf d)$; 역전파는 기저값에 색 기울기를 곱하기만 한다. 방향 기울기는 선택 사항.
- gsplat: `_spherical_harmonics`(참조) ↔ CUDA `sh_coeffs_to_color_fast`/`_vjp`; `sh_degree`로 활성 차수 제한, `masks`로 컬링된 Gaussian 스킵, `_maybe_evaluate_sh`에서 `+0.5`·`clamp_min(0)`.
- 노트북 `sh_eval(coeffs, dirs, degree)`가 이 연산의 최소 구현이다.
