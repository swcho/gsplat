# `rasterize_mode="antialiased"`는 무엇을 하는가?

> **한 줄 답**: eps2d 블러를 더하기 **전후의 공분산 행렬식 비** $\sqrt{\det_0/\det}$ 를 계산해 불투명도에 곱한다.
> 블러 때문에 부풀어 오른 밝기를 원래대로 되돌리는 보정이며, Mip-Splatting(Yu et al., CVPR 2024)의 기법이다.

---

## 0. 출발점: 고교 확률과 통계의 정규분포

정규분포(가우스 분포)의 확률밀도함수는

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

여기서 두 가지를 따로 봐야 한다.

| 양 | 값 | σ가 커지면 |
|---|---|---|
| **총량**(그래프 아래 넓이) $\int_{-\infty}^{\infty} f(x)\,dx$ | 항상 $1$ | 변하지 않음 |
| **봉우리 높이** $f(\mu)$ | $\dfrac{1}{\sigma\sqrt{2\pi}}$ | 반비례해서 낮아짐 |

즉 확률분포는 "넓이를 1로 고정"하기 위해 $1/(\sigma\sqrt{2\pi})$ 라는 **정규화 상수**를 앞에 붙인다.
넓게 퍼뜨리면(σ↑) 그만큼 낮아지고, 좁게 모으면(σ↓) 뾰족해진다. 총량은 언제나 보존된다.

**이 "총량 보존"이 뒤에서 깨지는 것이 문제의 핵심이다.**

---

## 1. 2차원으로 확장: $\sqrt{\det\Sigma}$ 가 "타원의 크기"

### 1-1. 2D 가우시안

화면 위의 픽셀 좌표 $\mathbf{x}=(x,y)$ 에 대한 2D 가우시안은

$$G(\mathbf{x}) = \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol\mu)^\top \Sigma^{-1} (\mathbf{x}-\boldsymbol\mu)\right)$$

$\Sigma$ 는 $2\times2$ **공분산 행렬**이다. 고교 수준에서 겁먹을 필요 없이, 가장 간단한 **대각 행렬**부터 보자.

$$\Sigma = \begin{bmatrix}\sigma_x^2 & 0 \\ 0 & \sigma_y^2\end{bmatrix}
\quad\Longrightarrow\quad
G(x,y)=\exp\!\left(-\frac{(x-\mu_x)^2}{2\sigma_x^2}\right)\exp\!\left(-\frac{(y-\mu_y)^2}{2\sigma_y^2}\right)$$

x방향 1D 가우시안과 y방향 1D 가우시안의 **곱**이다. 퍼진 정도는 x로 $\sigma_x$, y로 $\sigma_y$.
즉 이 splat이 차지하는 영역은 반지름이 $\sigma_x,\sigma_y$ 인 **타원**이고, 그 넓이는 $\pi\sigma_x\sigma_y$ 에 비례한다.

### 1-2. 행렬식이 "넓이 배율"이라는 기하 직관

고교 기하에서 배운 사실: 2×2 행렬 $A$ 로 평면을 변환하면, 도형의 **넓이는 $|\det A|$ 배**가 된다.
(예: $\begin{bmatrix}2&0\\0&3\end{bmatrix}$ 은 가로 2배·세로 3배 → 넓이 6배 = $\det$.)

공분산 $\Sigma$ 는 "단위원 → 실제 타원"으로 보내는 변환 $A$ 에 대해 $\Sigma = AA^\top$ 관계에 있고,
$\det\Sigma = (\det A)^2$ 이므로

$$\boxed{\ \sqrt{\det\Sigma} = |\det A| = \text{(단위원 → 타원)의 넓이 배율}\ }$$

대각 행렬로 확인하면 딱 맞는다:

$$\sqrt{\det\Sigma} = \sqrt{\sigma_x^2\sigma_y^2} = \sigma_x\sigma_y$$

**정리: $\sqrt{\det\Sigma}$ 는 그 splat이 화면에서 덮는 타원의 "크기"(넓이 ∝ $\pi\sqrt{\det\Sigma}$)다.**

### 1-3. 2D에서의 총량

높이 1짜리(정규화 없는) 2D 가우시안을 화면 전체에 적분하면

$$\iint \exp\!\left(-\tfrac12 \mathbf{d}^\top\Sigma^{-1}\mathbf{d}\right) dx\,dy = 2\pi\sqrt{\det\Sigma}$$

**총량은 $\sqrt{\det\Sigma}$ 에 정비례한다.** 1D에서 총량 $=\sigma\sqrt{2\pi}$ 였던 것의 2D판이다.

---

## 2. gsplat의 splat은 정규화를 하지 않는다

여기가 결정적이다. 3DGS/gsplat의 알파는

$$\alpha(\mathbf{x}) = o \cdot \exp\!\left(-\tfrac12(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\right)$$

앞에 $\dfrac{1}{2\pi\sqrt{\det\Sigma}}$ 같은 **정규화 상수가 없다**. 중심에서의 값은 언제나 정확히 $o$(불투명도)다.

| | 확률밀도 | gsplat splat |
|---|---|---|
| 고정되는 것 | 총량 $=1$ | **봉우리 높이 $=o$** |
| 따라 변하는 것 | 봉우리 높이 $\propto 1/\sqrt{\det\Sigma}$ | **총량 $\propto o\sqrt{\det\Sigma}$** |

즉 **$\Sigma$ 가 커지면 화면에 뿌려지는 총 밝기가 그대로 커진다.** 이게 아티팩트의 씨앗이다.

---

## 3. eps2d 블러: 없으면 사라지고, 있으면 부푼다

투영된 2D 공분산에 gsplat은 항상 최소 블러를 더한다 (`_torch_impl.py`의 `_fully_fused_projection`):

$$\Sigma_{2D} = J\Sigma_c J^\top + \epsilon I,\qquad \epsilon = \texttt{eps2d} = 0.3\ \text{px}^2$$

**왜 필요한가**: $\epsilon$ 이 없으면 화면상 크기가 1px보다 작은 Gaussian은 픽셀 중심들 **사이로 빠져** 샘플링되지 않는다.
카메라가 조금만 움직여도 나타났다 사라졌다 하는 **깜빡임(aliasing)** 이 생긴다. 그래서 최소 0.3px² 만큼은 항상 퍼뜨린다.
(고교 물리의 "분해능"과 같은 발상: 아무리 작은 점광원도 실제 센서에서는 최소 크기의 점으로 번져 찍힌다.)

**대신 대가가 있다.** 2절에서 봤듯 총 밝기는 $\sqrt{\det}$ 에 비례하는데, 블러를 더하면

$$\det_0 = \det(\Sigma) \quad\longrightarrow\quad \det = \det(\Sigma+\epsilon I) > \det_0$$

이므로 총 밝기가

$$\frac{\sqrt{\det}}{\sqrt{\det_0}} = \sqrt{\frac{\det(\Sigma+\epsilon I)}{\det\Sigma}}\ \ (>1)$$

배로 **부풀어 오른다**. 원본에 없던 밝기가 공짜로 생긴 것이다.

### 3-1. 되돌리기 = compensation

부푼 만큼 그 **역수**를 불투명도에 곱하면 원래 총량으로 되돌아온다:

$$\boxed{\ \rho = \sqrt{\frac{\det_0}{\det}} = \sqrt{\frac{\det(\Sigma)}{\det(\Sigma+\epsilon I)}}\ \le 1\ },\qquad o \leftarrow o\cdot\rho$$

gsplat 코드에서 정확히 이 두 줄이다.

- `_torch_impl.py` `_fully_fused_projection`:
  `compensations = sqrt(clamp(det_orig / det, min=MIN_COMPENSATION**2))`  ← $\rho$ 계산
  (`MIN_COMPENSATION = 0.005`, 즉 $\rho$ 의 하한은 0.005)
- `rendering.py`: `opacities = opacities * compensations`  ← 불투명도에 곱하기

`rasterize_mode="antialiased"` 일 때만 `calc_compensations=True` 가 되어 이 경로가 켜진다.
`"classic"` 이면 `compensations=None` → 곱하지 않는다.

---

## 4. 숫자로: 이 비는 언제 1이고 언제 ≪1인가

등방(isotropic) splat $\Sigma=\sigma^2 I$ 로 두면 계산이 아주 쉽다.

$$\det_0 = \sigma^4,\qquad \det = (\sigma^2+\epsilon)^2
\quad\Longrightarrow\quad
\rho = \sqrt{\frac{\sigma^4}{(\sigma^2+\epsilon)^2}} = \frac{\sigma^2}{\sigma^2+\epsilon}$$

$\epsilon=0.3$ 을 넣어 보면:

| σ (px) | $\sigma^2$ | $\rho = \sigma^2/(\sigma^2+0.3)$ | 해석 |
|---|---|---|---|
| 4.0 | 16.0 | **0.981** | 큰 splat — 블러 영향 거의 없음, ≈1 |
| 2.0 | 4.0 | **0.930** | 살짝 어두워짐 |
| 1.0 | 1.0 | **0.769** | 픽셀 크기 — 무시 못 할 보정 |
| 0.5 | 0.25 | **0.455** | 서브픽셀 — 절반 이하로 |
| 0.2 | 0.04 | **0.118** | ≪1 |
| 0.1 | 0.01 | **0.032** | 거의 지워짐 |

**큰 Gaussian에서는 $\rho\approx1$ (아무 일도 안 일어남), 서브픽셀 Gaussian에서만 $\rho\ll1$ 로 강하게 작동한다.**
$\sigma\ll\sqrt\epsilon$ 인 극한에서는 $\rho\approx\sigma^2/\epsilon$, 즉 **$\rho\propto\sigma^2$** 로 떨어진다.

---

## 5. 왜 하필 "줌아웃(멀어질 때)"에 문제가 되는가

투영 공식에서 화면상 크기는 깊이 $z$ 에 반비례한다. Jacobian $J$ 의 성분이 $f/z$ 이므로

$$\Sigma_{2D}^{\text{(블러 전)}} \sim \frac{f^2}{z^2}\Sigma_{\text{world}}
\quad\Longrightarrow\quad \sigma \propto \frac{1}{z}$$

카메라가 멀어지면(z↑) $\sigma$ 는 줄어드는데 **$\epsilon$ 은 0.3px² 로 고정**이다. 그래서 $\epsilon$ 의 **상대 비중이 커진다**.

- **가까울 때**: $\sigma^2 = 16 \gg 0.3$ → 블러는 2% 미만의 영향.
- **멀 때**: $\sigma^2 = 0.01 \ll 0.3$ → 실제 크기는 $\sigma^2=0.01$ 인데 화면에는 $0.31$ 크기로 그려진다. **30배 부푼다.**

classic 모드에서 이 splat이 화면에 뿌리는 총 밝기는 $o\cdot 2\pi\sqrt{\det} \approx o\cdot2\pi\epsilon$ 으로,
**σ가 아무리 작아져도 더 이상 줄어들지 않고 바닥값(floor)에 걸린다.**

결과적으로 **멀어질수록 물체가 점점 밝아지고 뭉개지는 아티팩트**가 생긴다.
원래는 멀어질수록 각 splat이 화면에서 차지하는 넓이가 $1/z^2$ 로 줄어 어두워져야 하는데, 그러지 않기 때문이다.
(현실의 사진에서 멀리 있는 세밀한 격자무늬가 회색으로 평균되어 **어두워지는** 것과 반대로 동작한다.)

antialiased 모드는 $\rho\propto\sigma^2$ 를 곱해 총 밝기를 $o\cdot2\pi\epsilon\cdot\frac{\sigma^2}{\epsilon} = o\cdot2\pi\sigma^2$ 로 되돌린다.
즉 **총 밝기가 다시 $\sigma^2\propto 1/z^2$ 에 정직하게 비례한다.**

---

## 6. "Mip filter"라는 이름의 유래

**mipmap**은 컴퓨터 그래픽스의 고전 기법이다. 텍스처를 축소해서 그릴 때, 원본 픽셀을 띄엄띄엄 뽑으면(point sampling)
모아레·깜빡임이 생기므로, **미리 여러 해상도로 평균낸 피라미드**를 만들어 두고 화면 크기에 맞는 레벨을 골라 쓴다.
이름은 라틴어 *multum in parvo*("작은 것 안에 많은 것")에서 왔다.

Mip-Splatting은 이 발상을 3DGS로 옮긴 것으로, 두 개의 필터를 제안한다.

1. **3D smoothing filter**: 월드 공간에서 Gaussian의 최소 크기를, 학습에 쓰인 카메라들의 최대 샘플링 주파수에 맞춰 제한. (고주파 성분 자체를 제거)
2. **2D Mip filter**: 화면 공간의 $\epsilon I$ 블러를 "픽셀 센서의 박스 필터를 근사한 것"으로 보고, **밝기를 물리적으로 맞추기 위해 $\rho$ 로 보정**.

두 번째가 바로 `rasterize_mode="antialiased"` 다. 블러를 단순한 "안전장치"가 아니라 **정직한 저역통과 필터**로 취급하고,
필터를 통과했으면 에너지도 그에 맞게 재정규화해야 한다 — 이것이 mipmap의 "축소할 때는 평균을 낸다"는 정신과 같기에 *Mip* filter라 부른다.

---

## 7. 한눈 요약

| | classic | antialiased |
|---|---|---|
| 2D 공분산 | $\Sigma+\epsilon I$ | $\Sigma+\epsilon I$ (동일) |
| 불투명도 | $o$ | $o\cdot\sqrt{\det_0/\det}$ |
| 큰 splat | — | 보정 ≈1 (사실상 동일) |
| 서브픽셀 splat | 총 밝기 바닥값에 고정 → 밝아짐 | 총 밝기 $\propto\sigma^2$ → 정상적으로 어두워짐 |
| 줌아웃 | 밝아지고 뭉개짐 | 밝기 보존 |
| 비용 | — | Gaussian당 나눗셈+제곱근 1회 (사실상 공짜) |

핵심 한 문장: **정규화하지 않는 splat에 블러를 더하면 밝기가 늘어나므로, 늘어난 만큼($\sqrt{\det/\det_0}$)의 역수를 불투명도에 곱해 돌려놓는다.**
