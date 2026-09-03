# %% [markdown]
# # 원근 투영의 Jacobian $J$ — 손으로, 수치로, 자동미분으로
#
# 투영 함수
# $$\pi(x,y,z) = \left(f_x\frac{x}{z} + c_x,\ \ f_y\frac{y}{z} + c_y\right)$$
# 를 미분하면 **2×3** 행렬
# $$J = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix}$$
# 가 나온다. 이 노트는 그 식을 세 가지 방법으로 교차 검증하고, $J$가 실제로 무엇을 하는지 본다.
#
# 1. 해석적 공식 (손으로 미분한 것)
# 2. 유한차분 수치 미분 (미분의 정의 그대로)
# 3. `torch.autograd.functional.jacobian` (자동미분)
# 4. $J$가 3D의 작은 이동을 화면 이동으로 어떻게 사상하는가 — $z=2$ vs $z=8$
# 5. $\Sigma_{2D} = J\,\Sigma_{3D}\,J^\top$ 로 3D 공분산을 2D 타원으로 밀어내기 (EWA splatting)
#
# 필요 패키지: numpy, torch, plotly, kaleido
# (gsplat은 **import하지 않는다** — JIT CUDA 빌드가 30분 이상 걸린다. 수식은 `_torch_impl._persp_proj`와 동일.)

# %%
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

torch.set_printoptions(precision=6, sci_mode=False, linewidth=120)
np.set_printoptions(precision=6, suppress=True)


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# 장난감 카메라 (walkthrough의 toy 씬과 같은 성격의 값)
FX, FY, CX, CY = 300.0, 300.0, 160.0, 120.0
print(f"fx={FX}  fy={FY}  cx={CX}  cy={CY}")
# 출력: fx=300.0  fy=300.0  cx=160.0  cy=120.0

# %% [markdown]
# ## 1. 투영 함수 $\pi$ 와 해석적 Jacobian
#
# 편미분 6개를 하나씩:
#
# $$\frac{\partial u}{\partial x} = \frac{f_x}{z},\qquad
#   \frac{\partial u}{\partial y} = 0,\qquad
#   \frac{\partial u}{\partial z} = -\frac{f_x x}{z^2}$$
#
# $$\frac{\partial v}{\partial x} = 0,\qquad
#   \frac{\partial v}{\partial y} = \frac{f_y}{z},\qquad
#   \frac{\partial v}{\partial z} = -\frac{f_y y}{z^2}$$
#
# $u$의 식에 $y$가 없어서 $\partial u/\partial y = 0$, $c_x$는 상수라 미분하면 사라진다.

# %%
def proj(p, fx=FX, fy=FY, cx=CX, cy=CY):
    """π(x,y,z) = (fx·x/z + cx, fy·y/z + cy).  p: (..., 3) → (..., 2)"""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)


def jacobian_analytic(p, fx=FX, fy=FY):
    """해석적 J. p: (..., 3) → (..., 2, 3).  gsplat _persp_proj와 같은 stack 순서."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    O = torch.zeros_like(z)
    return torch.stack(
        [fx / z, O, -fx * x / z**2,
         O, fy / z, -fy * y / z**2], dim=-1
    ).reshape(*p.shape[:-1], 2, 3)


P = torch.tensor([0.4, -0.3, 2.0], dtype=torch.float64)  # 카메라 좌표계의 한 3D 점
print("점 P        =", P.numpy())
print("π(P) [px]   =", proj(P).numpy())
print("해석적 J    =\n", jacobian_analytic(P).numpy())
# 출력: 점 P        = [ 0.4 -0.3  2. ]
# 출력: π(P) [px]   = [220.  75.]
# 출력: 해석적 J    =
#  [[150.   0. -30.]
#   [  0. 150.  22.5]]

# %% [markdown]
# 손으로 확인해 보자. $z=2$ 이므로
#
# - $f_x/z = 300/2 = 150$
# - $-f_x x/z^2 = -300\cdot 0.4/4 = -30$
# - $-f_y y/z^2 = -300\cdot(-0.3)/4 = +22.5$  ← $y<0$ 이라 부호가 뒤집힌다
#
# 정확히 일치한다. $u = 300\cdot 0.4/2 + 160 = 220$, $v = 300\cdot(-0.3)/2 + 120 = 75$ 도 맞다.

# %% [markdown]
# ## 2. 유한차분(수치 미분)과 비교
#
# 미분의 정의를 그대로 쓴다. 오차가 $O(h^2)$인 **중심차분**을 쓰면 훨씬 정확하다:
# $$\frac{\partial \pi_i}{\partial p_j} \approx \frac{\pi_i(p + h\,e_j) - \pi_i(p - h\,e_j)}{2h}$$
# $e_j$는 $j$번째 좌표만 1인 단위벡터 — "다른 변수는 고정하고 하나만 움직인다"는 편미분의 정의 그 자체다.

# %%
def jacobian_finite_diff(p, h=1e-5):
    J = torch.zeros(2, 3, dtype=p.dtype)
    for j in range(3):
        e = torch.zeros(3, dtype=p.dtype)
        e[j] = h
        J[:, j] = (proj(p + e) - proj(p - e)) / (2 * h)   # 중심차분
    return J


J_ana = jacobian_analytic(P)
J_fd = jacobian_finite_diff(P)
print("유한차분 J  =\n", J_fd.numpy())
print("최대 절대오차 =", (J_ana - J_fd).abs().max().item())
# 출력: 유한차분 J  =
#  [[150.        0.      -30.000000]
#   [  0.      150.       22.500000]]
# 출력: 최대 절대오차 = 1.4574617068774387e-09

# %%
# float32(3DGS CUDA 커널이 실제로 쓰는 정밀도)로도 확인.
# float32에서는 h를 너무 작게 잡으면 뺄셈의 자리수 손실(catastrophic cancellation)이 커지므로
# h를 1e-2 정도로 키우는 편이 오히려 정확하다.
P32 = P.float()
for h in (1e-1, 1e-2, 1e-3, 1e-4):
    e = (jacobian_analytic(P32) - jacobian_finite_diff(P32, h=h)).abs().max().item()
    print(f"  h={h:<7g} 최대 절대오차 = {e:.3e}   상대오차 = {e / 150.0:.2e}")
# 출력:   h=0.1     최대 절대오차 = 7.523e-02   상대오차 = 5.02e-04   ← h가 커서 절단오차(truncation)
# 출력:   h=0.01    최대 절대오차 = 1.068e-03   상대오차 = 7.12e-06   ← 최적점, 상대오차 ~1e-6
# 출력:   h=0.001   최대 절대오차 = 6.104e-03   상대오차 = 4.07e-05   ← 반올림오차(round-off) 시작
# 출력:   h=0.0001  최대 절대오차 = 3.143e-02   상대오차 = 2.10e-04   ← 자리수 손실이 지배
#   → float32에서도 상대오차 ~1e-6 수준까지 일치. 해석식과 수치미분이 같은 함수를 미분하고 있다.

# %% [markdown]
# ## 3. 자동미분(autograd)과 비교
#
# PyTorch가 계산 그래프를 따라 연쇄법칙을 적용해 뽑아 주는 Jacobian.
# 3DGS의 CUDA 커널은 성능 때문에 이 식을 **손으로 짜 넣지만**, 값은 자동미분과 정확히 같아야 한다.

# %%
J_auto = torch.autograd.functional.jacobian(proj, P)
print("autograd J  =\n", J_auto.numpy())
print("shape       =", tuple(J_auto.shape), " ← (출력 2개, 입력 3개)")
print("해석 vs autograd 최대 절대오차 =", (J_ana - J_auto).abs().max().item())
# 출력: autograd J  =
#  [[150.    0.  -30. ]
#   [  0.  150.   22.5]]
# 출력: shape       = (2, 3)  ← (출력 2개, 입력 3개)
# 출력: 해석 vs autograd 최대 절대오차 = 0.0

# %%
# 배치로도 확인: 무작위 점 1000개에 대해 해석식 vs autograd
torch.manual_seed(0)
pts = torch.randn(1000, 3, dtype=torch.float64)
pts[:, 2] = pts[:, 2].abs() + 0.5          # z > 0 (카메라 앞) 보장
J_batch_ana = jacobian_analytic(pts)                       # [1000,2,3]
J_batch_auto = torch.stack([torch.autograd.functional.jacobian(proj, p) for p in pts])
print("배치 최대 절대오차 =", (J_batch_ana - J_batch_auto).abs().max().item())
# 출력: 배치 최대 절대오차 = 9.094947017729282e-13  ← float64 반올림 한계

# %% [markdown]
# ## 4. $J$는 "3D의 작은 이동 → 화면 이동" 환율표다
#
# $$\begin{bmatrix}\Delta u\\ \Delta v\end{bmatrix} \approx J \begin{bmatrix}\Delta x\\ \Delta y\\ \Delta z\end{bmatrix}$$
#
# 같은 $(x, y)$를 가진 점을 $z=2$와 $z=8$에 두고, 각 축으로 $\delta$만큼 움직여 본다.

# %%
DELTA = 0.02  # 3D 공간에서 2cm 이동

for z in (2.0, 8.0):
    p = torch.tensor([0.4, -0.3, z], dtype=torch.float64)
    J = jacobian_analytic(p)
    print(f"\n--- z = {z} (화면 위치 {proj(p).numpy()}) ---")
    for j, name in enumerate("xyz"):
        d = torch.zeros(3, dtype=torch.float64)
        d[j] = DELTA
        lin = J @ d                          # 1차 근사
        true = proj(p + d) - proj(p)         # 실제 이동
        print(f"  Δ{name}={DELTA}:  J·d = [{lin[0]:8.4f},{lin[1]:8.4f}] px"
              f"   실제 = [{true[0]:8.4f},{true[1]:8.4f}] px"
              f"   근사오차 = {(lin-true).abs().max():.2e}")
# 출력:
# --- z = 2.0 (화면 위치 [220.  75.]) ---
#   Δx=0.02:  J·d = [  3.0000,  0.0000] px   실제 = [  3.0000,  0.0000] px   근사오차 = 0.00e+00
#   Δy=0.02:  J·d = [  0.0000,  3.0000] px   실제 = [  0.0000,  3.0000] px   근사오차 = 0.00e+00
#   Δz=0.02:  J·d = [ -0.6000,  0.4500] px   실제 = [ -0.5941,  0.4455] px   근사오차 = 5.94e-03
#
# --- z = 8.0 (화면 위치 [175.   108.75]) ---
#   Δx=0.02:  J·d = [  0.7500,  0.0000] px   실제 = [  0.7500,  0.0000] px   근사오차 = 0.00e+00
#   Δy=0.02:  J·d = [  0.0000,  0.7500] px   실제 = [  0.0000,  0.7500] px   근사오차 = 0.00e+00
#   Δz=0.02:  J·d = [ -0.0375,  0.0281] px   실제 = [ -0.0374,  0.0281] px   근사오차 = 9.35e-05
#
#   읽는 법:
#   · Δx, Δy 는 J가 **정확**하다 (z 고정 시 π가 x,y에 대해 일차함수이므로 오차 0.00e+00).
#   · Δz 만 근사오차가 생긴다 (1/z 가 비선형). z가 클수록 오차도 작아진다 (5.9e-3 → 9.4e-5).
#   · z가 4배 멀어지자 Δx 응답이 3.0 → 0.75 px 로 정확히 1/4  (= ∂u/∂x = f_x/z 의 1/z 의존).
#   · Δz 응답은 -0.600 → -0.0375 로 1/16  (= ∂u/∂z = -f_x·x/z² 의 1/z² 의존, x는 고정).
#     둘째 열이 첫째 열보다 깊이에 훨씬 예민하다는 것이 숫자로 보인다.

# %%
# J의 영공간(null space): 시선 방향 (x, y, z) 는 화면을 전혀 움직이지 않는다
p = torch.tensor([0.4, -0.3, 2.0], dtype=torch.float64)
J = jacobian_analytic(p)
print("J @ p =", (J @ p).numpy(), " ← 0벡터: 시선 방향으로 미끄러져도 같은 픽셀")
print("rank(J) =", torch.linalg.matrix_rank(J).item(), " ← 2×3인데 랭크 2, 1차원 정보가 사라진다")
# 출력: J @ p = [0. 0.]  ← 0벡터: 시선 방향으로 미끄러져도 같은 픽셀
# 출력: rank(J) = 2  ← 2×3인데 랭크 2, 1차원 정보가 사라진다

# %%
# 실제로 시선 방향으로 크게 움직여도 픽셀이 그대로인지 확인 (근사가 아니라 정확히 성립)
for s in (0.5, 1.0, 2.0, 5.0):
    print(f"  s={s}: π(s·P) = {proj(s * p).numpy()}")
# 출력:   s=0.5: π(s·P) = [220.  75.]
# 출력:   s=1.0: π(s·P) = [220.  75.]
# 출력:   s=2.0: π(s·P) = [220.  75.]
# 출력:   s=5.0: π(s·P) = [220.  75.]

# %% [markdown]
# ## 5. $\Sigma_{2D} = J\,\Sigma_{3D}\,J^\top$ — 3D 공분산을 화면 타원으로
#
# 이것이 3DGS(EWA splatting)에서 $J$를 계산하는 **진짜 이유**다.
# $\pi$가 비선형이라 3D Gaussian을 통과시키면 Gaussian이 아니게 되므로,
# $\mu_c$ 근방에서 $J$로 선형화한 뒤 선형변환의 공분산 법칙을 쓴다.
# 1변수 $\mathrm{Var}(aX) = a^2\mathrm{Var}(X)$ 의 행렬 버전:
#
# $$\Sigma_{2D} = J\,\Sigma_{3D}\,J^\top + \varepsilon I \qquad (2{\times}3)(3{\times}3)(3{\times}2) = (2{\times}2)$$
#
# 아래에서는 몬테카를로 샘플과 비교해 이 근사가 실제로 맞는지 본다.

# %%
def covar_from_rot_scale(axis_deg, scales):
    """z축 회전 + 축 길이 → Σ = R S Sᵀ Rᵀ (항상 양정치)."""
    t = np.deg2rad(axis_deg)
    R = torch.tensor([[np.cos(t), -np.sin(t), 0.0],
                      [np.sin(t),  np.cos(t), 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float64)
    S = torch.diag(torch.tensor(scales, dtype=torch.float64))
    M = R @ S
    return M @ M.T


SIGMA3D = covar_from_rot_scale(30.0, [0.30, 0.12, 0.10])
EPS2D = 0.3  # gsplat 기본 최소 블러 (px²)
print("Σ_3D =\n", SIGMA3D.numpy())
print("sqrt(eigvals) =", torch.linalg.eigvalsh(SIGMA3D).sqrt().numpy(), " ← scales와 일치")
# 출력: Σ_3D =
#  [[0.0711   0.032736 0.      ]
#   [0.032736 0.0333   0.      ]
#   [0.       0.       0.01    ]]
# 출력: sqrt(eigvals) = [0.1  0.12 0.3 ]  ← scales와 일치


def project_covar(p, sigma3d, eps2d=EPS2D):
    J = jacobian_analytic(p)
    return J @ sigma3d @ J.T + eps2d * torch.eye(2, dtype=p.dtype)


for z in (2.0, 8.0):
    p = torch.tensor([0.4, -0.3, z], dtype=torch.float64)
    c2 = project_covar(p, SIGMA3D)
    ev = torch.linalg.eigvalsh(c2).sqrt()
    print(f"z={z}: Σ_2D =\n{c2.numpy()}\n   1σ 반축 = {ev.numpy()} px,  radii(3.33σ) = "
          f"{torch.ceil(3.33 * c2.diagonal().sqrt()).numpy()}")
# 출력: z=2.0: Σ_2D =
# [[1609.05      729.804606]
#  [ 729.804606  754.6125  ]]
#    1σ 반축 = [18.335138 45.027605] px,  radii(3.33σ) = [134.  92.]
# 출력: z=8.0: Σ_2D =
# [[100.319531  46.008296]
#  [ 46.008296  47.1479  ]]
#    1σ 반축 = [ 4.538331 11.263702] px,  radii(3.33σ) = [34. 23.]
#   → 4배 멀어지면 Σ_2D 성분은 약 1/16 (J가 ∝1/z 이므로 JΣJᵀ 는 ∝1/z²),
#     1σ 반축과 radii는 약 1/4. 화면 위 "면적"이 1/16이 된다는 뜻이다.

# %%
# 몬테카를로 검증: 3D Gaussian에서 샘플을 뽑아 π로 투영하고, 그 표본공분산과 비교
def mc_cov2d(p, sigma3d, n=400_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = torch.linalg.cholesky(sigma3d)
    s = p + torch.randn(n, 3, generator=g, dtype=torch.float64) @ L.T
    uv = proj(s)
    return torch.cov(uv.T)


for z in (0.8, 2.0, 8.0):
    p = torch.tensor([0.4, -0.3, z], dtype=torch.float64)
    J_cov = project_covar(p, SIGMA3D, eps2d=0.0)   # eps2d 빼고 순수 JΣJᵀ 만 비교
    mc = mc_cov2d(p, SIGMA3D)
    rel = ((J_cov - mc).abs() / mc.abs()).max().item()
    print(f"z={z}:  JΣJᵀ 대각 = {J_cov.diagonal().numpy()},  MC 대각 = {mc.diagonal().numpy()}"
          f",  최대 상대오차 = {rel:.3f}")
# 출력: z=0.8:  JΣJᵀ 대각 = [10350.        4880.566406],  MC 대각 = [10906.894196  5131.39279 ],  최대 상대오차 = 0.051
# 출력: z=2.0:  JΣJᵀ 대각 = [1608.75    754.3125],  MC 대각 = [1620.120278  758.254849],  최대 상대오차 = 0.007
# 출력: z=8.0:  JΣJᵀ 대각 = [100.019531  46.8479  ],  MC 대각 = [100.010512  46.770233],  최대 상대오차 = 0.002
#   → 멀수록 1차 근사가 정확해진다: z=0.8 에서 5.1%, z=2 에서 0.7%, z=8 에서 0.2%.
#     Gaussian의 3D 크기(σ_max=0.30)가 깊이 z에 비해 클수록 선형화가 나빠진다
#     — EWA 근사의 본질적 한계다. 카메라에 아주 가까운 splat일수록 타원이 실제보다 작게 예측된다.

# %% [markdown]
# ## 6. 시각화
#
# 네 장으로 나눠 그린다.
#
# 1. **$J$ 성분 vs 깊이 $z$**: $f_x/z$ 는 $1/z$, $-f_x x/z^2$ 는 $1/z^2$ 로 감쇠
# 2. **$\partial u/\partial z$ vs 화면 위치**: 중심에서 멀수록 앞뒤 이동에 크게 반응
# 3. **$J$가 사상하는 작은 정육면체**: $z=2$ vs $z=8$
# 4. **$\Sigma_{2D} = J\Sigma J^\top$ 타원** vs 몬테카를로 샘플

# %%
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "① J 성분의 깊이 의존 (x=0.4 고정)",
        "② ∂u/∂z: 화면 중심에서 멀수록 크다 (z=2)",
        "③ 같은 3D 큐브(±2cm)의 화면 이동량 J·d — z=2 vs z=8",
        "④ Σ_2D = JΣJᵀ 타원 vs 몬테카를로 샘플",
    ),
)

# ① J 성분 vs z
zs = np.linspace(0.6, 8.0, 300)
x0 = 0.4
fig.add_trace(go.Scatter(x=zs, y=FX / zs, name="∂u/∂x = f_x/z  (∝1/z)",
                         line=dict(color="#3366cc", width=2.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=zs, y=-FX * x0 / zs**2, name="∂u/∂z = -f_x·x/z²  (∝1/z²)",
                         line=dict(color="#dc3912", width=2.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=zs, y=np.zeros_like(zs), name="∂u/∂y = 0",
                         line=dict(color="#999999", width=1.5, dash="dot")), row=1, col=1)
fig.update_xaxes(title_text="깊이 z", row=1, col=1)
fig.update_yaxes(title_text="편미분 값 (px / 단위길이)", range=[-90, 260], row=1, col=1)

# ② ∂u/∂z vs 화면 u 위치 (z=2 고정, x를 훑는다)
z0 = 2.0
xs = np.linspace(-1.0, 1.0, 200)
us = FX * xs / z0 + CX
dudz = -FX * xs / z0**2
fig.add_trace(go.Scatter(x=us, y=dudz, name="∂u/∂z (z=2)",
                         line=dict(color="#ff9900", width=2.5), showlegend=True), row=1, col=2)
fig.add_trace(go.Scatter(x=[CX], y=[0.0], mode="markers+text", text=["광축 x=0"],
                         textposition="top center", marker=dict(size=10, color="#109618"),
                         name="광축: ∂u/∂z = 0"), row=1, col=2)
fig.update_xaxes(title_text="화면 가로 좌표 u (px)", row=1, col=2)
fig.update_yaxes(title_text="∂u/∂z (px / 단위깊이)", row=1, col=2)

# ③ 같은 크기 3D 큐브가 J로 사상되는 "상대 이동량" (중심을 겹쳐 크기 비교)
cube = np.array([[dx, dy, dz] for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]) * 0.02
edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
for z, col in ((2.0, "#3366cc"), (8.0, "#dc3912")):
    pz = torch.tensor([0.4, -0.3, z], dtype=torch.float64)
    Jz = jacobian_analytic(pz).numpy()
    duv = cube @ Jz.T                            # Δuv = J·d  (중심 기준 상대 이동)
    first = True
    for a, b in edges:
        fig.add_trace(go.Scatter(x=[duv[a,0], duv[b,0]], y=[duv[a,1], duv[b,1]],
                                 mode="lines", line=dict(color=col, width=2),
                                 name=f"z={z:g}  (반폭 {np.abs(duv[:,0]).max():.2f} px)",
                                 legendgroup=f"cube{z}", showlegend=first), row=2, col=1)
        first = False
fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                         marker=dict(size=8, color="#109618", symbol="x"),
                         name="π(μ) (중심을 겹침)"), row=2, col=1)
fig.update_xaxes(title_text="Δu (px)", row=2, col=1)
fig.update_yaxes(title_text="Δv (px)", autorange="reversed", scaleanchor="x3",
                 scaleratio=1, row=2, col=1)

# ④ Σ_2D 타원 + MC 샘플 (z=8, 근사가 잘 맞는 쪽)
p8 = torch.tensor([0.4, -0.3, 8.0], dtype=torch.float64)
uv8 = proj(p8).numpy()
C8 = project_covar(p8, SIGMA3D, eps2d=0.0).numpy()
g = torch.Generator().manual_seed(1)
L3 = torch.linalg.cholesky(SIGMA3D)
samp = proj(p8 + torch.randn(4000, 3, generator=g, dtype=torch.float64) @ L3.T).numpy()
fig.add_trace(go.Scatter(x=samp[:, 0], y=samp[:, 1], mode="markers",
                         marker=dict(size=2, color="#aaaaaa", opacity=0.5),
                         name="3D Gaussian 샘플의 실제 투영"), row=2, col=2)
th = np.linspace(0, 2 * np.pi, 200)
circ = np.stack([np.cos(th), np.sin(th)])
Lc = np.linalg.cholesky(C8)
for k, dash in ((1, "solid"), (3, "dash")):
    e = uv8[:, None] + k * (Lc @ circ)
    fig.add_trace(go.Scatter(x=e[0], y=e[1], mode="lines",
                             line=dict(color="#dc3912", width=2, dash=dash),
                             name=f"JΣJᵀ {k}σ 타원"), row=2, col=2)
fig.add_trace(go.Scatter(x=[uv8[0]], y=[uv8[1]], mode="markers",
                         marker=dict(size=8, color="#109618", symbol="x"),
                         name="π(μ)"), row=2, col=2)
fig.update_xaxes(title_text="u (px)", row=2, col=2)
fig.update_yaxes(title_text="v (px)", autorange="reversed", scaleanchor="x4",
                 scaleratio=1, row=2, col=2)

fig.update_layout(
    height=880, width=1250,
    title_text="원근 투영의 Jacobian J = [[f_x/z, 0, -f_x·x/z²], [0, f_y/z, -f_y·y/z²]]",
    legend=dict(font=dict(size=10)), template="plotly_white",
)
_show(fig)
fig.write_image("expy.png", scale=2)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ### 그림 읽는 법
#
# - **①** 파란 선 $f_x/z$ 는 $1/z$ 로, 빨간 선 $-f_x x/z^2$ 는 $1/z^2$ 로 감쇠한다.
#   가까운 곳에서는 깊이 항이 훨씬 크게 튀고, 이래서 `_persp_proj`가 $x/z$를 시야각의 1.3배로 clamp한다.
#   회색 점선 $\partial u/\partial y = 0$ — $u$ 식에 $y$가 없어서다.
# - **②** $\partial u/\partial z = -f_x x/z^2$ 는 $x$(즉 화면 위치)에 **비례**한다.
#   광축($u = c_x = 160$)에서 정확히 0 — 정면 정중앙 물체는 앞뒤로 움직여도 화면에서 제자리다.
#   가장자리로 갈수록 커진다(확장 초점, focus of expansion).
# - **③** 3D의 같은 크기 큐브($\pm2\text{cm}$)를 $J$로 밀어낸 화면 이동량. 비교를 위해 두 중심을 원점에 겹쳤다.
#   가로 반폭이 $z=2$에서 3.60px, $z=8$에서 0.79px — 약 **1/4.6배**로 줄었다.
#   ($0.02\cdot(f_x/z + f_x x/z^2)$ 이므로 $1/z$ 항과 $1/z^2$ 항이 섞여 정확히 4배는 아니다.)
#   두 상자 모두 3차원 큐브인데 화면에서는 **평행사변형 12개로 눌린 그림자**로 그려진다 —
#   $J$의 랭크가 2라 깊이 방향 두께 하나가 사라지기 때문이다(§4의 영공간).
# - **④** 회색 점이 3D Gaussian 샘플을 실제 $\pi$로 투영한 것, 빨간 타원이 $J\Sigma J^\top$ 예측.
#   거의 완벽히 겹친다 = EWA 근사가 유효하다 (§5에서 상대오차 0.2%).
#   $z=0.8$로 바꾸면 5% 수준으로 벌어진다 — 가까운 Gaussian일수록 선형화가 나빠진다.

# %% [markdown]
# ## 7. 정리
#
# | 확인한 것 | 결과 |
# |---|---|
# | 해석적 $J$ vs 중심차분 (float64) | 최대 절대오차 $1.5\times10^{-9}$ |
# | 해석적 $J$ vs 중심차분 (float32, $h=10^{-2}$) | 상대오차 $7\times10^{-6}$ |
# | 해석적 $J$ vs `autograd.functional.jacobian` | 단일 점 오차 $0$, 무작위 1000개 점 $9\times10^{-13}$ |
# | $J\cdot(x,y,z)^\top$ | $\mathbf{0}$ — 시선 방향은 영공간, $\mathrm{rank}(J)=2$ |
# | $\Delta x,\Delta y$ 이동의 1차 근사 | **정확** ($z$ 고정 시 $\pi$가 일차함수) |
# | $\Delta z$ 이동의 1차 근사 | $z=2$에서 $5.9\times10^{-3}$px, $z=8$에서 $9.4\times10^{-5}$px 오차 |
# | $J\Sigma J^\top$ vs 몬테카를로 (40만 샘플) | $z=0.8$: 5.1%, $z=2$: 0.7%, $z=8$: 0.2% |
#
# 핵심: $J$는 **2×3**이고(출력 2 × 입력 3), 첫 두 열은 $1/z$ 로, 셋째 열은 $x, y$에 비례하며 $1/z^2$ 로
# 스케일된다. $c_x, c_y$는 상수라 미분에서 사라진다. 3DGS는 이 $J$로
# $\Sigma_{2D} = J\Sigma_c J^\top + \varepsilon I$ 를 계산해 3D Gaussian을 화면 위 타원으로 만든다.
