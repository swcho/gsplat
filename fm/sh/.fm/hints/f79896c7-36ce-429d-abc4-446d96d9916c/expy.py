# %% [markdown]
# # `sphere_grid`: 구면 적분을 격자 구적(quadrature)으로 근사하기
#
# 3DGS의 SH 설명에서 정규직교성 확인 $\int Y_i Y_j\,d\Omega = \delta_{ij}$ 나
# 계수 투영 $c_k = \int f(\mathbf d)\,Y_k(\mathbf d)\,d\Omega$ 처럼 **구면 위 적분**이 자주 등장한다.
# `sphere_grid`는 이 적분을 컴퓨터로 계산하기 위해
#
# 1. 구면을 $(\theta,\varphi)$ 격자로 덮는 단위 방향 벡터 `d[nθ,nφ,3]`,
# 2. 각 격자점의 적분 가중치 $w = \sin\theta\,\Delta\theta\,\Delta\varphi$
#
# 를 만들어 $\int_{S^2} f\,d\Omega \approx \sum_{ij} f(\mathbf d_{ij})\,w_{ij}$ 로 바꾼다.
# 아래에서 (a) 왜 $\sin\theta$가 필요한지, (b) 가중치 합이 $4\pi$로 수렴하는지,
# (c) 실제 적분·SH 정규직교성이 맞는지, (d) 셀 중심 오프셋 0.5의 역할을 단계별로 확인한다.

# %%
# 필요 패키지: numpy, torch, plotly, kaleido
import math
import os

import numpy as np
import torch


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def sphere_grid(n_theta: int = 128, n_phi: int = 256, device="cpu", offset: float = 0.5):
    """sh_walkthrough.py의 sphere_grid 그대로 (offset 인자만 실험용으로 추가)."""
    theta = (torch.arange(n_theta, device=device) + offset) * math.pi / n_theta          # 극각 θ ∈ (0, π)
    phi = (torch.arange(n_phi, device=device) + offset) * 2 * math.pi / n_phi - math.pi  # 방위각 φ ∈ (-π, π)
    th, ph = torch.meshgrid(theta, phi, indexing="ij")
    d = torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], dim=-1)
    w = th.sin() * (math.pi / n_theta) * (2 * math.pi / n_phi)
    return d, w


dirs, w = sphere_grid(8, 16)
print("d shape:", tuple(dirs.shape), " w shape:", tuple(w.shape))
print("모든 d가 단위 벡터인가:", torch.allclose(dirs.norm(dim=-1), torch.ones(8, 16)))
# 출력: d shape: (8, 16, 3)  w shape: (8, 16)
# 출력: 모든 d가 단위 벡터인가: True

# %% [markdown]
# ## (a) 왜 $\sin\theta$ 가중치가 필요한가
#
# 구면 좌표에서 면적 요소는 $d\Omega = \sin\theta\,d\theta\,d\varphi$ 이다.
# $(\theta,\varphi)$ 평면에서는 모든 격자 셀이 같은 크기 $\Delta\theta\times\Delta\varphi$ 지만,
# 구면 위에서는 **극($\theta\to0,\pi$) 근처 셀이 좁아진다** — 위도선의 둘레가 $2\pi\sin\theta$ 로 줄기 때문.
#
# 셀 하나의 정확한 구면 넓이는
# $$A = \int_{\theta_0}^{\theta_1}\!\!\int_{\varphi_0}^{\varphi_1}\sin\theta\,d\theta\,d\varphi
#      = (\cos\theta_0-\cos\theta_1)\,\Delta\varphi$$
# 이고, `sphere_grid`의 $w=\sin\theta_c\,\Delta\theta\,\Delta\varphi$ 는 이를 셀 중심값으로 근사한 것(중점 규칙)이다.

# %%
n_theta, n_phi = 8, 16
dth, dph = math.pi / n_theta, 2 * math.pi / n_phi
edges = torch.arange(n_theta + 1) * dth
exact_cell = (edges[:-1].cos() - edges[1:].cos()) * dph          # 정확한 셀 넓이 (θ행마다 같음)
_, w8 = sphere_grid(n_theta, n_phi)
print(f"{'θ행':>4} {'θ중심(deg)':>10} {'정확 넓이':>10} {'sinθ·dθ·dφ':>11} {'평면 넓이 dθdφ':>14}")
for i in range(n_theta):
    print(f"{i:>4} {math.degrees((i + .5) * dth):>10.1f} {exact_cell[i]:>10.4f} {w8[i, 0]:>11.4f} {dth * dph:>14.4f}")
# 출력:   θ행   θ중심(deg)      정확 넓이  sinθ·dθ·dφ     평면 넓이 dθdφ
# 출력:    0       11.2     0.0299      0.0301         0.1542
# 출력:    1       33.8     0.0851      0.0857         0.1542
# 출력:    2       56.2     0.1274      0.1282         0.1542
# 출력:    3       78.8     0.1503      0.1512         0.1542
# 출력:    4      101.2     0.1503      0.1512         0.1542
# 출력:    5      123.7     0.1274      0.1282         0.1542
# 출력:    6      146.2     0.0851      0.0857         0.1542
# 출력:    7      168.8     0.0299      0.0301         0.1542
# → 극 근처(0,7행)의 실제 넓이는 적도 근처(3,4행)의 1/5 수준. 평면 넓이 dθdφ로 똑같이 취급하면 극을 5배 과대평가한다.
#   sinθ·dθ·dφ(중점 근사)는 정확 넓이와 소수 3자리까지 일치한다.

# %% [markdown]
# ## (b) 가중치 합 $\sum w \to 4\pi$ (구 표면적), 가중치 없이 더하면?
#
# $\sum_{ij} w_{ij} = \Delta\theta\,\Delta\varphi\sum_i\sin\theta_i \cdot n_\varphi
# \;\xrightarrow{n\to\infty}\; \int_0^{2\pi}\!\!\int_0^{\pi}\sin\theta\,d\theta\,d\varphi = 4\pi.$
#
# 반면 $\sin\theta$ 없이 $\Delta\theta\,\Delta\varphi$ 만 더하면 $(\theta,\varphi)$ 직사각형의 넓이 $\pi\cdot2\pi = 2\pi^2 \approx 19.74$ 가 나온다 — 구의 표면적과 무관한 값이다.

# %%
print(f"{'nθ×nφ':>10} {'Σw':>10} {'Σw-4π':>12} {'Σ dθdφ (sinθ 없이)':>20}")
rows = []
for n in [4, 8, 16, 32, 64, 128, 256]:
    _, wn = sphere_grid(n, 2 * n)
    flat = n * 2 * n * (math.pi / n) * (2 * math.pi / (2 * n))
    rows.append((n, wn.sum().item()))
    print(f"{n:>4}×{2*n:<5} {wn.sum().item():>10.6f} {wn.sum().item() - 4 * math.pi:>12.2e} {flat:>20.4f}")
print("4π =", 4 * math.pi, "  2π² =", 2 * math.pi ** 2)
# 출력:      nθ×nφ         Σw        Σw-4π     Σ dθdφ (sinθ 없이)
# 출력:    4×8      12.895262     3.29e-01              19.7392
# 출력:    8×16     12.647482     8.11e-02              19.7392
# 출력:   16×32     12.586579     2.02e-02              19.7392
# 출력:   32×64     12.571419     5.05e-03              19.7392
# 출력:   64×128    12.567633     1.26e-03              19.7392
# 출력:  128×256    12.566688     3.17e-04              19.7392
# 출력:  256×512    12.566449     7.86e-05              19.7392
# 출력: 4π = 12.566370614359172   2π² = 19.739208802178716
# → n이 2배가 되면 오차가 1/4 → O(1/n²) 수렴 (중점 규칙의 전형적 차수). sinθ 없이는 항상 2π²으로 틀린다.

# %% [markdown]
# ## (c) 실제 함수 적분: $\int_{S^2} z^2\,d\Omega = \tfrac{4\pi}{3}$, 그리고 SH 정규직교성
#
# 대칭성으로 $\int x^2 = \int y^2 = \int z^2$ 이고 합은 $\int 1\,d\Omega = 4\pi$ 이므로 각각 $4\pi/3$.
# 격자 구적 $\sum z_{ij}^2\,w_{ij}$ 가 이 값을 재현하는지, 또 $\int Y_1^0 Y_1^0 = 1$, $\int Y_0^0 Y_1^0 = 0$ 이 나오는지 본다.

# %%
C0, C1 = 0.28209479177387814, 0.4886025119029199        # Y₀⁰ = 1/(2√π), Y₁⁰ = √(3/4π)·z
print(f"{'nθ×nφ':>10} {'∫z² (≈4π/3=' + f'{4*math.pi/3:.5f})':>22} {'∫Y₁⁰Y₁⁰ (≈1)':>14} {'∫Y₀⁰Y₁⁰ (≈0)':>14}")
for n in [8, 32, 128]:
    d, wn = sphere_grid(n, 2 * n)
    z = d[..., 2]
    I_z2 = (z ** 2 * wn).sum().item()
    I_11 = ((C1 * z) ** 2 * wn).sum().item()
    I_01 = (C0 * C1 * z * wn).sum().item()
    print(f"{n:>4}×{2*n:<5} {I_z2:>22.6f} {I_11:>14.6f} {I_01:>14.2e}")
# 출력:      nθ×nφ    ∫z² (≈4π/3=4.18879)   ∫Y₁⁰Y₁⁰ (≈1)   ∫Y₀⁰Y₁⁰ (≈0)
# 출력:    8×16                  4.272171       1.019906      -7.45e-08
# 출력:   32×64                  4.193847       1.001207       5.96e-08
# 출력:  128×256                 4.189106       1.000075       0.00e+00
# → z² 적분이 4π/3 으로, Y₁⁰의 노름은 1로, 서로 다른 기저의 내적은 0(부동소수 오차 수준)으로 수렴한다.
#   원문 노트북이 128×256 격자에서 Gram 행렬 ≈ I 를 확인한 것이 바로 이 구적이다.

# %% [markdown]
# ## (d) 셀 중심 오프셋 `+0.5`의 역할
#
# `theta = (arange(n) + 0.5) * π/n` 은 격자점을 셀 **경계**가 아니라 **중심**에 둔다.
# 오프셋 0(왼쪽 끝점)을 쓰면 $\theta=0$(북극)이 포함되어 $\sin0=0$ 으로 그 행 전체 가중치가 0이 되고,
# 반대쪽 $\theta=\pi$ 행은 빠져 적분이 한쪽으로 치우친다(항상 과소평가). 중점 규칙은 극 양쪽을 대칭으로 다뤄 오차가 더 작다.

# %%
print(f"{'nθ×nφ':>10} {'offset=0.5 (중점)':>18} {'offset=0 (좌측 끝점)':>20}")
for n in [8, 32, 128]:
    _, wc = sphere_grid(n, 2 * n, offset=0.5)
    _, wl = sphere_grid(n, 2 * n, offset=0.0)
    print(f"{n:>4}×{2*n:<5} {wc.sum().item() - 4 * math.pi:>18.2e} {wl.sum().item() - 4 * math.pi:>20.2e}")
# 출력:      nθ×nφ    offset=0.5 (중점)     offset=0 (좌측 끝점)
# 출력:    8×16              8.11e-02            -1.62e-01
# 출력:   32×64              5.05e-03            -1.01e-02
# 출력:  128×256             3.17e-04            -6.29e-04
# → 같은 해상도에서 끝점 격자는 오차가 약 2배 크고 부호가 음(항상 과소평가): θ=0 행이 sinθ=0으로 죽고 θ=π 쪽 끝 셀은 빠지기 때문.
#   중점 격자는 극 양쪽을 대칭으로 다뤄 항상 약간 과대평가하지만 크기가 절반이다.

# %% [markdown]
# ## 시각화
#
# 왼쪽: 16×32 격자의 방향 벡터 $\mathbf d$ 를 3D 산점도로, 색은 가중치 $w$ (극 근처가 작다).
# 오른쪽: 해상도에 따른 $|\sum w - 4\pi|$ 수렴 곡선 (로그-로그, 중점 vs 끝점). 기울기 −2 → $O(1/n^2)$.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

d16, w16 = sphere_grid(16, 32)
P = d16.reshape(-1, 3).numpy()
W = w16.reshape(-1).numpy()

ns = np.array([4, 8, 16, 32, 64, 128, 256])
err_mid = np.array([abs(sphere_grid(n, 2 * n, offset=0.5)[1].sum().item() - 4 * math.pi) for n in ns])
err_left = np.array([abs(sphere_grid(n, 2 * n, offset=0.0)[1].sum().item() - 4 * math.pi) for n in ns])

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
                    subplot_titles=("격자 방향 벡터 d (색 = 가중치 w = sinθ dθ dφ)", "|Σw − 4π| 수렴 (log-log)"),
                    column_widths=[0.5, 0.5], horizontal_spacing=0.12)
fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", showlegend=False,
                           marker=dict(size=3, color=W, colorscale="Viridis",
                                       colorbar=dict(title="w", x=0.40, len=0.7)),
                           hovertemplate="w=%{marker.color:.4f}<extra></extra>"), row=1, col=1)
fig.add_trace(go.Scatter(x=ns, y=err_mid, mode="lines+markers", name="offset=0.5 (중점)"), row=1, col=2)
fig.add_trace(go.Scatter(x=ns, y=err_left, mode="lines+markers", name="offset=0 (끝점)"), row=1, col=2)
fig.add_trace(go.Scatter(x=ns, y=err_mid[0] * (ns[0] / ns) ** 2, mode="lines", name="∝ 1/n² 참조",
                         line=dict(dash="dash", color="gray")), row=1, col=2)
fig.update_xaxes(type="log", title_text="nθ (nφ = 2nθ)", row=1, col=2)
fig.update_yaxes(type="log", title_text="|Σw − 4π|", row=1, col=2)
fig.update_layout(width=1200, height=520, title_text="sphere_grid: 구면 격자 구적",
                  scene=dict(aspectmode="cube"), legend=dict(x=0.75, y=0.98))
_show(fig)
fig.write_image(os.path.join(HERE, "expy.png"), scale=2)
print("saved:", os.path.join(HERE, "expy.png"))
# 출력: saved: /home/sungwoo/projects/swcho/gsplat/fm/sh/.fm/hints/f79896c7-36ce-429d-abc4-446d96d9916c/expy.png

# %% [markdown]
# ## 정리
#
# - `sphere_grid`는 $\int_{S^2} f\,d\Omega \approx \sum f(\mathbf d_{ij})\,w_{ij}$ 라는 **중점 규칙 격자 구적**을 위한 노드(`d`)와 가중치(`w`)를 만든다.
# - $w=\sin\theta\,\Delta\theta\,\Delta\varphi$ 의 $\sin\theta$ 는 극 근처 셀이 좁아지는 구면 기하를 보정한다. 없으면 $\sum w = 2\pi^2$ 로 틀리고, 있으면 $4\pi$ 로 $O(1/n^2)$ 수렴.
# - `+0.5` 오프셋은 $\theta=0$ 의 퇴화(가중치 0)와 한쪽 치우침을 피하는 셀 중심 배치다.
# - 이 구적이 노트북의 정규직교성 확인($\text{Gram}\approx I$)과 SH 계수 투영 계산의 토대다.
