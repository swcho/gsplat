# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # SH 기저의 정규직교성을 수치로 확인하기 — Gram 행렬 `einsum("abi,abj,ab->ij", B, B, w)`
#
# 실수형 SH 기저 $Y_i$ ($i = 0,\dots,15$, $\ell \le 3$)는 구면 위에서 **정규직교**한다.
#
# $$
# G_{ij} \;=\; \int_{S^2} Y_i(\mathbf d)\,Y_j(\mathbf d)\,d\Omega \;=\; \delta_{ij},
# \qquad d\Omega = \sin\theta\,d\theta\,d\varphi
# $$
#
# 적분을 컴퓨터로 확인하려면 $(\theta,\varphi)$ 격자 위에서 **구적(quadrature)** 으로 근사한다.
#
# $$
# G_{ij} \;\approx\; \sum_{a,b} B[a,b,i]\;B[a,b,j]\;w[a,b],
# \qquad w[a,b] = \sin\theta_a\,\Delta\theta\,\Delta\varphi
# $$
#
# 이 합이 바로 `torch.einsum("abi,abj,ab->ij", B, B, w)` 한 줄이다. 결과 `G`가 16×16 단위행렬과 얼마나 다른지
# 최대 절대 오차로 보면 정규직교성이 확인된다.
#
# 이 스크립트는 노트북 `sh_walkthrough.py`의 `sh_bases` / `sphere_grid`를 그대로 가져와 다음을 단계적으로 보여준다.
# 1. 아주 작은 격자에서 `einsum`의 의미를 for 루프로 손으로 풀어 보기
# 2. 16×16 Gram 행렬과 단위행렬의 최대 오차
# 3. 격자 해상도를 바꾸며 오차가 줄어드는 표
# 4. 가중치 `w`를 빼면 정규직교성이 깨지는 대비 실험
# 5. 부호 규약을 바꿔도 Gram은 여전히 단위행렬
# 6. 두 Gram 행렬(가중치 O / X) 히트맵 → `expy.png`

# %%
# 필요 패키지: torch, numpy, plotly, kaleido
import math
import os

import numpy as np
import torch

torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
DEVICE = "cpu"                        # 검증용이므로 CPU. float64로 계산해 구적 오차와 부동소수 오차를 구분한다.
torch.set_default_dtype(torch.float64)


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# %% [markdown]
# ## 0. 노트북의 `sh_bases`, `sphere_grid` 재사용
#
# - `sh_bases(dirs, 3)`: 단위 방향 `[..., 3]` → 16개 기저값 `[..., 16]` (순서 $k=\ell^2+\ell+m$, Condon–Shortley 부호)
# - `sphere_grid(nθ, nφ)`: 셀 중심 격자의 방향 `d[nθ,nφ,3]`와 적분 가중치 `w[nθ,nφ]`, $\sum w \approx 4\pi$

# %%
LM = [(l, m) for l in range(4) for m in range(-l, l + 1)]     # k번째 기저의 (ℓ, m)
C0 = 0.28209479177387814                                       # Y₀⁰ = 1/(2√π)


def sh_bases(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """단위 방향 dirs[..., 3] → SH 기저값 [..., (degree+1)²] (sh_walkthrough.py와 동일)."""
    x, y, z = dirs.unbind(-1)
    out = [torch.full_like(x, C0)]                                                      # (0, 0)
    if degree >= 1:
        c = 0.4886025119029199                                                          # √(3/4π)
        out += [-c * y, c * z, -c * x]                                                  # (1,-1) (1,0) (1,1)
    if degree >= 2:
        c1, c2, c3 = 1.0925484305920792, 0.31539156525252005, 0.5462742152960396        # √(15/4π) √(5/16π) √(15/16π)
        out += [c1 * x * y, -c1 * y * z, c2 * (3 * z * z - 1), -c1 * x * z, c3 * (x * x - y * y)]
    if degree >= 3:
        c1, c2, c3, c4, c5 = 0.5900435899266435, 2.890611442640554, 0.4570457994644658, 0.3731763325901154, 1.445305721320277
        out += [-c1 * y * (3 * x * x - y * y), c2 * x * y * z, -c3 * y * (5 * z * z - 1), c4 * z * (5 * z * z - 3),
                -c3 * x * (5 * z * z - 1), c5 * z * (x * x - y * y), -c1 * x * (x * x - 3 * y * y)]
    return torch.stack(out, dim=-1)


def sphere_grid(n_theta: int = 128, n_phi: int = 256, device=DEVICE):
    """구면을 (θ, φ) 격자로 덮는 방향 벡터 d[nθ,nφ,3]와 적분 가중치 w = sinθ dθ dφ (Σw ≈ 4π)."""
    theta = (torch.arange(n_theta, device=device) + 0.5) * math.pi / n_theta            # 극각: z축에서의 각
    phi = (torch.arange(n_phi, device=device) + 0.5) * 2 * math.pi / n_phi - math.pi    # 방위각
    th, ph = torch.meshgrid(theta, phi, indexing="ij")
    d = torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], dim=-1)
    w = th.sin() * (math.pi / n_theta) * (2 * math.pi / n_phi)
    return d, w


dirs, w = sphere_grid()
B = sh_bases(dirs, 3)
print("B:", tuple(B.shape), " w:", tuple(w.shape), " Σw =", round(w.sum().item(), 6), " 4π =", round(4 * math.pi, 6))
# 출력: B: (128, 256, 16)  w: (128, 256)  Σw = 12.566686  4π = 12.566371

# %% [markdown]
# ## 1. `einsum("abi,abj,ab->ij")`를 손으로 풀어 보기 (2×2 격자, 기저 2개)
#
# 첨자 문자열의 뜻:
# - 입력 `abi` = `B[a,b,i]`, `abj` = `B[a,b,j]`, `ab` = `w[a,b]`
# - 출력 `ij` → 출력에 **없는** 첨자 `a, b`는 곱한 뒤 **합산**된다
#
# $$
# G_{ij} = \sum_a \sum_b B_{abi}\,B_{abj}\,w_{ab}
# $$
#
# 격자를 $2\times 2$, 기저를 $Y_0^0, Y_1^0$ 두 개만 잡아 for 루프로 같은 값을 만들어 본다.
# 격자가 너무 거칠어 값 자체는 단위행렬과 멀지만, **einsum이 무엇을 계산하는지**는 정확히 드러난다.

# %%
d2, w2 = sphere_grid(2, 2)
B2 = sh_bases(d2, 1)[..., [0, 2]]                 # k=0 (Y₀⁰), k=2 (Y₁⁰ ∝ z) 두 개만
print("B2[a,b,i] =\n", B2)
print("w2[a,b] =\n", w2)

G_loop = torch.zeros(2, 2)
for i in range(2):
    for j in range(2):
        s = 0.0
        for a in range(2):
            for b in range(2):
                s += B2[a, b, i] * B2[a, b, j] * w2[a, b]
        G_loop[i, j] = s
G_einsum = torch.einsum("abi,abj,ab->ij", B2, B2, w2)
print("for 루프 G =\n", G_loop)
print("einsum  G =\n", G_einsum)
print("두 결과 동일?", torch.allclose(G_loop, G_einsum))
# 출력: B2[a,b,i] =
#  tensor([[[ 0.2821,  0.3455],
#          [ 0.2821,  0.3455]],
#         [[ 0.2821, -0.3455],
#          [ 0.2821, -0.3455]]])
# w2[a,b] =
#  tensor([[3.4894, 3.4894],
#         [3.4894, 3.4894]])
# for 루프 G =
#  tensor([[1.1107, 0.0000],
#         [0.0000, 1.6661]])
# einsum  G =
#  tensor([[1.1107, 0.0000],
#         [0.0000, 1.6661]])
# 두 결과 동일? True

# %% [markdown]
# 관찰: 2×2 격자에서도 $Y_0^0$과 $Y_1^0$의 **직교성**(비대각 0)은 대칭성 덕분에 정확히 나오지만,
# **정규성**(대각 1)은 아직 멀다(1.11, 1.67). 격자를 촘촘히 하면 두 값 모두 1에 수렴한다 → 다음 셀.

# %% [markdown]
# ## 2. 노트북과 같은 16×16 Gram 행렬 — 단위행렬과의 최대 차이

# %%
gram = torch.einsum("abi,abj,ab->ij", B, B, w)                    # [16, 16]
err = (gram - torch.eye(16)).abs()
print("Gram 행렬과 단위행렬의 최대 차이:", err.max().item())
print("대각 성분(≈1):", gram.diagonal())
print("가장 큰 비대각 성분의 절댓값:", (gram - torch.diag(gram.diagonal())).abs().max().item())
# 출력: Gram 행렬과 단위행렬의 최대 차이: 0.00017581221567919414
# 대각 성분(≈1): tensor([1.0000, 1.0000, 1.0001, 1.0000, 1.0000, 1.0000, 1.0001, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0002, 1.0000,
#         1.0000, 1.0000])
# 가장 큰 비대각 성분의 절댓값: 0.00011506576030109739

# %% [markdown]
# 대각은 1, 비대각은 0에서 모두 $10^{-4}$ 수준으로만 벗어난다(가장 큰 오차는 $z$만 쓰는 $Y_3^0$, k=12).
# 부동소수 오차($10^{-16}$)보다 훨씬 크므로, 남은 오차는 **격자 구적의 이산화 오차**(θ 방향 중점 규칙)이며
# 해상도를 올리면 줄어들어야 한다 → 다음 셀.

# %% [markdown]
# ## 3. 격자 해상도 vs 최대 오차
#
# θ 방향 중점(midpoint) 규칙의 오차는 $O(\Delta\theta^2)$ 이므로, 해상도를 2배 올리면 오차가 약 1/4로 줄어야 한다.

# %%
print(f"{'nθ×nφ':>10} {'max|G-I|':>14} {'비율(이전/현재)':>16}")
prev = None
for n in [8, 16, 32, 64, 128, 256, 512]:
    d_, w_ = sphere_grid(n, 2 * n)
    G_ = torch.einsum("abi,abj,ab->ij", sh_bases(d_, 3), sh_bases(d_, 3), w_)
    e = (G_ - torch.eye(16)).abs().max().item()
    ratio = f"{prev / e:6.2f}" if prev else "     -"
    print(f"{n:4d}×{2*n:<5d} {e:14.3e} {ratio:>16}")
    prev = e
# 출력:      nθ×nφ       max|G-I|        비율(이전/현재)
#    8×16         5.420e-02                -
#   16×32         1.174e-02             4.62
#   32×64         2.841e-03             4.13
#   64×128        7.046e-04             4.03
#  128×256        1.758e-04             4.01
#  256×512        4.393e-05             4.00
#  512×1024       1.098e-05             4.00

# %% [markdown]
# 해상도 2배 → 오차 약 1/4(비율이 4.00으로 수렴). 중점 규칙의 2차 수렴 $O(\Delta\theta^2)$이 그대로 보인다.
# 노트북 기본값(128×256)의 $1.8\times10^{-4}$는 "거의 0"으로 충분하며, 더 줄이고 싶으면 격자를 키우면 된다.

# %% [markdown]
# ## 4. 대비 실험 — 가중치 `w`를 빼면?
#
# `einsum("abi,abj->ij", B, B)`는 격자점을 **균등하게** 세는 셈이 되어, 극(θ≈0, π) 근처의 좁은 영역이
# 적도 근처와 같은 비중으로 들어간다. 그러면 적분이 아니라 "격자점 위 표본 내적"이 되어 정규직교성이 깨진다.
#
# (비교를 공정하게 하기 위해 $\sum = 4\pi$가 되도록 상수만 곱해 스케일을 맞춘다.)

# %%
n_pts = B.shape[0] * B.shape[1]
gram_now = torch.einsum("abi,abj->ij", B, B) * (4 * math.pi / n_pts)   # w 대신 균등 가중치 4π/N
err_now = (gram_now - torch.eye(16)).abs()
print("w 없이(균등 가중) Gram과 단위행렬의 최대 차이:", err_now.max().item())
print("대각 성분:", gram_now.diagonal())
print("최대 비대각 절댓값:", (gram_now - torch.diag(gram_now.diagonal())).abs().max().item())
# 가장 크게 틀어진 대각 항 확인
k_worst = int((gram_now.diagonal() - 1).abs().argmax())
print(f"가장 크게 틀린 기저: k={k_worst}, (ℓ,m)={LM[k_worst]}, G_kk={gram_now[k_worst, k_worst].item():.4f}")
# 출력: w 없이(균등 가중) Gram과 단위행렬의 최대 차이: 0.8593750000000004
# 대각 성분: tensor([1.0000, 0.7500, 1.5000, 0.7500, 0.7031, 0.9375, 1.7187, 0.9375, 0.7031, 0.6836, 0.8203, 1.0664, 1.8594, 1.0664,
#         0.8203, 0.6836])
# 최대 비대각 절댓값: 0.8592329428042207
# 가장 크게 틀린 기저: k=12, (ℓ,m)=(3, 0), G_kk=1.8594

# %% [markdown]
# 대각이 0.68~1.86으로 흩어지고, 비대각에도 0.86 크기의 성분이 생긴다. 특히 $z$에만 의존하는 $m=0$ 기저들
# ($Y_1^0, Y_2^0, Y_3^0$: k=2, 6, 12)이 크게 부풀고 서로 섞인다 — 극 근처($|z|\approx1$)가 과대 대표되기 때문이다.
# **`w = sinθ dθ dφ`가 있어야 격자 합이 진짜 구면 적분이 된다.**

# %% [markdown]
# ## 5. 부호 규약을 바꿔도 Gram은 단위행렬
#
# 노트북의 기저는 Condon–Shortley 부호를 쓴다($Y_1^{1} = -\sqrt{3/4\pi}\,x$ 등).
# 어떤 기저 $Y_k$에 $-1$을 곱하면 $G_{kk} = \int (-Y_k)^2 = G_{kk}$, $G_{kj} = -\int Y_kY_j = 0$ 이므로
# Gram 행렬은 변하지 않는다. 즉 **정규직교성은 부호 규약과 무관**하고, 부호는 계수의 부호에만 영향을 준다.

# %%
sign = torch.tensor([(-1.0) ** m if m > 0 else 1.0 for (l, m) in LM])   # m>0 기저의 부호를 뒤집는 다른 규약
B_alt = B * sign
gram_alt = torch.einsum("abi,abj,ab->ij", B_alt, B_alt, w)
print("부호 뒤집은 기저 개수:", int((sign < 0).sum()), "→ k =", [k for k in range(16) if sign[k] < 0])
print("규약 변경 후 max|G-I|  :", (gram_alt - torch.eye(16)).abs().max().item())
print("원래 Gram과의 최대 차이:", (gram_alt - gram).abs().max().item())
# 출력: 부호 뒤집은 기저 개수: 4 → k = [3, 7, 13, 15]
# 규약 변경 후 max|G-I|  : 0.00017581221567919414
# 원래 Gram과의 최대 차이: 2.0218737892530766e-16

# %% [markdown]
# ## 6. 시각화 — Gram 행렬 히트맵 (가중치 O vs X)
#
# 값이 0을 중심으로 양·음으로 갈리므로 **발산형(diverging) 색상**(0 = 중립 회백색)을 쓴다.
# 왼쪽은 대각만 1(균일한 파란 대각선), 오른쪽은 대각의 진하기가 흩어지고 $m=0$ 기저들 사이(0,0)-(2,0), (1,0)-(3,0) 등
# 비대각에도 0이 아닌 성분이 나타난다.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

labels = [f"{l},{m:+d}" for (l, m) in LM]
vmax = float(max(gram.abs().max(), gram_now.abs().max()))
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                    subplot_titles=(f"w = sinθ dθ dφ 포함  (max|G−I| = {err.max().item():.1e})",
                                    f"w 없이 균등 가중  (max|G−I| = {err_now.max().item():.2f})"))
for col, G_ in enumerate([gram, gram_now], start=1):
    fig.add_trace(go.Heatmap(z=G_.numpy(), x=labels, y=labels, zmin=-vmax, zmax=vmax, zmid=0.0,
                             colorscale="RdBu", showscale=(col == 2),
                             colorbar=dict(title="G_ij", thickness=12, len=0.9),
                             hovertemplate="Y(%{y}) · Y(%{x})<br>G = %{z:.4f}<extra></extra>"),
                  row=1, col=col)
    fig.update_yaxes(autorange="reversed", row=1, col=col, tickfont=dict(size=9))
    fig.update_xaxes(tickangle=-60, row=1, col=col, tickfont=dict(size=9))
fig.update_layout(title="SH 기저 Gram 행렬  G_ij = Σ_ab B[a,b,i] B[a,b,j] w[a,b]   (축 라벨: ℓ,m)",
                  width=1050, height=520, margin=dict(t=90, b=60), template="plotly_white")
_show(fig)

out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expy.png")
fig.write_image(out_png, scale=2)
print("저장:", out_png)
# 출력: 저장: /home/sungwoo/projects/swcho/gsplat/fm/sh/.fm/hints/9fa7bd29-2486-48b6-89e1-1a9f53c71fd2/expy.png

# %% [markdown]
# ## 정리
#
# | 단계 | 코드 | 확인 내용 |
# |---|---|---|
# | 기저·격자 | `B = sh_bases(dirs, 3)`, `dirs, w = sphere_grid()` | `B[nθ,nφ,16]`, `Σw ≈ 4π` |
# | Gram 구적 | `einsum("abi,abj,ab->ij", B, B, w)` | $G_{ij}\approx\int Y_iY_j\,d\Omega$ |
# | 판정 | `(gram - eye(16)).abs().max()` | 128×256에서 $\approx 1.8\times10^{-4}$, 해상도 2배마다 1/4 |
# | 반례 | `w` 생략 | 대각 0.68~1.86, 비대각 0.86 → 정규직교성 붕괴 |
# | 불변성 | 기저 부호 뒤집기 | Gram 불변 → 정규직교성은 부호 규약과 무관 |
