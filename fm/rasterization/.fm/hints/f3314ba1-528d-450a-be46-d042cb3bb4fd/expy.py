# %% [markdown]
# # 3D 공분산 $\Sigma$의 고유값 제곱근 = 정렬된 `scales`
#
# 3DGS에서 Gaussian의 모양은 학습 파라미터 `quats`(회전 $R$)와 `scales`($s = (s_0, s_1, s_2)$)로부터
#
# $$\Sigma = (R\,\mathrm{diag}(s))\,(R\,\mathrm{diag}(s))^\top = R\,\mathrm{diag}(s)^2\,R^\top$$
#
# 로 만들어진다. $\Sigma$를 직접 최적화하면 양정치성이 깨질 수 있어서 항상 유효한 $\Sigma$가 나오는 이 형태를 쓴다.
#
# 그런데 이 식은 그 자체로 **대칭행렬의 고유분해(스펙트럼 분해)** 와 똑같은 모양이다.
#
# $$\Sigma = Q\,\Lambda\,Q^\top,\qquad Q\ \text{직교},\ \ \Lambda = \mathrm{diag}(\lambda_0,\lambda_1,\lambda_2)$$
#
# $R$은 회전행렬이라 직교($R^\top R = I$)이고 $\mathrm{diag}(s)^2$는 대각이므로,
# $Q = R$, $\Lambda = \mathrm{diag}(s)^2$가 바로 고유분해다. 따라서
#
# $$\lambda_k = s_k^2 \;\Longrightarrow\; \sqrt{\lambda_k} = |s_k|$$
#
# 즉 **$\Sigma$의 고유값 제곱근은 `scales`를 정렬한 것과 같다**
# (`eigh`가 고유값을 오름차순으로 주므로 "정렬"이 붙는다).
# 기하학적으로는 $\sqrt{\lambda_k}$가 곧 1$\sigma$ 타원체의 **주축 반길이**이고, 그게 정의상 `scales`다.
# 그리고 고유벡터는 $R$의 열(= 주축 방향)이 된다(부호·순서 차이 제외).

# %%
# 필요 패키지: torch, numpy, plotly, kaleido
# (gsplat은 import하지 않는다 — JIT CUDA 빌드가 걸린다. 로직만 그대로 옮겨 온다.)
import math

import numpy as np
import torch
import torch.nn.functional as F

torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# %% [markdown]
# ## 1. `quat_to_rotmat` / `covar_from_quat_scale` — walkthrough와 동일한 로직
#
# `gsplat.cuda._math._quat_to_rotmat`은 쿼터니언을 정규화한 뒤 $(w,x,y,z)$로 회전행렬을 만든다.
# 공분산은 $M = R\,\mathrm{diag}(s)$를 만들고 $\Sigma = M M^\top$.

# %%
def quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """쿼터니언(w-first) → 회전행렬. gsplat `_quat_to_rotmat`와 동일."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2), 2 * (x * y - w * z),   2 * (x * z + w * y),
            2 * (x * y + w * z),   1 - 2 * (x**2 + z**2), 2 * (y * z - w * x),
            2 * (x * z - w * y),   2 * (y * z + w * x),   1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return R.reshape(quats.shape[:-1] + (3, 3))


def covar_from_quat_scale(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Σ = R S Sᵀ Rᵀ. walkthrough 2절과 동일."""
    R = quat_to_rotmat(q)          # [N,3,3]  (내부에서 q를 정규화)
    M = R * s[..., None, :]        # R @ diag(s)
    return M @ M.transpose(-1, -2)  # R S Sᵀ Rᵀ


def quat_z(deg: float) -> torch.Tensor:
    """z축 회전 쿼터니언 (w, x, y, z) — gsplat은 w-first 규약."""
    h = math.radians(deg) / 2
    return torch.tensor([math.cos(h), 0.0, 0.0, math.sin(h)])


# %% [markdown]
# ## 2. 장난감 Gaussian 하나로 $\Sigma$ 만들기
#
# walkthrough의 toy Gaussian 0과 같은 값: z축 30° 회전, `scales = (0.30, 0.12, 0.10)`.

# %%
q0 = quat_z(30.0)
s0 = torch.tensor([0.30, 0.12, 0.10])

R0 = quat_to_rotmat(q0)
Sigma0 = covar_from_quat_scale(q0[None], s0[None])[0]

print("R =\n", R0)
print("\nΣ =\n", Sigma0)
# 출력:
# R =
#  tensor([[ 0.8660, -0.5000,  0.0000],
#         [ 0.5000,  0.8660,  0.0000],
#         [ 0.0000,  0.0000,  1.0000]])
#
# Σ =
#  tensor([[0.0711, 0.0327, 0.0000],
#         [0.0327, 0.0333, 0.0000],
#         [0.0000, 0.0000, 0.0100]])

# %% [markdown]
# ## 3. `eigh`로 고유값·고유벡터 뽑기
#
# $\Sigma$는 대칭이므로 `torch.linalg.eigh`를 쓴다. 고유값은 **오름차순**으로 나온다.

# %%
evals, evecs = torch.linalg.eigh(Sigma0)

print("eigvals(Σ)       =", evals)
print("sqrt(eigvals(Σ)) =", evals.sqrt())
print("sorted(scales)   =", s0.sort().values)
print("max|Δ| =", (evals.sqrt() - s0.sort().values).abs().max().item())
# 출력:
# eigvals(Σ)       = tensor([0.0100, 0.0144, 0.0900])
# sqrt(eigvals(Σ)) = tensor([0.1000, 0.1200, 0.3000])
# sorted(scales)   = tensor([0.1000, 0.1200, 0.3000])
# max|Δ| = 7.450580596923828e-09

# %% [markdown]
# $\sqrt{\lambda} = (0.10, 0.12, 0.30)$ — `scales = (0.30, 0.12, 0.10)`을 오름차순 정렬한 것과 정확히 같다.
# 고유값 자체는 $s^2$이므로 $(0.01, 0.0144, 0.09)$.
#
# ## 4. 고유벡터 = $R$의 열 (부호·순서 제외)
#
# `eigh`가 준 $k$번째 고유벡터는 $s$를 오름차순 정렬했을 때 $k$번째로 오는 축, 즉
# $R[:,\ \mathrm{argsort}(s)_k]$와 같다. 부호는 $\pm$ 자유도가 있어 일치하지 않을 수 있다.

# %%
order = torch.argsort(s0)               # 오름차순 정렬 시 원래 축 인덱스
R_reordered = R0[:, order]              # R의 열을 같은 순서로 재배열

print("argsort(scales) =", order.tolist())
print("\neigh 고유벡터 (열 기준):\n", evecs)
print("\nR의 열을 같은 순서로:\n", R_reordered)

# 각 열끼리 부호를 맞춘 뒤 비교
signs = torch.sign((evecs * R_reordered).sum(dim=0))
print("\n열별 부호 =", signs.tolist())
print("부호 맞춘 뒤 max|Δ| =", (evecs * signs - R_reordered).abs().max().item())
# 출력:
# argsort(scales) = [2, 1, 0]
#
# eigh 고유벡터 (열 기준):
#  tensor([[ 0.0000,  0.5000, -0.8660],
#         [ 0.0000, -0.8660, -0.5000],
#         [ 1.0000,  0.0000, -0.0000]])
#
# R의 열을 같은 순서로:
#  tensor([[ 0.0000, -0.5000,  0.8660],
#         [ 0.0000,  0.8660,  0.5000],
#         [ 1.0000,  0.0000,  0.0000]])
#
# 열별 부호 = [1.0, -1.0, -1.0]
# 부호 맞춘 뒤 max|Δ| = 5.960464477539063e-08

# %% [markdown]
# ## 5. 랜덤 배치로 일반성 확인
#
# 임의의 쿼터니언·scales 1000개에 대해 $\sqrt{\lambda(\Sigma)} = \mathrm{sort}(s)$가 항상 성립하는지 본다.

# %%
torch.manual_seed(0)
Nb = 1000
qb = torch.randn(Nb, 4, dtype=torch.float64)
sb = torch.rand(Nb, 3, dtype=torch.float64) * 0.5 + 0.01

Sb = covar_from_quat_scale(qb, sb)
sqrt_eig = torch.linalg.eigvalsh(Sb).sqrt()      # [N,3] 오름차순
sorted_s = sb.sort(dim=-1).values                # [N,3] 오름차순

print("N =", Nb)
print("max|sqrt(eigvals(Σ)) - sort(scales)| =", (sqrt_eig - sorted_s).abs().max().item())
print("Σ 대칭성 max|Σ - Σᵀ|              =", (Sb - Sb.transpose(-1, -2)).abs().max().item())
print("det(Σ) vs (s0·s1·s2)² max|Δ|      =",
      (torch.linalg.det(Sb) - sb.prod(dim=-1) ** 2).abs().max().item())
print("trace(Σ) vs Σsᵢ² max|Δ|           =",
      (torch.diagonal(Sb, dim1=-2, dim2=-1).sum(-1) - (sb**2).sum(-1)).abs().max().item())
# 출력:
# N = 1000
# max|sqrt(eigvals(Σ)) - sort(scales)| = 2.048708425128609e-15
# Σ 대칭성 max|Σ - Σᵀ|              = 0.0
# det(Σ) vs (s0·s1·s2)² max|Δ|      = 1.0408340855860843e-17
# trace(Σ) vs Σsᵢ² max|Δ|           = 3.885780586188048e-16

# %% [markdown]
# 참고로 따라오는 항등식들:
#
# - $\det \Sigma = \prod_k \lambda_k = (s_0 s_1 s_2)^2$ → 타원체 부피 $\frac{4}{3}\pi\, s_0 s_1 s_2$
# - $\mathrm{tr}\,\Sigma = \sum_k \lambda_k = \sum_k s_k^2$
#
# ## 6. 시각화 — 1$\sigma$ 타원체의 주축 길이가 곧 `scales`
#
# $\Sigma$의 1$\sigma$ 등고면은 $\{x : x^\top \Sigma^{-1} x = 1\}$이고,
# 이는 단위구를 $M = R\,\mathrm{diag}(s)$로 보낸 상이다. 중심에서 주축 방향으로 그린 화살표의
# 길이를 재면 정확히 $s_k = \sqrt{\lambda_k}$가 나온다.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

R_np = R0.numpy().astype(np.float64)
s_np = s0.numpy().astype(np.float64)
Sig_np = Sigma0.numpy().astype(np.float64)
evals_np, evecs_np = np.linalg.eigh(Sig_np)

# --- 1σ 타원체: 단위구를 M = R diag(s)로 변환 ---
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 40)
sph = np.stack([np.outer(np.cos(u), np.sin(v)),
                np.outer(np.sin(u), np.sin(v)),
                np.outer(np.ones_like(u), np.cos(v))], axis=0)      # [3,60,40]
M_np = R_np * s_np[None, :]
ell = np.einsum("ij,jab->iab", M_np, sph)                            # [3,60,40]

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "xy"}]],
    subplot_titles=("1σ 타원체와 주축 (화살표 길이 = √λ = scales)",
                    "z=0 단면: 장축/단축 반길이 = 0.30 / 0.12"),
    horizontal_spacing=0.06,
)

fig.add_trace(
    go.Surface(x=ell[0], y=ell[1], z=ell[2], opacity=0.28, showscale=False,
               colorscale=[[0, "#6aa9ff"], [1, "#6aa9ff"]], name="1σ 타원체",
               hoverinfo="skip"),
    row=1, col=1,
)

axis_colors = ["#d62728", "#2ca02c", "#1f77b4"]
for k in range(3):
    d = evecs_np[:, k] * np.sqrt(evals_np[k])          # 고유벡터 × √λ
    lab = d + evecs_np[:, k] * 0.055                   # 라벨은 화살표 끝에서 조금 더 바깥
    fig.add_trace(
        go.Scatter3d(x=[0, d[0]], y=[0, d[1]], z=[0, d[2]], mode="lines+markers",
                     line=dict(color=axis_colors[k], width=7),
                     marker=dict(size=[2, 5], color=axis_colors[k]),
                     name=f"√λ{k} = {np.sqrt(evals_np[k]):.2f}"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(x=[lab[0]], y=[lab[1]], z=[lab[2]], mode="text",
                     text=[f"√λ{k}={np.sqrt(evals_np[k]):.2f}"],
                     textfont=dict(color=axis_colors[k], size=12),
                     showlegend=False, hoverinfo="skip"),
        row=1, col=1,
    )

# --- 오른쪽: z=0 평면 단면 (z축 회전이라 xy 블록만 보면 된다) ---
Sig2 = Sig_np[:2, :2]
ev2, ec2 = np.linalg.eigh(Sig2)
t = np.linspace(0, 2 * np.pi, 200)
circ = np.stack([np.cos(t), np.sin(t)])
M2 = ec2 * np.sqrt(ev2)[None, :]
pts = M2 @ circ
fig.add_trace(go.Scatter(x=pts[0], y=pts[1], mode="lines",
                         line=dict(color="#6aa9ff", width=2.5), name="1σ 타원(z=0)"),
              row=1, col=2)
for k in range(2):
    d = ec2[:, k] * np.sqrt(ev2[k])
    fig.add_trace(go.Scatter(x=[0, d[0]], y=[0, d[1]], mode="lines+markers+text",
                             line=dict(color=axis_colors[k + 1], width=4),  # 3D 패널의 λ1/λ2와 같은 색
                             marker=dict(size=[3, 7]),
                             text=["", f" {np.sqrt(ev2[k]):.2f}"], textposition="top right",
                             showlegend=False),
                  row=1, col=2)

lim = 0.36
fig.update_xaxes(range=[-lim, lim], title_text="x", row=1, col=2)
fig.update_yaxes(range=[-lim, lim], title_text="y", scaleanchor="x", scaleratio=1, row=1, col=2)
fig.update_layout(
    title="Σ = R diag(s)² Rᵀ 의 고유분해: √λ = sort(scales) = (0.10, 0.12, 0.30)",
    width=1200, height=560,
    scene=dict(  # 3D는 라벨이 잘리지 않도록 조금 더 여유 있는 범위
        xaxis=dict(range=[-0.44, 0.44], title="x"),
        yaxis=dict(range=[-0.44, 0.44], title="y"),
        zaxis=dict(range=[-0.44, 0.44], title="z"),
        aspectmode="cube",
        camera=dict(eye=dict(x=0.95, y=-1.35, z=1.25)),
    ),
)

_show(fig)
fig.write_image("expy.png", scale=2)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 정리
#
# - $\Sigma = R\,\mathrm{diag}(s)^2\,R^\top$는 이미 고유분해 형태 → 고유값 $\lambda_k = s_k^2$, 고유벡터 = $R$의 열.
# - 따라서 **$\sqrt{\lambda_k}$는 `scales`를 (오름차순) 정렬한 값**이고, 이는 1$\sigma$ 타원체의 주축 반길이다.
# - `eigh`/`eigvalsh`는 고유값을 오름차순으로 주므로 `scales`가 정렬 안 된 상태라면 순서가 달라 보일 뿐이다.
# - 고유벡터의 부호는 임의($\pm$)라서 $R$의 열과 부호까지 같지는 않다.
