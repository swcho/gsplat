# %% [markdown]
# # world→camera 행렬 $[R\,|\,\mathbf t]$에서 카메라 위치 복원하기 — $\mathbf o_{\text{cam}} = -R^\top\mathbf t$
#
# 이 스크립트는 다음을 작은 실행 예제로 확인한다.
#
# 1. 회전 행렬은 직교: $R^\top R = I$, 따라서 $R^{-1} = R^\top$
# 2. 카메라 자세($R_{c2w}$)와 위치 $\mathbf o$로 c2w를 만들고 w2c $= \text{c2w}^{-1}$를 구한 뒤,
#    w2c의 $[R\,|\,\mathbf t]$에서 $-R^\top\mathbf t$가 원래 위치 $\mathbf o$와 일치함
# 3. $4\times4$ 역행렬이 정확히 $\begin{pmatrix} R^\top & -R^\top\mathbf t\\ \mathbf 0^\top & 1\end{pmatrix}$와 같음
# 4. `sh_walkthrough.py`의 예제(카메라 $(0,0,-5)$, $R=I$, $\mathbf t=(0,0,5)$) 손계산 재현
# 5. 잘못된 공식 $-\mathbf t$, $-R\mathbf t$를 쓰면 얼마나 틀리는지 표
# 6. 여러 카메라에서 Gaussian 중심으로의 시점 방향 $\mathbf d = (\boldsymbol\mu - \mathbf o)/\|\cdot\|$ 계산과 3D 시각화
#
# 필요 패키지: torch, numpy, plotly, kaleido (PNG 저장용)

# %%
import os
import math

import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go

torch.manual_seed(0)
np.random.seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# %% [markdown]
# ## 1. 회전 행렬은 직교한다: $R^\top R = I$
#
# 축-각(axis-angle) 표현에서 회전 행렬을 만드는 로드리게스 공식:
#
# $$
# R = I + \sin\theta\,K + (1-\cos\theta)\,K^2,\qquad
# K = \begin{pmatrix} 0 & -k_z & k_y\\ k_z & 0 & -k_x\\ -k_y & k_x & 0\end{pmatrix}
# $$
#
# ($\mathbf k$는 단위 회전축). 이렇게 만든 $R$의 열들이 서로 수직인 단위벡터라서 $R^\top R = I$가 된다.
# 즉 **역행렬을 따로 계산할 필요 없이 전치가 곧 역행렬**이다.

# %%
def skew(k: torch.Tensor) -> torch.Tensor:
    kx, ky, kz = k
    return torch.tensor([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=k.dtype)


def axis_angle_to_R(axis: torch.Tensor, theta: float) -> torch.Tensor:
    k = F.normalize(axis, dim=0)
    K = skew(k)
    return torch.eye(3, dtype=k.dtype) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


R = axis_angle_to_R(torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64), theta=0.9)
print("R =\n", R)
print("Rᵀ R =\n", R.T @ R)
print("‖Rᵀ R − I‖_max =", (R.T @ R - torch.eye(3, dtype=R.dtype)).abs().max().item())
print("det R =", torch.linalg.det(R).item(), " (회전이면 +1)")
print("‖R⁻¹ − Rᵀ‖_max =", (torch.linalg.inv(R) - R.T).abs().max().item())
# 출력:
# R =
#  tensor([[ 0.6937, -0.0268,  0.7198],
#         [ 0.3151,  0.9099, -0.2698],
#         [-0.6477,  0.4139,  0.6396]], dtype=torch.float64)
# Rᵀ R =
#  tensor([[ 1.0000, -0.0000, -0.0000],
#         [-0.0000,  1.0000,  0.0000],
#         [-0.0000,  0.0000,  1.0000]], dtype=torch.float64)
# ‖Rᵀ R − I‖_max = 2.220446049250313e-16
# det R = 1.0  (회전이면 +1)
# ‖R⁻¹ − Rᵀ‖_max = 1.1102230246251565e-16

# %% [markdown]
# ## 2. c2w → w2c → $-R^\top\mathbf t$ 가 원래 카메라 위치인가
#
# 카메라의 **자세** $R_{c2w}$(카메라 축을 world에서 본 것, 열벡터 = 카메라 x, y, z축)와 **위치** $\mathbf o$로
# camera→world 행렬을 만든다:
#
# $$
# \text{c2w} = \begin{pmatrix} R_{c2w} & \mathbf o \\ \mathbf 0^\top & 1\end{pmatrix}
# $$
#
# gsplat의 `viewmat`은 그 역행렬 world→camera $= [R\,|\,\mathbf t]$ 이다. 여기서 $R = R_{c2w}^\top$, $\mathbf t = -R\,\mathbf o$.
# 이 $R, \mathbf t$만 갖고 $\mathbf o$를 되찾는 식이 $-R^\top\mathbf t$ 이다.

# %%
def make_c2w(R_c2w: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    c2w = torch.eye(4, dtype=R_c2w.dtype)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = o
    return c2w


def cam_pos_from_w2c(w2c: torch.Tensor) -> torch.Tensor:
    """카드의 공식: o_cam = −Rᵀ t  (sh_walkthrough.py의 view_dirs와 같은 계산)."""
    R, t = w2c[:3, :3], w2c[:3, 3]
    return -R.T @ t


o_true = torch.tensor([1.5, -2.0, 4.0], dtype=torch.float64)                       # 카메라 world 위치
R_c2w = axis_angle_to_R(torch.tensor([0.3, 1.0, -0.2], dtype=torch.float64), 1.3)  # 임의 자세
c2w = make_c2w(R_c2w, o_true)
w2c = torch.linalg.inv(c2w)                                                        # gsplat viewmat 형태
R, t = w2c[:3, :3], w2c[:3, 3]

print("w2c (viewmat) =\n", w2c)
print("t            =", t.tolist())
print("−Rᵀ t        =", cam_pos_from_w2c(w2c).tolist())
print("원래 위치 o   =", o_true.tolist())
print("오차 ‖−Rᵀt − o‖ =", (cam_pos_from_w2c(w2c) - o_true).norm().item())
print("검산: R·o + t =", (R @ o_true + t).tolist(), " ← 카메라 자신은 camera 좌표로 원점")
# 출력:
# w2c (viewmat) =
#  tensor([[ 0.3258,  0.0132, -0.9453,  3.3189],
#         [ 0.3758,  0.9157,  0.1423,  0.6987],
#         [ 0.8675, -0.4016,  0.2934, -3.2782],
#         [ 0.0000, -0.0000,  0.0000,  1.0000]], dtype=torch.float64)
# t            = [3.3189361547919884, 0.6986818618743109, -3.278186458440461]
# −Rᵀ t        = [1.5000000000000016, -1.9999999999999996, 3.9999999999999987]
# 원래 위치 o   = [1.5, -2.0, 4.0]
# 오차 ‖−Rᵀt − o‖ = 2.094764613337708e-15
# 검산: R·o + t = [4.440892098500626e-16, -1.1102230246251565e-16, -4.440892098500626e-16]  ← 카메라 자신은 camera 좌표로 원점

# %% [markdown]
# ## 3. $4\times4$ 역행렬 vs. 닫힌 형태 $\begin{pmatrix} R^\top & -R^\top\mathbf t\\ \mathbf 0^\top & 1\end{pmatrix}$
#
# $\mathbf x_c = R\mathbf x_w + \mathbf t$ 를 $\mathbf x_w$에 대해 풀면 $\mathbf x_w = R^\top\mathbf x_c - R^\top\mathbf t$.
# 그러므로 w2c의 역행렬(= c2w)은 위 블록 형태이고, **4번째 열이 곧 카메라 위치**다.
# `torch.linalg.inv`로 직접 구한 값과 비교해 본다.

# %%
def inv_rigid(w2c: torch.Tensor) -> torch.Tensor:
    R, t = w2c[:3, :3], w2c[:3, 3]
    out = torch.eye(4, dtype=w2c.dtype)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


c2w_closed = inv_rigid(w2c)
c2w_numeric = torch.linalg.inv(w2c)
print("닫힌 형태 c2w =\n", c2w_closed)
print("‖닫힌 형태 − linalg.inv‖_max =", (c2w_closed - c2w_numeric).abs().max().item())
print("c2w 4번째 열 (위 3성분) =", c2w_numeric[:3, 3].tolist(), " == 카메라 위치")
print("c2w @ (0,0,0,1)ᵀ       =", (c2w_numeric @ torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64))[:3].tolist(),
      " ← camera 원점을 world로 보낸 것")
# 출력:
# 닫힌 형태 c2w =
#  tensor([[ 0.3258,  0.3758,  0.8675,  1.5000],
#         [ 0.0132,  0.9157, -0.4016, -2.0000],
#         [-0.9453,  0.1423,  0.2934,  4.0000],
#         [ 0.0000,  0.0000,  0.0000,  1.0000]], dtype=torch.float64)
# ‖닫힌 형태 − linalg.inv‖_max = 8.881784197001252e-16
# c2w 4번째 열 (위 3성분) = [1.5000000000000007, -1.9999999999999996, 3.9999999999999996]  == 카메라 위치
# c2w @ (0,0,0,1)ᵀ       = [1.5000000000000007, -1.9999999999999996, 3.9999999999999996]  ← camera 원점을 world로 보낸 것

# %% [markdown]
# ## 4. 노트북 예제 재현: 카메라 $(0,0,-5)$, $R = I$
#
# `sh_walkthrough.py` 4절: `viewmat = eye(4); viewmat[:3,3] = (0,0,5)`.
# $\mathbf t = -R\,\mathbf o = -(0,0,-5) = (0,0,5)$ 이므로, 복원하면 $-R^\top\mathbf t = -(0,0,5) = (0,0,-5)$.
# 이어서 두 Gaussian 중심 $(0,0,0)$, $(2,1,3)$으로의 시점 방향을 정규화까지 계산한다.

# %%
def view_dirs(means: torch.Tensor, viewmat: torch.Tensor) -> torch.Tensor:
    """3DGS의 시점 방향: 카메라 위치 −Rᵀt 에서 각 Gaussian 중심을 향하는 벡터 (노트북과 동일)."""
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    cam_pos = -R.T @ t
    return means - cam_pos


viewmat_nb = torch.eye(4)
viewmat_nb[:3, 3] = torch.tensor([0.0, 0.0, 5.0])
means_nb = torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, 3.0]])
print("카메라 위치 −Rᵀt =", cam_pos_from_w2c(viewmat_nb).tolist())
print("원점 Gaussian의 camera 좌표 R·μ + t =", (viewmat_nb[:3, :3] @ means_nb[0] + viewmat_nb[:3, 3]).tolist(), " (카메라 앞 5)")
print("시점 방향 d       =\n", F.normalize(view_dirs(means_nb, viewmat_nb), dim=-1))
print("손계산 (2,1,8)/√69 =", (torch.tensor([2.0, 1.0, 8.0]) / math.sqrt(69)).tolist())
# 출력:
# 카메라 위치 −Rᵀt = [0.0, 0.0, -5.0]
# 원점 Gaussian의 camera 좌표 R·μ + t = [0.0, 0.0, 5.0]  (카메라 앞 5)
# 시점 방향 d       =
#  tensor([[0.0000, 0.0000, 1.0000],
#         [0.2408, 0.1204, 0.9631]])
# 손계산 (2,1,8)/√69 = [0.24077171087265015, 0.12038585543632507, 0.9630868434906006]

# %% [markdown]
# ## 5. 잘못된 공식과 비교: $-\mathbf t$, $-R\,\mathbf t$
#
# - $-\mathbf t$: $R = I$일 때만 우연히 맞는다. 회전이 있으면 $\mathbf t = -R\mathbf o$ 에 들어간 회전을 되돌리지 못한다.
# - $-R\,\mathbf t$: 되돌리는 방향이 반대다. $R$을 한 번 더 곱해 회전이 두 배가 된다.
#
# 회전각을 0°에서 150°까지 키우며 세 공식의 오차 $\|\hat{\mathbf o} - \mathbf o\|$ 를 표로 본다.

# %%
o_fixed = torch.tensor([2.0, 1.0, -3.0], dtype=torch.float64)
axis = torch.tensor([0.2, 1.0, 0.4], dtype=torch.float64)
print(f"{'회전각(°)':>9} | {'−Rᵀt 오차':>10} | {'−t 오차':>10} | {'−Rt 오차':>10}")
print("-" * 50)
for deg in [0, 30, 60, 90, 120, 150]:
    w2c_i = torch.linalg.inv(make_c2w(axis_angle_to_R(axis, math.radians(deg)), o_fixed))
    R_i, t_i = w2c_i[:3, :3], w2c_i[:3, 3]
    err_ok = (-R_i.T @ t_i - o_fixed).norm().item()
    err_t = (-t_i - o_fixed).norm().item()
    err_Rt = (-R_i @ t_i - o_fixed).norm().item()
    print(f"{deg:>9d} | {err_ok:>10.2e} | {err_t:>10.4f} | {err_Rt:>10.4f}")
# 출력:
#    회전각(°) |    −Rᵀt 오차 |      −t 오차 |     −Rt 오차
# --------------------------------------------------
#         0 |   0.00e+00 |     0.0000 |     0.0000
#        30 |   2.22e-16 |     1.9345 |     3.7372
#        60 |   9.93e-16 |     3.7372 |     6.4730
#        90 |   1.09e-15 |     5.2852 |     7.4744
#       120 |   9.93e-16 |     6.4730 |     6.4730
#       150 |   1.20e-15 |     7.2197 |     3.7372
# (−t 는 각이 클수록 나빠지고, −Rt 는 회전이 두 배(2θ)가 되어 90°에서 최악·180°에서 되돌아온다 — 둘 다 회전이 있으면 틀림)

# %% [markdown]
# ## 6. 여러 카메라에서 Gaussian 중심으로의 시점 방향
#
# 원점 근처를 둘러싼 카메라 5대를 look-at으로 만든다(카메라 $+z$축이 시선 방향).
# 각 카메라의 w2c에서 $-R^\top\mathbf t$로 위치를 되찾고, 두 Gaussian 중심 $\boldsymbol\mu$에 대해
#
# $$
# \mathbf d = \frac{\boldsymbol\mu - \mathbf o_{\text{cam}}}{\|\boldsymbol\mu - \mathbf o_{\text{cam}}\|}
# $$
#
# 를 계산한다. 이 $\mathbf d$가 SH 평가 `sh_eval(coeffs, d)`에 들어가는 방향이다.
# (검산으로 c2w를 직접 써서 구한 방향과 일치하는지도 본다.)

# %%
def look_at_c2w(o: torch.Tensor, target: torch.Tensor, up=(0.0, 1.0, 0.0)) -> torch.Tensor:
    """카메라 z축 = 전방(target − o), x축 = up × z, y축 = z × x. (좌표 관례 세부는 이 카드의 논지와 무관)"""
    z = F.normalize(target - o, dim=0)
    x = F.normalize(torch.linalg.cross(torch.tensor(up, dtype=o.dtype), z), dim=0)
    y = torch.linalg.cross(z, x)
    return make_c2w(torch.stack([x, y, z], dim=1), o)


means = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, -0.8]], dtype=torch.float64)      # Gaussian 중심 2개
target = means.mean(0)
angles = torch.linspace(0, 2 * math.pi, 6, dtype=torch.float64)[:-1]              # 카메라 5대, 원형 배치
cam_pos_true = torch.stack([4.0 * torch.cos(angles), 1.5 + 0.8 * torch.sin(2 * angles), 4.0 * torch.sin(angles)], dim=1)

viewmats = torch.stack([torch.linalg.inv(look_at_c2w(o, target)) for o in cam_pos_true])   # [5,4,4] = gsplat viewmats
cam_pos_rec = torch.stack([cam_pos_from_w2c(v) for v in viewmats])                         # −Rᵀt
print("복원 위치 최대 오차 =", (cam_pos_rec - cam_pos_true).abs().max().item())

dirs = F.normalize(means[None, :, :] - cam_pos_rec[:, None, :], dim=-1)          # [5 cams, 2 gaussians, 3]
dirs_check = F.normalize(means[None] - torch.stack([torch.linalg.inv(v)[:3, 3] for v in viewmats])[:, None], dim=-1)
print("c2w 4번째 열로 구한 방향과의 최대 차이 =", (dirs - dirs_check).abs().max().item())
print("방향 벡터 길이(정규화 확인) =", dirs.norm(dim=-1).flatten().tolist())
print("카메라 0 → Gaussian 0 방향 d =", dirs[0, 0].tolist())
print("카메라 0 의 시선축(c2w z열)  =", torch.linalg.inv(viewmats[0])[:3, 2].tolist(), " (target을 보므로 비슷하지만 같지는 않음)")
# 출력:
# 복원 위치 최대 오차 = 1.9984014443252818e-15
# c2w 4번째 열로 구한 방향과의 최대 차이 = 7.216449660063518e-16
# 방향 벡터 길이(정규화 확인) = [1.0, 0.9999999999999999, 1.0, 0.9999999999999999, 1.0, 1.0, 0.9999999999999999, 1.0, 1.0, 1.0]
# 카메라 0 → Gaussian 0 방향 d = [-0.9363291775690445, -0.3511234415883916, 1.8148786872587925e-17]
# 카메라 0 의 시선축(c2w z열)  = [-0.9363344128893162, -0.33440514746047, -0.10700964718735043]  (target을 보므로 비슷하지만 같지는 않음)

# %% [markdown]
# ## 7. 시각화 — 카메라 위치·축, Gaussian 중심, 시점 방향
#
# - 파란 점: $-R^\top\mathbf t$로 복원한 카메라 위치 (검은 ×: 원래 위치 — 겹쳐야 정상)
# - 빨강/초록/파랑 짧은 선: 카메라 x/y/z축 (c2w의 열벡터; z가 시선 방향)
# - 주황 다이아몬드: Gaussian 중심 $\boldsymbol\mu$
# - 회색 선분: 카메라 → Gaussian 시점 방향 $\mathbf d$ (길이 1.2로 그림)

# %%
fig = go.Figure()
P = cam_pos_rec.numpy()
fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers+text",
                           marker=dict(size=7, color="royalblue", opacity=0.6), text=[f"cam{i}" for i in range(len(P))],
                           textposition="top center", name="카메라 위치 −Rᵀt"))
T = cam_pos_true.numpy()
fig.add_trace(go.Scatter3d(x=T[:, 0], y=T[:, 1], z=T[:, 2], mode="markers",
                           marker=dict(size=3, color="black", symbol="x"), name="원래 위치 o (−Rᵀt 점과 겹침)"))
M = means.numpy()
fig.add_trace(go.Scatter3d(x=M[:, 0], y=M[:, 1], z=M[:, 2], mode="markers",
                           marker=dict(size=8, color="darkorange", symbol="diamond"), name="Gaussian 중심 μ"))

axis_colors = ["red", "green", "blue"]
axis_names = ["카메라 x축", "카메라 y축", "카메라 z축(시선)"]
for k in range(3):
    xs, ys, zs = [], [], []
    for v in viewmats:
        c2w_v = torch.linalg.inv(v)
        o = c2w_v[:3, 3].numpy(); a = c2w_v[:3, k].numpy() * 0.8
        xs += [o[0], o[0] + a[0], None]; ys += [o[1], o[1] + a[1], None]; zs += [o[2], o[2] + a[2], None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=axis_colors[k], width=5), name=axis_names[k]))

xs, ys, zs = [], [], []
for i in range(len(P)):
    for j in range(len(M)):
        d = dirs[i, j].numpy() * 1.2
        xs += [P[i, 0], P[i, 0] + d[0], None]; ys += [P[i, 1], P[i, 1] + d[1], None]; zs += [P[i, 2], P[i, 2] + d[2], None]
fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="gray", width=3, dash="dot"), name="시점 방향 d = (μ − o)/‖·‖"))

fig.update_layout(title="w2c에서 복원한 카메라 위치(−Rᵀt)와 Gaussian 시점 방향",
                  scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                  width=900, height=700, legend=dict(x=0.0, y=1.0))
_show(fig)

png_path = os.path.join(HERE, "expy.png")
try:
    fig.write_image(png_path, scale=2)
    print("저장:", png_path)
except Exception as e:  # kaleido 미설치 등
    print("PNG 저장 실패:", e)
# 출력:
# 저장: /home/sungwoo/projects/swcho/gsplat/fm/sh/.fm/hints/29154e9d-02f6-4621-874e-151de750672d/expy.png

# %% [markdown]
# ## 정리
#
# | 항목 | 확인 결과 |
# |---|---|
# | $R^\top R = I$ | 오차 $\sim 10^{-16}$ → $R^{-1} = R^\top$ |
# | $-R^\top\mathbf t$ vs 원래 위치 | 일치 (오차 $\sim 10^{-15}$) |
# | $\text{w2c}^{-1}$ vs $[R^\top\,|\,-R^\top\mathbf t]$ | 일치, 4번째 열 = 카메라 위치 |
# | 노트북 예제 | $-R^\top\mathbf t = (0,0,-5)$, $\mathbf d = (2,1,8)/\sqrt{69}$ |
# | $-\mathbf t$, $-R\mathbf t$ | 회전각이 0이 아니면 수 단위로 틀림 |
#
# 카메라는 camera 좌표계의 원점이므로 $\mathbf 0 = R\,\mathbf o_{\text{cam}} + \mathbf t$, 양변에 $R^\top$을 곱하면 $\mathbf o_{\text{cam}} = -R^\top\mathbf t$.
