# %% [markdown]
# # `covar_from_quat_scale`가 $\Sigma$를 만드는 방식
#
# 3D Gaussian Splatting에서 Gaussian의 모양은 공분산 $\Sigma$(3×3 대칭 **양정치**)로 표현되지만,
# 실제 학습 파라미터는 쿼터니언 `quats`(회전 $R$)와 `scales`(축 길이 $s$)다.
# $\Sigma$의 6개 성분을 직접 최적화하면 양정치성이 쉽게 깨지므로, 항상 유효한 $\Sigma$가 나오도록
# 아래처럼 **구성해서** 만든다.
#
# $$\Sigma = R\,S\,S^\top R^\top,\qquad S=\mathrm{diag}(s)$$
#
# gsplat 구현은 이걸 단 3줄로 처리한다:
#
# ```python
# def covar_from_quat_scale(q, s):
#     R = _quat_to_rotmat(q)          # [N,3,3]  (내부에서 q를 정규화)
#     M = R * s[..., None, :]         # R @ diag(s)
#     return M @ M.transpose(-1, -2)  # R S Sᵀ Rᵀ
# ```
#
# 이 노트북은 (1) 쿼터니언 → $R$, (2) `R * s[None,:]`가 왜 $R\,\mathrm{diag}(s)$인지,
# (3) `M @ M.T`가 $R S S^\top R^\top$과 같은지, (4) $\Sigma$의 고유값 제곱근이 곧 `sorted(s)`인지를
# 하나씩 확인하고, 마지막으로 $\Sigma$를 3D 타원체로 그려본다.
#
# 필요 패키지: numpy, plotly, kaleido (정적 PNG 저장용)

# %%
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

np.set_printoptions(precision=4, suppress=True)
HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# %% [markdown]
# ## 1. 쿼터니언 → 회전행렬 $R$
#
# gsplat의 `_quat_to_rotmat`은 **w-first** 규약 $q=(w,x,y,z)$를 쓰고, 먼저 $q$를 단위 길이로 정규화한다.
# (정규화하지 않으면 $R$이 $\|q\|^2$배 스케일된 행렬이 되어 회전이 아니게 된다.)
#
# $$R=\begin{bmatrix}
# 1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy)\\
# 2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx)\\
# 2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
# \end{bmatrix}$$

# %%
def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """gsplat `_quat_to_rotmat`와 동일한 로직 (w-first, 내부 정규화). q: [..., 4] -> [..., 3, 3]"""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack(
        [
            1 - 2 * (y**2 + z**2), 2 * (x * y - w * z),   2 * (x * z + w * y),
            2 * (x * y + w * z),   1 - 2 * (x**2 + z**2), 2 * (y * z - w * x),
            2 * (x * z - w * y),   2 * (y * z + w * x),   1 - 2 * (x**2 + y**2),
        ],
        axis=-1,
    )
    return R.reshape(q.shape[:-1] + (3, 3))


# z축 30° 회전 쿼터니언 (일부러 정규화되지 않은 크기로 넣어본다)
theta = np.deg2rad(30.0)
q = 2.5 * np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])  # w-first
s = np.array([0.30, 0.12, 0.10])

R = quat_to_rotmat(q)
print("q (정규화 전) =", q, " |q| =", np.linalg.norm(q))
print("R =\n", R)
print("R Rᵀ =\n", R @ R.T)
print("det(R) =", np.linalg.det(R))
# 출력:
# q (정규화 전) = [2.4148 0.     0.     0.647 ] |q| = 2.5
# R =
#  [[ 0.866 -0.5     0.   ]
#  [ 0.5     0.866   0.   ]
#  [ 0.      0.      1.   ]]
# R Rᵀ =
#  [[ 1. -0.  0.]
#  [-0.  1.  0.]
#  [ 0.  0.  1.]]
# det(R) = 1.0
# → |q|=2.5여도 내부 정규화 덕분에 R Rᵀ=I, det=+1인 정직한 회전행렬이 나온다.

# %% [markdown]
# ## 2. `R * s[..., None, :]` 는 왜 $R\,\mathrm{diag}(s)$ 인가?
#
# `s[..., None, :]`는 shape `(1,3)`이므로 **행 방향으로 브로드캐스트**된다.
# 즉 $j$번째 **열** 전체에 $s_j$가 곱해진다:
#
# $$(R\,\mathrm{diag}(s))_{ij} = \sum_k R_{ik}\,[\mathrm{diag}(s)]_{kj} = R_{ij}s_j$$
#
# 이것은 정확히 elementwise 곱 `R[i,j] * s[j]`이다.
# (반대로 `s[..., :, None]`이면 열 방향 브로드캐스트라 $\mathrm{diag}(s)\,R$이 된다 — 완전히 다른 행렬!)

# %%
S = np.diag(s)
M_broadcast = R * s[..., None, :]  # gsplat이 쓰는 방식
M_matmul = R @ S  # 명시적 행렬곱

print("s[..., None, :].shape =", s[..., None, :].shape)
print("M (broadcast) =\n", M_broadcast)
print("R @ diag(s)   =\n", M_matmul)
print("max|차이| =", np.abs(M_broadcast - M_matmul).max())
print("참고) diag(s) @ R (틀린 버전) =\n", S @ R)
# 출력:
# s[..., None, :].shape = (1, 3)
# M (broadcast) =
#  [[ 0.2598 -0.06    0.    ]
#  [ 0.15    0.1039  0.    ]
#  [ 0.      0.      0.1   ]]
# R @ diag(s)   =
#  [[ 0.2598 -0.06    0.    ]
#  [ 0.15    0.1039  0.    ]
#  [ 0.      0.      0.1   ]]
# max|차이| = 0.0
# 참고) diag(s) @ R (틀린 버전) =
#  [[ 0.2598 -0.15    0.    ]
#  [ 0.06    0.1039  0.    ]
#  [ 0.      0.      0.1   ]]

# %% [markdown]
# ## 3. `M @ M.T` $=R S S^\top R^\top=\Sigma$
#
# $M = RS$ 이므로
#
# $$M M^\top = (RS)(RS)^\top = R\,S\,S^\top R^\top$$
#
# 그리고 $S$가 대각이라 $SS^\top=\mathrm{diag}(s^2)$이다. 어떤 $M$에 대해서도 $MM^\top$은
# 자동으로 **대칭 & 반양정치**이고, $s_i>0$이면 $M$이 가역이라 **양정치**가 보장된다.
# 이게 $\Sigma$를 직접 파라미터화하지 않는 이유다.

# %%
def covar_from_quat_scale(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    R = quat_to_rotmat(q)  # [..., 3, 3]
    M = R * s[..., None, :]  # R @ diag(s)
    return M @ np.swapaxes(M, -1, -2)  # R S Sᵀ Rᵀ


Sigma = covar_from_quat_scale(q, s)
Sigma_ref = R @ S @ S.T @ R.T

print("Σ = M @ M.T =\n", Sigma)
print("R S Sᵀ Rᵀ  =\n", Sigma_ref)
print("max|차이| =", np.abs(Sigma - Sigma_ref).max())
print("대칭성 max|Σ - Σᵀ| =", np.abs(Sigma - Sigma.T).max())
print("고유값(양정치?) =", np.linalg.eigvalsh(Sigma))
# 출력:
# Σ = M @ M.T =
#  [[0.0711 0.0327 0.    ]
#  [0.0327 0.0333 0.    ]
#  [0.     0.     0.01  ]]
# R S Sᵀ Rᵀ  =
#  [[0.0711 0.0327 0.    ]
#  [0.0327 0.0333 0.    ]
#  [0.     0.     0.01  ]]
# max|차이| = 6.938893903907228e-18
# 대칭성 max|Σ - Σᵀ| = 0.0
# 고유값(양정치?) = [0.01   0.0144 0.09  ]

# %% [markdown]
# ## 4. $\sqrt{\mathrm{eig}(\Sigma)} = \mathrm{sorted}(s)$
#
# $\Sigma = R\,\mathrm{diag}(s^2)\,R^\top$ 는 그 자체가 고유분해다.
# 고유벡터는 $R$의 열(= 타원체의 주축 방향), 고유값은 $s_i^2$.
# 따라서 고유값의 제곱근이 곧 스케일이고, `eigvalsh`가 오름차순을 주므로 `sorted(s)`와 일치한다.

# %%
evals, evecs = np.linalg.eigh(Sigma)
print("sqrt(eigvals(Σ)) =", np.sqrt(evals))
print("sorted(s)        =", np.sort(s))
print("일치? ", np.allclose(np.sqrt(evals), np.sort(s)))
print("고유벡터(열) =\n", evecs, "\n← R의 열(부호/순서 차이는 있을 수 있음):\n", R)
# 출력:
# sqrt(eigvals(Σ)) = [0.1  0.12 0.3 ]
# sorted(s)        = [0.1  0.12 0.3 ]
# 일치?  True
# 고유벡터(열) =
#  [[ 0.     0.5   -0.866]
#  [ 0.    -0.866 -0.5  ]
#  [ 1.     0.    -0.   ]]
# ← R의 열(부호/순서 차이는 있을 수 있음):
#  [[ 0.866 -0.5    0.   ]
#  [ 0.5    0.866  0.   ]
#  [ 0.     0.     1.   ]]

# %% [markdown]
# ## 5. 배치 동작 확인 (`[N,4] × [N,3] → [N,3,3]`)
#
# `s[..., None, :]` 형태로 쓴 덕분에 임의의 배치 차원에 그대로 동작한다.

# %%
rng = np.random.default_rng(0)
qs = rng.normal(size=(5, 4))
ss = rng.uniform(0.05, 0.4, size=(5, 3))

cov_batch = covar_from_quat_scale(qs, ss)
cov_loop = np.stack([covar_from_quat_scale(qs[i], ss[i]) for i in range(5)])
print("cov_batch.shape =", cov_batch.shape)
print("배치 vs 루프 max|차이| =", np.abs(cov_batch - cov_loop).max())
print(
    "각 Σ의 sqrt(eig) vs sorted(s) 최대오차 =",
    np.abs(np.sqrt(np.linalg.eigvalsh(cov_batch)) - np.sort(ss, axis=-1)).max(),
)
# 출력:
# cov_batch.shape = (5, 3, 3)
# 배치 vs 루프 max|차이| = 0.0
# 각 Σ의 sqrt(eig) vs sorted(s) 최대오차 = 1.6653345369377348e-16

# %% [markdown]
# ## 6. 시각화: $\Sigma$가 그리는 타원체
#
# 단위 구 $\|u\|=1$의 점들을 $M=R\,\mathrm{diag}(s)$로 보내면 $x = Mu$가 되고,
# 이 점들이 만드는 표면이 바로 $x^\top \Sigma^{-1} x = 1$ 인 타원체다.
#
# - 왼쪽: $R=I$ (축 정렬) — 반축 길이가 정확히 $s=(0.30,0.12,0.10)$
# - 오른쪽: $R$ = z축 30° 회전 — **같은 모양이 통째로 회전**만 했다 (고유값 불변)
#
# 즉 $s$가 "얼마나 길쭉한가", $R$이 "어느 방향을 보는가"를 나누어 담당한다.

# %%
def ellipsoid_mesh(M, n=60):
    """단위 구를 M으로 밀어 x = M u 표면을 만든다."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    sph = np.stack(
        [
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
        ]
    )  # [3,n,n]
    pts = M @ sph.reshape(3, -1)
    return [c.reshape(n, n) for c in pts]


def axes_traces(M, colors=("#d1495b", "#00798c", "#edae49")):
    """주축 R·(s_i e_i) 을 선분으로."""
    tr = []
    for i, col in enumerate(colors):
        a = M[:, i]
        tr.append(
            go.Scatter3d(
                x=[-a[0], a[0]], y=[-a[1], a[1]], z=[-a[2], a[2]],
                mode="lines", line=dict(color=col, width=7), showlegend=False,
            )
        )
    return tr


M_id = np.eye(3) * s[None, :]  # R = I
M_rot = R * s[None, :]  # R = z축 30°

fig = go.Figure()
for M, scene in [(M_id, "scene"), (M_rot, "scene2")]:
    X, Y, Z = ellipsoid_mesh(M)
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z, colorscale="Blues", opacity=0.55,
                   showscale=False, scene=scene)
    )
    for t in axes_traces(M):
        t.scene = scene
        fig.add_trace(t)

rng_ax = dict(range=[-0.35, 0.35], showbackground=True, backgroundcolor="#f7f7f7")
cam = dict(eye=dict(x=1.4, y=1.4, z=1.0))
fig.update_layout(
    title="Σ = R diag(s) diag(s)ᵀ Rᵀ &nbsp;|&nbsp; 왼쪽 R=I, 오른쪽 R=Rot_z(30°),&nbsp; s=(0.30, 0.12, 0.10)",
    width=1100, height=520, margin=dict(l=0, r=0, t=60, b=0),
    scene=dict(domain=dict(x=[0.0, 0.48]), aspectmode="cube",
               xaxis=rng_ax, yaxis=rng_ax, zaxis=rng_ax, camera=cam),
    scene2=dict(domain=dict(x=[0.52, 1.0]), aspectmode="cube",
                xaxis=rng_ax, yaxis=rng_ax, zaxis=rng_ax, camera=cam),
)
_show(fig)

out = HERE / "expy.png"
try:
    fig.write_image(str(out), scale=2)
    print("saved:", out)
except Exception as e:  # kaleido 미설치 등
    print("PNG 저장 실패 (pip install kaleido):", e)
# 출력:
# saved: .../expy.png

# %% [markdown]
# ## 정리
#
# | 코드 | 의미 |
# |---|---|
# | `R = _quat_to_rotmat(q)` | $q$를 정규화 후 회전행렬 $R$ ($RR^\top=I$, $\det R=1$) |
# | `M = R * s[..., None, :]` | 열 $j$에 $s_j$ 곱 $\Rightarrow M = R\,\mathrm{diag}(s)$ |
# | `M @ M.T` | $\Sigma = R S S^\top R^\top$, 항상 대칭 양정치 |
#
# - $\mathrm{diag}$를 실제로 만들지 않고 브로드캐스트 한 번으로 끝내므로 배치에서 싸다.
# - $\Sigma$의 고유벡터 = $R$의 열(주축), 고유값 = $s_i^2$.
# - gsplat CUDA 커널에서는 이 계산이 투영 커널에 융합되어 있고,
#   별도로 `quat_scale_to_covar_preci`로도 호출할 수 있다
#   (precision $\Sigma^{-1}$은 $R\,\mathrm{diag}(1/s)$ 로 같은 방식으로 만든다).
