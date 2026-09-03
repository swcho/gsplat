# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---
#
# 필요 패키지: torch, numpy, plotly, kaleido
# (gsplat 자체는 import하지 않는다 — numpy/torch 토이 재구현으로 개념만 확인)

# %% [markdown]
# # 3D Gaussian의 공분산 $\Sigma = R\,S\,S^\top R^\top$ 은 왜 항상 양의 정부호인가
#
# 한 Gaussian의 확률밀도는
#
# $$f(\Delta) \propto \exp\!\left(-\tfrac12\,\Delta^\top \Sigma^{-1}\Delta\right)$$
#
# 이다. 식에 들어가는 것이 $\Sigma$가 아니라 $\Sigma^{-1}$ 이므로 **역행렬이 존재해야** 하고,
# 지수의 부호가 뒤집히지 않으려면 **양의 정부호**여야 한다. 부정부호가 되면
# 중심에서 멀어질수록 밝아지는 발산하는 괴물이 되어 화면을 태운다.
#
# gsplat은 $\Sigma$를 직접 파라미터로 두지 않고 quaternion $q$와 스케일 $s$로부터
#
# $$\Sigma = R(q)\,S\,S^\top R(q)^\top, \qquad S = \mathrm{diag}(s_1,s_2,s_3)$$
#
# 로 **합성**한다. 아래에서 다음을 순서대로 직접 확인한다.
#
# 1. 축정렬 Gaussian: $\Sigma_0 = \mathrm{diag}(s_k^2)$, 등고면은 반축 $s_k$의 타원체
# 2. 회전을 씌우면 $\Sigma = R\,\mathrm{diag}(s_k^2)\,R^\top$ — 고유값 $s_k^2$, 고유벡터 $R$의 열
# 3. **핵심**: $M = RS$ 로 묶으면 $\Sigma = MM^\top$ 이고 $x^\top\Sigma x = \lVert M^\top x\rVert^2 \ge 0$
# 4. 경사하강 비교: $\Sigma$ 6원소 직접 파라미터화 vs. $(q, \log s)$ 재매개화
# 5. $\Sigma^{-1} = R S^{-1}S^{-1}R^\top$ 닫힌 형식 (역행렬 루틴 불필요)
# 6. $M$은 $\Sigma$의 제곱근 → split/MCMC 샘플링에 그대로 재사용
# 7. 화면 투영 $\Sigma' = (JWM)(JWM)^\top$ 도 Gram이지만 rank가 떨어질 수 있다 → `+0.3·I`
# 8. quaternion 노름은 gauge 자유도 (미정규화 저장이 가능한 이유)

# %%
import math

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


torch.manual_seed(0)
np.random.seed(0)
torch.set_printoptions(precision=5, sci_mode=False)

EPS2D = 0.3  # gsplat/cuda/_torch_impl.py:309 의 eps2d 기본값
SPLIT_SHRINK = 1.6  # ops.py:211 split 시 크기 축소 계수
RADIUS_SIGMA = 3.33  # _torch_impl.py:337 타일 반경 (3σ 규칙)

print(f"eps2d = {EPS2D}, split shrink = {SPLIT_SHRINK}, radius = {RADIUS_SIGMA}σ")
# 출력: eps2d = 0.3, split shrink = 1.6, radius = 3.33σ


# %% [markdown]
# ## 0. gsplat과 같은 방식으로 $R(q)$ 와 $\Sigma$ 를 만든다
#
# `gsplat/utils.py:123` `normalized_quat_to_rotmat` 과 같은 공식 (규약은 `wxyz`).
# 성분이 전부 quaternion의 **2차 다항식**이라 삼각함수도, 특이점도 없다.
#
# $$
# R(q) = \begin{pmatrix}
# 1-2(y^2+z^2) & 2(xy - wz) & 2(xz + wy)\\
# 2(xy + wz) & 1-2(x^2+z^2) & 2(yz - wx)\\
# 2(xz - wy) & 2(yz + wx) & 1-2(x^2+y^2)
# \end{pmatrix}
# $$
#
# 공분산 합성은 `gsplat/cuda/include/Utils.cuh:285` 와 `_math.py:700` 을 그대로 옮긴 것.

# %%
def quat_to_rotmat(quat: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """wxyz quaternion -> [...,3,3] 회전행렬 (gsplat/utils.py:123과 동일한 공식)."""
    if normalize:
        quat = torch.nn.functional.normalize(quat, dim=-1)
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def quat_scale_to_covar_preci(quats, scales, normalize=True):
    """Utils.cuh:285 / _math.py:700 의 토이 재구현. (covar, preci) 반환."""
    R = quat_to_rotmat(quats, normalize=normalize)          # [...,3,3]
    M = R * scales[..., None, :]                            # R @ diag(s)  (열마다 s_k 곱)
    covar = torch.einsum("...ij,...kj->...ik", M, M)        # M @ M^T
    P = R * (1.0 / scales[..., None, :])                    # R @ diag(1/s)
    preci = torch.einsum("...ij,...kj->...ik", P, P)        # P @ P^T = Σ^{-1}
    return covar, preci


# 축정렬 확인: q = (1,0,0,0) 은 항등회전이므로 Σ = diag(s^2) 여야 한다
q_id = torch.tensor([1.0, 0.0, 0.0, 0.0])
s_demo = torch.tensor([2.0, 0.5, 0.1])
covar0, preci0 = quat_scale_to_covar_preci(q_id, s_demo)
print("R(1,0,0,0) =\n", quat_to_rotmat(q_id))
print("Σ0 =\n", covar0)
print("diag(s^2) =", [round(v, 5) for v in (s_demo**2).tolist()])
# 출력: R(1,0,0,0) = I (항등행렬)
# 출력: Σ0 = diag(4.0, 0.25, 0.01)
# 출력: diag(s^2) = [4.0, 0.25, 0.01]   → 항등회전에서 Σ = diag(s^2)


# %% [markdown]
# ## 1. 등고면은 반축 $s_k$ 의 타원체 — 이차형식과 타원체 방정식은 같은 식
#
# $$\Delta^\top \Sigma_0^{-1}\Delta
# = \frac{\Delta_1^2}{s_1^2}+\frac{\Delta_2^2}{s_2^2}+\frac{\Delta_3^2}{s_3^2}$$
#
# 이므로 $\Delta^\top\Sigma_0^{-1}\Delta = 1$ 은 고교 기하의 타원체 방정식과 글자 그대로 같다.
# 즉 축 방향으로 정확히 $\pm s_k$ 에서 이 값이 1이 된다.

# %%
for k in range(3):
    d = torch.zeros(3)
    d[k] = s_demo[k]                      # 축 방향으로 s_k 만큼
    val = d @ preci0 @ d
    print(f"Δ = s_{k+1}·e_{k+1} 에서 Δᵀ Σ⁻¹ Δ = {val:.6f}  (밀도 = exp(-1/2) = {math.exp(-0.5):.4f})")
# 출력: Δ = s_1·e_1 에서 Δᵀ Σ⁻¹ Δ = 1.000000  (밀도 = exp(-1/2) = 0.6065)
# 출력: Δ = s_2·e_2 에서 Δᵀ Σ⁻¹ Δ = 1.000000
# 출력: Δ = s_3·e_3 에서 Δᵀ Σ⁻¹ Δ = 1.000000


# %% [markdown]
# ## 2. 회전을 씌워도 고유값은 정확히 $s_k^2$
#
# $$\Sigma \mathbf{r}_k = R S^2 R^\top \mathbf{r}_k = R S^2 \mathbf{e}_k = s_k^2\,\mathbf{r}_k$$
#
# 즉 **고유값 = $s_k^2$ (반축 길이의 제곱), 고유벡터 = $R$의 열 (타원체의 축 방향)**.
# 고유값이 전부 실수의 제곱이므로 음수가 될 방법이 없다.

# %%
q_rand = torch.tensor([0.3, -0.7, 0.2, 0.6])
R_rand = quat_to_rotmat(q_rand)
covar, preci = quat_scale_to_covar_preci(q_rand, s_demo)

evals, evecs = torch.linalg.eigh(covar)
print("Σ 고유값        =", [round(v, 6) for v in evals.tolist()])
print("s^2 (정렬)      =", sorted(round(v, 6) for v in (s_demo**2).tolist()))
# |R의 열| 과 고유벡터가 (부호/순서 무시) 일치하는지: |R^T V| 가 순열행렬이어야 한다
print("|Rᵀ V| (순열행렬이어야) =\n", (R_rand.T @ evecs).abs().round(decimals=4))
print("Rᵀ R - I 의 최대오차 =", (R_rand.T @ R_rand - torch.eye(3)).abs().max().item())
# 출력: Σ 고유값 = [0.01, 0.25, 4.0], s^2 (정렬) = [0.01, 0.25, 4.0]  → 정확히 일치
# 출력: |Rᵀ V| 는 순열행렬 (각 행/열에 1이 하나) → 고유벡터 = R의 열
# 출력: Rᵀ R - I 최대오차 ≈ 1.2e-07  (정규화한 quaternion → R은 직교)


# %% [markdown]
# ## 3. 핵심 증명: $M = RS$ 로 묶으면 $\Sigma = MM^\top$ (Gram 형태)
#
# $$\Sigma = R S S^\top R^\top = (RS)(RS)^\top = MM^\top$$
#
# $$x^\top \Sigma x = x^\top M M^\top x = (M^\top x)^\top (M^\top x) = \lVert M^\top x\rVert^2 \;\ge\; 0$$
#
# "어떤 벡터든 자기 자신과의 내적은 길이의 제곱이라 음수가 될 수 없다" — 이게 전부다.
# **$M$이 무엇으로 만들어졌는지 전혀 보지 않았다.** 등호는 $M^\top x = 0$ 일 때뿐이고,
# $\det M = \det R \cdot \det S = 1\cdot s_1s_2s_3 \ne 0$ 이면 $x = 0$ 만 해당된다.
#
# 아래에서 랜덤 파라미터 200,000개로, 재매개화와 "$\Sigma$ 6원소 직접 파라미터화"를 비교한다.

# %%
N = 200_000

# (a) 재매개화: 완전히 랜덤한 quaternion(정규화 전) + 랜덤 log-scale
#     수학적 보장을 보려면 float64로 계산한다. float32에서는 아래에서 따로 확인한다.
quats_r = torch.randn(N, 4, dtype=torch.float64)
log_s_r = torch.randn(N, 3, dtype=torch.float64) * 2.0   # scales = exp(·) → 몇 자릿수에 걸쳐 퍼짐
scales_r = torch.exp(log_s_r)
covars_r, _ = quat_scale_to_covar_preci(quats_r, scales_r)

# 항등식 xᵀΣx = ‖Mᵀx‖² 를 수치로 확인 (Σ 성분이 1e7까지 퍼지므로 상대오차로 본다)
R_all = quat_to_rotmat(quats_r[:1000])
M_all = R_all * scales_r[:1000, None, :]
xs = torch.randn(1000, 3, dtype=torch.float64)
lhs = torch.einsum("ni,nij,nj->n", xs, covars_r[:1000], xs)
rhs = torch.einsum("nji,nj->ni", M_all, xs).pow(2).sum(-1)   # ‖Mᵀx‖²
rel_id = (lhs - rhs).abs() / lhs.abs().clamp_min(1e-300)
print(f"max |xᵀΣx - ‖Mᵀx‖²| / |xᵀΣx| = {rel_id.max():.3e}   (같은 식이므로 0)")
print(f"xᵀΣx < 0 인 샘플 수 = {(lhs < 0).sum().item()} / 1000")

# (b) Σ의 대칭 6원소를 직접 랜덤 파라미터로 (스케일은 (a)와 비슷하게 맞춤)
t = torch.randn(N, 6, dtype=torch.float64) * covars_r.abs().mean()
direct = torch.zeros(N, 3, 3, dtype=torch.float64)
iu = torch.triu_indices(3, 3)
direct[:, iu[0], iu[1]] = t
direct = direct + direct.transpose(-1, -2) - torch.diag_embed(torch.diagonal(direct, dim1=-2, dim2=-1))

min_eig_r = torch.linalg.eigvalsh(covars_r).min(dim=-1).values
min_eig_d = torch.linalg.eigvalsh(direct).min(dim=-1).values

print(f"\n[재매개화 RSSᵀRᵀ]  최소고유값 < 0 인 비율: {(min_eig_r < 0).float().mean()*100:.3f}%"
      f"  (최솟값 {min_eig_r.min():.3e})")
print(f"[Σ 6원소 직접]     최소고유값 < 0 인 비율: {(min_eig_d < 0).float().mean()*100:.3f}%")
print(f"[Σ 6원소 직접]     양의 정부호인 비율:     {(min_eig_d > 0).float().mean()*100:.3f}%")

# float32에서는? — 조건수가 1e14까지 가므로 고유값 루틴이 "음수"를 보고할 수 있다.
# 단 trace로 정규화한 상대 최소고유값을 보면 float32 eps(≈1.2e-7) 수준의 반올림일 뿐이다.
covars32, _ = quat_scale_to_covar_preci(quats_r.float(), scales_r.float())
min_eig_32 = torch.linalg.eigvalsh(covars32).min(dim=-1).values
rel_32 = min_eig_32 / torch.diagonal(covars32, dim1=-2, dim2=-1).sum(-1)
print(f"\n[동일 파라미터, float32] 최소고유값 < 0 인 비율: {(min_eig_32 < 0).float().mean()*100:.3f}%"
      f"  (최솟값 {min_eig_32.min():.3e})")
print(f"  → 상대 최소고유값(λ_min/tr Σ)의 최솟값 = {rel_32.min():.3e}"
      f"  (float32 eps ≈ {torch.finfo(torch.float32).eps:.1e} → 구조적 음수가 아니라 반올림)")
# 출력: max |xᵀΣx - ‖Mᵀx‖²| / |xᵀΣx| = 6.579e-14   (float64 반올림 수준)
# 출력: xᵀΣx < 0 인 샘플 수 = 0 / 1000
# 출력: [재매개화 RSSᵀRᵀ]  최소고유값 < 0 인 비율: 0.000%  (최솟값 9.137e-09 > 0)
# 출력: [Σ 6원소 직접]     최소고유값 < 0 인 비율: 98.773%
# 출력: [Σ 6원소 직접]     양의 정부호인 비율:     1.226%
# 출력: [동일 파라미터, float32] 최소고유값 < 0 인 비율: 0.275%  (최솟값 -3.476e+01)
# 출력:   → 상대 최소고유값(λ_min/tr Σ)의 최솟값 = -1.810e-07  (float32 eps ≈ 1.2e-07)
# 출력:     → 구조적 음수가 아니라 반올림
# → 수학적 보장은 완전하다. float32에서 보이는 음수는 "고유값 루틴의 오차"이지
#   Σ 자체의 성질이 아니다 — 그래서 코드는 Σ를 수치 분해하지 않고 5절의 닫힌 형식을 쓴다.


# %% [markdown]
# ## 4. 경사하강으로 목표 $\Sigma$ 에 맞추기 — 두 파라미터화의 차이
#
# 실제 학습은 랜덤 샘플링이 아니라 gradient step이다. 목표 공분산 $\Sigma^\star$ 를 주고
# $\lVert \Sigma - \Sigma^\star\rVert_F^2$ 를 Adam으로 줄이면서, **학습 경로 중 몇 스텝이
# 부정부호 상태를 지나가는지** 센다. 부정부호 스텝은 렌더러가 터지는 스텝이다.
#
# 실제 3DGS는 매 스텝 랜덤한 한 장의 뷰만 렌더하므로 gradient에 미니배치 잡음이 섞인다.
# 이를 모사하려고 목표에 매 스텝 대칭 잡음 $\mathcal{N}(0,\sigma^2)$ 를 더한다 ($\sigma = 0.02$).
# 이 잡음이 **부정부호로 밀어내는 힘**이고, 두 파라미터화의 차이가 여기서 드러난다.
#
# 두 파라미터화 모두 표현력은 같다 (둘 다 6 자유도). 다른 것은 **안전성**뿐이다.

# %%
q_star = torch.nn.functional.normalize(torch.tensor([0.5, 0.5, -0.5, 0.5]), dim=-1)
s_star = torch.tensor([1.0, 0.35, 0.04])          # 얇은 판 모양 (실제 3DGS에서 흔하다)
covar_star, _ = quat_scale_to_covar_preci(q_star, s_star)
STEPS = 1000
GRAD_NOISE = 0.02                                 # 미니배치 잡음 대용


def _noisy_target(sigma):
    n = torch.randn(3, 3) * sigma
    return covar_star + (n + n.T) / 2             # 목표도 대칭으로 유지


# (a) 재매개화: (quat, log-scale) 학습. 초기값은 워크스루와 같은 방식
torch.manual_seed(0)
p_quat = torch.nn.Parameter(torch.rand(4))
p_logs = torch.nn.Parameter(torch.zeros(3))
opt_a = torch.optim.Adam([{"params": [p_quat], "lr": 1e-2},
                          {"params": [p_logs], "lr": 3e-2}], eps=1e-15)
hist_a, bad_a = [], 0
for step in range(STEPS):
    covar, _ = quat_scale_to_covar_preci(p_quat, torch.exp(p_logs))
    loss = (covar - _noisy_target(GRAD_NOISE)).pow(2).sum()
    opt_a.zero_grad(); loss.backward(); opt_a.step()
    me = torch.linalg.eigvalsh(covar.detach()).min().item()
    hist_a.append(me); bad_a += me <= 0
err_a = (covar.detach() - covar_star).pow(2).sum().item()

# (b) Σ 6원소 직접 학습 (대칭은 구성상 보장, 양정부호는 보장 없음)
torch.manual_seed(0)
p_tri = torch.nn.Parameter(torch.zeros(6))
opt_b = torch.optim.Adam([p_tri], lr=3e-2, eps=1e-15)
hist_b, bad_b = [], 0
for step in range(STEPS):
    C = torch.zeros(3, 3)
    C = C.index_put((iu[0], iu[1]), p_tri)
    C = C + C.T - torch.diag(torch.diagonal(C))
    loss = (C - _noisy_target(GRAD_NOISE)).pow(2).sum()
    opt_b.zero_grad(); loss.backward(); opt_b.step()
    me = torch.linalg.eigvalsh(C.detach()).min().item()
    hist_b.append(me); bad_b += me <= 0
err_b = (C.detach() - covar_star).pow(2).sum().item()

print(f"목표 Σ* 고유값 = {[round(v, 5) for v in torch.linalg.eigvalsh(covar_star).tolist()]}")
print(f"[재매개화 (q, log s)] ‖Σ-Σ*‖²_F {err_a:.3e},  부정부호 스텝 {bad_a}/{STEPS}"
      f",  경로 최소고유값 {min(hist_a):+.3e}")
print(f"[Σ 6원소 직접]        ‖Σ-Σ*‖²_F {err_b:.3e},  부정부호 스텝 {bad_b}/{STEPS}"
      f",  경로 최소고유값 {min(hist_b):+.3e}")
# 출력: 목표 Σ* 고유값 = [0.0016, 0.1225, 1.0]
# 출력: [재매개화 (q, log s)] ‖Σ-Σ*‖²_F 3.063e-03, 부정부호 스텝 0/1000, 경로 최소고유값 +2.547e-02
# 출력: [Σ 6원소 직접]        ‖Σ-Σ*‖²_F 1.717e-03, 부정부호 스텝 491/1000, 경로 최소고유값 -4.685e-02
# → 최종 오차는 비슷한데(둘 다 표현력 6자유도), 직접 파라미터화는 학습 경로의 절반 이상에서
#   부정부호 = 렌더 불가 상태였다. 재매개화는 단 한 스텝도 그렇지 않다 — 될 수가 없다.


# %% [markdown]
# ## 5. $\Sigma^{-1}$ 은 역행렬 루틴 없이 닫힌 형식으로 나온다
#
# $$\Sigma^{-1} = (R S S^\top R^\top)^{-1} = (R^\top)^{-1}(SS^\top)^{-1}R^{-1} = R\,S^{-1}S^{-\top}R^\top$$
#
# ($R^{-1} = R^\top$ 이므로 $(R^\top)^{-1} = R$). 대각행렬의 역행렬은 성분의 역수뿐이므로
# 코드는 `1.0f / scale[k]` 로 $S^{-1}$을 만들어 같은 Gram 곱을 한 번 더 한다 (`Utils.cuh:285`).
# **$\Sigma^{-1}$ 역시 Gram 형태이므로 역행렬도 자동으로 양의 정부호다.**

# %%
inv_lin = torch.linalg.inv(covar_star)                        # 일반 역행렬 루틴
_, preci_cf = quat_scale_to_covar_preci(q_star, s_star)       # 닫힌 형식
print(f"‖Σ⁻¹(닫힌형식) - inv(Σ)‖_max = {(preci_cf - inv_lin).abs().max():.3e}")
print(f"Σ Σ⁻¹ - I 최대오차          = {(covar_star @ preci_cf - torch.eye(3)).abs().max():.3e}")
print(f"Σ⁻¹ 고유값 = {[round(v, 4) for v in torch.linalg.eigvalsh(preci_cf).tolist()]}  (= 1/s^2, 전부 양수)")
print(f"1/s^2      = {sorted(round(v, 4) for v in (1.0 / s_star**2).tolist())}")
print(f"조건수 κ(Σ) = (s_max/s_min)² = {(s_star.max()/s_star.min())**2:.1f}")
# 출력: ‖Σ⁻¹(닫힌형식) - inv(Σ)‖_max = 0.000e+00  (이 대각 케이스에서는 완전히 동일)
# 출력: Σ Σ⁻¹ - I 최대오차 = 5.960e-08
# 출력: Σ⁻¹ 고유값 = [1.0, 8.1633, 625.0]  (= 1/s^2, 전부 양수)
# 출력: 1/s^2      = [1.0, 8.1633, 625.0]
# 출력: 조건수 κ(Σ) = 625.0
# → 닫힌 형식은 역행렬 루틴을 부르지 않으므로 특이행렬 분기·실패 경로가 아예 없다.


# %% [markdown]
# ## 6. $M$ 은 "$\Sigma$ 의 제곱근" — split/MCMC 샘플링에 그대로 재사용된다
#
# $\varepsilon \sim N(0,I)$ 일 때
#
# $$\mathrm{Cov}(M\varepsilon) = M\,\mathrm{Cov}(\varepsilon)\,M^\top = M I M^\top = MM^\top = \Sigma$$
#
# 1변수에서 $Z\sim N(0,1)$ 일 때 $\sigma Z \sim N(0,\sigma^2)$ 였던 것의 3차원 판이고,
# $M$이 $\sigma$의 자리를 차지한다. `gsplat/strategy/ops.py:196` split 코드가 정확히 이것이다.
#
# ```python
# samples = torch.einsum("nij,nj,bnj->bni", rotmats, scales, torch.randn(2, N, 3))
# ```
#
# $\Sigma$의 6원소만 갖고 있었다면 이 샘플링을 위해 Cholesky/고유값 분해를 매번 돌려야 했다 —
# 재매개화는 그 분해를 **미리, 파라미터 형태로** 들고 있는 셈이다.

# %%
NS = 400_000
R_star = quat_to_rotmat(q_star)
eps = torch.randn(NS, 3)
samples = torch.einsum("ij,j,nj->ni", R_star, s_star, eps)    # ops.py:200 과 동일
emp = (samples.T @ samples) / NS
print("경험적 공분산 Cov(Mε) =\n", emp)
print("이론값 Σ =\n", covar_star)
print(f"최대 상대오차 = {((emp - covar_star).abs() / covar_star.abs().max()).max():.4f}")

# split은 자식의 scales를 log(s/1.6)으로 줄인다 (ops.py:211) → 공분산은 1/1.6² 배
# (covar_star는 대각 성분이 0인 곳이 있어 원소별 비율은 0/0 = nan 이 된다 → 대각/trace로 본다)
covar_child, _ = quat_scale_to_covar_preci(q_star, s_star / SPLIT_SHRINK)
ratio_diag = (torch.diagonal(covar_child) / torch.diagonal(covar_star))
print(f"\nsplit 후 대각 성분별 Σ_child/Σ_parent = {[round(v, 6) for v in ratio_diag.tolist()]}")
print(f"trace 비 = {(torch.diagonal(covar_child).sum() / torch.diagonal(covar_star).sum()):.6f}"
      f"   (= 1/1.6² = {1/SPLIT_SHRINK**2:.6f})")
print(f"log 공간에서는 상수 감산: -log(1.6) = {-math.log(SPLIT_SHRINK):.4f}")
# 출력: 경험적 공분산 Cov(Mε) = [[0.12284, -0.00002, -0.00039], [-0.00002, 0.00159, -0.00005],
# 출력:                          [-0.00039, -0.00005, 0.99938]] ≈ 이론값 diag(0.1225, 0.0016, 1.0)
# 출력: 최대 상대오차 = 0.0006  → M은 Σ의 제곱근 (Cov(Mε) = MMᵀ = Σ)
# 출력: split 후 대각 성분별 Σ_child/Σ_parent = [0.390625, 0.390625, 0.390625]
# 출력: trace 비 = 0.390625   (= 1/1.6² = 0.390625)
# 출력: log 공간에서는 상수 감산: -log(1.6) = -0.4700
# → 크기 조절이 log-scale 파라미터에서는 그냥 상수 뺄셈이다 (양수 제약도 자동으로 유지)


# %% [markdown]
# ## 7. 화면 투영: Gram은 상속되지만 **순**부등호는 깨질 수 있다
#
# 렌더링에 쓰이는 것은 3D $\Sigma$가 아니라 화면 위 2×2 공분산이다 (EWA splatting).
#
# $$\Sigma' = J\,W\,\Sigma\,W^\top J^\top = (JWM)(JWM)^\top$$
#
# $A = JWM$ 으로 묶으면 또 Gram 형태라 양의 준정부호는 공짜다. 그런데 $J$는 **2×3** 야코비안이므로
# $A$는 2×3, rank가 최대 2다. rank가 1로 떨어지면 $\det\Sigma' = 0$ 이 되고
# $\Sigma'^{-1}$ (코드의 `conics`)이 존재하지 않는다.
#
# **어떤 자세에서 그렇게 되는가?** $J$의 영공간은 시선 방향이므로, 시선 방향으로 얇은 것은
# 애초에 화면에 보이지도 않는다 (그 방향은 $J$가 이미 지워버린다). 위험한 것은 반대로
# **얇은 판이 시선을 품고 있을 때** — 즉 판을 옆에서(edge-on) 보는 경우다. 이때 타원체는
# 화면에서 **선분**으로 투영되어 폭이 0이 된다. 아래에서 $z$(시선)와 $y$ 방향으로는 두껍고
# $x$(화면 가로) 방향으로만 두께를 $10^0 \to 10^{-6}$ 로 줄여 이 붕괴를 재현한다.
#
# gsplat의 처방은 대각선에 상수를 더하는 것 (`_torch_impl.py:309`, `eps2d = 0.3`).
# $\Sigma' + \varepsilon I$ 의 고유값은 원래 고유값 $+\varepsilon$ 이므로 **최소고유값 $\ge 0.3$ 이 강제**된다.
# 대가로 없던 흐림이 더해져 어두워지므로 행렬식 비로 밝기를 보정한다.
#
# $$\text{compensation} = \sqrt{\dfrac{\det\Sigma'_{\text{orig}}}{\det(\Sigma'+\varepsilon I)}}$$

# %%
fx = fy = 800.0
W_cam = torch.eye(3)                    # world = camera (설명 단순화)
t_cam = torch.tensor([0.0, 0.0, 5.0])   # 카메라 앞 5m
tz = t_cam[2]
J = torch.tensor([[fx / tz, 0.0, -fx * t_cam[0] / tz**2],
                  [0.0, fy / tz, -fy * t_cam[1] / tz**2]])   # _torch_impl.py:100
print("J =\n", J, "\nJ의 영공간 = 시선(z) 방향 → z 두께는 화면에 아예 나타나지 않는다\n")

# edge-on 판 Gaussian: 시선(z)·세로(y)로는 두껍고, 화면 가로(x)로만 얇아진다
thin = torch.logspace(0, -6, 25)
rows = []
for th in thin:
    s = torch.tensor([float(th), 0.05, 0.05])          # x(화면 가로) 방향만 얇게
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    Sig3, _ = quat_scale_to_covar_preci(q, s)
    Sig2 = J @ W_cam @ Sig3 @ W_cam.T @ J.T            # [2,2]
    det_orig = torch.det(Sig2)
    Sig2d = Sig2 + torch.eye(2) * EPS2D
    det_dil = torch.det(Sig2d)
    comp = torch.sqrt(torch.clamp(det_orig / det_dil, min=0.0))
    rows.append((float(th),
                 torch.linalg.eigvalsh(Sig2).min().item(),
                 torch.linalg.eigvalsh(Sig2d).min().item(),
                 det_orig.item(), det_dil.item(), comp.item()))

_hdr = ("thickness", "minEig(Σ')", "minEig(+εI)", "det(Σ')", "comp")
print(f"{_hdr[0]:>10} {_hdr[1]:>12} {_hdr[2]:>12} {_hdr[3]:>11} {_hdr[4]:>7}")
for th, me, med, do, dd, cp in rows[::4]:
    print(f"{th:10.1e} {me:12.3e} {med:12.3e} {do:11.3e} {cp:7.4f}")

# 얇은 판 Gaussian은 dilation 없이는 conics 계산이 불가능하다
th, me, med, do, dd, cp = rows[-1]
print(f"\n두께 {th:.0e}: det(Σ') = {do:.3e} → 역행렬 불가."
      f"  +0.3·I 후 det = {dd:.4f}, 최소고유값 {med:.4f}, 밝기 보정 {cp:.5f}")
r_px = math.ceil(RADIUS_SIGMA * math.sqrt(EPS2D))
print(f"dilation의 물리적 의미: 화면에서 최소 √0.3 = {math.sqrt(EPS2D):.3f}px 폭의 저역통과 필터"
      f" (타일 반경 {RADIUS_SIGMA}σ → {r_px}px)")
# 출력: J = [[160, 0, -0], [0, 160, -0]]  (fx/tz = 800/5 = 160, 3열은 0 → z가 영공간)
# 출력:  thickness   minEig(Σ')  minEig(+εI)     det(Σ')    comp
# 출력:    1.0e+00    6.400e+01    6.430e+01   1.638e+06  0.9977
# 출력:    1.0e-01    6.400e+01    6.430e+01   1.638e+04  0.9971
# 출력:    1.0e-02    2.560e+00    2.860e+00   1.638e+02  0.9439
# 출력:    1.0e-03    2.560e-02    3.256e-01   1.638e+00  0.2797
# 출력:    1.0e-04    2.560e-04    3.003e-01   1.638e-02  0.0291
# 출력:    1.0e-05    2.560e-06    3.000e-01   1.638e-04  0.0029
# 출력:    1.0e-06    2.560e-08    3.000e-01   1.638e-06  0.0003
# 출력:   (두께 1e-1 까지는 최소고유값이 64 = 반대 축(0.05)이 결정 → 아직 안전.
# 출력:    두께가 그보다 작아지면 minEig = 160²·두께² 로 붕괴한다)
# 출력: 두께 1e-06: det(Σ') = 1.638e-06 → 역행렬 불가.
# 출력:   +0.3·I 후 det = 19.2900, 최소고유값 0.3000, 밝기 보정 0.00029
# 출력: dilation의 물리적 의미: 최소 √0.3 = 0.548px 폭의 저역통과 필터 (타일 반경 3.33σ → 2px)
# → 최소고유값은 0으로 붕괴하지만 +0.3·I 후에는 항상 ≥ 0.3 → conics 계산이 언제나 안전하다.
#   대신 compensation이 1 → 0 으로 떨어져 밝기를 깎는다 (에일리어싱 대신 흐림을 택함).


# %% [markdown]
# ## 8. quaternion 노름은 gauge 자유도 — 그래서 미정규화 저장이 가능하다
#
# $q$와 $\lambda q$ ($\lambda>0$)는 정규화하면 같은 단위 quaternion이 되므로 **같은 회전**을 준다.
# 노름 방향의 gradient는 손실에 영향을 주지 않고 정규화 층이 걸러내므로,
# gsplat은 `torch.rand((N,4))` 로 초기화한 자유 4-벡터를 그대로 학습시키고
# 렌더 직전에만 `F.normalize` 한다 (`rendering.py:1283`).
#
# **함정**: 정규화를 건너뛰고 미정규화 $q$를 공식에 직접 넣으면 $R$은 더 이상 직교가 아니다.
# 그래도 $\Sigma = RSS^\top R^\top$ 은 Gram 형태이므로 **양의 정부호는 유지된다**
# (3절의 증명이 $R$의 직교성을 전혀 쓰지 않았다). 깨지는 것은 정부호성이 아니라 **기하**다.

# %%
q_base = torch.tensor([0.3, -0.7, 0.2, 0.6])
print("λ 배율에 따른 회전행렬 차이 (정규화 O):")
R_ref = quat_to_rotmat(q_base, normalize=True)
for lam in [0.1, 1.0, 3.0, 50.0]:
    R_l = quat_to_rotmat(q_base * lam, normalize=True)
    print(f"  λ={lam:5.1f}  ‖λq‖={q_base.norm()*lam:7.3f}  ‖R(λq)-R(q)‖_max = {(R_l-R_ref).abs().max():.2e}")

print("\n정규화를 건너뛰면 (normalize=False):")
for lam in [0.5, 1.0, 2.0]:
    qq = torch.nn.functional.normalize(q_base, dim=-1) * lam    # ‖qq‖ = λ
    R_u = quat_to_rotmat(qq, normalize=False)
    Sig_u, _ = quat_scale_to_covar_preci(qq, s_demo, normalize=False)
    ev = torch.linalg.eigvalsh(Sig_u)
    ortho = (R_u.T @ R_u - torch.eye(3)).abs().max()
    print(f"  ‖q‖={lam:4.1f}  ‖RᵀR-I‖={ortho:8.3f}  최소고유값={ev.min():.3e} (>0 유지)"
          f"  축 길이 √eig = {[round(float(v)**0.5, 3) for v in ev]}"
          f"  vs 참값 {sorted(round(v, 3) for v in s_demo.tolist())}")
# 출력: 정규화하면 λ=0.1~50 어느 배율이든 R이 동일 (오차 ≤ 3e-08) → 노름은 gauge 자유도
# 출력: 정규화를 건너뛰면
# 출력:   ‖q‖=0.5  ‖RᵀR-I‖= 0.651  최소고유값=4.396e-03  축 길이 [0.066, 0.288, 1.668]
# 출력:   ‖q‖=1.0  ‖RᵀR-I‖= 0.000  최소고유값=1.000e-02  축 길이 [0.1, 0.5, 2.0]  ← 참값
# 출력:   ‖q‖=2.0  ‖RᵀR-I‖=41.633  최소고유값=2.377e-02  축 길이 [0.154, 3.171, 9.122]
# 출력: → 최소고유값은 어느 경우에도 양수로 유지되지만 (Gram 형태 덕분),
# 출력:   축 길이가 참값 [0.1, 0.5, 2.0]에서 크게 벗어난다 (‖q‖=2 → 최대축 2.0 → 9.12)
# → 양정부호는 Gram 형태가, 올바른 크기는 정규화가 각각 담보한다


# %% [markdown]
# ## 시각화
#
# - **①** 단위구가 $M = RS$ 를 거쳐 타원체가 되는 과정. 반축 길이는 정확히 $s_k$,
#   축 방향은 $R$의 열벡터 = $\Sigma$의 고유벡터
# - **②** 최소고유값 분포 (float64): 재매개화(파랑)는 20만 개 전부 0의 오른쪽,
#   $\Sigma$ 6원소 직접(빨강)은 98.7%가 0의 왼쪽 = 사용 불가
# - **③** Adam 학습 경로의 최소고유값. 직접 파라미터화는 0선 아래(붉은 영역)를 절반 이상 지나간다
# - **④** edge-on 판 Gaussian의 투영 2D 공분산 최소고유값 vs. 화면 가로 방향 두께.
#   dilation 없이는 0으로 붕괴하고, `+0.3·I` 는 하한선 0.3을 강제한다

# %%
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "scene"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
    subplot_titles=(
        "① 단위구 --(M = R S)--> 타원체 (반축 = s_k, 축 = R의 열)<br>"
        "<sub>가독성을 위해 s = (1.6, 0.8, 0.35)</sub>",
        "② 최소고유값 분포 (랜덤 파라미터 200k개, float64)",
        "③ Adam 학습 경로의 최소고유값 (목표 Σ* 피팅, 미니배치 잡음 σ=0.02)",
        "④ edge-on 판의 투영 2D 공분산 최소고유값 vs 화면 가로 두께",
    ),
    vertical_spacing=0.11, horizontal_spacing=0.09,
)

# ① 단위구 → 타원체
# 시각적으로 세 축이 모두 보이도록 s_demo(=2, 0.5, 0.1) 대신 덜 극단적인 s_vis를 쓴다.
# (s_3 = 0.1 이면 20:5:1 이라 화면에서 칼날처럼 보여 축 구분이 안 된다)
s_vis = torch.tensor([1.6, 0.8, 0.35])
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 30)
uu, vv = np.meshgrid(u, v)
sphere = np.stack([np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)], axis=-1)
M_np = (R_rand * s_vis[None, :]).numpy()           # M = R @ diag(s)
ell = sphere @ M_np.T                              # 각 점에 M을 적용
fig.add_trace(go.Surface(x=sphere[..., 0], y=sphere[..., 1], z=sphere[..., 2],
                         opacity=0.18, colorscale=[[0, "#9aa3af"], [1, "#9aa3af"]],
                         showscale=False, name="단위구", hoverinfo="skip"), row=1, col=1)
fig.add_trace(go.Surface(x=ell[..., 0], y=ell[..., 1], z=ell[..., 2],
                         opacity=0.55, colorscale=[[0, "#2f6fdb"], [1, "#7fb0ff"]],
                         showscale=False, name="타원체", hoverinfo="skip"), row=1, col=1)
for k, col in enumerate(["#d64545", "#2f9e44", "#f08c00"]):
    ax = (R_rand[:, k] * s_vis[k]).numpy()
    fig.add_trace(go.Scatter3d(x=[-ax[0], ax[0]], y=[-ax[1], ax[1]], z=[-ax[2], ax[2]],
                               mode="lines", line=dict(color=col, width=7),
                               name=f"① 축 s_{k+1}={s_vis[k]:.2f} (R의 {k+1}번째 열)"), row=1, col=1)

# ② 최소고유값 히스토그램 (양/음 모두 보이도록 asinh 스케일 대신 부호별 분리)
def _asinh(x, sc=1e-3):
    return np.arcsinh(np.asarray(x) / sc)

fig.add_trace(go.Histogram(x=_asinh(min_eig_r.numpy()), nbinsx=70,
                           name="② RSSᵀRᵀ 재매개화 (분포)",
                           marker_color="#2f6fdb", opacity=0.75), row=1, col=2)
fig.add_trace(go.Histogram(x=_asinh(min_eig_d.numpy()), nbinsx=70,
                           name="② Σ 6원소 직접 (분포)",
                           marker_color="#d64545", opacity=0.65), row=1, col=2)
fig.add_vline(x=0.0, line=dict(color="black", width=2, dash="dot"),
              row=1, col=2, exclude_empty_subplots=False)
fig.add_annotation(x=_asinh(-3.0), y=0.92, yref="y domain",
                   text=f"음수(사용 불가) {float((min_eig_d<0).float().mean())*100:.1f}%",
                   showarrow=False, font=dict(color="#d64545", size=10),
                   row=1, col=2, exclude_empty_subplots=False)
fig.add_annotation(x=_asinh(3.0), y=0.92, yref="y domain",
                   text="양수(사용 가능)", showarrow=False,
                   font=dict(color="#2f6fdb", size=10),
                   row=1, col=2, exclude_empty_subplots=False)

# ③ 학습 경로
steps = np.arange(STEPS)
fig.add_hrect(y0=_asinh(min(min(hist_b), -1.0)) * 1.05, y1=0.0, fillcolor="#d64545",
              opacity=0.08, line_width=0, layer="below",
              row=2, col=1, exclude_empty_subplots=False)
fig.add_trace(go.Scatter(x=steps, y=_asinh(hist_a), mode="lines",
                         name="③ (q, log s) 재매개화 (학습 경로)",
                         line=dict(color="#2f6fdb", width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=steps, y=_asinh(hist_b), mode="lines",
                         name="③ Σ 6원소 직접 (학습 경로)",
                         line=dict(color="#d64545", width=2)), row=2, col=1)
fig.add_hline(y=0.0, line=dict(color="black", width=1.5, dash="dot"),
              row=2, col=1, exclude_empty_subplots=False)
fig.add_annotation(x=STEPS * 0.55, y=_asinh(-0.2),
                   text=f"부정부호 {bad_b}/{STEPS} 스텝 → 렌더 불가",
                   showarrow=False, font=dict(color="#d64545", size=10),
                   row=2, col=1, exclude_empty_subplots=False)

# ④ 투영 최소고유값
th_arr = np.array([r[0] for r in rows])
fig.add_trace(go.Scatter(x=th_arr, y=[r[1] for r in rows], mode="lines+markers",
                         name="④ Σ' (dilation 없음)", line=dict(color="#d64545", width=2),
                         marker=dict(size=4)), row=2, col=2)
fig.add_trace(go.Scatter(x=th_arr, y=[r[2] for r in rows], mode="lines+markers",
                         name="④ Σ' + 0.3·I", line=dict(color="#2f6fdb", width=2),
                         marker=dict(size=4)), row=2, col=2)
# 주의(plotly 7 / kaleido에서 실측): 로그 축에서 add_hline/add_vline 은 **원래 데이터값**을,
# add_annotation 은 **log10 값**을 받는다. 섞으면 선이 1e-48 같은 곳에 그려진다.
fig.add_hline(y=EPS2D, line=dict(color="#2f6fdb", width=1, dash="dot"),
              row=2, col=2, exclude_empty_subplots=False)
fig.add_annotation(x=np.log10(1e-4), y=np.log10(EPS2D) + 0.5,
                   text="eps2d = 0.3 하한", showarrow=False,
                   font=dict(color="#2f6fdb", size=10),
                   row=2, col=2, exclude_empty_subplots=False)

tickvals = [-1e0, -1e-2, 0, 1e-2, 1e0, 1e2]
ticktext = ["-1", "-1e-2", "0", "1e-2", "1", "100"]
fig.update_xaxes(title_text="최소고유값 (asinh 스케일)", row=1, col=2, tickangle=0,
                 tickvals=_asinh(tickvals), ticktext=ticktext)
fig.update_yaxes(title_text="개수", type="log", row=1, col=2)
fig.update_xaxes(title_text="Adam step", row=2, col=1)
fig.update_yaxes(title_text="최소고유값 (asinh 스케일)", row=2, col=1,
                 tickvals=_asinh(tickvals), ticktext=ticktext)
fig.update_xaxes(title_text="판 두께 (화면 가로 방향 s_1)", type="log", autorange="reversed", row=2, col=2)
fig.update_yaxes(title_text="2D 공분산 최소고유값 (px²)", type="log", row=2, col=2)
fig.update_layout(
    height=1020, width=1360, template="plotly_white",
    title=dict(text="Σ = R S Sᵀ Rᵀ — Gram 형태가 양정부호를 구조적으로 보장한다",
               x=0.5, xanchor="center", y=0.985, yanchor="top", font=dict(size=19)),
    legend=dict(orientation="h", y=-0.07, x=0.5, xanchor="center"),
    scene=dict(aspectmode="data",
               camera=dict(eye=dict(x=1.55, y=1.55, z=0.95)),
               xaxis=dict(title="x", nticks=5), yaxis=dict(title="y", nticks=5),
               zaxis=dict(title="z", nticks=5)),
    barmode="overlay", margin=dict(t=130, b=120),
)
fig.write_image("expy.png", scale=2)
print("expy.png 저장 완료")
_show(fig)
# 출력: expy.png 저장 완료
