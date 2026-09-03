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
# (gsplat 자체는 import하지 않는다 — 토이 재구현으로 개념만 확인)

# %% [markdown]
# # 왜 Gaussian 파라미터를 "비제약 공간"에 저장하는가
#
# 3DGS의 파라미터에는 **수학적 제약**이 붙어 있다.
#
# | 파라미터 | 제약 | gsplat이 저장하는 값 | 렌더 직전 활성화 |
# |---|---|---|---|
# | `scales` | $s > 0$ | $\tilde s = \log s$ | `torch.exp` |
# | `opacities` | $o \in (0,1)$ | $\tilde o = \mathrm{logit}(o)$ | `torch.sigmoid` |
# | `quats` | $\lVert q \rVert = 1$ | 자유로운 4-벡터 | 커널 내부 `normalize` |
# | `means`, `sh0/shN` | 없음 | 그대로 | 없음 |
#
# Adam은 **제약 없는 $\mathbb{R}^n$에서 도는 옵티마이저**다. 제약이 있는 값을 그대로
# 넘기면 step 한 번이 제약을 깨뜨릴 수 있고, 이를 clamp로 막으면 gradient가 죽고
# Adam의 모멘트 상태가 오염된다. 그래서 **재매개화(reparametrization)** 로 제약을
# 제거해 버리는 것이 gsplat(및 원논문)의 선택이다.
#
# 아래에서 다섯 가지를 직접 확인한다.
#
# 1. `scales`: 직접 최적화 + clamp vs. log 공간 최적화
# 2. log 공간이 주는 **상대(곱셈) 업데이트** 성질
# 3. `opacities`: sigmoid의 gradient 계수 $o(1-o)$와 3DGS 임계값들의 위치
# 4. 공분산 $\Sigma = RSS^\top R^\top$의 양정부호 자동 보장, `quats`의 norm 자유도
# 5. 밀도화(split / reset / relocate)가 활성화 공간을 왕복하는 방식

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

# gsplat/examples/simple_trainer.py의 실제 기본값
LR_SCALES = 5e-3
LR_OPACITIES = 5e-2
INIT_OPACITY = 0.1
PRUNE_OPA = 0.005
RESET_OPA = 0.01
SPLIT_SHRINK = 1.6

print(f"scales lr = {LR_SCALES}, opacities lr = {LR_OPACITIES}")
print(f"logit(init_opacity=0.1) = {torch.logit(torch.tensor(INIT_OPACITY)):.4f}")
# 출력: scales lr = 0.005, opacities lr = 0.05
# 출력: logit(init_opacity=0.1) = -2.1972

# %% [markdown]
# ## 1. `scales`: 제약 공간에서 Adam을 돌리면 무슨 일이 생기나
#
# 크기가 크게 다른 Gaussian 두 개를 동시에 최적화한다. 실제 씬에서 흔한 상황이다 —
# SfM 포인트 밀도가 균일하지 않으니 초기 `scales = log(knn 평균거리)`는
# 몇 자릿수에 걸쳐 퍼져 있다.
#
# - Gaussian A: $s_0 = 1.0 \to s^\star = 0.1$ (10배 축소)
# - Gaussian B: $s_0 = 10^{-3} \to s^\star = 10^{-4}$ (똑같이 10배 축소)
#
# 손실은 $\mathcal L = \sum_i (s_i - s_i^\star)^2$. **하나의 lr = 5e-3** 로 둘을 함께 학습한다.
#
# 핵심은 Adam의 step 크기가 gradient 크기와 (거의) 무관하게 $\approx \text{lr}$ 이라는 점이다.
# 즉 제약 공간에서 최적화하면 한 스텝이 **절대량 0.005** 만큼 움직인다. Gaussian B는
# 전체 크기가 0.001이니 첫 스텝에 음수로 튀어 나가고, `clamp(min=eps)`에 달라붙는다.

# %%
STEPS = 6000
init = torch.tensor([1.0, 1e-3])
target = torch.tensor([0.1, 1e-4])
EPS = 1e-8

# (a) 제약 공간 직접 최적화 + 매 스텝 clamp (= projected gradient)
p_direct = torch.nn.Parameter(init.clone())
opt_d = torch.optim.Adam([p_direct], lr=LR_SCALES, eps=1e-15)
traj_direct, clamp_hits = [], 0
for _ in range(STEPS):
    loss = ((p_direct - target) ** 2).sum()
    opt_d.zero_grad()
    loss.backward()
    opt_d.step()
    with torch.no_grad():
        clamp_hits += int((p_direct < EPS).any())
        p_direct.clamp_(min=EPS)  # 제약 유지를 위한 사영
    traj_direct.append(p_direct.detach().clone())
traj_direct = torch.stack(traj_direct).numpy()

# (b) 비제약(log) 공간 최적화 — gsplat 방식
p_log = torch.nn.Parameter(torch.log(init.clone()))
opt_l = torch.optim.Adam([p_log], lr=LR_SCALES, eps=1e-15)
traj_log = []
for _ in range(STEPS):
    s = torch.exp(p_log)  # 렌더 직전 활성화
    loss = ((s - target) ** 2).sum()
    opt_l.zero_grad()
    loss.backward()
    opt_l.step()
    traj_log.append(torch.exp(p_log).detach().clone())
traj_log = torch.stack(traj_log).numpy()

rel = lambda v: np.abs(v - target.numpy()) / target.numpy()
print(f"clamp에 걸린 스텝 수 (direct): {clamp_hits} / {STEPS}")
print(f"direct 최종 s = {traj_direct[-1]}  상대오차 = {rel(traj_direct[-1])}")
print(f"log    최종 s = {traj_log[-1]}  상대오차 = {rel(traj_log[-1])}")
print(f"direct: B의 마지막 1000스텝 범위 = [{traj_direct[-1000:, 1].min():.2e}, {traj_direct[-1000:, 1].max():.2e}]")
# 출력: clamp에 걸린 스텝 수 (direct): 2517 / 6000
# 출력: direct 최종 s = [0.10000001 0.00028257]  상대오차 = [7.4505806e-08 1.8257101e+00]
# 출력: log    최종 s = [0.1000098  0.00010001]  상대오차 = [9.79751348e-05 1.03827915e-04]
# 출력: direct: B의 마지막 1000스텝 범위 = [1.00e-08, 4.20e-04]

# %% [markdown]
# 결과를 읽어 보자.
#
# - **Gaussian A (큰 것)** 는 제약 공간에서도 잘 수렴한다. 크기 0.1이 Adam의
#   step 규모 $\approx$ lr $= 0.005$ 보다 충분히 크기 때문이다.
# - **Gaussian B (작은 것)** 는 제약 공간에서 완전히 실패한다. 6000 스텝 중 **2517 스텝**이
#   `clamp` 벽에 부딪히고(0 아래로 튀어나감 → 사영 → 다시 튀어나감), 마지막 1000 스텝에서도
#   $10^{-8}$ 과 $4.2\times10^{-4}$ 사이를 왕복한다. 최종 상대오차 **183%**. 이유는 단순하다 —
#   전체 크기가 $10^{-3}$ 인 값을 0.005짜리 스텝으로 다루려니 한 걸음이 값의 5배다.
# - **log 공간**에서는 두 Gaussian의 궤적이 **완전히 겹친다**(상대오차 $10^{-4}$ 수준으로 동일).
#   $\log 0.1 - \log 1.0 = \log 10^{-4} - \log 10^{-3} = -2.303$ 이므로
#   "10배 줄이기"라는 과제가 초기 크기와 무관하게 **같은 거리 문제**가 되기 때문이다.
#
# ## 2. log 공간 = 곱셈 업데이트 = 스케일 불변
#
# 체인룰을 보면 이유가 분명하다. $s = e^{\tilde s}$ 이므로
#
# $$\frac{\partial \mathcal L}{\partial \tilde s} = \frac{\partial \mathcal L}{\partial s}\cdot e^{\tilde s} = s\,\frac{\partial \mathcal L}{\partial s}$$
#
# 그리고 업데이트는 덧셈이지만 실공간에서는 곱셈이 된다.
#
# $$\tilde s \leftarrow \tilde s - \eta \quad\Longleftrightarrow\quad s \leftarrow s \cdot e^{-\eta}$$
#
# 즉 **lr이 "절대 길이"가 아니라 "한 스텝당 상대 변화율" 단위**가 된다.
# lr $=5\times10^{-3}$ 은 곧 "스텝당 최대 0.5% 크기 변화"다. 씬 스케일이 바뀌어도,
# Gaussian 크기가 몇 자릿수에 걸쳐 있어도 하나의 lr이 통한다.
# (`means`만 예외적으로 `1.6e-4 * scene_scale` 처럼 lr에 직접 스케일을 곱하는 이유가
# 이것이다 — `means`는 log 공간에 넣을 수 없는 부호 있는 좌표이므로 손으로 보정해야 한다.)

# %%
step_rel_direct = np.abs(np.diff(traj_direct, axis=0)) / np.maximum(traj_direct[:-1], EPS)
step_rel_log = np.abs(np.diff(traj_log, axis=0)) / traj_log[:-1]
print("스텝당 상대 변화율 |Δs|/s, 초기 50스텝 중간값")
print(f"  direct: A={np.median(step_rel_direct[:50, 0]):.4f}  B={np.median(step_rel_direct[:50, 1]):.4f}")
print(f"  log   : A={np.median(step_rel_log[:50, 0]):.4f}  B={np.median(step_rel_log[:50, 1]):.4f}")
print(f"  참고: exp(lr)-1 = {math.exp(LR_SCALES) - 1:.4f}")
# 출력: 스텝당 상대 변화율 |Δs|/s, 초기 50스텝 중간값
# 출력:   direct: A=0.0055  B=0.3965
# 출력:   log   : A=0.0047  B=0.0047
# 출력:   참고: exp(lr)-1 = 0.0050

# %% [markdown]
# 제약 공간에서 Gaussian B는 스텝당 **약 40%** 씩 요동친다 — A(0.55%)의 70배다.
# log 공간에서는 A와 B가 **똑같이 약 0.5%**, 즉 `exp(lr) - 1` 과 일치한다.
# lr 하나가 모든 크기의 Gaussian에 대해 같은 의미를 갖는다는 뜻이다.
#
# ## 3. `opacities`: sigmoid와 gradient 계수 $o(1-o)$
#
# 불투명도는 알파 블렌딩 $C = \sum_i c_i \alpha_i \prod_{j<i}(1-\alpha_j)$ 에서
# 반드시 $(0,1)$ 이어야 의미가 있다. logit 공간에 저장하면 경계는 $\pm\infty$ 로
# 밀려나므로 clamp가 아예 필요 없다. 대가는 gradient 계수다.
#
# $$o = \sigma(\tilde o), \qquad \frac{\partial o}{\partial \tilde o} = o\,(1-o)$$
#
# $o \to 0$ 또는 $o \to 1$ 에서 gradient가 0으로 사그라든다(포화). 이것이 3DGS에
# **prune과 opacity reset이 필요한 구조적 이유**다 — 한 번 투명해진 Gaussian은
# 스스로 되살아나기 어렵다.

# %%
o_grid = np.linspace(1e-4, 1 - 1e-4, 400)
logit_grid = np.log(o_grid / (1 - o_grid))
dody = o_grid * (1 - o_grid)

for name, o in [("init 0.1", INIT_OPACITY), ("reset 0.01", RESET_OPA), ("prune 0.005", PRUNE_OPA)]:
    g = o * (1 - o)
    print(f"{name:12s} logit={math.log(o / (1 - o)):8.4f}  do/dõ={g:.5f}  (0.5 대비 {g / 0.25:.3f}x)")
# 출력: init 0.1     logit= -2.1972  do/dõ=0.09000  (0.5 대비 0.360x)
# 출력: reset 0.01   logit= -4.5951  do/dõ=0.00990  (0.5 대비 0.040x)
# 출력: prune 0.005  logit= -5.2933  do/dõ=0.00498  (0.5 대비 0.020x)

# %% [markdown]
# prune 임계값($o = 0.005$)에 있는 Gaussian의 gradient는 $o = 0.5$ 대비 **1/50**.
# 그래서 gsplat은 그런 Gaussian을 되살리려 하지 않고 그냥 잘라낸다
# (`default.py:351` — `torch.sigmoid(params["opacities"]) < prune_opa`).
#
# 참고로 logit 공간의 lr은 5e-2로 `scales`보다 10배 크다. 유효 범위가
# $\mathrm{logit}(0.005) \approx -5.3$ 부터 $\mathrm{logit}(0.995) \approx +5.3$ 까지
# 폭 10이 넘는 축이기 때문이다.
#
# ## 4. 공분산의 양정부호와 quaternion의 norm 자유도
#
# `scales`를 log에 넣는 더 근본적인 이유: 공분산은
#
# $$\Sigma = R\,S\,S^\top R^\top, \qquad S = \mathrm{diag}(e^{\tilde s_1}, e^{\tilde s_2}, e^{\tilde s_3})$$
#
# 로 합성되고, 이 형태는 고유값이 정확히 $e^{2\tilde s_k} > 0$ 이므로 **항상 양의 정부호**다.
# 만약 $\Sigma$ 의 대칭 6원소를 직접 파라미터로 두면 최적화 도중 부정부호가 되어
# EWA 투영과 $\exp(-\frac12\Delta^\top\Sigma'^{-1}\Delta)$ 가 발산한다.

# %%
def quat_to_rotmat(q):
    """gsplat normalized_quat_to_rotmat의 토이 버전 (w, x, y, z 순)."""
    q = q / q.norm(dim=-1, keepdim=True)  # ← 저장은 미정규화, 사용 직전 정규화
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


N = 20000
DT = torch.float64  # 고유값 판정이므로 배정도로 (float32에서는 반올림으로 ±1e-2가 섞인다)

# (a) gsplat 방식: 자유 quat + log-scale → RSSᵀRᵀ
quats = torch.rand(N, 4, dtype=DT)        # simple_trainer.py:331과 동일하게 미정규화 랜덤
log_s = torch.randn(N, 3, dtype=DT)
R = quat_to_rotmat(quats)
S = torch.exp(log_s)
Sigma = R @ torch.diag_embed(S**2) @ R.transpose(-1, -2)
eig_repar = torch.linalg.eigvalsh(Sigma).min(dim=-1).values

# (b) 대칭 6원소를 직접 파라미터로 두는 가상의 방식
tri = torch.randn(N, 6, dtype=DT)
i, j = torch.tril_indices(3, 3)
Sig_free = torch.zeros(N, 3, 3, dtype=DT)
Sig_free[:, i, j] = tri
Sig_free = Sig_free + Sig_free.transpose(-1, -2) - torch.diag_embed(Sig_free.diagonal(dim1=-2, dim2=-1))
eig_free = torch.linalg.eigvalsh(Sig_free).min(dim=-1).values

print(f"RSSᵀRᵀ 재매개화: 최소고유값 < 0 인 비율 = {(eig_repar < 0).float().mean():.4f}"
      f"  (min = {eig_repar.min():.3e})")
print(f"Σ 6원소 직접   : 최소고유값 < 0 인 비율 = {(eig_free < 0).float().mean():.4f}")

# quaternion의 norm은 회전에 영향이 없다 (gauge 자유도) → 노름 제약을 학습에 강제할 필요 없음
q = torch.rand(1, 4)
print(f"‖q‖={q.norm():.4f} 와 ‖7q‖={7 * q.norm():.4f} 의 회전행렬 차이 = "
      f"{(quat_to_rotmat(q) - quat_to_rotmat(7 * q)).abs().max():.2e}")
# 출력: RSSᵀRᵀ 재매개화: 최소고유값 < 0 인 비율 = 0.0000  (min = 1.950e-04)
# 출력: Σ 6원소 직접   : 최소고유값 < 0 인 비율 = 0.9872
# 출력: ‖q‖=1.1907 와 ‖7q‖=8.3346 의 회전행렬 차이 = 1.19e-07

# %% [markdown]
# 자유 대칭행렬 6원소를 랜덤하게 두면 **98.7%** 가 부정부호다. 반면 재매개화는 **정확히 0%**.
# `quats`도 노름이 0.9든 6.3든 같은 회전을 주므로, gsplat은 노름 제약을 **학습에 강제하지 않고**
# 커널 진입 직전에 `F.normalize`로 흡수한다 (`rendering.py:1283`,
# `.cu` 주석 "quats … No need to be normalized").
#
# ## 5. 부수 효과: 밀도화도 활성화 공간을 왕복해야 한다
#
# 파라미터 공간(비제약)과 의미 공간(물리 단위)이 다르므로, **판단과 연산은 항상
# 활성화 후 물리 단위에서** 하고 결과를 다시 저장 공간으로 되돌려야 한다.
# gsplat `strategy/ops.py`가 정확히 그렇게 한다.

# %%
raw_scales = torch.log(torch.tensor([[0.05, 0.05, 0.02]]))
raw_opac = torch.logit(torch.tensor([0.30]))

# (1) split: 실공간에서 1/1.6배 → log 공간에서는 log(1.6) 상수 감산 (ops.py:211)
split_gsplat = torch.log(torch.exp(raw_scales) / SPLIT_SHRINK)
split_shift = raw_scales - math.log(SPLIT_SHRINK)
print(f"split  log(exp(s)/1.6) == s - log(1.6) ? {torch.allclose(split_gsplat, split_shift)}"
      f"  (shift = -{math.log(SPLIT_SHRINK):.4f})")

# (2) revised opacity: 겹친 두 개의 합성 알파가 원래 값과 같아지도록 (ops.py:213)
o = torch.sigmoid(raw_opac)
o_new = 1.0 - torch.sqrt(1.0 - o)
print(f"revised opacity {o.item():.4f} → {o_new.item():.4f}, "
      f"두 개 합성 1-(1-o')² = {(1 - (1 - o_new) ** 2).item():.4f}  "
      f"→ 다시 logit {torch.logit(o_new).item():.4f}")

# (3) reset_opa: 저장 공간에서 직접 clamp (ops.py:287)
raw_many = torch.logit(torch.tensor([0.002, 0.05, 0.9]))
cap = torch.logit(torch.tensor(RESET_OPA)).item()
print(f"reset_opa: sigmoid(clamp(p, max=logit(0.01))) = "
      f"{torch.sigmoid(torch.clamp(raw_many, max=cap)).tolist()}")

# (4) prune 판정은 활성화 후 (default.py:351,354)
print(f"prune 판정: sigmoid(opacities) < {PRUNE_OPA} 그리고 exp(scales).max() > 0.1·scene_scale")
# 출력: split  log(exp(s)/1.6) == s - log(1.6) ? True  (shift = -0.4700)
# 출력: revised opacity 0.3000 → 0.1633, 두 개 합성 1-(1-o')² = 0.3000  → 다시 logit -1.6336
# 출력: reset_opa: sigmoid(clamp(p, max=logit(0.01))) = [0.0020000003278255463, 0.009999998845160007, 0.009999998845160007]
# 출력: prune 판정: sigmoid(opacities) < 0.005 그리고 exp(scales).max() > 0.1·scene_scale

# %% [markdown]
# `reset_opa`가 대입이 아니라 `clamp(max=...)` 인 것도 눈여겨볼 만하다 —
# 이미 0.01보다 투명한 것(0.002)은 그대로 두고, 불투명한 것만 끌어내린다.
#
# ## 시각화

# %%
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "① scales 궤적 (동일 lr=5e-3) · 옅은 띠 = direct의 진동 범위",
        "② 스텝당 상대 변화율 |Δs|/s (60스텝 중간값)",
        "③ sigmoid 활성화와 gradient 계수 o(1-o)",
        "④ 공분산 최소고유값: 재매개화 vs Σ 6원소 직접",
    ),
    specs=[[{}, {}], [{"secondary_y": True}, {}]],
    vertical_spacing=0.15, horizontal_spacing=0.135,
)
C = {"dA": "#d62728", "dB": "#ff7f0e", "lA": "#1f77b4", "lB": "#17becf"}


def envelope(v, w=60):
    """진동하는 궤적을 [w스텝 구간 min, max] 띠 + 중간값으로 압축."""
    n = (len(v) // w) * w
    m = v[:n].reshape(-1, w)
    x = np.arange(n).reshape(-1, w).mean(1)
    return x, np.maximum(m.min(1), 1e-30), m.max(1), np.median(m, axis=1)


def band(x, lo, hi, rgba, row, col):
    """진동 범위를 옅은 띠로 (legend에는 넣지 않는다)."""
    for y, fill in ((hi, None), (lo, "tonexty")):
        fig.add_trace(go.Scatter(x=x, y=y, line=dict(width=0), fill=fill,
                                 fillcolor=rgba, showlegend=False, hoverinfo="skip"),
                      row=row, col=col)


for k, (lbl, col, tgt) in enumerate(
    [("A (s₀=1.0)", "A", 0.1), ("B (s₀=1e-3)", "B", 1e-4)]
):
    rgba = "rgba(214,39,40,0.13)" if col == "A" else "rgba(255,127,14,0.16)"
    # 제약공간(direct): clamp 진동이 심하므로 min–max 띠 + 기하평균으로 표현
    x, lo, hi, gm = envelope(traj_direct[:, k])
    band(x, lo, hi, rgba, 1, 1)
    fig.add_trace(go.Scatter(x=x, y=gm, name=f"direct {lbl}",
                             line=dict(color=C["d" + col], width=2.2)), row=1, col=1)
    # log 공간은 매끄러우므로 그대로
    fig.add_trace(go.Scatter(x=np.arange(STEPS), y=traj_log[:, k], name=f"log {lbl}",
                             line=dict(color=C["l" + col], width=2.2, dash="dash")),
                  row=1, col=1)
    fig.add_hline(y=tgt, line=dict(color="gray", dash="dot", width=1), row=1, col=1)

    # ②는 띠 없이 중간값만 (범위가 너무 넓어 띠가 패널을 덮는다)
    xr, _, _, gmr = envelope(step_rel_direct[:, k])
    fig.add_trace(go.Scatter(x=xr, y=gmr, line=dict(color=C["d" + col], width=2.2),
                             showlegend=False), row=1, col=2)
    _, _, _, gml = envelope(step_rel_log[:, k])
    fig.add_trace(go.Scatter(x=xr, y=gml, line=dict(color=C["l" + col], width=2.2, dash="dash"),
                             showlegend=False), row=1, col=2)

fig.add_hline(y=math.exp(LR_SCALES) - 1, line=dict(color="black", dash="dot", width=1),
              annotation_text="exp(lr)-1 = 0.5%", annotation_font_size=9,
              annotation_position="bottom right", row=1, col=2)

fig.add_trace(go.Scatter(x=logit_grid, y=o_grid, name="σ(õ)",
                         line=dict(color="#2ca02c", width=2.2)), row=2, col=1)
fig.add_trace(go.Scatter(x=logit_grid, y=dody, name="do/dõ = o(1-o)",
                         line=dict(color="#9467bd", width=2.2, dash="dash")),
              row=2, col=1, secondary_y=True)
for name, o_v in [("init 0.1", INIT_OPACITY), ("reset 0.01", RESET_OPA), ("prune 0.005", PRUNE_OPA)]:
    fig.add_vline(x=math.log(o_v / (1 - o_v)), line=dict(color="gray", dash="dot", width=1),
                  annotation_text=name, annotation_font_size=9,
                  annotation_position="bottom right", annotation_textangle=-90,
                  row=2, col=1)

fig.add_trace(go.Histogram(x=eig_free.numpy(), name="Σ 6원소 직접", nbinsx=80,
                           marker_color="#d62728", opacity=0.65), row=2, col=2)
fig.add_trace(go.Histogram(x=eig_repar.numpy(), name="R S Sᵀ Rᵀ", nbinsx=80,
                           marker_color="#1f77b4", opacity=0.65), row=2, col=2)
fig.add_vline(x=0.0, line=dict(color="black", width=1.2), row=2, col=2)

fig.update_yaxes(type="log", title_text="scale s", range=[-5.2, 0.3], row=1, col=1)
fig.update_xaxes(title_text="step", row=1, col=1)
fig.update_yaxes(type="log", title_text="|Δs|/s", range=[-4.2, 1.0], row=1, col=2)
fig.update_xaxes(title_text="step", row=1, col=2)
fig.update_xaxes(title_text="저장값 õ = logit(o)", range=[-8, 8], row=2, col=1)
fig.update_yaxes(title_text="o = σ(õ)", range=[-0.03, 1.03], row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text="do/dõ = o(1-o)", range=[-0.008, 0.27], dtick=0.05,
                 row=2, col=1, secondary_y=True)
fig.update_xaxes(title_text="min eigenvalue of Σ", range=[-4, 4], row=2, col=2)
fig.update_yaxes(type="log", title_text="count", row=2, col=2)
fig.update_layout(
    height=860, width=1220, barmode="overlay",
    title_text="비제약 공간 저장 + 활성화: 무엇을 사는가",
    legend=dict(orientation="h", y=-0.11, font_size=9),
    template="plotly_white",
)
_show(fig)
fig.write_image("expy.png", scale=2)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 한 줄 요약
#
# 제약을 **옵티마이저의 문제로 남기지 말고 파라미터화로 제거해 버린다.**
# `scales`는 log에, `opacities`는 logit에 저장해 (1) 제약을 자동으로 만족하고
# (2) lr을 절대량이 아닌 상대 변화율 단위로 만들고 (3) 공분산의 양정부호를 보장한다.
# 대가는 활성화 함수의 gradient 계수(포화)와, 밀도화 코드가 매번
# `exp`/`sigmoid` ↔ `log`/`logit`을 왕복해야 하는 번거로움이다.
