# %% [markdown]
# # `means` 학습률 지수 감쇠 스케줄 실습
#
# 질문: **`means`의 학습률 감쇠 스케줄은 어떻게 설정하는가?**
#
# 답: `torch.optim.lr_scheduler.ExponentialLR`에
# $\gamma = 0.01^{1/T}$ (`gamma = 0.01 ** (1.0 / MAX_STEPS)`)를 주어,
# 총 $T$ 스텝에 걸쳐 초기 학습률의 1%까지 지수 감쇠시킨다.
#
# $$\eta_t = \eta_0\,\gamma^{\,t},\qquad \gamma^{\,T}=0.01$$
#
# 이 스크립트는 gsplat을 import하지 않고 (JIT 빌드가 매우 느리다) numpy/torch/plotly로
# 스케줄만 그대로 재현·검증한다.

# %%
# 필요 패키지: numpy, torch, plotly, kaleido
import math

import numpy as np
import plotly.graph_objects as go
import torch
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
print("torch", torch.__version__)
# 출력: torch 2.9.1+cu128

# %% [markdown]
# ## 1단계: `gamma`는 "1%"라는 요구조건을 푼 결과다
#
# 원하는 조건: $\eta_T = 0.01\,\eta_0$. $\eta_t=\eta_0\gamma^t$ 이므로
#
# $$\gamma^T = 0.01 \;\Longleftrightarrow\; \gamma = 0.01^{1/T} = \exp\!\Big(\frac{\ln 0.01}{T}\Big)$$

# %%
for T in (2_000, 7_000, 30_000):
    gamma = 0.01 ** (1.0 / T)
    print(
        f"T={T:6,d}  gamma={gamma:.10f}  "
        f"1스텝 감소율={100*(gamma-1):+.5f}%  gamma^T={gamma**T:.6f}"
    )
# 출력: T= 2,000  gamma=0.9977000638  1스텝 감소율=-0.22999%  gamma^T=0.010000
# 출력: T= 7,000  gamma=0.9993423349  1스텝 감소율=-0.06577%  gamma^T=0.010000
# 출력: T=30,000  gamma=0.9998465061  1스텝 감소율=-0.01535%  gamma^T=0.010000

# %%
# ln 을 이용한 동치 표현 + 반감기 t_(1/2) = T * ln2 / ln100
T = 30_000
gamma = 0.01 ** (1.0 / T)
print("exp(ln(0.01)/T) =", math.exp(math.log(0.01) / T))
print("반감기(step)     =", math.log(2) / (-math.log(gamma)))
print("T * ln2/ln100    =", T * math.log(2) / math.log(100))
# 출력: exp(ln(0.01)/T) = 0.9998465061085267
# 출력: 반감기(step)     = 4515.449934958809
# 출력: T * ln2/ln100    = 4515.449934959717

# %% [markdown]
# ## 2단계: 진짜 `ExponentialLR`로 확인
#
# gsplat 학습 루프와 동일하게 `means` 전용 Adam 하나에만 스케줄러를 붙이고,
# 매 iteration `optimizer.step()` 뒤에 `scheduler.step()`을 호출한다.

# %%
MAX_STEPS = 2_000  # 워크스루 데모 값
scene_scale = 1.0
means = torch.nn.Parameter(torch.zeros(10, 3))  # [N,3] 위치 파라미터
opt_means = torch.optim.Adam([means], lr=1.6e-4 * scene_scale, eps=1e-15)
sched = torch.optim.lr_scheduler.ExponentialLR(
    opt_means, gamma=0.01 ** (1.0 / MAX_STEPS)
)

lr_trace = []
for step in range(MAX_STEPS):
    lr_trace.append(sched.get_last_lr()[0])
    means.grad = torch.zeros_like(means)  # 더미 grad
    opt_means.step()
    opt_means.zero_grad(set_to_none=True)
    sched.step()  # (6) means lr 감쇠

lr_trace = np.array(lr_trace)
lr0 = 1.6e-4 * scene_scale
closed_form = lr0 * (0.01 ** (1.0 / MAX_STEPS)) ** np.arange(MAX_STEPS)
print(f"lr[0]      = {lr_trace[0]:.6e}   (초기값 1.6e-4 * scene_scale)")
print(f"lr[last]   = {lr_trace[-1]:.6e}")
print(f"최종 lr 후 한 스텝 = {sched.get_last_lr()[0]:.6e}  (= 0.01 * lr0)")
print(f"lr[last]/lr[0] = {lr_trace[-1]/lr_trace[0]:.6f}")
print(f"닫힌 형태와의 최대 오차 = {np.abs(lr_trace - closed_form).max():.3e}")
# 출력: lr[0]      = 1.600000e-04   (초기값 1.6e-4 * scene_scale)
# 출력: lr[last]   = 1.603688e-06
# 출력: 최종 lr 후 한 스텝 = 1.600000e-06  (= 0.01 * lr0)
# 출력: lr[last]/lr[0] = 0.010023
# 출력: 닫힌 형태와의 최대 오차 = 1.491e-19

# %% [markdown]
# `scheduler.step()`을 $T$번 호출한 뒤의 학습률이 정확히 $0.01\,\eta_0$이다.
# (루프 안에서 기록한 마지막 값은 $T-1$번 곱한 시점이라 아주 살짝 크다.)
#
# ## 3단계: 총 스텝 수가 달라도 "진행률" 기준 곡선은 같다
#
# $$\eta_t = \eta_0\big(0.01^{1/T}\big)^{sT} = \eta_0 \cdot 100^{-s},\qquad s=\tfrac{t}{T}$$
#
# $T$가 식에서 사라지므로, 2,000스텝 데모와 30,000스텝 본 학습은 진행률 축에서 완전히 겹친다.

# %%
def lr_curve(T, lr0=1.6e-4, n=400):
    t = np.linspace(0, T, n)
    return t, lr0 * (0.01 ** (1.0 / T)) ** t


for T in (2_000, 7_000, 30_000):
    t, lr = lr_curve(T)
    s = t / T
    print(f"T={T:6,d}  s=0.5에서 lr={np.interp(0.5, s, lr):.4e}  (이론 1.6e-4*100^-0.5)")
print("이론값:", 1.6e-4 * 100 ** (-0.5))
# 출력: T= 2,000  s=0.5에서 lr=1.6000e-05  (이론 1.6e-4*100^-0.5)
# 출력: T= 7,000  s=0.5에서 lr=1.6000e-05  (이론 1.6e-4*100^-0.5)
# 출력: T=30,000  s=0.5에서 lr=1.6000e-05  (이론 1.6e-4*100^-0.5)
# 출력: 이론값: 1.6000000000000003e-05

# %% [markdown]
# ## 4단계: 왜 감쇠가 필요한가 — 노이즈가 섞인 목표 위치 따라가기
#
# 실제 3DGS에서는 매 스텝 **다른 이미지 한 장**만 보고 위치를 갱신하므로 기울기에 잡음이 섞인다.
# 잡음이 있는 SGD에서 학습률이 일정하면, 파라미터는 정답 주변을 계속 떠도는
# "잔여 진동(residual jitter)"을 남긴다. 학습률을 줄이면 그 진동 폭도 함께 줄어든다.
#
# 아래는 2차 함수(볼록한 그릇) 위에서 잡음 섞인 기울기로 내려가는 최소 모형이다:
#
# $$L(\mu)=\tfrac12\|\mu-\mu^\star\|^2,\qquad
#   \hat g = (\mu-\mu^\star) + \varepsilon,\;\; \varepsilon\sim\mathcal N(0,\sigma^2 I)$$

# %%
def run_sgd(schedule, T=2_000, lr0=0.05, sigma=0.3, seed=0):
    """schedule: step -> lr 배율. 반환: 정답까지의 거리 궤적."""
    g = torch.Generator().manual_seed(seed)
    mu_star = torch.tensor([1.0, -2.0, 0.5])
    mu = torch.zeros(3)
    dist = []
    for t in range(T):
        grad = (mu - mu_star) + sigma * torch.randn(3, generator=g)
        mu = mu - lr0 * schedule(t) * grad
        dist.append(torch.linalg.norm(mu - mu_star).item())
    return np.array(dist)


T = 2_000
gam = 0.01 ** (1.0 / T)
d_const = run_sgd(lambda t: 1.0, T=T)
d_decay = run_sgd(lambda t: gam**t, T=T)
print(f"고정 lr  : 마지막 200스텝 평균 거리 = {d_const[-200:].mean():.4f}")
print(f"지수 감쇠: 마지막 200스텝 평균 거리 = {d_decay[-200:].mean():.4f}")
print(f"개선 배수 = {d_const[-200:].mean() / d_decay[-200:].mean():.1f}x")
# 출력: 고정 lr  : 마지막 200스텝 평균 거리 = 0.0712
# 출력: 지수 감쇠: 마지막 200스텝 평균 거리 = 0.0175
# 출력: 개선 배수 = 4.1x

# %% [markdown]
# 초반 수렴 속도는 거의 같지만, 후반의 남은 오차가 한 자릿수 줄었다.
# 3DGS에서 이 차이가 곧 "픽셀 이하로 정렬된 선명한 렌더링" 대 "미세하게 흔들려 뭉개진 렌더링"이다.
#
# ## 5단계: MCMCStrategy — 감쇠하는 lr이 곧 담금질 온도
#
# `simple_trainer.py`는 MCMC 전략에 `lr=schedulers[0].get_last_lr()[0]`을 넘기고,
# `gsplat/strategy/mcmc.py`는 `noise_scale = lr * noise_lr` (기본 `noise_lr=5e5`)로
# `means`에 더할 탐색 노이즈 크기를 정한다. 학습률이 100배 줄면 노이즈도 100배 줄어든다.

# %%
NOISE_LR = 5e5
for s in (0.0, 0.25, 0.5, 0.75, 1.0):
    lr_s = 1.6e-4 * 100 ** (-s)
    print(f"진행률 {s:4.0%}  means lr={lr_s:.3e}  noise_scale={lr_s*NOISE_LR:.4f}")
# 출력: 진행률   0%  means lr=1.600e-04  noise_scale=80.0000
# 출력: 진행률  25%  means lr=5.060e-05  noise_scale=25.2982
# 출력: 진행률  50%  means lr=1.600e-05  noise_scale=8.0000
# 출력: 진행률  75%  means lr=5.060e-06  noise_scale=2.5298
# 출력: 진행률 100%  means lr=1.600e-06  noise_scale=0.8000

# %% [markdown]
# ## 6단계: 시각화
#
# 1. 진행률 축에서 겹치는 감쇠 곡선 (선형 y축)
# 2. 같은 곡선을 로그 y축으로 — $\ln\eta_t = \ln\eta_0 - \frac{t}{T}\ln 100$ 이므로 직선
# 3. 감쇠 스케줄 후보 비교 (지수 vs 선형 vs 고정)
# 4. 고정 lr vs 지수 감쇠의 수렴 궤적

# %%
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "① 진행률 축: T가 달라도 같은 곡선",
        "② 로그 y축에서는 직선 (기울기 = -ln100)",
        "③ 스케줄 비교 (T=2,000)",
        "④ 잡음 SGD 수렴: 고정 lr vs 지수 감쇠",
    ),
)

# 세 곡선이 정확히 겹치므로, 겹침이 보이도록 굵기/점선을 달리 준다
styles = {
    2_000: dict(color="#4C78A8", width=9, dash="solid"),
    7_000: dict(color="#F58518", width=5, dash="dash"),
    30_000: dict(color="#54A24B", width=2, dash="dot"),
}
for T_ in (2_000, 7_000, 30_000):
    t, lr = lr_curve(T_)
    for r, c in ((1, 1), (1, 2)):
        fig.add_trace(
            go.Scatter(
                x=t / T_,
                y=lr,
                name=f"T={T_:,}",
                line=styles[T_],
                legendgroup=str(T_),
                showlegend=(r == 1 and c == 1),
            ),
            row=r,
            col=c,
        )
fig.update_yaxes(type="log", row=1, col=2)

T_ = 2_000
t = np.arange(T_)
lr0 = 1.6e-4
fig.add_trace(
    go.Scatter(x=t, y=lr0 * (0.01 ** (1.0 / T_)) ** t, name="지수 (gsplat)",
               line=dict(color="#4C78A8", width=2)),
    row=2, col=1,
)
fig.add_trace(
    go.Scatter(x=t, y=lr0 * (1 - t / T_), name="선형 (0으로 죽음)",
               line=dict(color="#E45756", width=2, dash="dash")),
    row=2, col=1,
)
fig.add_trace(
    go.Scatter(x=t, y=np.full_like(t, lr0, dtype=float), name="고정",
               line=dict(color="#9D755D", width=2, dash="dot")),
    row=2, col=1,
)

fig.add_trace(
    go.Scatter(x=np.arange(T), y=d_const, name="고정 lr",
               line=dict(color="#E45756", width=1)),
    row=2, col=2,
)
fig.add_trace(
    go.Scatter(x=np.arange(T), y=d_decay, name="지수 감쇠 lr",
               line=dict(color="#4C78A8", width=1)),
    row=2, col=2,
)
fig.update_yaxes(type="log", row=2, col=2)

fig.update_xaxes(title_text="진행률 t/T", row=1, col=1)
fig.update_xaxes(title_text="진행률 t/T", row=1, col=2)
fig.update_xaxes(title_text="step", row=2, col=1)
fig.update_xaxes(title_text="step", row=2, col=2)
fig.update_yaxes(title_text="means lr", row=1, col=1)
fig.update_yaxes(title_text="means lr (log)", row=1, col=2)
fig.update_yaxes(title_text="means lr", row=2, col=1)
fig.update_yaxes(title_text="‖μ − μ*‖ (log)", row=2, col=2)
fig.update_layout(
    title="ExponentialLR(gamma = 0.01 ** (1/MAX_STEPS)) — means 학습률 감쇠",
    template="plotly_white",
    width=1100,
    height=760,
)

_show(fig)
fig.write_image("expy.png", scale=2)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 정리
#
# - `gamma = 0.01 ** (1.0 / MAX_STEPS)`는 "$T$ 스텝 뒤 1%"라는 조건 $\gamma^T=0.01$의 유일한 해다.
# - `ExponentialLR`은 `scheduler.step()`마다 `lr`에 $\gamma$를 곱한다 — **매 iteration**, `optimizer.step()` 뒤에.
# - 스케줄이 붙는 건 `means`(+ 옵션인 pose / post-processing)뿐이고, 나머지 파라미터는 고정 lr이다.
# - 진행률 $t/T$ 기준으로는 항상 $100^{-t/T}$ 라 총 스텝 수를 바꿔도 스케줄을 다시 튜닝할 필요가 없다.
# - MCMC 전략에서는 이 lr이 그대로 탐색 노이즈 크기가 되어 담금질(annealing) 효과를 낸다.
