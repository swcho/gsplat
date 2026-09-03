# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 같은 계수, 다른 `sh_degree` — 방향 지도는 어떻게 변하는가
#
# 질문: 같은 Gaussian, 같은 SH 계수 `[16,3]`를 두고 **평가 차수만** 0 → 3으로 바꾸면 방향별 색 지도가 어떻게 달라지는가?
#
# 3DGS/gsplat의 SH 평가는 활성화 차수 $L$까지의 $K=(L+1)^2$개 계수만 쓴다.
#
# $$
# \mathbf c_L(\mathbf d) = \sum_{k=0}^{(L+1)^2-1} \mathbf c_k\,Y_k(\mathbf d), \qquad
# \text{최종색} = \max\big(\mathbf c_L(\mathbf d) + 0.5,\ 0\big)
# $$
#
# 그러므로 $L$을 바꾸는 것은 **계수를 바꾸는 게 아니라 급수를 어디서 끊을지**를 바꾸는 것이다.
# - $L=0$: $\mathbf c_0 Y_0^0$ 은 상수 → 방향과 무관한 **단색**.
# - $L$이 오르면 $\ell=L$ 블록(기저 $2L+1$개)이 통째로 더해져 더 잘게 진동하는 세부가 붙는다.
# - $K$ 이후의 계수는 값이 무엇이든 결과에 영향이 없다.
#
# 아래에서 이를 숫자와 그림으로 확인한다. (`sh_bases`, `sphere_grid`는 sh_walkthrough.py의 것을 그대로 사용)

# %%
# 필요 패키지: torch, numpy, plotly, kaleido(png 저장)
import math
import os

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


DEVICE = "cpu"
torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

C0 = 0.28209479177387814  # Y₀⁰ = 1/(2√π)
LM = [(l, m) for l in range(4) for m in range(-l, l + 1)]


def sh_bases(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """단위 방향 dirs[..., 3] → SH 기저값 [..., (degree+1)²]. gsplat _eval_sh_bases_fast와 같은 값·순서."""
    x, y, z = dirs.unbind(-1)
    out = [torch.full_like(x, C0)]
    if degree >= 1:
        c = 0.4886025119029199
        out += [-c * y, c * z, -c * x]
    if degree >= 2:
        c1, c2, c3 = 1.0925484305920792, 0.31539156525252005, 0.5462742152960396
        out += [c1 * x * y, -c1 * y * z, c2 * (3 * z * z - 1), -c1 * x * z, c3 * (x * x - y * y)]
    if degree >= 3:
        c1, c2, c3, c4, c5 = 0.5900435899266435, 2.890611442640554, 0.4570457994644658, 0.3731763325901154, 1.445305721320277
        out += [-c1 * y * (3 * x * x - y * y), c2 * x * y * z, -c3 * y * (5 * z * z - 1), c4 * z * (5 * z * z - 3),
                -c3 * x * (5 * z * z - 1), c5 * z * (x * x - y * y), -c1 * x * (x * x - 3 * y * y)]
    return torch.stack(out, dim=-1)


def sphere_grid(n_theta: int = 128, n_phi: int = 256, device=DEVICE):
    """구면을 (θ, φ) 격자로 덮는 방향 d[nθ,nφ,3]와 적분 가중치 w = sinθ dθ dφ (Σw ≈ 4π)."""
    theta = (torch.arange(n_theta, device=device) + 0.5) * math.pi / n_theta
    phi = (torch.arange(n_phi, device=device) + 0.5) * 2 * math.pi / n_phi - math.pi
    th, ph = torch.meshgrid(theta, phi, indexing="ij")
    d = torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], dim=-1)
    w = th.sin() * (math.pi / n_theta) * (2 * math.pi / n_phi)
    return d, w


dirs, w = sphere_grid()
B = sh_bases(dirs, 3)                       # [128, 256, 16]  전 구면 격자에서의 기저값
print("기저 텐서:", tuple(B.shape), " Σw =", round(w.sum().item(), 5), "(≈ 4π =", round(4 * math.pi, 5), ")")
# 출력: 기저 텐서: (128, 256, 16)  Σw = 12.56669 (≈ 4π = 12.56637 )

# %% [markdown]
# ## 1. Gaussian 하나의 계수 `[16,3]` 만들기
#
# sh_walkthrough.py 4.1절과 같은 방식: DC는 기본색 $(0.7, 0.5, 0.3)$에서 $(\mathbf{rgb}-0.5)/C_0$ 로, 나머지 15개는 $\mathcal N(0, 0.25^2)$.
# 이 계수 텐서는 이후 **한 번도 바꾸지 않는다**.

# %%
coeffs = torch.zeros(16, 3, device=DEVICE)
coeffs[0] = (torch.tensor([0.7, 0.5, 0.3], device=DEVICE) - 0.5) / C0
coeffs[1:] = torch.randn(15, 3, device=DEVICE) * 0.25
print("DC 계수 c₀ =", coeffs[0].tolist())
print("c₀·Y₀⁰ + 0.5 =", (coeffs[0] * C0 + 0.5).tolist(), " ← L=0 단색")
# 출력: DC 계수 c₀ = [0.708981454372406, 0.0, -0.708981454372406]
# 출력: c₀·Y₀⁰ + 0.5 = [0.699999988079071, 0.5, 0.30000001192092896]  ← L=0 단색


def eval_deg(c: torch.Tensor, L: int) -> torch.Tensor:
    """전 구면 격자에서 차수 L까지만 써서 평가한 원색(활성화 전) [nθ, nφ, 3]."""
    K = (L + 1) ** 2
    return torch.einsum("abk,kc->abc", B[..., :K], c[:K])


raw = [eval_deg(coeffs, L) for L in range(4)]                   # 활성화 전
img = [torch.clamp_min(r + 0.5, 0.0) for r in raw]              # gsplat rasterization()이 하는 +0.5, clamp_min(0)

# %% [markdown]
# ## 2. 차수별 통계 — 방향 의존 정도와 이전 차수 대비 변화량
#
# 구면 가중 표준편차(각 채널, $w=\sin\theta\,d\theta\,d\varphi$ 가중)로 "방향에 따라 색이 얼마나 달라지는지"를 잰다.
#
# $$
# \sigma_L^2 = \frac{1}{4\pi}\int_{S^2}\big(\mathbf c_L(\mathbf d) - \bar{\mathbf c}\big)^2 d\Omega
# = \frac{1}{4\pi}\sum_{k=1}^{(L+1)^2-1}\mathbf c_k^2 \quad(\text{정규직교성, Parseval})
# $$
#
# 기대: $L=0$이면 $\sigma_0 = 0$ (단색). $L$이 오르면 $\sigma_L$은 **단조 증가**하고, 그 증가분은 새로 켜진 블록 계수의 제곱합이다.

# %%
def wmean(x):                         # 구면 가중 평균 [..., 3] → [3]
    return (x * w[..., None]).sum((0, 1)) / w.sum()


def wstd(x):
    mu = wmean(x)
    return ((((x - mu) ** 2) * w[..., None]).sum((0, 1)) / w.sum()).sqrt()


def wrms(x):
    return (((x ** 2) * w[..., None]).sum((0, 1)) / w.sum()).sqrt()


print("L  K   σ(방향 의존 정도, RGB)          Parseval 예측       RMS(L − (L−1), RGB)      clamp 걸린 픽셀 비율")
for L in range(4):
    K = (L + 1) ** 2
    sig = wstd(raw[L])
    pred = (coeffs[1:K] ** 2).sum(0).div(4 * math.pi).sqrt()
    diff = wrms(raw[L] - raw[L - 1]) if L > 0 else torch.zeros(3)
    clamped = ((raw[L] + 0.5) < 0).float().mul(w[..., None]).sum((0, 1)) / w.sum()
    print(f"{L}  {K:2d}  {sig.numpy().round(4)}   {pred.numpy().round(4)}   {diff.numpy().round(4)}   "
          f"{(100 * clamped).numpy().round(2)} %")
# 출력: L  K   σ(방향 의존 정도, RGB)          Parseval 예측       RMS(L − (L−1), RGB)      clamp 걸린 픽셀 비율
# 출력: 0   1  [0. 0. 0.]   [0. 0. 0.]   [0. 0. 0.]   [0. 0. 0.] %
# 출력: 1   4  [0.088  0.1801 0.0567]   [0.088  0.1801 0.0567]   [0.088  0.1801 0.0567]   [0. 0. 0.] %
# 출력: 2   9  [0.1723 0.2313 0.2075]   [0.1723 0.2313 0.2075]   [0.1481 0.1451 0.1996]   [0.   1.25 7.5 ] %
# 출력: 3  16  [0.259  0.3617 0.2734]   [0.259  0.3617 0.2734]   [0.1934 0.2781 0.178 ]   [ 0.    9.88 15.73] %

# %% [markdown]
# 읽는 법
# - $L=0$: 표준편차 0 → 어느 방향에서 봐도 같은 색 $(0.7, 0.5, 0.3)$.
# - $L$이 오를수록 $\sigma_L$이 커지고(방향별 세부 증가), 측정값이 Parseval 예측 $\sqrt{\sum_{k<K}\mathbf c_k^2/4\pi}$ 과 정확히 일치한다.
#   즉 "세부가 붙는다"는 말은 **켜진 블록 계수의 에너지가 지도에 그대로 더해진다**는 뜻이다.
# - RMS 변화량은 각 단계마다 0이 아니다 — **계수를 하나도 바꾸지 않았는데** 결과가 달라진다. 오직 급수 절단 위치가 달라졌기 때문.
# - clamp 비율: $L\le1$에서는 진폭이 작아 어느 채널도 $\mathbf c+0.5<0$ 이 되지 않는다. 차수가 오르며 진폭이 커지면
#   DC가 낮은 채널부터(파랑 $0.3$ → 초록 $0.5$) 음수로 떨어지는 방향이 생기고 그 비율이 늘어난다(파랑 0 → 7.5 → 15.7 %).
#   빨강은 DC가 $0.7$로 높아 $L=3$까지도 clamp에 걸리지 않는다. 일반적으로는 상쇄 때문에 단조라는 보장은 없다.

# %% [markdown]
# ## 3. 증분 지도 = 그 차수 블록의 기여
#
# $L$차 결과에서 $(L-1)$차 결과를 빼면 정확히 $\ell=L$ 블록의 기저 $2L+1$개 × 해당 계수의 합이 나와야 한다.
#
# $$
# \mathbf c_L(\mathbf d) - \mathbf c_{L-1}(\mathbf d) = \sum_{k=L^2}^{(L+1)^2-1} \mathbf c_k\,Y_k(\mathbf d)
# $$

# %%
for L in range(1, 4):
    k0, k1 = L ** 2, (L + 1) ** 2
    block = torch.einsum("abk,kc->abc", B[..., k0:k1], coeffs[k0:k1])
    err = (raw[L] - raw[L - 1] - block).abs().max().item()
    print(f"ℓ={L} 블록 (k={k0}..{k1 - 1}, 기저 {k1 - k0}개): |증분 − 블록 기여| 최대 = {err:.2e}")
# 출력: ℓ=1 블록 (k=1..3, 기저 3개): |증분 − 블록 기여| 최대 = 3.73e-08
# 출력: ℓ=2 블록 (k=4..8, 기저 5개): |증분 − 블록 기여| 최대 = 1.04e-07
# 출력: ℓ=3 블록 (k=9..15, 기저 7개): |증분 − 블록 기여| 최대 = 1.49e-07

# %% [markdown]
# ## 4. 활성화 차수 이후의 계수는 무시된다
#
# $L=1$로 평가할 때 $k\ge4$ 계수를 0으로 두든, 완전히 다른 난수로 바꾸든 결과는 **비트 단위로 같다**.
# 반대로 $L=3$으로 평가하면 그 차이가 그대로 드러난다.

# %%
c_zero, c_other = coeffs.clone(), coeffs.clone()
c_zero[4:] = 0.0
c_other[4:] = torch.randn(12, 3) * 5.0                    # 고차 계수를 크게 바꿔도
print("L=1:  |eval(coeffs) − eval(c_zero)|  =", (eval_deg(coeffs, 1) - eval_deg(c_zero, 1)).abs().max().item())
print("L=1:  |eval(coeffs) − eval(c_other)| =", (eval_deg(coeffs, 1) - eval_deg(c_other, 1)).abs().max().item())
print("L=3:  |eval(coeffs) − eval(c_other)| =", round((eval_deg(coeffs, 3) - eval_deg(c_other, 3)).abs().max().item(), 4))
# 출력: L=1:  |eval(coeffs) − eval(c_zero)|  = 0.0
# 출력: L=1:  |eval(coeffs) − eval(c_other)| = 0.0
# 출력: L=3:  |eval(coeffs) − eval(c_other)| = 14.5949

# %% [markdown]
# ## 5. 시각화 — 위: 누적 결과 $\max(\mathbf c_L+0.5,0)$, 아래: 증분 $\mathbf c_L-\mathbf c_{L-1}$
#
# 등장방형 지도(가로 = 방위각 $\varphi$, 세로 = 극각 $\theta$). 증분은 0을 회색(0.5)으로 두고 ±0.5 범위를 색으로 표시한다.
# 아래 행이 각각 $\ell=1,2,3$ 기저의 "무늬"를 그대로 닮는 것을 볼 수 있다 — 위 3절의 등식이 눈으로 확인된다.

# %%
def to_u8(x):
    return (x.clamp(0, 1) * 255).to(torch.uint8).numpy()


titles = [f"sh_degree={L} (계수 {(L + 1) ** 2}개 사용)" for L in range(4)] + \
         ["증분 없음 (L=0 자체)"] + [f"증분 ℓ={L} 블록 (L={L} − L={L - 1})" for L in range(1, 4)]
fig = make_subplots(rows=2, cols=4, subplot_titles=titles, horizontal_spacing=0.03, vertical_spacing=0.12)
for L in range(4):
    fig.add_trace(go.Image(z=to_u8(img[L]), x0=-180 + 360 / 512, dx=360 / 256, y0=180 / 256, dy=180 / 128), row=1, col=L + 1)
    inc = raw[L] - raw[L - 1] if L > 0 else torch.zeros_like(raw[0])
    fig.add_trace(go.Image(z=to_u8(inc + 0.5), x0=-180 + 360 / 512, dx=360 / 256, y0=180 / 256, dy=180 / 128), row=2, col=L + 1)
for r in (1, 2):
    for c in range(1, 5):
        fig.update_xaxes(title_text="φ (deg)" if r == 2 else None, row=r, col=c)
        fig.update_yaxes(title_text="θ (deg)" if c == 1 else None, autorange="reversed", row=r, col=c)
fig.update_layout(title="같은 Gaussian, 같은 계수 — 활성화 차수만 바꿔 평가한 방향 지도 (위: 누적, 아래: 증분)",
                  width=1500, height=640, margin=dict(l=50, r=20, t=90, b=50))
_show(fig)
fig.write_image(os.path.join(HERE, "expy.png"), scale=1)
print("저장:", os.path.join(HERE, "expy.png"))
# 출력: 저장: <hint dir>/expy.png

# %% [markdown]
# ## 6. 3DGS의 `sh_degree_interval` 과의 연결
#
# 학습 코드(simple_trainer.py)는
#
# ```python
# sh_degree_to_use = min(step // sh_degree_interval, sh_degree)   # 기본 interval=1000, sh_degree=3
# ```
#
# 로 렌더링에 쓰는 차수를 스텝에 따라 0 → 1 → 2 → 3으로 올린다. 이 노트북에서 한 일이 정확히 그것이다:
# **파라미터 `[16,3]`는 처음부터 다 존재하지만**, 초기 1000스텝은 DC만 쓰여 시점 무관 기본색이 먼저 잡히고,
# 이후 1000스텝마다 한 블록씩 켜지며 방향별 세부가 추가된다.
#
# - 꺼진 블록의 계수는 forward에 참여하지 않으므로 gradient도 0 → 그동안은 초기값(0)에 머문다.
#   그래서 실제 학습에서는 4절처럼 "무시된 계수가 크게 다른" 상황은 생기지 않고, 차수가 켜질 때 증분 지도가 0에서 서서히 자란다.
# - 이렇게 저차부터 안정화하는 것은 1.3절의 "저차 = 부드러운 근사, 고차 = 세부"를 학습 순서에 반영한 커리큘럼이다.
#   기본색이 잡히기 전에 고차가 자유로우면, 적은 시점에서의 오차를 고차 진동으로 과적합하기 쉽다(4.2절 표).
#
# **요약**: 차수 0에서는 방향 무관 단색, 차수를 올릴수록 해당 블록의 기저 무늬가 더해져 방향별 색 변화가 생긴다.
# 결과가 달라지는 이유는 계수가 아니라 **급수를 끊는 위치**($K=(L+1)^2$) 가 바뀌기 때문이며, $K$ 이후 계수는 완전히 무시된다.
