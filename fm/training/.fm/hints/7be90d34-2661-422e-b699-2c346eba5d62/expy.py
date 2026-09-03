# %% [markdown]
# # PSNR을 MSE로부터 계산하기
#
# 핵심 한 줄:
#
# $$\text{PSNR} = 10\log_{10}\!\left(\frac{\text{MAX}^2}{\text{MSE}}\right)
#   \;\overset{\text{MAX}=1}{=}\; -10\log_{10}(\text{MSE})$$
#
# gsplat 학습 워크스루의 `eval_psnr()`이 하는 일:
#
# ```python
# mse = F.mse_loss(render.clamp(0, 1), gt)
# psnrs.append(-10.0 * math.log10(mse.item()))
# ```
#
# 이 스크립트는 gsplat을 import하지 않고 numpy/torch만으로 같은 계산을 재현한다.

# %%
# 필요 패키지: torch, numpy, plotly, kaleido
import math

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


torch.manual_seed(0)
print("torch", torch.__version__)
# 출력: torch 2.9.1+cu128

# %% [markdown]
# ## 1. MSE와 PSNR을 직접 구현해 보기
#
# MSE는 "차이의 제곱의 평균"이다(분산과 같은 모양, 기준만 GT):
#
# $$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(x_i - y_i)^2,\qquad N = H\cdot W\cdot C$$

# %%
def mse_manual(x: torch.Tensor, y: torch.Tensor) -> float:
    """F.mse_loss와 동일한 계산을 손으로."""
    return float(((x - y) ** 2).sum() / x.numel())


def psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    """PSNR = 10*log10(MAX^2 / MSE).  MAX=1이면 -10*log10(mse)."""
    return 10.0 * math.log10(data_range**2 / mse)


gt = torch.rand(1, 100, 120, 3)                 # 정답 (이미 [0,1])
render = (gt + 0.05 * torch.randn_like(gt))     # 렌더 결과 (약간의 오차)

m_torch = F.mse_loss(render.clamp(0, 1), gt).item()
m_manual = mse_manual(render.clamp(0, 1), gt)
print(f"F.mse_loss = {m_torch:.8f}")
print(f"수동 계산   = {m_manual:.8f}")
print(f"-10*log10(mse) = {-10.0 * math.log10(m_torch):.4f} dB")
print(f"10*log10(1/mse) = {psnr_from_mse(m_torch):.4f} dB   <- 같은 값")
# 출력: F.mse_loss = 0.00235927
# 출력: 수동 계산   = 0.00235927
# 출력: -10*log10(mse) = 26.2722 dB
# 출력: 10*log10(1/mse) = 26.2722 dB   <- 같은 값

# %% [markdown]
# ## 2. `clamp(0, 1)`이 왜 필요한가
#
# 3DGS 렌더 출력은 알파 합성 결과라 $[0,1]$을 벗어날 수 있다. 화면에는 어차피
# 잘려서 보이므로 지표도 **보이는 값** 기준으로 재야 한다.
#
# $$\text{clamp}(x) = \min(\max(x,0),\,1)$$

# %%
# 일부러 범위를 벗어나는 렌더를 만든다 (밝은 영역이 1을 넘어감)
render_over = gt * 1.15 + 0.03
frac_out = float(((render_over < 0) | (render_over > 1)).float().mean())

mse_raw = F.mse_loss(render_over, gt).item()
mse_clamped = F.mse_loss(render_over.clamp(0, 1), gt).item()

print(f"범위 밖 픽셀 비율 : {frac_out * 100:.1f}%")
print(f"clamp 없이 : MSE={mse_raw:.6f} -> PSNR={-10 * math.log10(mse_raw):6.2f} dB")
print(f"clamp 적용 : MSE={mse_clamped:.6f} -> PSNR={-10 * math.log10(mse_clamped):6.2f} dB")
# 출력: 범위 밖 픽셀 비율 : 15.7%
# 출력: clamp 없이 : MSE=0.012948 -> PSNR= 18.88 dB
# 출력: clamp 적용 : MSE=0.009775 -> PSNR= 20.10 dB
# 출력: (clamp 하나로 1.2 dB 차이 — 눈에 보이지도 않는 범위 밖 오차를 빼기 때문)

# %% [markdown]
# ## 3. dB 눈금 감각: 10배 규칙과 3 dB 규칙
#
# * MSE가 $10$배 줄면 PSNR은 정확히 $+10$ dB
# * MSE가 절반이 되면 $10\log_{10}2 \approx +3.01$ dB

# %%
for mse in [1e-1, 1e-2, 1e-3, 1e-4]:
    rmse = math.sqrt(mse)
    print(f"MSE={mse:.0e} -> PSNR={-10 * math.log10(mse):5.1f} dB, "
          f"RMSE={rmse:.4f} ([0,255] 환산 {rmse * 255:5.1f})")

print()
print(f"MSE 절반일 때 이득: {10 * math.log10(2):.3f} dB")
print(f"27 dB의 MSE = {10 ** (-27 / 10):.6f}, RMSE = {math.sqrt(10 ** (-27 / 10)):.4f}")
# 출력: MSE=1e-01 -> PSNR= 10.0 dB, RMSE=0.3162 ([0,255] 환산  80.6)
# 출력: MSE=1e-02 -> PSNR= 20.0 dB, RMSE=0.1000 ([0,255] 환산  25.5)
# 출력: MSE=1e-03 -> PSNR= 30.0 dB, RMSE=0.0316 ([0,255] 환산   8.1)
# 출력: MSE=1e-04 -> PSNR= 40.0 dB, RMSE=0.0100 ([0,255] 환산   2.6)
# 출력:
# 출력: MSE 절반일 때 이득: 3.010 dB
# 출력: 27 dB의 MSE = 0.001995, RMSE = 0.0447

# %% [markdown]
# ## 4. 스케일을 섞으면 안 된다
#
# $[0,255]$ 스케일에서 MSE를 쟀다면 $\text{MAX}=255$를 써야 한다.
# $\text{MAX}$와 MSE의 스케일이 맞기만 하면 결과는 동일하다.

# %%
gt255, r255 = gt * 255.0, render.clamp(0, 1) * 255.0
mse255 = F.mse_loss(r255, gt255).item()

print(f"MSE(0~1)   = {m_torch:.6f} -> PSNR = {psnr_from_mse(m_torch, 1.0):.4f} dB")
print(f"MSE(0~255) = {mse255:.4f} -> PSNR = {psnr_from_mse(mse255, 255.0):.4f} dB")
print(f"틀린 조합 (255 스케일 MSE에 MAX=1) = {psnr_from_mse(mse255, 1.0):.4f} dB  <- 엉터리")
# 출력: MSE(0~1)   = 0.002359 -> PSNR = 26.2722 dB
# 출력: MSE(0~255) = 153.4118 -> PSNR = 26.2722 dB
# 출력: 틀린 조합 (255 스케일 MSE에 MAX=1) = -21.8586 dB  <- 엉터리

# %% [markdown]
# ## 5. `eval_psnr()` 재현 — 이미지별 PSNR 평균
#
# 노트북은 이미지마다 PSNR을 구한 뒤 그 값들을 평균한다.
# 로그는 비선형이므로 "PSNR의 평균" $\ne$ "평균 MSE의 PSNR" (옌센 부등식).

# %%
@torch.no_grad()
def eval_psnr(renders, gts):
    """워크스루의 eval_psnr()과 동일한 구조."""
    psnrs = []
    for r, g in zip(renders, gts):
        mse = F.mse_loss(r.clamp(0, 1), g)
        psnrs.append(-10.0 * math.log10(mse.item()))
    return float(np.mean(psnrs)), psnrs


gts = [torch.rand(1, 64, 64, 3) for _ in range(5)]
# 이미지마다 오차 크기를 다르게 (어려운 뷰 / 쉬운 뷰)
sigmas = [0.02, 0.03, 0.05, 0.09, 0.15]
renders = [g + s * torch.randn_like(g) for g, s in zip(gts, sigmas)]

mean_psnr, per_image = eval_psnr(renders, gts)
mses = [F.mse_loss(r.clamp(0, 1), g).item() for r, g in zip(renders, gts)]
psnr_of_mean_mse = -10.0 * math.log10(float(np.mean(mses)))

for i, (s, mse, p) in enumerate(zip(sigmas, mses, per_image)):
    print(f"  img{i} sigma={s:.2f}  MSE={mse:.6f}  PSNR={p:6.2f} dB")
print(f"PSNR 평균          = {mean_psnr:.3f} dB   <- eval_psnr()이 돌려주는 값")
print(f"평균 MSE의 PSNR    = {psnr_of_mean_mse:.3f} dB   <- 다른 값 (옌센 부등식)")
# 출력:   img0 sigma=0.02  MSE=0.000381  PSNR= 34.19 dB
# 출력:   img1 sigma=0.03  MSE=0.000882  PSNR= 30.55 dB
# 출력:   img2 sigma=0.05  MSE=0.002394  PSNR= 26.21 dB
# 출력:   img3 sigma=0.09  MSE=0.007253  PSNR= 21.39 dB
# 출력:   img4 sigma=0.15  MSE=0.019006  PSNR= 17.21 dB
# 출력: PSNR 평균          = 25.910 dB   <- eval_psnr()이 돌려주는 값
# 출력: 평균 MSE의 PSNR    = 22.231 dB   <- 다른 값 (옌센 부등식)

# %% [markdown]
# ## 6. 시각화
#
# 왼쪽: $\text{PSNR} = -10\log_{10}(\text{MSE})$ 곡선 (MSE는 로그축).
# 오른쪽: 학습이 진행되며 MSE가 지수적으로 줄 때 PSNR은 거의 선형으로 오른다 —
# 그래서 진척도를 읽기에는 PSNR이 편하다.

# %%
mse_grid = np.logspace(-5, -0.5, 200)
psnr_grid = -10.0 * np.log10(mse_grid)

# 가짜 학습 곡선: MSE가 지수적으로 감소
steps = np.arange(0, 3001, 50)
mse_curve = 0.05 * np.exp(-steps / 900.0) + 0.0015
psnr_curve = -10.0 * np.log10(mse_curve)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("PSNR = -10·log10(MSE)", "학습 곡선: MSE(로그) vs PSNR"),
    specs=[[{}, {"secondary_y": True}]],
)

fig.add_trace(
    go.Scatter(x=mse_grid, y=psnr_grid, mode="lines",
               line=dict(color="#2563eb", width=3), name="PSNR(MSE)"),
    row=1, col=1,
)
marks = [1e-1, 1e-2, 1e-3, 1e-4]
fig.add_trace(
    go.Scatter(x=marks, y=[-10 * np.log10(m) for m in marks], mode="markers+text",
               marker=dict(color="#dc2626", size=10),
               text=[f"{-10 * np.log10(m):.0f} dB" for m in marks],
               textposition="top right", name="10배 = 10 dB"),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=[m_torch], y=[-10 * np.log10(m_torch)], mode="markers+text",
               marker=dict(color="#059669", size=12, symbol="star"),
               text=["예제 렌더"], textposition="bottom left", name="예제 렌더"),
    row=1, col=1,
)

fig.add_trace(
    go.Scatter(x=steps, y=mse_curve, mode="lines", name="MSE",
               line=dict(color="#dc2626", width=2)),
    row=1, col=2, secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=steps, y=psnr_curve, mode="lines", name="PSNR (dB)",
               line=dict(color="#2563eb", width=3)),
    row=1, col=2, secondary_y=True,
)

fig.update_xaxes(type="log", title_text="MSE (로그축)", row=1, col=1)
fig.update_yaxes(title_text="PSNR (dB)", row=1, col=1)
fig.update_xaxes(title_text="step", row=1, col=2)
fig.update_yaxes(type="log", title_text="MSE", row=1, col=2, secondary_y=False)
fig.update_yaxes(title_text="PSNR (dB)", row=1, col=2, secondary_y=True)
fig.update_layout(
    title="MSE → PSNR (MAX=1, 즉 [0,1] 정규화 기준)",
    template="plotly_white", width=1100, height=460,
    legend=dict(orientation="h", y=-0.18),
)

_show(fig)
fig.write_image("expy.png", scale=2)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 정리
#
# $$\underbrace{\text{clamp}(x,0,1)}_{\text{보이는 값으로}}
#   \;\to\; \underbrace{\tfrac{1}{N}\sum (x_i-y_i)^2}_{\texttt{F.mse\_loss}}
#   \;\to\; \underbrace{-10\log_{10}(\text{MSE})}_{\text{dB}}$$
#
# * $\text{MAX}=1$ 덕분에 $10\log_{10}(\text{MAX}^2/\text{MSE})$가 `-10*log10(mse)`로 줄어든다.
# * MSE $\times \tfrac{1}{10}$ = PSNR $+10$ dB, MSE 절반 = $+3$ dB.
# * PSNR은 MSE의 단조 감소 변환이라 최적화 목표로는 동치지만, 3DGS는 학습에 L1+SSIM을 쓰고
#   PSNR은 평가 지표로만 쓴다(`simple_trainer.py`는 SSIM/LPIPS도 함께 기록).
