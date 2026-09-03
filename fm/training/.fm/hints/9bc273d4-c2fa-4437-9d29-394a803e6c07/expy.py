# %% [markdown]
# # 손실의 SSIM 항은 무슨 일을 하는가
#
# 3DGS 손실은 다음과 같다 (`examples/simple_trainer.py:961`).
#
# $$\mathcal{L} = (1-\lambda)\,\mathcal{L}_{L1} + \lambda\,(1-\mathrm{SSIM}), \qquad \lambda = 0.2$$
#
# 이 스크립트는 gsplat을 import하지 않고 numpy/torch로 SSIM을 직접 구현해
# **왜 L1만으로는 안 되고, SSIM 항이 무엇을 강제하는지**를 단계적으로 확인한다.
#
# 1. L1은 "평평하게 뭉개기"에 상을 준다 (숫자 2줄로 확인)
# 2. 11×11 가우시안 창($\sigma=1.5$)의 정체 — `gsplat/losses.py:_gaussian_kernel_1d`
# 3. 창 통계 $\mu,\sigma^2,\sigma_{12}$로 SSIM 지도 만들기 — `torch_ssim_loss`와 동일 식
# 4. blur / 노이즈 / 위상반전 예측을 L1과 SSIM이 어떻게 다르게 채점하는가
# 5. 2D Gaussian 몇 개로 텍스처 이미지 맞추기: L1 단독 vs $0.8L_1+0.2(1-\mathrm{SSIM})$

# %%
# 필요 패키지: numpy, torch, plotly, kaleido  (gsplat은 import하지 않는다)
import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots

torch.manual_seed(0)
np.random.seed(0)
DEV = "cpu"


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


print(f"torch {torch.__version__}")
# 출력: torch 2.9.1+cu128

# %% [markdown]
# ## 1단계: L1은 '평평하게 뭉개기'를 선호한다
#
# 정답이 $0.4, 0.6, 0.4, 0.6, \dots$ 로 오르내리는 작은 무늬라고 하자.
# - 예측 A = 전부 $0.5$ (무늬를 버리고 평균으로 도망)
# - 예측 B = $0.6, 0.4, 0.6, \dots$ (무늬는 살렸지만 위상이 반대)
#
# 오차의 절댓값 기댓값 $E[|X-c|]$을 최소화하는 $c$는 중앙값이므로,
# L1 입장에서는 A가 B보다 정확히 두 배 유리하다.

# %%
gt_1d = np.array([0.4, 0.6] * 8)
pred_flat = np.full_like(gt_1d, 0.5)  # 평평하게 뭉갠 예측
pred_phase = np.array([0.6, 0.4] * 8)  # 무늬는 있지만 위상 반대

for name, p in [("A 평평(뭉갬)", pred_flat), ("B 위상반전", pred_phase)]:
    print(
        f"{name}: L1={np.abs(p - gt_1d).mean():.4f}  "
        f"std={p.std():.4f} (GT std={gt_1d.std():.4f})  "
        f"corr={0.0 if p.std() == 0 else np.corrcoef(p, gt_1d)[0, 1]:+.1f}"
    )
# 출력: A 평평(뭉갬): L1=0.1000  std=0.0000 (GT std=0.1000)  corr=+0.0
# 출력: B 위상반전: L1=0.2000  std=0.1000 (GT std=0.1000)  corr=-1.0
# → L1만 보면 무늬를 없앤 A가 2배 좋다. "국소 대비(std)가 사라졌다"는 사실을 L1은 보지 못한다.

# %% [markdown]
# ## 2단계: 11×11 가우시안 창 ($\sigma = 1.5$)
#
# `gsplat/losses.py`의 `_gaussian_kernel_1d(window_size, sigma=1.5)` →
# `create_ssim_window`가 만드는 창이다.
#
# $$w_{1\mathrm{D}}(k)\propto\exp\!\left(-\frac{(k-5)^2}{2\cdot1.5^2}\right),\quad k=0..10,
# \qquad w_{2\mathrm{D}}(i,j)=w_{1\mathrm{D}}(i)\,w_{1\mathrm{D}}(j)$$
#
# - 크기 11: $\sigma=1.5$면 $\pm5$ 픽셀이 $\pm3.3\sigma$ → 가중치가 거의 다 담긴다.
# - 홀수: 중심 픽셀이 하나로 정해지고 `padding = 11 // 2 = 5`로 출력 크기가 유지된다.
# - 1D 두 개의 곱(separable) → 계산이 가볍고 회전 대칭이라 무늬 방향에 편향이 없다.


# %%
def gaussian_1d(window_size=11, sigma=1.5):
    x = torch.arange(window_size, dtype=torch.float32)
    g = torch.exp(-((x - window_size // 2) ** 2) / (2 * sigma**2))
    return g / g.sum()


def ssim_window(window_size=11, channel=3):
    w1 = gaussian_1d(window_size).unsqueeze(1)
    w2 = (w1 @ w1.t()).unsqueeze(0).unsqueeze(0)
    return w2.expand(channel, 1, window_size, window_size).contiguous()


w1d = gaussian_1d()
WIN = ssim_window(11, 1)
print("1D 가중치:", np.round(w1d.numpy(), 4))
print(f"합={w1d.sum():.6f}  중심가중치={w1d[5]:.4f}  끝가중치={w1d[0]:.6f}")
print(f"2D 창 합={WIN.sum():.6f}  (11x11 균등창이면 중심가중치는 {1/121:.4f})")
# 출력: 1D 가중치: [0.001  0.0076 0.036  0.1094 0.213  0.266  0.213  0.1094 0.036  0.0076
# 출력:  0.001 ]
# 출력: 합=1.000000  중심가중치=0.2660  끝가중치=0.001028
# 출력: 2D 창 합=1.000000  (11x11 균등창이면 중심가중치는 0.0083)
# → 중심 ±2픽셀에 가중치의 91%가 몰려 있고 양 끝은 0.001. 창 하나가 사실상 좁은 동네만
#    보므로 SSIM이 재는 것은 "국소" 대비/구조다. 균등창이라면 중심도 0.0083이었을 것.

# %% [markdown]
# ## 3단계: 창 통계로 SSIM 지도 만들기
#
# 각 픽셀 위치에서 창 안의 가중 통계를 구하는 일은 곧 합성곱이다.
#
# $$\mu_1 = w * x,\quad \sigma_1^2 = w*(x^2)-\mu_1^2,\quad \sigma_{12}=w*(xy)-\mu_1\mu_2$$
#
# $$\mathrm{SSIM}=\underbrace{\frac{2\mu_1\mu_2+C_1}{\mu_1^2+\mu_2^2+C_1}}_{\text{밝기}}\cdot
# \underbrace{\frac{2\sigma_{12}+C_2}{\sigma_1^2+\sigma_2^2+C_2}}_{\text{대비}\times\text{구조}},
# \qquad C_1=0.01^2,\ C_2=0.03^2$$
#
# 아래 `ssim_map`은 `gsplat/losses.py`의 `torch_ssim_loss`와 같은 식이고,
# `ssim_loss = 1 - ssim_map.mean()`이 `ssim_loss()`의 fallback 경로와 같다.

# %%
C1, C2 = 0.01**2, 0.03**2


def ssim_parts(img1, img2, window_size=11):
    """(밝기 인자, 대비·구조 인자) 지도를 따로 반환. img: (B,C,H,W), 값 [0,1]."""
    ch = img1.shape[1]
    win = ssim_window(window_size, ch).to(img1)
    pad = window_size // 2
    c = lambda t: F.conv2d(t, win, padding=pad, groups=ch)
    mu1, mu2 = c(img1), c(img2)
    s1 = c(img1 * img1) - mu1**2
    s2 = c(img2 * img2) - mu2**2
    s12 = c(img1 * img2) - mu1 * mu2
    lum = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)
    cs = (2 * s12 + C2) / (s1 + s2 + C2)
    return lum, cs


def ssim_map(img1, img2, window_size=11):
    lum, cs = ssim_parts(img1, img2, window_size)
    return lum * cs


def ssim_loss(img1, img2, window_size=11):
    return 1.0 - ssim_map(img1, img2, window_size).mean()


def l1_loss(a, b):
    return (a - b).abs().mean()


def total_loss(pred, gt, lam=0.2):
    return torch.lerp(l1_loss(pred, gt), ssim_loss(pred, gt), lam)


# 자기 자신과 비교하면 SSIM = 1 → loss 0 인지 확인
t = torch.rand(1, 3, 32, 32)
print(f"자기비교 ssim_loss = {ssim_loss(t, t).item():.3e}")
# 출력: 자기비교 ssim_loss = 0.000e+00

# %% [markdown]
# ### 대비 인자가 '평평함'을 어떻게 벌하는가
#
# 예측 창이 평평하면 $\sigma_1=0,\ \sigma_{12}=0$이므로 대비·구조 인자는
#
# $$\frac{C_2}{\sigma_2^2+C_2}$$
#
# 즉 정답의 국소 대비 $\sigma_2$가 클수록 0으로 떨어진다.
# 반대로 무늬 세기가 같고 방향까지 같아야($\rho=1$, $\sigma_1=\sigma_2$) 1점이 된다.

# %%
sigma_gt = np.linspace(0, 0.3, 7)
print("GT 국소표준편차 σ₂ → 평평한 예측(σ₁=0)의 대비·구조 인자")
for s in sigma_gt:
    print(f"  σ₂={s:.2f} → {C2 / (s**2 + C2):.4f}")
# 출력: GT 국소표준편차 σ₂ → 평평한 예측(σ₁=0)의 대비·구조 인자
# 출력:   σ₂=0.00 → 1.0000
# 출력:   σ₂=0.05 → 0.2647
# 출력:   σ₂=0.10 → 0.0826
# 출력:   σ₂=0.15 → 0.0385
# 출력:   σ₂=0.20 → 0.0220
# 출력:   σ₂=0.25 → 0.0142
# 출력:   σ₂=0.30 → 0.0099

# 상관계수 ρ와 대비비 r=σ₁/σ₂를 바꿔 본다 (σ₂=0.1 고정)
s2v = 0.1
print("\nr=σ₁/σ₂,  ρ  → 대비·구조 인자 (C2 무시 근사: 2ρr/(1+r²))")
for r in [0.0, 0.5, 1.0, 2.0]:
    row = []
    for rho in [1.0, 0.0, -1.0]:
        s1v = r * s2v
        val = (2 * rho * s1v * s2v + C2) / (s1v**2 + s2v**2 + C2)
        row.append(f"ρ={rho:+.0f}:{val:+.3f}")
    print(f"  r={r:.1f}  " + "  ".join(row))
# 출력:
# 출력: r=σ₁/σ₂,  ρ  → 대비·구조 인자 (C2 무시 근사: 2ρr/(1+r²))
# 출력:   r=0.0  ρ=+1:+0.083  ρ=+0:+0.083  ρ=-1:+0.083
# 출력:   r=0.5  ρ=+1:+0.813  ρ=+0:+0.067  ρ=-1:-0.679
# 출력:   r=1.0  ρ=+1:+1.000  ρ=+0:+0.043  ρ=-1:-0.914
# 출력:   r=2.0  ρ=+1:+0.804  ρ=+0:+0.018  ρ=-1:-0.768
# → 무늬가 없으면(r=0) 상관계수와 무관하게 저득점, 세기와 방향이 모두 맞을 때만 1에 가깝다.

# %% [markdown]
# ## 4단계: 이미지 3종 예측 채점 — blur vs 노이즈 vs 위상반전
#
# 텍스처가 있는 정답 이미지를 만들고 세 가지 '틀린 방식'을 비교한다.
# L1 점수와 SSIM 점수의 **순위가 뒤집히는지** 보는 것이 핵심이다.


# %%
def make_target(H=96, W=96):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = 0.35 + 0.25 * (xx / W)  # 부드러운 밝기 변화
    freq = 0.25 + 0.55 * (xx / W)  # 오른쪽으로 갈수록 촘촘한 무늬
    tex = 0.18 * np.sin(freq * yy) * np.sin(freq * xx)
    img = np.clip(base + tex, 0, 1)
    rgb = np.stack([img, img * 0.92 + 0.05, img * 0.8 + 0.12], 0)
    return torch.from_numpy(np.clip(rgb, 0, 1))[None].float()


def box_blur(img, k=9):
    ch = img.shape[1]
    ker = torch.ones(ch, 1, k, k) / (k * k)
    return F.conv2d(img, ker, padding=k // 2, groups=ch)


gt = make_target()
pred_blur = box_blur(gt, 9)  # 뭉갠 예측 (L1만 쓰면 도달하는 곳)
pred_noise = (gt + 0.06 * torch.randn_like(gt)).clamp(0, 1)  # 대비는 있지만 노이즈
pred_inv = (gt.mean() + (gt - gt.mean()) * -1).clamp(0, 1)  # 무늬 위상 반전

rows = []
for name, p in [("blur(9x9)", pred_blur), ("noise(σ=.06)", pred_noise), ("위상반전", pred_inv)]:
    rows.append((name, l1_loss(p, gt).item(), ssim_loss(p, gt).item(), total_loss(p, gt).item()))
    print(f"{name:14s} L1={rows[-1][1]:.4f}  1-SSIM={rows[-1][2]:.4f}  total={rows[-1][3]:.4f}")
# 출력: blur(9x9)      L1=0.0757  1-SSIM=0.7018  total=0.2009
# 출력: noise(σ=.06)   L1=0.0480  1-SSIM=0.2237  total=0.0831
# 출력: 위상반전           L1=0.1696  1-SSIM=1.5929  total=0.4543
# → blur의 L1은 노이즈의 1.6배(0.076 vs 0.048)인데 1-SSIM은 3.1배(0.702 vs 0.224)다.
#    즉 SSIM은 뭉개짐을 유독 크게 벌한다. 위상반전은 상관이 음수라 1-SSIM이 1을 넘는다.

# %%
# SSIM 지도를 보면 어디서 벌점이 나는지 알 수 있다 (오른쪽=고주파 텍스처 영역)
m_blur = ssim_map(pred_blur, gt)[0].mean(0)
lum_b, cs_b = ssim_parts(pred_blur, gt)
print(f"blur 예측의 SSIM 지도: 평균={m_blur.mean():.4f}, 최소={m_blur.min():.4f}")
print(f"  밝기 인자 평균={lum_b.mean():.4f}   대비·구조 인자 평균={cs_b.mean():.4f}")
left, right = m_blur[:, :32].mean(), m_blur[:, -32:].mean()
print(f"  왼쪽(저주파) SSIM={left:.4f}  vs  오른쪽(고주파) SSIM={right:.4f}")
# 출력: blur 예측의 SSIM 지도: 평균=0.2982, 최소=-0.2901
# 출력:   밝기 인자 평균=0.9904   대비·구조 인자 평균=0.3050
# 출력:   왼쪽(저주파) SSIM=0.4672  vs  오른쪽(고주파) SSIM=0.2236
# → 밝기 인자는 0.990으로 거의 만점인데 대비·구조 인자는 0.305. 벌점이 사실상 전부
#    '구조'에서 나온다. 무늬가 촘촘한 오른쪽이 왼쪽보다 절반 이하로 더 크게 깎인다.

# %% [markdown]
# ## 5단계: 실제로 학습시켜 보기 — 2D Gaussian 몇 개로 텍스처 맞추기
#
# 3DGS를 아주 얇게 흉내낸 장난감 모델이다. 각 Gaussian은 중심 $\mu$, 스케일 $s$,
# 색 $c$, 불투명도 $\alpha$를 갖고 픽셀 색은 정규화된 가중합으로 만든다.
#
# $$w_i(x) = \alpha_i\exp\!\left(-\tfrac12\left[\tfrac{(x-\mu_{ix})^2}{s_{ix}^2}+\tfrac{(y-\mu_{iy})^2}{s_{iy}^2}\right]\right),
# \qquad C(x)=\frac{\sum_i w_i(x)\,c_i}{\sum_i w_i(x)+\varepsilon}$$
#
# Gaussian 개수를 일부러 부족하게 주면(=학습 초기 상황) 표현력이 모자라
# "평균색으로 덮기"와 "무늬 살리기" 사이의 트레이드오프가 드러난다.
# 같은 초기값에서 **L1 단독**과 $0.8L_1+0.2(1-\mathrm{SSIM})$을 각각 학습해 비교한다.


# %%
H = W = 96
N_GS = 120
yy = torch.arange(H, dtype=torch.float32).view(1, H, 1)
xx = torch.arange(W, dtype=torch.float32).view(1, 1, W)


def init_params(seed=1):
    g = torch.Generator().manual_seed(seed)
    return {
        "mu": torch.stack(
            [torch.rand(N_GS, generator=g) * H, torch.rand(N_GS, generator=g) * W], -1
        ),
        "log_s": torch.full((N_GS, 2), float(np.log(6.0))),
        "color": torch.full((N_GS, 3), 0.5),
        "logit_a": torch.zeros(N_GS),
    }


def render(p):
    s = p["log_s"].exp().clamp(0.7, 40.0)
    a = torch.sigmoid(p["logit_a"]).view(N_GS, 1, 1)
    dy = (yy - p["mu"][:, 0].view(N_GS, 1, 1)) / s[:, 0].view(N_GS, 1, 1)
    dx = (xx - p["mu"][:, 1].view(N_GS, 1, 1)) / s[:, 1].view(N_GS, 1, 1)
    w = a * torch.exp(-0.5 * (dy**2 + dx**2))  # (N,H,W)
    num = (w.unsqueeze(1) * p["color"].view(N_GS, 3, 1, 1)).sum(0)  # (3,H,W)
    return (num / (w.sum(0, keepdim=True) + 1e-6)).unsqueeze(0).clamp(0, 1)


def fit(loss_kind, steps=400, lr=0.06, seed=1):
    p = {k: v.clone().requires_grad_(True) for k, v in init_params(seed).items()}
    opt = torch.optim.Adam(
        [
            {"params": [p["mu"]], "lr": lr * 8},
            {"params": [p["log_s"]], "lr": lr},
            {"params": [p["color"]], "lr": lr},
            {"params": [p["logit_a"]], "lr": lr},
        ]
    )
    hist = []
    for i in range(steps):
        img = render(p)
        l1 = l1_loss(img, gt)
        sl = ssim_loss(img, gt)
        loss = l1 if loss_kind == "l1" else torch.lerp(l1, sl, 0.2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append((l1.item(), sl.item()))
    with torch.no_grad():
        out = render(p)
    return out, np.array(hist)


img_l1, hist_l1 = fit("l1")
img_mix, hist_mix = fit("mix")

for name, img, h in [("L1 단독", img_l1, hist_l1), ("0.8L1+0.2(1-SSIM)", img_mix, hist_mix)]:
    print(
        f"{name:20s} 최종 L1={l1_loss(img, gt).item():.4f}  "
        f"1-SSIM={ssim_loss(img, gt).item():.4f}  "
        f"이미지 std={img.std().item():.4f} (GT std={gt.std().item():.4f})"
    )
# 출력: L1 단독                최종 L1=0.0438  1-SSIM=0.3996  이미지 std=0.0871 (GT std=0.1044)
# 출력: 0.8L1+0.2(1-SSIM)    최종 L1=0.0448  1-SSIM=0.3108  이미지 std=0.0958 (GT std=0.1044)
# → L1 값은 거의 같은데(0.0438 vs 0.0448) 1-SSIM은 0.400 → 0.311로 줄고 이미지 대비(std)는
#    GT(0.1044)에 더 가까워진다. "L1이 거의 똑같이 만족하는 해들" 중에서
#    SSIM 항이 무늬가 살아있는 해를 골라주는 것이다.

# %%
# 국소 대비(11x11 창 기준 표준편차)를 직접 재서 '뭉개짐'을 정량화한다


def local_std(img):
    win = ssim_window(11, img.shape[1]).to(img)
    c = lambda t: F.conv2d(t, win, padding=5, groups=img.shape[1])
    var = (c(img * img) - c(img) ** 2).clamp_min(0)
    return var.sqrt()[0].mean(0)


sd_gt, sd_l1, sd_mix = local_std(gt), local_std(img_l1), local_std(img_mix)
print(f"국소 std 평균 — GT={sd_gt.mean():.4f}  L1단독={sd_l1.mean():.4f}  혼합={sd_mix.mean():.4f}")
print(
    f"고주파 영역(오른쪽 1/3) — GT={sd_gt[:, -32:].mean():.4f}  "
    f"L1단독={sd_l1[:, -32:].mean():.4f}  혼합={sd_mix[:, -32:].mean():.4f}"
)
print(f"학습 곡선 끝 10스텝 평균 1-SSIM: L1단독={hist_l1[-10:,1].mean():.4f}  혼합={hist_mix[-10:,1].mean():.4f}")
# 출력: 국소 std 평균 — GT=0.0852  L1단독=0.0553  혼합=0.0677
# 출력: 고주파 영역(오른쪽 1/3) — GT=0.1017  L1단독=0.0512  혼합=0.0732
# 출력: 학습 곡선 끝 10스텝 평균 1-SSIM: L1단독=0.4001  혼합=0.3105
# → L1 단독 결과의 국소 대비는 GT의 2/3 수준(고주파 영역은 절반)으로 주저앉는다(=뭉개짐).
#    SSIM 항을 섞으면 같은 개수의 Gaussian으로도 국소 대비를 22%(고주파 43%) 더 살린다.

# %% [markdown]
# ## 6단계: 그림으로 정리
#
# 1행: 가우시안 창의 정체 / 2행: blur 예측의 SSIM 지도 / 3행: 학습 결과 비교

# %%
def to_img(t):  # (1,3,H,W) → (H,W,3) uint8
    return (t[0].permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)


fig = make_subplots(
    rows=3,
    cols=3,
    subplot_titles=(
        "1D 가우시안 창 (11, σ=1.5)",
        "2D 창 w₂D = w₁D⊗w₁D",
        "평평한 예측(σ₁=0)의 대비·구조 인자",
        "정답 (오른쪽이 고주파 텍스처)",
        "blur 예측",
        "blur 예측의 SSIM 지도 (초록=1, 빨강=0)",
        "GT 국소 std (11×11 창)",
        "L1 단독 학습 결과",
        "0.8·L1 + 0.2·(1-SSIM) 결과",
    ),
    vertical_spacing=0.10,
    horizontal_spacing=0.07,
)

fig.add_trace(
    go.Bar(x=list(range(11)), y=w1d.numpy(), marker_color="#4C78A8", name="w1D"), row=1, col=1
)
fig.add_trace(
    go.Heatmap(z=WIN[0, 0].numpy(), colorscale="Viridis", showscale=False), row=1, col=2
)
s_grid = np.linspace(0, 0.3, 200)
fig.add_trace(
    go.Scatter(
        x=s_grid, y=C2 / (s_grid**2 + C2), mode="lines", line=dict(color="#E45756", width=3),
        name="C2/(σ₂²+C2)",
    ),
    row=1,
    col=3,
)

fig.add_trace(go.Image(z=to_img(gt)), row=2, col=1)
fig.add_trace(go.Image(z=to_img(pred_blur)), row=2, col=2)
# 창의 zero-padding 때문에 이미지 경계 6픽셀은 통계가 왜곡되므로 잘라서 본다
crop = lambda a: np.flipud(a[6:-6, 6:-6])
fig.add_trace(
    go.Heatmap(
        z=crop(m_blur.detach().numpy()), colorscale="RdYlGn", zmin=0, zmax=1, showscale=False
    ),
    row=2,
    col=3,
)

fig.add_trace(
    go.Heatmap(
        z=crop(sd_gt.numpy()), colorscale="Magma", zmin=0, zmax=0.15, showscale=False
    ),
    row=3,
    col=1,
)
fig.add_trace(go.Image(z=to_img(img_l1)), row=3, col=2)
fig.add_trace(go.Image(z=to_img(img_mix)), row=3, col=3)

fig.update_xaxes(title_text="창 내 위치 k", row=1, col=1)
fig.update_yaxes(title_text="가중치", row=1, col=1)
fig.update_xaxes(title_text="정답의 국소 표준편차 σ₂", row=1, col=3)
fig.update_yaxes(title_text="대비·구조 인자", row=1, col=3)
for r, c in [(1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]:
    fig.update_xaxes(showticklabels=False, row=r, col=c)
    fig.update_yaxes(showticklabels=False, scaleanchor=None, row=r, col=c)

fig.update_layout(
    height=1000,
    width=1250,
    showlegend=False,
    title_text=(
        f"SSIM 항의 역할 — blur 예측: L1={l1_loss(pred_blur, gt).item():.3f} (작다) 인데 "
        f"1-SSIM={ssim_loss(pred_blur, gt).item():.3f} (크다) | "
        f"학습 결과 1-SSIM: L1단독 {hist_l1[-1,1]:.3f} → 혼합 {hist_mix[-1,1]:.3f}"
    ),
    template="plotly_white",
    font=dict(size=12),
)
_show(fig)
fig.write_image("expy.png", scale=2)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 결론
#
# - L1은 픽셀별 절댓값 오차만 보므로 **무늬를 평균색으로 지우는 해**를 선호한다
#   (1단계: 뭉갠 예측의 L1이 위상반전 예측의 절반).
# - SSIM은 픽셀마다 **11×11 가우시안 창**($\sigma=1.5$) 안의 $\mu,\sigma^2,\sigma_{12}$로
#   밝기·대비·구조를 채점한다. 예측이 평평하면 대비·구조 인자가
#   $C_2/(\sigma_2^2+C_2)$로 0에 수렴 → 뭉개짐에 큰 벌점 (3~4단계: blur 예측의
#   밝기 인자는 0.990인데 대비·구조 인자는 0.305, 벌점이 전부 구조에서 나온다).
# - 그래서 $1-\mathrm{SSIM}$을 $\lambda=0.2$로 섞으면, L1이 거의 똑같이 만족하는 해들 중에서
#   **국소 대비/구조가 살아있는 해**가 선택된다 (5단계: 최종 L1은 0.044로 사실상 같은데
#   1−SSIM은 0.400 → 0.311, 국소 std는 0.0553 → 0.0677).
#
# 실제 학습에서 이 항의 무게는 `ssim_lambda: float = 0.2`
# (`examples/simple_trainer.py:171`)이고, `torch.lerp(l1loss, ssimloss, 0.2)`로 합쳐진다.
