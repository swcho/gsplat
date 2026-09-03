# %% [markdown]
# # `sh0 = (rgb − 0.5)/C0`, `shN = 0` 초기화는 정말 rgb를 그대로 돌려주는가?
#
# 3DGS/gsplat의 색 공식은
#
# $$
# \mathbf c(\mathbf d) = \operatorname{clamp}_{\ge 0}\Big(\sum_{k=0}^{15} \mathbf c_k\,Y_k(\mathbf d) + 0.5\Big)
# $$
#
# 이다. 고차 계수 $\mathbf c_{1..15}$(= `shN`)를 0으로 두면 합에는 DC 항 하나만 남고,
# $Y_0^0 = C_0 = \tfrac{1}{2\sqrt\pi}$ 는 **방향과 무관한 상수**이므로
#
# $$
# \mathbf c(\mathbf d) = c_0\,C_0 + 0.5 = \frac{\mathbf{rgb}-0.5}{C_0}\,C_0 + 0.5 = \mathbf{rgb}
# $$
#
# 즉 **어느 방향에서 평가해도 원래 `rgb`가 그대로 복원**되어야 한다. 아래에서 이를 수치로 확인하고,
# 대조 실험(shN 노이즈, 잘못된 초기화, clamp)으로 각 요소가 왜 필요한지 본다.
#
# 필요 패키지: torch, numpy, plotly, kaleido (`expy.png` 저장용)

# %%
import os
import math

import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots

torch.manual_seed(0)
DEVICE = "cpu"
torch.set_printoptions(precision=5, sci_mode=False, linewidth=120)
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


C0 = 0.28209479177387814  # Y₀⁰ = 1/(2√π)


def sh_bases(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """단위 방향 dirs[..., 3] → SH 기저값 [..., (degree+1)²]  (노트북 sh_walkthrough.py와 동일)."""
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


def gs_color(coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """3DGS 색 공식: coeffs[N,16,3], dirs[N,3] → clamp(ΣcₖYₖ(d) + 0.5, 0)."""
    d = F.normalize(dirs, dim=-1)
    return torch.clamp_min(torch.einsum("nk,nkc->nc", sh_bases(d, 3), coeffs) + 0.5, 0.0)


def init_coeffs(rgb: torch.Tensor, shN: torch.Tensor | None = None) -> torch.Tensor:
    """3DGS 초기화: sh0 = (rgb − 0.5)/C0, shN = 0 → coeffs[N,16,3]."""
    sh0 = (rgb - 0.5) / C0
    if shN is None:
        shN = torch.zeros(rgb.shape[0], 15, 3, device=rgb.device)
    return torch.cat([sh0[:, None], shN], dim=1)


print("C0 =", C0, " 1/C0 =", 1 / C0)
# 출력: C0 = 0.28209479177387814  1/C0 = 3.544907701811032

# %% [markdown]
# ## 1. 실험: rgb 4개 → 초기화 → 랜덤 방향 1000개에서 평가
#
# rgb 하나당 서로 다른 방향 1000개에서 색을 평가한 뒤, `rgb`와의 **최대 절대 오차**와
# 방향에 따른 **표준편차**를 본다. 둘 다 부동소수점 오차 수준(~1e-7)이어야 한다.

# %%
rgb = torch.tensor([[0.8, 0.3, 0.1],
                    [0.1, 0.6, 0.9],
                    [0.5, 0.5, 0.5],
                    [0.02, 0.97, 0.40]], device=DEVICE)          # SfM 포인트 색 4개
N, M = rgb.shape[0], 1000
coeffs = init_coeffs(rgb)                                          # [4,16,3]
print("sh0 =", coeffs[:, 0])

dirs = F.normalize(torch.randn(M, 3, device=DEVICE), dim=-1)       # 랜덤 방향 1000개
# 각 rgb에 대해 1000개 방향 모두 평가: coeffs를 방향 수만큼 복제
colors = gs_color(coeffs.repeat_interleave(M, dim=0), dirs.repeat(N, 1)).view(N, M, 3)

err = (colors - rgb[:, None]).abs()
print(f"복원 색 최대 오차   = {err.max().item():.3e}")
print("방향별 표준편차(채널) =", colors.std(dim=1))
print("복원 색 (방향 평균)  =", colors.mean(dim=1))
# 출력: sh0 = tensor([[ 1.06347, -0.70898, -1.41796],
# 출력:         [-1.41796,  0.35449,  1.41796],
# 출력:         [ 0.00000,  0.00000,  0.00000],
# 출력:         [-1.70156,  1.66611, -0.35449]])
# 출력: 복원 색 최대 오차   = 1.118e-08
# 출력: 방향별 표준편차(채널) = tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])   ← 방향 무관
# 출력: 복원 색 (방향 평균)  = tensor([[0.80000, 0.30000, 0.10000], [0.10000, 0.60000, 0.90000], [0.50000, 0.50000, 0.50000], [0.02000, 0.97000, 0.40000]])

# %% [markdown]
# ## 2. 왜 방향 의존성이 사라지는가 — 항별 기여 분해
#
# $\sum_k \mathbf c_k Y_k(\mathbf d)$ 를 $k=0$ (DC)과 $k\ge1$ (고차)로 나눠 보면,
# 고차 기저 $Y_k(\mathbf d)$ 자체는 방향마다 크게 달라지지만 **계수가 0이므로 곱이 0**이다.
# 남는 것은 $c_0 C_0$ 뿐이고 이것은 상수다. 참고로 rgb=0.5인 세 번째 색은 `sh0`까지 0이 되어 $+0.5$ 오프셋만으로 복원된다.

# %%
B = sh_bases(dirs, 3)                                              # [1000,16]
print("고차 기저 Y₁..Y₁₅의 방향별 표준편차 (0이 아님) =", B[:, 1:].std(dim=0)[:4], "...")
print("DC 기저 Y₀의 방향별 표준편차 (상수)          =", B[:, 0].std().item())

c = coeffs[0]                                                      # 첫 rgb의 계수 [16,3]
dc_term = B[:, :1] @ c[:1]                                         # [1000,3]  c₀·Y₀(d)
hi_term = B[:, 1:] @ c[1:]                                         # [1000,3]  Σ_{k≥1} cₖ·Yₖ(d)
print("DC 항  c₀·C0     (모든 방향 동일) =", dc_term[0], " ↔ rgb−0.5 =", rgb[0] - 0.5)
print("고차 항 최대 절대값               =", hi_term.abs().max().item())
# 출력: 고차 기저 Y₁..Y₁₅의 방향별 표준편차 (0이 아님) = tensor([0.28153, 0.28186, 0.28314, 0.28298]) ...
# 출력: DC 기저 Y₀의 방향별 표준편차 (상수)          = 8.94517029337294e-08   (float32 반올림 잔차)
# 출력: DC 항  c₀·C0     (모든 방향 동일) = tensor([ 0.30000, -0.20000, -0.40000])  ↔ rgb−0.5 = tensor([ 0.30000, -0.20000, -0.40000])
# 출력: 고차 항 최대 절대값               = 0.0

# %% [markdown]
# ## 3. 대조 실험 A — `shN`에 작은 랜덤값을 넣으면
#
# 고차 계수가 0이 아니면 $\sum_{k\ge1}\mathbf c_k Y_k(\mathbf d)$ 가 방향마다 달라져 색이 "흔들린다".
# 노이즈 크기 $\sigma$를 키우며 방향별 표준편차를 재 보면, 흔들림이 $\sigma$에 거의 비례해 커진다.
# (학습에서 `shN`의 학습률을 `sh0`의 1/20로 두는 이유 — 기본색이 잡히기 전에 고차 항이 요동치지 않게.)

# %%
print("σ(shN 노이즈)   방향별 색 표준편차(R,G,B)   방향별 최대 편차")
for sigma in [0.0, 0.01, 0.05, 0.2]:
    shN = sigma * torch.randn(1, 15, 3, device=DEVICE)
    cf = init_coeffs(rgb[:1], shN).repeat_interleave(M, dim=0)
    col = gs_color(cf, dirs)                                       # [1000,3]
    print(f"{sigma:8.2f}        {col.std(dim=0).numpy().round(4)}        {(col - rgb[0]).abs().max().item():.4f}")
# 출력: σ(shN 노이즈)   방향별 색 표준편차(R,G,B)   방향별 최대 편차
# 출력:     0.00        [0. 0. 0.]        0.0000
# 출력:     0.01        [0.0097 0.0125 0.0098]        0.0318
# 출력:     0.05        [0.0686 0.0552 0.0407]        0.1875
# 출력:     0.20        [0.1573 0.2217 0.1228]        0.6066

# %% [markdown]
# ## 4. 대조 실험 B — 초기화 식의 각 요소를 빠뜨리면
#
# | 초기화 | 복원 색 $c_0C_0+0.5$ | 결과 |
# |---|---|---|
# | $(\mathbf{rgb}-0.5)/C_0$ (정답) | $\mathbf{rgb}$ | 정확 |
# | $\mathbf{rgb}/C_0$ ( −0.5 생략) | $\mathbf{rgb}+0.5$ | 전체가 0.5만큼 밝아짐 (1을 넘음) |
# | $\mathbf{rgb}-0.5$ ( /C0 생략) | $0.282\,(\mathbf{rgb}-0.5)+0.5$ | 회색 0.5 쪽으로 3.5배 눌림(저대비) |
# | $\mathbf{rgb}$ 그대로 | $0.282\,\mathbf{rgb}+0.5$ | 밝고 뿌연 회색 |

# %%
variants = {
    "(rgb-0.5)/C0 (정답)": (rgb - 0.5) / C0,
    "rgb/C0   (−0.5 생략)": rgb / C0,
    "rgb-0.5  (/C0 생략)": rgb - 0.5,
    "rgb 그대로": rgb,
}
d1 = dirs[:1].repeat(N, 1)                                        # 아무 방향 하나 (방향 무관이므로 결과는 동일)
print(f"{'초기화':<24}  {'입력 rgb':<22} → 복원 색")
for name, sh0 in variants.items():
    cf = torch.cat([sh0[:, None], torch.zeros(N, 15, 3)], dim=1)
    col = gs_color(cf, d1)
    for i in range(2):                                             # 앞의 두 색만 표시
        print(f"{name if i == 0 else '':<24}  {str(rgb[i].numpy().round(2)):<22} → {col[i].numpy().round(3)}")
# 출력: 초기화                    입력 rgb               → 복원 색
# 출력: (rgb-0.5)/C0 (정답)       [0.8 0.3 0.1]          → [0.8 0.3 0.1]
# 출력:                           [0.1 0.6 0.9]          → [0.1 0.6 0.9]
# 출력: rgb/C0   (−0.5 생략)      [0.8 0.3 0.1]          → [1.3 0.8 0.6]
# 출력:                           [0.1 0.6 0.9]          → [0.6 1.1 1.4]
# 출력: rgb-0.5  (/C0 생략)       [0.8 0.3 0.1]          → [0.585 0.444 0.387]
# 출력:                           [0.1 0.6 0.9]          → [0.387 0.528 0.613]
# 출력: rgb 그대로                [0.8 0.3 0.1]          → [0.726 0.585 0.528]
# 출력:                           [0.1 0.6 0.9]          → [0.528 0.669 0.754]

# %% [markdown]
# ## 5. 대조 실험 C — `clamp(0)`에 걸리는 경우
#
# `shN = 0`이면 $c_0C_0+0.5=\mathbf{rgb}\ge0$ 이라 clamp는 아무 일도 하지 않는다.
# 그러나 어두운 색(rgb < 0.5)에 큰 `shN` 노이즈가 얹히면 합이 음수가 되는 방향이 생기고,
# 그 방향에서 clamp가 0으로 잘라 **방향 평균이 원래 rgb보다 위로 편향**된다 (평균이 0인 고차 항이 더는 평균 0이 아니게 됨).

# %%
rgb_dark = torch.tensor([[0.05, 0.10, 0.40]], device=DEVICE)
print("rgb_dark =", rgb_dark[0].numpy())
print("σ       clamp 걸린 방향 비율(R,G,B)   방향 평균 색(R,G,B)")
for sigma in [0.0, 0.1, 0.3]:
    shN = sigma * torch.randn(1, 15, 3, device=DEVICE)
    cf = init_coeffs(rgb_dark, shN).repeat_interleave(M, dim=0)
    raw = torch.einsum("nk,nkc->nc", B, cf) + 0.5                  # clamp 전
    col = torch.clamp_min(raw, 0.0)
    frac = (raw < 0).float().mean(dim=0)
    print(f"{sigma:4.1f}    {frac.numpy().round(3)}               {col.mean(dim=0).numpy().round(3)}")
# 출력: rgb_dark = [0.05 0.1  0.4 ]
# 출력: σ       clamp 걸린 방향 비율(R,G,B)   방향 평균 색(R,G,B)
# 출력:  0.0    [0. 0. 0.]               [0.05 0.1  0.4 ]
# 출력:  0.1    [0.31  0.159 0.   ]               [0.063 0.109 0.404]
# 출력:  0.3    [0.441 0.373 0.187]               [0.193 0.19  0.435]

# %% [markdown]
# ## 6. 시각화 — 등장방형(equirectangular) 지도
#
# 구면의 모든 방향 $(\theta,\varphi)$ 을 평면에 펼쳐 색을 그린다.
# 왼쪽: `shN = 0` → 방향과 무관한 **단색**. 오른쪽: `shN`에 $\sigma=0.1$ 노이즈 → 방향마다 다른 **얼룩**.
# (색 자체가 데이터이므로 별도 팔레트 없이 복원된 RGB를 그대로 그린다.)

# %%
def sphere_grid(n_theta=128, n_phi=256):
    theta = (torch.arange(n_theta) + 0.5) * math.pi / n_theta
    phi = (torch.arange(n_phi) + 0.5) * 2 * math.pi / n_phi - math.pi
    th, ph = torch.meshgrid(theta, phi, indexing="ij")
    return torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], dim=-1)


grid = sphere_grid()                                               # [128,256,3]
Bg = sh_bases(grid, 3)                                             # [128,256,16]


def sphere_image(cf16):                                            # cf16: [16,3] → uint8 이미지
    img = torch.clamp(torch.einsum("abk,kc->abc", Bg, cf16) + 0.5, 0, 1)
    return (img * 255).round().to(torch.uint8).numpy()


cf_clean = init_coeffs(rgb[:1])[0]
cf_noisy = init_coeffs(rgb[:1], 0.1 * torch.randn(1, 15, 3))[0]

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.06,
                    subplot_titles=("shN = 0 : 모든 방향에서 rgb=(0.8,0.3,0.1) 그대로",
                                    "shN ~ N(0, 0.1²) : 방향마다 색이 흔들림"))
for col_i, cf in enumerate([cf_clean, cf_noisy], start=1):
    fig.add_trace(go.Image(z=sphere_image(cf), x0=-180 + 360 / 512, dx=360 / 256, y0=180 / 256, dy=180 / 128,
                           hovertemplate="φ=%{x:.0f}°, θ=%{y:.0f}°<br>RGB=%{z}<extra></extra>"),
                  row=1, col=col_i)
    fig.update_xaxes(title_text="방위각 φ (°)", showgrid=False, row=1, col=col_i)
    fig.update_yaxes(title_text="극각 θ (°)" if col_i == 1 else None, showgrid=False, autorange="reversed", row=1, col=col_i)
fig.update_layout(width=1100, height=380, margin=dict(l=60, r=20, t=60, b=50),
                  paper_bgcolor="white", plot_bgcolor="white", font=dict(size=12),
                  title_text="DC 항만 있는 초기화는 방향과 무관하게 rgb를 복원한다")
_show(fig)
png_path = os.path.join(HERE, "expy.png")
fig.write_image(png_path, scale=2)
print("저장:", png_path)
# 출력: 저장: <hint dir>/expy.png

# %% [markdown]
# ## 정리
#
# - `shN = 0`이면 $\sum_k \mathbf c_kY_k(\mathbf d) = c_0 C_0$ 만 남고, $C_0$는 상수 → 색은 방향 무관.
# - $c_0 = (\mathbf{rgb}-0.5)/C_0$ 를 넣으면 $c_0C_0 + 0.5 = \mathbf{rgb}$ 로 정확히 복원된다 (오차 ~1e-7, 부동소수점 한계).
# - `−0.5`를 빼지 않으면 0.5만큼 밝아지고, `/C0`를 빼면 색이 회색 0.5 쪽으로 3.5배 눌린다.
# - `shN`이 0에서 벗어나면 방향별로 색이 흔들리고, 어두운 색에서는 `clamp(0)`이 평균을 위로 편향시킨다.
