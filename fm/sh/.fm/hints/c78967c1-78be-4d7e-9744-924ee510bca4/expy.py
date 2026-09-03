# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (gsplat)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 환경 조명을 SH 차수 0~3으로 사영·복원하면 하늘과 태양은 각각 어떻게 되는가?
#
# `sh_walkthrough.py`의 가짜 환경 조명은 **하늘(부드러운 그라디언트) + 태양(좁고 날카로운 봉우리)** 의 합이다.
# 두 성분을 **분리**해서 각각 차수 $L$까지의 SH로 사영·복원하면, 어느 쪽이 저차에서 잘 맞고 어느 쪽이 안 맞는지가
# 수치로 드러난다. 확인할 것은 세 가지다.
#
# 1. 성분별·차수별 구면 MSE 표 — 하늘은 $L=1$이면 거의 완벽, 태양은 $L=3$(16개 계수)에서도 오차가 크다
# 2. 태양 방향을 지나는 대원(great circle) 단면 — 봉우리가 **낮고 넓게 퍼지고**, 주변에 **음의 물결(ringing)** 이 생긴다
# 3. 태양 로브 폭을 바꾸면 "충분한 차수"가 어떻게 달라지는가 — 폭이 좁아질수록 필요한 차수가 급격히 올라간다
#
# 사영은 노트북과 같이 격자 구적으로 계산한다:
# $$c_k = \int_{S^2} f(\mathbf d)\,Y_k(\mathbf d)\,d\Omega \approx \sum_{\text{grid}} f(\mathbf d)\,Y_k(\mathbf d)\,w(\mathbf d),
# \qquad f_L(\mathbf d) = \sum_{k<(L+1)^2} c_k Y_k(\mathbf d)$$
#
# 필요 패키지: numpy, torch, scipy, plotly, kaleido (정적 PNG 저장용)

# %%
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import sph_harm_y

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


DEVICE = "cpu"                       # CUDA 초기화 편차를 피하려고 CPU 고정
torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# ---- sh_walkthrough.py에서 그대로 가져온 함수들 ------------------------------------------------
C0 = 0.28209479177387814             # Y₀⁰ = 1/(2√π)


def sh_bases(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """단위 방향 dirs[..., 3] → SH 기저값 [..., (degree+1)²]  (gsplat 규약, degree ≤ 3)."""
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


def project_to_sh(f_vals: torch.Tensor, bases: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """f_vals[nθ,nφ,D], bases[nθ,nφ,K], w[nθ,nφ] → 계수 [K, D]   (c_k = ∫ f Y_k dΩ)"""
    return torch.einsum("abk,abd,ab->kd", bases, f_vals, w)


def reconstruct(coeffs: torch.Tensor, bases: torch.Tensor, degree: int) -> torch.Tensor:
    """계수 [K,D]와 기저 [...,K]로 차수 degree까지만 써서 복원 → [..., D]"""
    K = (degree + 1) ** 2
    return torch.einsum("...k,kd->...d", bases[..., :K], coeffs[:K])


# ---- 비교용: 3차를 넘는 차수까지 실수 SH를 scipy로 만든다 -------------------------------------------
def sh_bases_any(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """임의 차수의 실수 정규직교 SH [..., (degree+1)²]. m>0: √2·Re(Y_ℓ^m), m<0: √2·Im(Y_ℓ^|m|), m=0: Re(Y_ℓ^0).
    부호 규약은 gsplat과 다를 수 있으나 같은 공간을 펼치는 정규직교 기저이므로 사영·복원 결과(MSE)는 동일하다."""
    d = dirs.detach().cpu().numpy().astype(np.float64)
    theta = np.arccos(np.clip(d[..., 2], -1, 1))          # 극각
    phi = np.arctan2(d[..., 1], d[..., 0])                # 방위각
    cols = []
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)         # scipy ≥1.15: (n, m, θ_polar, φ_azimuth)
            if m == 0:
                cols.append(Y.real)
            elif m > 0:
                cols.append(math.sqrt(2) * Y.real)
            else:
                cols.append(math.sqrt(2) * Y.imag)
    return torch.tensor(np.stack(cols, axis=-1), dtype=torch.float32, device=dirs.device)


dirs, w = sphere_grid()
L_MAX = 16
B_all = sh_bases_any(dirs, L_MAX)                          # [nθ, nφ, 289]
B3 = sh_bases(dirs, 3)                                    # gsplat 규약 16개 (교차 검증용)

gram = torch.einsum("abi,abj,ab->ij", B_all, B_all, w)
print("scipy 기저 개수:", B_all.shape[-1], " Gram−I 최대 차이:", (gram - torch.eye(B_all.shape[-1])).abs().max().item())
# 두 기저가 같은 공간을 펼치는지: ℓ≤3 부분공간의 사영 연산자가 같은가
P_g = torch.einsum("abk,cdk->abcd", B3, B3)[0, 0]         # 한 방향에서의 재생핵 K(d0, d)
P_s = torch.einsum("abk,cdk->abcd", B_all[..., :16], B_all[..., :16])[0, 0]
print("ℓ≤3 재생핵(gsplat 기저 vs scipy 기저) 최대 차이:", (P_g - P_s).abs().max().item())
# 출력: scipy 기저 개수: 289  Gram−I 최대 차이: 0.00085   (float32 격자 구적 오차 수준)
# 출력: ℓ≤3 재생핵(gsplat 기저 vs scipy 기저) 최대 차이: 2.7e-06   → 두 기저는 같은 부분공간을 펼친다

# %% [markdown]
# ## 1. 하늘과 태양을 분리해서 사영하기
#
# 노트북의 `env_radiance`의 R 채널은 $0.25 + 0.35\,\text{sky}(\mathbf d) + 3.0\,\text{sun}(\mathbf d)$ 이다.
# 여기서
# $$\text{sky}(\mathbf d) = \tfrac12 (d_z + 1), \qquad \text{sun}(\mathbf d) = \exp\!\Big(-\frac{1 - \mathbf d\cdot\mathbf s}{\sigma}\Big),\ \ \sigma = 0.02$$
#
# - **sky는 $d_z$의 1차식** → 정확히 $\ell\le1$ 안에 들어 있다. $L=1$이면 오차가 0이어야 한다.
# - **sun은 $\mathbf d\cdot\mathbf s$의 지수함수** → 모든 차수에 성분이 퍼져 있다. 로브가 좁을수록($\sigma$ 작을수록) 고차 비중이 크다.
#
# 오차는 구면 평균 MSE와, 함수 자체의 분산으로 나눈 **상대 MSE**(1.0 = DC만 쓴 것과 같은 수준) 두 가지로 본다.

# %%
SUN_DIR = F.normalize(torch.tensor([0.5, 0.3, 0.8]), dim=0)


def sky_part(d):
    return 0.25 + 0.35 * 0.5 * (d[..., 2] + 1)


def sun_part(d, sigma=0.02, amp=3.0):
    return amp * torch.exp(-(1 - d @ SUN_DIR) / sigma)


def sphere_mean(v):
    return (v * w).sum() / w.sum()


def mse_table(f_vals, degrees):
    """f_vals[nθ,nφ] → {L: (MSE, 상대MSE)}"""
    coeffs = project_to_sh(f_vals[..., None], B_all, w)                   # [289, 1]
    var = sphere_mean((f_vals - sphere_mean(f_vals)) ** 2).item()
    out = {}
    for L in degrees:
        rec = reconstruct(coeffs, B_all, L)[..., 0]
        mse = sphere_mean((rec - f_vals) ** 2).item()
        out[L] = (mse, mse / var)
    return out


DEGREES = [0, 1, 2, 3, 4, 8, 16]
f_sky, f_sun = sky_part(dirs), sun_part(dirs)
f_env = f_sky + f_sun
tabs = {"하늘(sky)": mse_table(f_sky, DEGREES), "태양(sun)": mse_table(f_sun, DEGREES), "합(env R채널)": mse_table(f_env, DEGREES)}

print(f"{'성분':<14}" + "".join(f"{'L=' + str(L):>12}" for L in DEGREES))
print(f"{'(계수 개수)':<14}" + "".join(f"{(L + 1) ** 2:>12}" for L in DEGREES))
for name, t in tabs.items():
    print(f"{name:<14}" + "".join(f"{t[L][0]:>12.5f}" for L in DEGREES) + "   ← MSE")
    print(f"{'':<14}" + "".join(f"{t[L][1]:>12.4f}" for L in DEGREES) + "   ← 상대 MSE (÷분산)")
# 출력:
# 성분               L=0      L=1      L=2      L=3      L=4      L=8     L=16
# (계수 개수)           1        4        9       16       25       81      289
# 하늘(sky)      0.01021  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000   ← MSE
#                1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   ← 상대 MSE
# 태양(sun)      0.04410  0.04151  0.03752  0.03258  0.02717  0.00880  0.00014   ← MSE
#                1.0000   0.9412   0.8508   0.7387   0.6160   0.1996   0.0031   ← 상대 MSE
# 합(env R채널)  0.06262  0.04151  0.03752  0.03258  0.02717  0.00880  0.00014   ← MSE (L≥1에서는 태양 오차만 남는다)
#                1.0000   0.6628   0.5991   0.5202   0.4338   0.1405   0.0022   ← 상대 MSE

# %% [markdown]
# 하늘은 $L=1$에서 오차가 부동소수 수준으로 떨어진다(정확히 1차 다항식이니까). 태양은 $L=3$에서도 상대 오차가 절반 넘게 남고,
# $L=16$(289개 계수)에 가서야 눈에 띄게 줄어든다. 3DGS가 쓰는 16개 계수로는 태양 봉우리의 에너지 대부분을 담지 못한다는 뜻이다.
#
# ## 2. 태양 방향을 지나는 대원 단면 — 퍼짐과 음의 물결
#
# 태양 방향 $\mathbf s$와 그에 수직인 $\mathbf u$로 대원 $\mathbf d(\alpha) = \cos\alpha\,\mathbf s + \sin\alpha\,\mathbf u$ 를 만들고
# 원본 $f$와 복원 $f_L$을 $\alpha$에 대해 그린다. 확인할 수치는
# - **봉우리 높이** $f_L(\mathbf s)$ vs $f(\mathbf s)=3.0$ → 얼마나 낮아졌나
# - **반치폭(FWHM)** → 얼마나 넓게 퍼졌나
# - **최솟값** $\min_\alpha f_L$ → 원본은 항상 $\ge 0$인데 복원은 음수가 나오면 ringing

# %%
u = F.normalize(torch.linalg.cross(SUN_DIR, torch.tensor([0.0, 0.0, 1.0])), dim=0)     # s에 수직
alpha = torch.linspace(-math.pi, math.pi, 1441)
circle = alpha[:, None].cos() * SUN_DIR + alpha[:, None].sin() * u                      # [n, 3]
Bc = sh_bases_any(circle, L_MAX)                                                        # 대원 위 기저
coeffs_sun = project_to_sh(f_sun[..., None], B_all, w)                                  # [289, 1]
f_sun_c = sun_part(circle)


def fwhm_deg(y):
    """봉우리(α=0) 기준 반치폭(도). 음수 바닥은 0으로 보고 계산."""
    half = y.max() / 2
    above = (y >= half).nonzero().squeeze(-1)
    return math.degrees((alpha[above.max()] - alpha[above.min()]).item())


curves = {}
print(f"{'':>8}{'봉우리 f_L(s)':>14}{'FWHM(도)':>10}{'최솟값':>10}{'음수 영역 비율':>14}")
print(f"{'원본':>8}{f_sun_c.max().item():>14.3f}{fwhm_deg(f_sun_c):>10.1f}{f_sun_c.min().item():>10.4f}{0.0:>14.3f}")
for L in [0, 1, 2, 3, 8, 16]:
    y = reconstruct(coeffs_sun, Bc, L)[..., 0]
    curves[L] = y
    neg_frac = (y < 0).float().mean().item()
    print(f"{'L=' + str(L):>8}{y[720].item():>14.3f}{fwhm_deg(y):>10.1f}{y.min().item():>10.4f}{neg_frac:>14.3f}")

# 구면 전체에서의 음수 비율(면적 가중)도 확인
rec3 = reconstruct(coeffs_sun, B_all, 3)[..., 0]
print(f"\nL=3 복원의 구면 전체 최솟값 = {rec3.min().item():.4f}  (원본 최솟값 = {f_sun.min().item():.2e})")
print(f"L=3 복원에서 값이 음수인 구면 면적 비율 = {sphere_mean((rec3 < 0).float()).item():.3f}")
# 출력:
#            봉우리 f_L(s)  FWHM(도)    최솟값  음수 영역 비율
#   원본          3.000      19.0    0.0000       0.000
#   L=0           0.030     360.0    0.0300       0.000   ← 평균값만 남는다(상수)
#   L=1           0.118     141.0   -0.0582       0.390
#   L=2           0.259      89.0   -0.0498       0.333
#   L=3           0.445      65.5   -0.1031       0.440   ← 3DGS의 16개: 높이 15%, 폭 3.4배, 음수 골
#   L=8           1.668      30.5   -0.1397       0.432
#   L=16          2.831      20.0   -0.0181       0.404
#
# L=3 복원의 구면 전체 최솟값 = -0.1030  (원본 최솟값 = 1.14e-43)
# L=3 복원에서 값이 음수인 구면 면적 비율 = 0.462

# %% [markdown]
# 원본 태양은 반치폭 약 19°의 뾰족한 봉우리(높이 3.0)다. $L=3$ 복원은 봉우리가 **높이 0.45(원본의 15%)로 낮아지고 반치폭은 65°로 3.4배 퍼진다**.
# 그리고 봉우리 바깥에서 **최솟값이 음수**가 되어 원본에는 없는 어두운 고리(ringing)가 생긴다 —
# 이것이 3DGS가 색 평가 뒤 `clamp_min(0)`을 붙이는 이유 중 하나다.
# 유한 차수의 SH 절단은 구면 위의 **저역 통과 필터**이고, 날카로운 입력을 통과시키면 1D 푸리에의 Gibbs 현상과 같은 것이 나타난다.
#
# ## 3. 태양 로브 폭 $\sigma$ 를 바꾸면 필요한 차수는?
#
# $\text{sun}(\mathbf d)=\exp(-(1-\mathbf d\cdot\mathbf s)/\sigma)$ 의 각 반경은 대략 $\sqrt{2\sigma}$ rad. $\sigma$를 0.005~0.5로 바꾸며
# "상대 MSE가 5% 미만이 되는 최소 차수"를 찾는다. 구면 위에서 각 크기 $\Delta\theta$의 특징을 표현하려면 대략
# $\ell \gtrsim \pi/\Delta\theta$ 가 필요하다는 경험 법칙과 비교해 보자.

# %%
SIGMAS = [0.005, 0.02, 0.05, 0.1, 0.2, 0.5]
THRESH = 0.05
print(f"{'σ':>7}{'각반경(도)':>10}" + "".join(f"{'L=' + str(L):>8}" for L in DEGREES) + f"{'  상대MSE<5% 최소 L':>20}")
width_rows = []
for s in SIGMAS:
    t = mse_table(sun_part(dirs, sigma=s), list(range(L_MAX + 1)))
    need = next((L for L in range(L_MAX + 1) if t[L][1] < THRESH), None)
    ang = math.degrees(math.sqrt(2 * s))
    width_rows.append((s, ang, need))
    print(f"{s:>7.3f}{ang:>10.1f}" + "".join(f"{t[L][1]:>8.3f}" for L in DEGREES) + f"{('L=' + str(need)) if need is not None else '>16':>20}")
# 출력:
#       σ  각반경(도)    L=0    L=1    L=2    L=3    L=4    L=8   L=16   상대MSE<5% 최소 L
#   0.005       5.7  1.000  0.985  0.961  0.928  0.887  0.670  0.236          >16
#   0.020      11.5  1.000  0.941  0.851  0.739  0.616  0.200  0.003         L=12   ← 노트북의 태양
#   0.050      18.1  1.000  0.858  0.664  0.465  0.294  0.018  0.000          L=7
#   0.100      25.6  1.000  0.730  0.434  0.211  0.085  0.000  0.000          L=5
#   0.200      36.2  1.000  0.520  0.182  0.045  0.008  0.000  0.000          L=3   ← 16개 계수로 충분한 최소 폭
#   0.500      57.3  1.000  0.194  0.019  0.001  0.000  0.000  0.000          L=2

# %% [markdown]
# 로브가 넓어질수록($\sigma\uparrow$) 필요한 차수가 뚝 떨어진다. $\sigma=0.5$(각반경 약 57°)면 $L=2$로 충분하고 $\sigma=0.2$(약 36°)가 $L=3$로 되는 경계다.
# 노트북의 $\sigma=0.02$(약 11°)는 $L=12$(169개 계수)는 되어야 오차 5% 아래로 내려간다 — $L=3$로는 한참 부족하다. 즉 **16개 계수는 "각 크기 수십 도 이상의 부드러운 변화"까지가 한계**이며,
# 이것이 3DGS가 3차에서 멈추는 이유이자 거울 반사 같은 날카로운 시점 의존성을 잘 못 그리는 이유다.
#
# ## 4. 그림으로 확인

# %%
fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                    subplot_titles=("태양 방향을 지나는 대원 단면: 원본 vs SH 복원 (σ=0.02)",
                                    "성분별 상대 MSE (÷분산)"))
deg = alpha.numpy() * 180 / math.pi
fig.add_trace(go.Scatter(x=deg, y=f_sun_c.numpy(), name="원본 sun", line=dict(color="black", width=2.5)), row=1, col=1)
palette = {0: "#9e9e9e", 1: "#4c72b0", 2: "#55a868", 3: "#c44e52", 8: "#8172b2", 16: "#dd8452"}
for L, y in curves.items():
    fig.add_trace(go.Scatter(x=deg, y=y.numpy(), name=f"L={L} ({(L + 1) ** 2}개)",
                             line=dict(color=palette[L], width=3 if L == 3 else 1.5, dash="solid" if L <= 3 else "dot")),
                  row=1, col=1)
fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"), row=1, col=1)
fig.update_xaxes(title_text="태양 방향으로부터의 각도 α (도)", range=[-90, 90], row=1, col=1)
fig.update_yaxes(title_text="밝기", row=1, col=1)

for name, color in [("하늘(sky)", "#4c72b0"), ("태양(sun)", "#c44e52")]:
    fig.add_trace(go.Bar(x=[f"L={L}" for L in DEGREES], y=[max(tabs[name][L][1], 1e-7) for L in DEGREES],
                         name=name, marker_color=color), row=1, col=2)
fig.update_yaxes(type="log", title_text="상대 MSE (log)", row=1, col=2)
fig.update_layout(title="하늘은 L=1에서 끝, 태양은 L=3에서도 흐릿하게 퍼지고 음의 물결이 남는다",
                  width=1250, height=520, barmode="group", legend=dict(orientation="h", y=-0.2),
                  template="plotly_white")
_show(fig)
png_path = os.path.join(HERE, "expy.png")
fig.write_image(png_path, scale=2)
print("저장:", png_path)
# 출력: 저장: <hint dir>/expy.png   (막대의 하늘 L≥1 값은 log 축 표시용 하한 1e-7로 클램프한 것 — 실제로는 ≈0)

# %% [markdown]
# ## 정리
#
# | | 하늘(sky) | 태양(sun, σ=0.02) |
# |---|---|---|
# | 함수의 성질 | $d_z$의 1차식 → $\ell\le1$에 완전히 포함 | $\mathbf d\cdot\mathbf s$의 좁은 지수 로브 → 모든 차수에 에너지 분산 |
# | $L=0$ | 평균만 (상대 MSE 1.0) | 평균만 — 태양이 사라지고 구면 전체가 옅게 밝아짐 |
# | $L=1$ | **오차 ≈ 0** (정확) | 봉우리가 아주 넓고 낮은 언덕으로 |
# | $L=3$ (3DGS 16개) | 오차 ≈ 0 | 봉우리 높이 15%, 반치폭 3.4배, 구면의 46%가 **음수(ringing)** |
# | 상대 오차 5% 미만에 필요한 차수 | 1 | 12 (169개 계수) |
#
# 답: **하늘의 부드러운 그라디언트는 낮은 차수에서 금방 맞지만, 태양처럼 좁고 날카로운 봉우리는 16개 계수로는
# 흐릿하게 퍼지고 그 주변에 음의 물결(ringing)이 생긴다.**
