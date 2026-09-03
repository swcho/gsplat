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
# # Spherical Harmonics(SH) 이해하기 — 3DGS의 시점 의존 색을 중심으로
#
# 3D Gaussian Splatting은 Gaussian마다 색을 RGB 값 하나가 아니라 **SH 계수 16개 × 3채널**로 저장한다.
# 이 노트북은 그 SH가 무엇인지, 어디에 쓰이는지, "DC 계수"와 "SH 평가"가 무슨 뜻인지를
# 순수 PyTorch 코드와 그림으로 따라간다. gsplat 자체는 필요 없다(마지막 교차 검증 셀만 선택적으로 import).
#
# 구성
# 1. SH란 무엇인가 — 구면 위 함수의 "푸리에 급수"
# 2. SH의 응용처 — 왜 그래픽스와 3DGS가 SH를 고르는가
# 3. DC 계수 — 0차 항의 의미와 3DGS 초기화 `(rgb − 0.5) / C0`
# 4. SH 평가 — 계수 + 방향 → 색, 그리고 학습에서 계수를 맞추는 과정
#
# **실행 방법**: VSCode에서 `# %%` 셀 단위로 실행하거나 `python examples/sh_walkthrough.py`. GPU 없이도 동작한다.

# %%
import math

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 그림 제목의 한글 표시용: 설치된 CJK 폰트가 있으면 사용, 없으면 DejaVu로 폴백(한글은 □로 나오지만 실행에는 영향 없음)
_ko_fonts = [f for f in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]
             if f in {x.name for x in fm.fontManager.ttflist}]
plt.rcParams["font.family"] = _ko_fonts + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

# %% [markdown]
# ## 1. SH란 무엇인가
#
# ### 1.1 구면 위 함수와 그 "주파수 분해"
#
# 1차원 주기 함수는 푸리에 급수로 사인·코사인의 합으로 쓸 수 있다. **구면 위에 정의된 함수** $f(\mathbf d)$
# ($\mathbf d$는 단위 방향 벡터, $\|\mathbf d\|=1$)에 대해 같은 역할을 하는 기저가 **Spherical Harmonics** $Y_\ell^m$ 이다.
#
# $$
# f(\mathbf d)\;\approx\;\sum_{\ell=0}^{L}\;\sum_{m=-\ell}^{\ell} c_\ell^m\,Y_\ell^m(\mathbf d)
# $$
#
# - $\ell$ = **차수(degree, band)**: 0, 1, 2, … 클수록 구면 위에서 더 빠르게 진동하는 성분(고주파).
# - $m$ = 차수 $\ell$ 안의 인덱스, $-\ell \le m \le \ell$ → 차수 $\ell$에는 $2\ell+1$개의 기저가 있다.
# - 차수 $L$까지 쓰면 기저 개수는 $\sum_{\ell=0}^{L}(2\ell+1) = (L+1)^2$. 3DGS의 $L=3$ → **16개**.
#
# 기저는 **정규직교**한다. 구면 전체에 대한 적분 $d\Omega = \sin\theta\,d\theta\,d\varphi$ 에 대해
#
# $$
# \int_{S^2} Y_\ell^m(\mathbf d)\,Y_{\ell'}^{m'}(\mathbf d)\,d\Omega = \delta_{\ell\ell'}\,\delta_{mm'}
# $$
#
# 덕분에 계수는 푸리에 계수처럼 **내적(사영)** 한 번으로 구해진다.
#
# $$
# c_\ell^m = \int_{S^2} f(\mathbf d)\,Y_\ell^m(\mathbf d)\,d\Omega
# $$
#
# ### 1.2 정의 (실수형 SH)
#
# 구면 좌표 $\mathbf d = (\sin\theta\cos\varphi,\ \sin\theta\sin\varphi,\ \cos\theta)$ 에서, 연관 르장드르 함수 $P_\ell^m$을 써서
#
# $$
# Y_\ell^m(\theta,\varphi)=
# \begin{cases}
# \sqrt{2}\,K_\ell^{m}\cos(m\varphi)\,P_\ell^{m}(\cos\theta) & m>0\\[2pt]
# K_\ell^{0}\,P_\ell^{0}(\cos\theta) & m=0\\[2pt]
# \sqrt{2}\,K_\ell^{|m|}\sin(|m|\varphi)\,P_\ell^{|m|}(\cos\theta) & m<0
# \end{cases},
# \qquad
# K_\ell^m=\sqrt{\frac{2\ell+1}{4\pi}\,\frac{(\ell-|m|)!}{(\ell+|m|)!}}
# $$
#
# 실제 코드에서는 이 식을 직교좌표 $(x,y,z)$의 **다항식**으로 풀어 쓴다. 저차 항은 다음과 같다
# (부호는 Condon–Shortley 규약, 3DGS/gsplat이 쓰는 것과 같다).
#
# $$
# \begin{aligned}
# \ell=0:\quad & Y_0^0 = \tfrac{1}{2\sqrt{\pi}} \approx 0.2821 \\
# \ell=1:\quad & Y_1^{-1} = -\sqrt{\tfrac{3}{4\pi}}\,y,\qquad Y_1^{0} = \sqrt{\tfrac{3}{4\pi}}\,z,\qquad Y_1^{1} = -\sqrt{\tfrac{3}{4\pi}}\,x \\
# \ell=2:\quad & Y_2^{-2} = \sqrt{\tfrac{15}{4\pi}}\,xy,\quad Y_2^{-1} = -\sqrt{\tfrac{15}{4\pi}}\,yz,\quad
#               Y_2^{0} = \sqrt{\tfrac{5}{16\pi}}\,(3z^2-1),\quad Y_2^{1} = -\sqrt{\tfrac{15}{4\pi}}\,xz,\quad
#               Y_2^{2} = \sqrt{\tfrac{15}{16\pi}}\,(x^2-y^2)
# \end{aligned}
# $$
#
# 즉 $\ell$차 SH는 **$x,y,z$의 $\ell$차 동차 다항식을 단위구에 제한한 것**이다. 0차는 상수, 1차는 선형(방향 성분 자체),
# 2차는 이차식… 그래서 $\ell$이 커질수록 구면 위에서 더 잘게 진동한다.
#
# 계수 배열의 순서는 3DGS/gsplat 규약 $k = \ell^2 + \ell + m$ 을 따른다 (k=0 → (0,0), k=1..3 → (1,−1),(1,0),(1,1), …).

# %%
LM = [(l, m) for l in range(4) for m in range(-l, l + 1)]     # k번째 기저의 (ℓ, m)
C0 = 0.28209479177387814                                       # Y₀⁰ = 1/(2√π)


def sh_bases(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """단위 방향 dirs[..., 3] → SH 기저값 [..., (degree+1)²].
    gsplat/cuda/_torch_impl.py의 _eval_sh_bases_fast와 같은 값·순서를 (ℓ, m)별로 풀어 쓴 것."""
    x, y, z = dirs.unbind(-1)
    out = [torch.full_like(x, C0)]                                                      # (0, 0)
    if degree >= 1:
        c = 0.4886025119029199                                                          # √(3/4π)
        out += [-c * y, c * z, -c * x]                                                  # (1,-1) (1,0) (1,1)
    if degree >= 2:
        c1, c2, c3 = 1.0925484305920792, 0.31539156525252005, 0.5462742152960396        # √(15/4π) √(5/16π) √(15/16π)
        out += [c1 * x * y, -c1 * y * z, c2 * (3 * z * z - 1), -c1 * x * z, c3 * (x * x - y * y)]
    if degree >= 3:
        c1, c2, c3, c4, c5 = 0.5900435899266435, 2.890611442640554, 0.4570457994644658, 0.3731763325901154, 1.445305721320277
        out += [-c1 * y * (3 * x * x - y * y), c2 * x * y * z, -c3 * y * (5 * z * z - 1), c4 * z * (5 * z * z - 3),
                -c3 * x * (5 * z * z - 1), c5 * z * (x * x - y * y), -c1 * x * (x * x - 3 * y * y)]
    return torch.stack(out, dim=-1)


def sphere_grid(n_theta: int = 128, n_phi: int = 256, device=DEVICE):
    """구면을 (θ, φ) 격자로 덮는 방향 벡터 d[nθ,nφ,3]와 적분 가중치 w = sinθ dθ dφ (Σw ≈ 4π)."""
    theta = (torch.arange(n_theta, device=device) + 0.5) * math.pi / n_theta            # 극각: z축에서의 각
    phi = (torch.arange(n_phi, device=device) + 0.5) * 2 * math.pi / n_phi - math.pi    # 방위각
    th, ph = torch.meshgrid(theta, phi, indexing="ij")
    d = torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], dim=-1)
    w = th.sin() * (math.pi / n_theta) * (2 * math.pi / n_phi)
    return d, w


dirs, w = sphere_grid()
B = sh_bases(dirs, 3)                                             # [nθ, nφ, 16]
print("기저 텐서:", tuple(B.shape), " 적분 가중치 합 =", w.sum().item(), "(≈ 4π =", 4 * math.pi, ")")

# 정규직교성 확인: ∫ Yᵢ Yⱼ dΩ ≈ δᵢⱼ  (격자 구적)
gram = torch.einsum("abi,abj,ab->ij", B, B, w)
print("Gram 행렬과 단위행렬의 최대 차이:", (gram - torch.eye(16, device=DEVICE)).abs().max().item())

# %%
# 16개 기저를 등장방형(equirectangular) 지도로 그린다: 가로 = 방위각 φ, 세로 = 극각 θ. 빨강 +, 파랑 −.
fig, axes = plt.subplots(4, 7, figsize=(17, 6))
for ax in axes.flat:
    ax.axis("off")
vmax = B.abs().max().item()
for k, (l, m) in enumerate(LM):
    ax = axes[l, m + 3]
    ax.imshow(B[..., k].cpu(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=[-180, 180, 180, 0])
    ax.set_title(f"k={k}  Y({l},{m:+d})", fontsize=9)
fig.suptitle("SH 기저 (행: 차수 ℓ, 열: m). 아래로 갈수록 구면 위 진동이 잦아진다 (고주파)", y=0.99)
plt.tight_layout(); plt.show()

# %%
# 같은 것을 3D로: 반지름 = |Y|, 색 = 부호. 양자역학 궤도 그림과 같은 모양이 나온다.
fig = plt.figure(figsize=(15, 5.5))
d_np, B_np = dirs.cpu().numpy(), B.cpu().numpy()
for k in range(9):
    ax = fig.add_subplot(2, 5, k + 1 if k < 5 else k + 2, projection="3d")
    r = np.abs(B_np[..., k])
    xyz = d_np * r[..., None]
    ax.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2], facecolors=plt.cm.RdBu_r(0.5 + 0.5 * np.sign(B_np[..., k])),
                    rstride=4, cstride=4, linewidth=0, antialiased=False, shade=True)
    lim = r.max(); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_title(f"Y{LM[k]}", fontsize=10); ax.set_axis_off()
fig.suptitle("ℓ ≤ 2 기저의 3D 모양 (반지름 = |Y|, 빨강 +, 파랑 −)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 1.3 저차 = 부드러운 근사, 고차 = 세부
#
# 구면 위 함수를 차수 $L$까지의 SH로 사영하면 **저역 통과 필터**를 거친 것과 같다. 실험으로 확인해 보자.
# 하늘(위쪽 밝음)과 태양(좁고 강한 봉우리)이 있는 가짜 환경 조명 $f(\mathbf d)\in\mathbb R^3$(RGB)을 만들고
# 차수 0, 1, 2, 3으로 사영·복원해서 비교한다.
#
# 사영은 위 적분식을 격자 구적으로 계산한다:
# $c_k = \sum_{\text{grid}} f(\mathbf d)\,Y_k(\mathbf d)\,w(\mathbf d)$

# %%
def env_radiance(d: torch.Tensor) -> torch.Tensor:
    """가짜 환경 조명. d[...,3] → RGB[...,3]. 위쪽은 밝은 하늘색, 아래쪽은 어두운 갈색, 태양 방향에 좁은 봉우리."""
    sun_dir = F.normalize(torch.tensor([0.5, 0.3, 0.8], device=d.device), dim=0)
    sky = 0.5 * (d[..., 2] + 1)                                  # z: −1(아래) → 0, +1(위) → 1
    sun = torch.exp(-(1 - d @ sun_dir) / 0.02)                    # 태양 주변 좁은 로브
    r = 0.25 + 0.35 * sky + 3.0 * sun
    g = 0.20 + 0.50 * sky + 2.8 * sun
    b = 0.15 + 0.75 * sky + 2.5 * sun
    return torch.stack([r, g, b], dim=-1)


def project_to_sh(f_vals: torch.Tensor, bases: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """f_vals[nθ,nφ,D], bases[nθ,nφ,K], w[nθ,nφ] → 계수 [K, D]   (c_k = ∫ f Y_k dΩ)"""
    return torch.einsum("abk,abd,ab->kd", bases, f_vals, w)


def reconstruct(coeffs: torch.Tensor, bases: torch.Tensor, degree: int) -> torch.Tensor:
    """계수 [K,D]와 기저 [...,K]로 차수 degree까지만 써서 복원 → [..., D]"""
    K = (degree + 1) ** 2
    return torch.einsum("...k,kd->...d", bases[..., :K], coeffs[:K])


f = env_radiance(dirs)                                           # [nθ, nφ, 3]
coeffs_env = project_to_sh(f, B, w)                              # [16, 3]
print("SH 계수 [k, RGB]:\n", coeffs_env)

fig, axes = plt.subplots(1, 5, figsize=(20, 3.2))
tone = lambda img: (img / 1.5).clamp(0, 1).cpu()                 # 표시용 톤매핑
axes[0].imshow(tone(f)); axes[0].set_title("원본 f(d)")
for L in range(4):
    rec = reconstruct(coeffs_env, B, L)
    err = ((rec - f) ** 2 * w[..., None]).sum() / w.sum()        # 구면 평균 MSE
    axes[L + 1].imshow(tone(rec)); axes[L + 1].set_title(f"L={L} ({(L + 1) ** 2}개 계수)  MSE={err.item():.4f}")
for ax in axes:
    ax.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# 차수가 올라갈수록 하늘의 부드러운 그라디언트는 금방 맞지만, 태양처럼 **좁고 날카로운 봉우리는 16개 계수로는 흐릿하게 퍼진다**
# (그리고 봉우리 주변에 음의 물결, ringing이 생긴다). SH는 "부드러운 구면 함수"를 소수의 계수로 압축하는 데 강하고,
# 고주파를 표현하려면 차수를 급격히 올려야 한다 — 이것이 3DGS가 **3차(16개)에서 멈추는** 이유이자,
# 거울 반사 같은 날카로운 시점 의존성은 3DGS가 잘 못 그리는 이유다.

# %% [markdown]
# ## 2. SH의 응용처
#
# SH가 쓰이는 곳은 공통적으로 **"방향(구면 위 위치)에 따라 값이 부드럽게 변하는 함수"를 소수의 숫자로 저장·연산**해야 하는 경우다.
#
# | 분야 | 무엇을 SH로 표현하나 | 비고 |
# |---|---|---|
# | 실시간 렌더링 — 환경 조명 | 방향별 입사 광량(environment map) | Ramamoorthi & Hanrahan(2001): 확산 반사 조명은 **2차(9개 계수)**로 오차 ~1% |
# | Precomputed Radiance Transfer(PRT) | 물체 표면의 방향별 반사/가림 전달 함수 | 조명 SH · 전달 SH의 내적 = 한 점의 밝기 |
# | 신경 장면 표현 | 시점 방향에 따른 색 | PlenOctrees, Plenoxels, **3DGS** — MLP 없이 SH 계수만 저장해 실시간 렌더링 |
# | 공간 음향 (Ambisonics) | 청취 위치에서의 방향별 음장 | 1차 = 4채널(B-format), 3차 = 16채널 |
# | 측지학·행성과학 | 지구 중력장·자기장(EGM, IGRF) | 수천 차수까지 사용 |
# | 양자역학 | 원자 궤도의 각도 부분 (s, p, d, f 궤도 = ℓ=0,1,2,3) | 위의 3D 그림이 바로 그 모양 |
# | 우주론 | 우주배경복사(CMB) 온도 지도의 각도 파워 스펙트럼 $C_\ell$ | |
# | 의료영상 | 확산 MRI(HARDI)의 방향별 확산 신호 | 짝수 차수만 사용(대칭) |
#
# ### 3DGS에서의 역할
#
# 각 Gaussian이 "어느 방향에서 보면 어떤 색인가"를 갖는다. 실제 물체는 보는 각도에 따라 색이 달라지므로
# (하이라이트, 프레넬, 반투명) 이를 RGB 하나로는 표현할 수 없다. 3DGS는 Gaussian $i$의 색을
#
# $$
# \mathbf c_i(\mathbf d) = \max\!\Big(0,\ \sum_{k=0}^{15} \mathbf c_{i,k}\,Y_k(\mathbf d) + 0.5\Big),
# \qquad \mathbf d = \frac{\boldsymbol\mu_i - \mathbf o_{\text{cam}}}{\|\boldsymbol\mu_i - \mathbf o_{\text{cam}}\|}
# $$
#
# 로 두고, 계수 $\mathbf c_{i,k}\in\mathbb R^3$ (16 × 3 = 48개 실수)를 학습한다. 뉴럴 네트워크 없이
# **덧셈·곱셈 48번**으로 색이 나오므로 수백만 개 Gaussian을 실시간으로 그릴 수 있다.

# %% [markdown]
# ## 3. DC 계수
#
# "DC"는 신호처리 용어 **direct current(직류)**, 즉 **주파수 0 성분**에서 왔다. SH에서는 $\ell = 0$ 항, 계수 $c_0^0$이다.
# $Y_0^0 = \tfrac{1}{2\sqrt\pi}$ 는 방향과 무관한 **상수**이므로,
#
# $$
# c_0^0 = \int_{S^2} f(\mathbf d)\,Y_0^0\,d\Omega = \frac{1}{2\sqrt\pi}\int_{S^2} f\,d\Omega
# \quad\Longrightarrow\quad
# c_0^0\,Y_0^0 = \frac{1}{4\pi}\int_{S^2} f\,d\Omega = \overline{f}
# $$
#
# 즉 **DC 항만으로 복원한 값 $c_0^0 Y_0^0$은 함수의 구면 평균**이다. 나머지 고차 항은 모두 평균이 0인 "변동"만 담는다
# (각 $Y_\ell^m$, $\ell\ge1$은 $Y_0^0$과 직교하므로 구면 적분이 0).

# %%
mean_f = (f * w[..., None]).sum(dim=(0, 1)) / w.sum()            # 구면 평균 (RGB)
print("f의 구면 평균         :", mean_f)
print("DC 항만으로 복원 c₀·Y₀:", coeffs_env[0] * C0)
print("고차 항(ℓ≥1)만의 구면 평균:", (reconstruct(coeffs_env, B, 3) - coeffs_env[0] * C0).mul(w[..., None]).sum(dim=(0, 1)) / w.sum(), "← 0")

# %% [markdown]
# ### 3DGS의 초기화 `sh0 = (rgb − 0.5) / C0` 가 나오는 이유
#
# 3DGS 색 공식은 SH 합에 $+0.5$를 더한 뒤 0 미만을 잘라낸다. 학습 시작 시 고차 계수를 0으로 두면
#
# $$
# \mathbf c(\mathbf d) = c_0\,Y_0^0 + 0.5 \quad(\text{모든 방향에서 동일})
# $$
#
# 이것이 SfM 포인트 색 $\mathbf{rgb}$와 같아지려면 $c_0 = (\mathbf{rgb} - 0.5)/Y_0^0 = (\mathbf{rgb}-0.5)/C_0$ 이어야 한다.
# $+0.5$ 오프셋은 **계수가 전부 0일 때 색이 검정이 아니라 중간 회색(0.5)** 이 되게 해서, 계수의 부호가 양·음 대칭으로
# 움직일 수 있게 하는 장치다. gsplat은 파라미터를 `sh0`(DC, [N,1,3])와 `shN`(나머지 15개, [N,15,3])으로 나누고
# `shN`의 학습률을 1/20로 둔다 — 시점 의존성은 기본색이 잡힌 뒤 천천히 배우라는 뜻이다.

# %%
rgb = torch.tensor([[0.8, 0.3, 0.1], [0.1, 0.6, 0.9]], device=DEVICE)      # SfM 색 2개
sh0 = (rgb - 0.5) / C0                                                     # [N,3]  DC 계수
shN = torch.zeros(2, 15, 3, device=DEVICE)                                 # 고차 계수는 0으로 시작
coeffs = torch.cat([sh0[:, None], shN], dim=1)                             # [N,16,3]

some_dir = F.normalize(torch.randn(2, 3, device=DEVICE), dim=-1)           # 아무 방향
color = torch.clamp_min(torch.einsum("nk,nkc->nc", sh_bases(some_dir, 3), coeffs) + 0.5, 0.0)
print("sh0     =", sh0)
print("복원된 색 =", color, " (rgb와 같다, 방향 무관)")

# %% [markdown]
# ## 4. SH 평가 (SH evaluation)
#
# **"평가"는 계수와 방향이 주어졌을 때 함수값을 계산하는 것**, 즉 급수를 실제로 더하는 일이다.
#
# $$
# \mathbf c(\mathbf d) = \sum_{k=0}^{K-1} \mathbf c_k\,Y_k(\mathbf d)
# = \underbrace{\big[\,Y_0(\mathbf d)\ \cdots\ Y_{K-1}(\mathbf d)\,\big]}_{1\times K}
#   \underbrace{\begin{bmatrix}\mathbf c_0^\top\\ \vdots\\ \mathbf c_{K-1}^\top\end{bmatrix}}_{K\times 3}
# $$
#
# 두 단계다.
# 1. 방향 $\mathbf d$에서 기저값 $Y_k(\mathbf d)$ 16개 계산 (위 `sh_bases` — 다항식 몇 개)
# 2. 계수와 내적 (채널마다 16번의 곱셈-덧셈)
#
# 3DGS에서는 카메라마다·Gaussian마다 한 번씩 수행된다. 방향은 **카메라 위치에서 Gaussian 중심을 보는 방향**이고,
# 카메라 위치는 world→camera 행렬 $[R\,|\,\mathbf t]$에서 $\mathbf o_{\text{cam}} = -R^\top\mathbf t$ 로 복원한다.
# gsplat에서는 `spherical_harmonics(sh_degree, means, viewmats, coeffs)`가 이 두 단계를 CUDA 커널 하나로 처리하고,
# `rasterization()`은 그 결과에 `+0.5`, `clamp_min(0)`을 적용해 래스터라이저에 넘긴다.
#
# 아래는 그 과정을 그대로 옮긴 함수다.

# %%
def sh_eval(coeffs: torch.Tensor, dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """coeffs[N,K,3], dirs[N,3](정규화 불필요) → 색 [N,3].  gsplat _spherical_harmonics와 같은 계산."""
    K = (degree + 1) ** 2
    d = F.normalize(dirs, dim=-1)
    return torch.einsum("nk,nkc->nc", sh_bases(d, degree), coeffs[:, :K])


def view_dirs(means: torch.Tensor, viewmat: torch.Tensor) -> torch.Tensor:
    """3DGS의 시점 방향: 카메라 위치 −Rᵀt 에서 각 Gaussian 중심을 향하는 벡터."""
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    cam_pos = -R.T @ t
    return means - cam_pos


# 예: 카메라가 (0,0,-5)에서 +z를 보는 viewmat  (world→camera: t = −R·cam_pos)
viewmat = torch.eye(4, device=DEVICE); viewmat[:3, 3] = torch.tensor([0.0, 0.0, 5.0], device=DEVICE)
means = torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, 3.0]], device=DEVICE)
print("카메라 위치 −Rᵀt =", (-viewmat[:3, :3].T @ viewmat[:3, 3]).tolist())
print("시점 방향 d       =", F.normalize(view_dirs(means, viewmat), dim=-1))

# %% [markdown]
# ### 4.1 한 Gaussian의 색을 모든 방향에서 보기
#
# 고차 계수를 랜덤으로 준 Gaussian 하나의 색 $\mathbf c(\mathbf d)$를 방향 지도로 그려 본다.
# 차수를 0 → 3으로 올리며 평가하면 점차 세부가 붙는다. 이것이 학습 코드의 `sh_degree_interval`
# (simple_trainer.py: `sh_degree_to_use = min(step // sh_degree_interval, sh_degree)`) 이 하는 일이다 —
# 처음 1000스텝은 DC만, 1000스텝마다 한 차수씩 활성화해서 기본색부터 안정적으로 잡는다.

# %%
one = torch.zeros(1, 16, 3, device=DEVICE)
one[0, 0] = (torch.tensor([0.7, 0.5, 0.3], device=DEVICE) - 0.5) / C0     # 기본색 (DC)
one[0, 1:] = torch.randn(15, 3, device=DEVICE) * 0.25                     # 시점 의존성

fig, axes = plt.subplots(1, 4, figsize=(17, 3))
for L in range(4):
    Kd = (L + 1) ** 2
    img = torch.clamp_min(torch.einsum("abk,kc->abc", B[..., :Kd], one[0, :Kd]) + 0.5, 0.0)   # 모든 격자 방향에서 평가
    axes[L].imshow(img.clamp(0, 1).cpu(), extent=[-180, 180, 180, 0]); axes[L].set_title(f"sh_degree={L} 로 평가 ({Kd}개 계수 사용)")
    axes[L].set_xlabel("φ (deg)"); axes[L].set_ylabel("θ (deg)")
fig.suptitle("같은 Gaussian, 같은 계수 — 활성화 차수만 바꿔 평가한 시점별 색")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 4.2 계수는 어떻게 얻는가 — 학습은 "관측으로부터의 역평가"
#
# 3DGS는 계수를 적분으로 구하지 않는다(진짜 $f$를 모른다). 대신 여러 카메라에서 관측한 색과 렌더 결과의 차이를
# 역전파해서 계수를 갱신한다. 이를 Gaussian 하나에 대해 축소 재현해 보자.
#
# - 정답: 확산 기본색 + 특정 방향의 광택 하이라이트를 가진 함수 $f^\star(\mathbf d)$
# - 관측: 카메라 $n_{\text{view}}$대 방향에서의 $f^\star$ 값
# - 방법 A — 최소제곱 해: $\min_{\mathbf c}\sum_j \|\,Y(\mathbf d_j)\,\mathbf c - f^\star(\mathbf d_j)\|^2$ (선형이므로 닫힌 형태)
# - 방법 B — Adam 경사하강 (실제 학습과 같은 방식)
#
# 관측 수가 적으면 고차가 **과적합**(관측 사이에서 색이 요동)하고, 차수가 낮으면 하이라이트를 **표현 못 한다**.

# %%
def f_star(d: torch.Tensor) -> torch.Tensor:
    """정답 시점 의존 색: 기본색 + 한 방향으로의 넓은 광택 로브."""
    h = F.normalize(torch.tensor([-0.4, 0.6, 0.7], device=d.device), dim=0)
    spec = torch.clamp_min(d @ h, 0) ** 8
    base = torch.tensor([0.6, 0.35, 0.2], device=d.device)
    return base + 0.6 * spec[..., None]


f_true = f_star(dirs)                                                             # 평가용 정답 (전 구면)


def fit_lstsq(obs_dirs, obs_rgb, degree):
    A = sh_bases(obs_dirs, degree)                                                # [n, K]
    return torch.linalg.lstsq(A, obs_rgb).solution                                # [K, 3]


def sphere_mse(coeffs, degree):
    rec = torch.einsum("abk,kc->abc", B[..., :(degree + 1) ** 2], coeffs)
    return (((rec - f_true) ** 2) * w[..., None]).sum() / w.sum()


print("전 구면 MSE (행: 관측 뷰 수, 열: 차수)")
print("        L=0      L=1      L=2      L=3")
for n_view in [8, 20, 60, 300]:
    obs_d = F.normalize(torch.randn(n_view, 3, device=DEVICE), dim=-1)
    obs_c = f_star(obs_d)
    row = [sphere_mse(fit_lstsq(obs_d, obs_c, L), L).item() for L in range(4)]
    print(f"n={n_view:4d}  " + "  ".join(f"{v:.5f}" for v in row))

# %%
# 방법 B: Adam으로 계수 학습 (3DGS 학습과 같은 방식; sh0와 shN에 다른 학습률)
obs_d = F.normalize(torch.randn(60, 3, device=DEVICE), dim=-1)
obs_c = f_star(obs_d)
sh0 = torch.zeros(1, 3, device=DEVICE, requires_grad=True)
shN = torch.zeros(15, 3, device=DEVICE, requires_grad=True)
opt = torch.optim.Adam([{"params": [sh0], "lr": 2.5e-2}, {"params": [shN], "lr": 2.5e-2 / 20}])
hist = []
A_obs = sh_bases(obs_d, 3)
for step in range(4000):
    degree_to_use = min(step // 500, 3)                                           # sh_degree_interval 흉내
    Kd = (degree_to_use + 1) ** 2
    coeffs = torch.cat([sh0, shN], dim=0)                                         # [16,3]
    pred = torch.clamp_min(A_obs[:, :Kd] @ coeffs[:Kd] + 0.5, 0.0)
    loss = F.l1_loss(pred, obs_c)
    opt.zero_grad(); loss.backward(); opt.step()
    hist.append(loss.item())

coeffs_gd = torch.cat([sh0, shN], dim=0).detach().clone()
coeffs_gd[0] += 0.5 / C0                          # 학습식의 +0.5 오프셋을 DC 계수로 흡수해 f*와 직접 비교 (0.5 = (0.5/C0)·Y₀⁰)
coeffs_ls = fit_lstsq(obs_d, obs_c, 3)

fig, axes = plt.subplots(1, 4, figsize=(18, 3.2))
axes[0].plot(hist); axes[0].set_yscale("log"); axes[0].set_title("Adam 학습 L1 loss (500스텝마다 차수 +1)")
for ax, (name, img) in zip(axes[1:], [("정답 f*(d)", f_true),
                                       ("최소제곱 L=3", torch.einsum("abk,kc->abc", B, coeffs_ls)),
                                       ("Adam L=3", torch.einsum("abk,kc->abc", B, coeffs_gd))]):
    ax.imshow(img.clamp(0, 1).cpu(), extent=[-180, 180, 180, 0]); ax.set_title(name); ax.axis("off")
plt.tight_layout(); plt.show()
print(f"전 구면 MSE   최소제곱: {sphere_mse(coeffs_ls, 3).item():.5f}   Adam: {sphere_mse(coeffs_gd, 3).item():.5f}")

# %% [markdown]
# ## 5. gsplat 구현과 교차 검증 (선택)
#
# 아래 셀은 이 노트북의 `sh_bases`/`sh_eval`이 gsplat의 참조 구현과 같은 값을 내는지 확인한다.
# `import gsplat`은 CUDA 확장 JIT 빌드를 유발할 수 있으므로 기본은 꺼 두었다.
#
# - [gsplat/cuda/_torch_impl.py](../gsplat/cuda/_torch_impl.py) `_eval_sh_bases_fast`: Sloan(2013)의 점화식으로 기저를 계산 (4차, 25개까지)
# - 같은 파일 `_spherical_harmonics`: 정규화 + 기저 × 계수 합
# - [gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu](../gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu): 같은 계산의 CUDA 커널(fwd/bwd)
# - [gsplat/rendering.py](../gsplat/rendering.py) `_maybe_evaluate_sh`: `+0.5`, `clamp_min(0)`, 컬링된 Gaussian 마스킹

# %%
COMPARE_WITH_GSPLAT = False
if COMPARE_WITH_GSPLAT:
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast, _spherical_harmonics

    d = F.normalize(torch.randn(10000, 3, device=DEVICE), dim=-1)
    print("기저 차이 :", (sh_bases(d, 3) - _eval_sh_bases_fast(16, d)).abs().max().item())
    c = torch.randn(10000, 16, 3, device=DEVICE)
    print("평가 차이 :", (sh_eval(c, d, 3) - _spherical_harmonics(3, d, c)).abs().max().item())

# %% [markdown]
# ## 정리
#
# | 용어 | 뜻 | 3DGS/gsplat에서 |
# |---|---|---|
# | SH $Y_\ell^m$ | 구면 위 함수의 정규직교 기저. $\ell$차 = $x,y,z$의 $\ell$차 다항식 | 3차까지 16개 사용 (`sh_degree=3`, K=16) |
# | 계수 $c_k$ | $f$를 기저에 사영한 값 $\int fY_k\,d\Omega$ | Gaussian당 `[16,3]`, 학습 파라미터 (`sh0` + `shN`) |
# | DC 계수 $c_0$ | $\ell=0$ 상수항. $c_0Y_0^0$ = 구면 평균 | 시점 무관 기본색. 초기값 `(rgb−0.5)/C0` |
# | SH 평가 | 방향 $\mathbf d$에서 $\sum_k c_kY_k(\mathbf d)$ 계산 | 카메라·Gaussian마다 1회, 결과에 `+0.5`, `clamp_min(0)` |
# | 차수 활성화 | 저차부터 순차적으로 사용 | `sh_degree_interval`=1000 스텝마다 +1 |
#
# **한계와 확장**: 16개 SH는 부드러운 시점 의존성(광택, 프레넬)까지만 표현한다. 날카로운 반사·굴절을 위해
# 후속 연구는 SH 대신 작은 MLP(시점 의존 디코더), 구면 가우시안(Spherical Gaussians), 또는 반사 방향 기반 인코딩을 쓴다.
