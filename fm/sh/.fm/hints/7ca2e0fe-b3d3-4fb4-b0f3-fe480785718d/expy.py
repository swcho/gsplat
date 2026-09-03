# %% [markdown]
# # DC 항 $c_0^0Y_0^0$ = 함수의 구면 평균 — 코드로 확인하기
#
# 이 스크립트는 다음 사실을 작은 실험으로 단계별로 확인한다.
#
# $$
# c_0^0\,Y_0^0=\frac{1}{4\pi}\int_{S^2} f\,d\Omega=\overline f
# $$
#
# 1. 1D 비유: 푸리에 급수의 상수항 $a_0/2$ = 구간 평균
# 2. $Y_0^0=1/(2\sqrt\pi)$ 가 정규 조건 $\int (Y_0^0)^2 d\Omega = 1$ 에서 나오는 것을 확인
# 3. 여러 함수(상수, $z$, $z^2$, 환경 조명)에 대해 "격자 구적 평균" vs `coeffs[0] * C0` 표
# 4. 고차 항($\ell\ge1$)만의 구면 평균이 0임을 확인
# 5. $c_0$을 $\delta$만큼 바꾸면 평균이 정확히 $\delta\cdot C_0$ 만큼 바뀜(선형)
# 6. 시각화: 원본 / DC만 / 고차 항만 지도 3장
#
# 필요 패키지: numpy, torch, plotly, kaleido(PNG 저장용)

# %%
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEVICE = "cpu"
torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False, linewidth=120)
HERE = os.path.dirname(os.path.abspath(__file__))


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# %% [markdown]
# ## 0. 도구 준비 — sh_walkthrough.py의 `sh_bases`, `sphere_grid`, `env_radiance`, `project_to_sh` 재사용
#
# - `sphere_grid`: 구면을 $(\theta,\varphi)$ 격자로 덮는 방향 벡터 `d`와 적분 가중치 `w = sinθ dθ dφ` ($\sum w\approx4\pi$)
# - `project_to_sh`: $c_k=\int f\,Y_k\,d\Omega$ 를 격자 구적 $\sum f\,Y_k\,w$ 로 계산
#
# 격자 구적이므로 "구면 평균"은 그냥 `(f * w).sum() / w.sum()` 이다 — 적분을 표면적으로 나눈 것.

# %%
C0 = 0.28209479177387814  # Y₀⁰ = 1/(2√π)


def sh_bases(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """단위 방향 dirs[..., 3] → SH 기저값 [..., (degree+1)²] (3DGS/gsplat 순서·부호)."""
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
    """구면 (θ, φ) 격자의 방향 d[nθ,nφ,3]와 적분 가중치 w = sinθ dθ dφ (Σw ≈ 4π)."""
    theta = (torch.arange(n_theta, device=device) + 0.5) * math.pi / n_theta
    phi = (torch.arange(n_phi, device=device) + 0.5) * 2 * math.pi / n_phi - math.pi
    th, ph = torch.meshgrid(theta, phi, indexing="ij")
    d = torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], dim=-1)
    w = th.sin() * (math.pi / n_theta) * (2 * math.pi / n_phi)
    return d, w


def env_radiance(d: torch.Tensor) -> torch.Tensor:
    """가짜 환경 조명 RGB: 위쪽 밝은 하늘, 아래쪽 어두운 갈색, 태양 방향에 좁은 봉우리."""
    sun_dir = F.normalize(torch.tensor([0.5, 0.3, 0.8], device=d.device), dim=0)
    sky = 0.5 * (d[..., 2] + 1)
    sun = torch.exp(-(1 - d @ sun_dir) / 0.02)
    r = 0.25 + 0.35 * sky + 3.0 * sun
    g = 0.20 + 0.50 * sky + 2.8 * sun
    b = 0.15 + 0.75 * sky + 2.5 * sun
    return torch.stack([r, g, b], dim=-1)


def project_to_sh(f_vals: torch.Tensor, bases: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """f_vals[nθ,nφ,D], bases[nθ,nφ,K], w[nθ,nφ] → 계수 [K, D]   (c_k = ∫ f Y_k dΩ)"""
    return torch.einsum("abk,abd,ab->kd", bases, f_vals, w)


def sphere_mean(f_vals: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """구면 평균 = ∫ f dΩ / 4π (격자 구적). f_vals[nθ,nφ,D] → [D]"""
    return (f_vals * w[..., None]).sum(dim=(0, 1)) / w.sum()


dirs, w = sphere_grid()
B = sh_bases(dirs, 3)  # [nθ, nφ, 16]
print("적분 가중치 합 Σw =", w.sum().item(), " / 4π =", 4 * math.pi)
# 출력: 적분 가중치 합 Σw = 12.56668758392334  / 4π = 12.566370614359172
# (128×256 격자 중점 구적의 float32 오차 ~3e-4; 아래 표의 1e-5 수준 차이는 모두 이 구적 오차)

# %% [markdown]
# ## 1. 1D 비유 — 푸리에 급수의 $a_0/2$ 는 구간 평균
#
# 주기 $2\pi$ 함수 $f(x)=\frac{a_0}{2}+\sum_{n\ge1}(a_n\cos nx+b_n\sin nx)$ 에서
# $a_0=\frac1\pi\int_0^{2\pi}f\,dx$ 이므로
#
# $$
# \frac{a_0}{2}=\frac{1}{2\pi}\int_0^{2\pi}f(x)\,dx=\overline f\quad(\text{구간 평균})
# $$
#
# $\cos nx,\sin nx$ ($n\ge1$)의 한 주기 적분은 0이라 평균에 기여하지 않는다.
# 구면에서는 주기 길이 $2\pi$ 자리에 표면적 $4\pi$ 가 들어간다.

# %%
x = torch.linspace(0, 2 * math.pi, 4001, device=DEVICE)
dx = x[1] - x[0]
f1d = 1.7 + 0.8 * torch.cos(x) - 0.5 * torch.sin(2 * x) + 0.3 * torch.cos(3 * x)  # 상수항 1.7

a0 = (f1d * 1.0).sum() * dx / math.pi               # a₀ = (1/π) ∫ f dx  (사다리꼴 대신 단순 리만합; 오차 ~1e-3)
interval_mean = f1d.sum() * dx / (2 * math.pi)      # (1/2π) ∫ f dx
print(f"a₀/2 = {a0.item() / 2:.5f}    구간 평균 = {interval_mean.item():.5f}    (참값 1.7)")
for n in (1, 2, 3):
    print(f"  ∫cos({n}x)dx / 2π = {(torch.cos(n * x).sum() * dx / (2 * math.pi)).item():+.5f}  ← 0")
# 출력: a₀/2 = 1.70070    구간 평균 = 1.70070    (참값 1.7)
# 출력:   ∫cos(1x)dx / 2π = +0.00025  ← 0
# 출력:   ∫cos(2x)dx / 2π = +0.00025  ← 0
# 출력:   ∫cos(3x)dx / 2π = +0.00025  ← 0
# (끝점을 한 번 더 세는 리만합 오차 ≈ 1/4000 = 0.00025)

# %% [markdown]
# ## 2. $Y_0^0=\dfrac{1}{2\sqrt\pi}$ 는 어디서 오나
#
# $Y_0^0$ 은 상수 $k$ 다. 정규 조건 $\int_{S^2}(Y_0^0)^2\,d\Omega=1$ 에 넣으면
#
# $$
# k^2\cdot\underbrace{\int_{S^2}1\,d\Omega}_{4\pi}=1\;\Longrightarrow\;k=\frac{1}{\sqrt{4\pi}}=\frac{1}{2\sqrt\pi}
# $$
#
# 그리고 $(Y_0^0)^2=\dfrac{1}{4\pi}$ 는 정확히 표면적의 역수다 — 이것이 "사영 두 번 = 평균"의 핵심.

# %%
print("1/(2√π)           =", 1 / (2 * math.sqrt(math.pi)))
print("C0 (코드 상수)     =", C0)
print("∫ (Y₀⁰)² dΩ (구적) =", (B[..., 0] ** 2 * w).sum().item(), " ← 1")
print("(Y₀⁰)²             =", C0 ** 2, "   1/(4π) =", 1 / (4 * math.pi))
# 출력: 1/(2√π)           = 0.28209479177387814
# 출력: C0 (코드 상수)     = 0.28209479177387814
# 출력: ∫ (Y₀⁰)² dΩ (구적) = 1.0000253915786743  ← 1
# 출력: (Y₀⁰)²             = 0.07957747154594766    1/(4π) = 0.07957747154594767

# %% [markdown]
# ## 3. 여러 함수에 대해 "구면 평균" vs "`coeffs[0] * C0`" 표
#
# $$
# c_0^0=\int f\,Y_0^0\,d\Omega=\frac{1}{2\sqrt\pi}\int f\,d\Omega
# \quad\Longrightarrow\quad
# c_0^0\,Y_0^0=\frac{1}{4\pi}\int f\,d\Omega=\overline f
# $$
#
# 해석적으로 아는 참값과 비교한다:
# - 상수 $2$: 평균 $2$
# - $z=\cos\theta$: 위·아래가 상쇄되어 평균 $0$
# - $z^2$: $\frac{1}{4\pi}\int\cos^2\theta\,d\Omega=\frac13$
# - $\max(z,0)$ (위쪽 반구만 밝음): $\frac{1}{4\pi}\int_{\text{상반구}}\cos\theta\,d\Omega=\frac14$
# - `env_radiance`: 참값은 모름 → 격자 평균과 DC 복원이 서로 같은지만 본다

# %%
z = dirs[..., 2]
cases = {
    "상수 2":       (torch.full_like(z, 2.0)[..., None],           2.0),
    "z":            (z[..., None],                                  0.0),
    "z²":           ((z ** 2)[..., None],                           1 / 3),
    "max(z,0)":     (torch.clamp_min(z, 0)[..., None],              0.25),
    "env (R)":      (env_radiance(dirs)[..., :1],                   None),
    "env (G)":      (env_radiance(dirs)[..., 1:2],                  None),
    "env (B)":      (env_radiance(dirs)[..., 2:3],                  None),
}
print(f"{'함수':<10} {'구면 평균(구적)':>16} {'c₀·C0':>12} {'차이':>10} {'참값':>8}")
for name, (fv, truth) in cases.items():
    c = project_to_sh(fv, B, w)                   # [16, 1]
    m = sphere_mean(fv, w).item()
    dc = (c[0, 0] * C0).item()
    print(f"{name:<10} {m:>16.6f} {dc:>12.6f} {abs(m - dc):>10.1e} {'' if truth is None else f'{truth:>8.4f}'}")
# 출력: 함수                구면 평균(구적)        c₀·C0         차이       참값
# 출력: 상수 2               2.000000     2.000052    5.2e-05   2.0000
# 출력: z                  0.000000    -0.000000    4.2e-09   0.0000
# 출력: z²                 0.333350     0.333357    7.4e-06   0.3333
# 출력: max(z,0)           0.250019     0.250025    6.3e-06   0.2500
# 출력: env (R)            0.454999     0.455009    9.5e-06
# 출력: env (G)            0.477999     0.478012    1.3e-05
# 출력: env (B)            0.549999     0.550015    1.5e-05
# → 모든 경우 "구면 평균 ≈ c₀·C0" (차이는 Σw≠4π 인 격자 구적 오차 수준), 해석 참값 2, 0, 1/3, 1/4 과도 일치

# %% [markdown]
# ## 4. 고차 항($\ell\ge1$)의 구면 평균은 0
#
# $Y_\ell^m$ ($\ell\ge1$)은 $Y_0^0$과 직교하므로
# $\int Y_\ell^m\,d\Omega=\frac{1}{Y_0^0}\int Y_\ell^m Y_0^0\,d\Omega=0$.
# 따라서 복원값에서 DC를 뺀 "고차 항만" 지도는 평균이 0이다 — 양의 영역과 음의 영역이 정확히 상쇄된다.

# %%
basis_means = sphere_mean(B, w)                    # 16개 기저 각각의 구면 평균
print("각 기저의 구면 평균 (k=0..15):")
print(basis_means)
print("  k=0 의 평균 = C0 =", C0, ",  k≥1 최대 |평균| =", basis_means[1:].abs().max().item())

f_env = env_radiance(dirs)                         # [nθ, nφ, 3]
coeffs_env = project_to_sh(f_env, B, w)            # [16, 3]
rec_full = torch.einsum("abk,kd->abd", B, coeffs_env)          # L=3 복원
rec_dc = (coeffs_env[0] * C0).expand_as(rec_full)              # DC만 (단색)
rec_high = rec_full - rec_dc                                   # 고차 항만
print("고차 항만의 구면 평균 (RGB):", sphere_mean(rec_high, w), " ← 0")
print("고차 항만의 최소/최대      :", rec_high.min().item(), rec_high.max().item(), " (양·음 모두 존재)")
# 출력: 각 기저의 구면 평균 (k=0..15):
# 출력: tensor([ 0.282095, -0.000000, -0.000000,  0.000000,  0.000000, -0.000000,  0.000016, -0.000000,  0.000000,  0.000000,
# 출력:          0.000000,  0.000000, -0.000000,  0.000000, -0.000000,  0.000000])
# 출력:   k=0 의 평균 = C0 = 0.28209479177387814 ,  k≥1 최대 |평균| = 1.5845549569348805e-05
# 출력: 고차 항만의 구면 평균 (RGB): tensor([0.000002, 0.000002, 0.000001])  ← 0
# 출력: 고차 항만의 최소/최대      : -0.4398643374443054 0.6639716625213623  (양·음 모두 존재)
# → k=0 만 평균이 C0 이고 나머지는 0 (k=6, Y₂⁰=√(5/16π)(3z²−1) 의 1.6e-5 는 θ 격자 구적 오차)

# %% [markdown]
# ## 5. $c_0$ 을 $\delta$ 만큼 바꾸면 평균은 $\delta\cdot C_0$ 만큼 바뀐다
#
# 평균은 $c_0$ 에 대해 선형이고 기울기가 $C_0=Y_0^0$ 이다. 3DGS 초기화 `sh0 = (rgb − 0.5)/C0` 는 이 관계를 거꾸로 쓴 것:
# 원하는 평균 색 `rgb − 0.5` 를 얻으려면 $c_0$ 을 $C_0$ 로 나눈 값으로 둔다.

# %%
print(f"{'δ (c₀ 변화)':>12} {'평균 변화(구적)':>16} {'δ·C0':>10}")
for delta in (-2.0, -1.0, 0.5, 1.0, 1.0 / C0):
    c2 = coeffs_env.clone()
    c2[0, 0] += delta                                              # R 채널 DC만 흔든다
    rec2 = torch.einsum("abk,kd->abd", B, c2)
    dmean = (sphere_mean(rec2, w) - sphere_mean(rec_full, w))[0].item()
    print(f"{delta:>12.4f} {dmean:>16.6f} {delta * C0:>10.6f}")
# 출력:  δ (c₀ 변화)    평균 변화(구적)       δ·C0
# 출력:      -2.0000        -0.564190  -0.564190
# 출력:      -1.0000        -0.282095  -0.282095
# 출력:       0.5000         0.141047   0.141047
# 출력:       1.0000         0.282095   0.282095
# 출력:       3.5449         1.000000   1.000000   ← δ = 1/C0 이면 평균이 정확히 1 증가

# %% [markdown]
# ## 6. 시각화 — 원본 / DC만 / 고차 항만 (등장방형 지도, 밝기 채널)
#
# 가로축 = 방위각 $\varphi$, 세로축 = 극각 $\theta$ (위 = 천정). 비교하기 쉽게 RGB 평균(밝기) 한 채널로 그린다.
#
# - 왼쪽: 원본 $f$ — 하늘 그라디언트 + 태양 봉우리
# - 가운데: DC 항만 $c_0^0Y_0^0$ — 온 구면이 **평균값 하나**로 칠해진 단색 지도
# - 오른쪽: 고차 항만 $f_{L=3}-c_0^0Y_0^0$ — 평균 0, 양(+)·음(−)이 상쇄되는 발산형 색 지도
#
# 왼쪽·가운데는 같은 색 스케일(순차형, 단일 색상)을 공유하고, 오른쪽은 0을 회색 중앙으로 둔 발산형 스케일을 쓴다.

# %%
lum = lambda img: img.mean(dim=-1).cpu().numpy()   # RGB → 밝기
img_f, img_dc, img_high = lum(f_env), lum(rec_dc), lum(rec_high)
mean_lum = float(img_dc[0, 0])

theta_deg = np.linspace(0, 180, img_f.shape[0])
phi_deg = np.linspace(-180, 180, img_f.shape[1])
zmax_seq = 1.5                                                 # 태양 봉우리는 클리핑(표시용 톤매핑)
zmax_div = float(np.abs(img_high).max())

fig = make_subplots(
    rows=1, cols=3, horizontal_spacing=0.06,
    subplot_titles=(
        "원본 f(d)  (밝기)",
        f"DC 항만: c₀·Y₀⁰ = 구면 평균 = {mean_lum:.3f}",
        "고차 항만 (ℓ≥1): 평균 0, +/− 상쇄",
    ),
)
seq_scale = "Blues"                                            # 순차형: 단일 색상, 밝음→어두움
div_scale = [[0.0, "#b2182b"], [0.5, "#e0e0e0"], [1.0, "#2166ac"]]   # 발산형: 두 색 + 회색 중앙
common = dict(x=phi_deg, y=theta_deg, hovertemplate="φ=%{x:.0f}°, θ=%{y:.0f}°<br>값=%{z:.3f}<extra></extra>")
fig.add_trace(go.Heatmap(z=img_f, zmin=0, zmax=zmax_seq, colorscale=seq_scale, showscale=True,
                         colorbar=dict(title="밝기", x=0.29, len=0.9, thickness=12), **common), row=1, col=1)
fig.add_trace(go.Heatmap(z=img_dc, zmin=0, zmax=zmax_seq, colorscale=seq_scale, showscale=False, **common), row=1, col=2)
fig.add_trace(go.Heatmap(z=img_high, zmin=-zmax_div, zmax=zmax_div, colorscale=div_scale, zmid=0, showscale=True,
                         colorbar=dict(title="편차", x=1.0, len=0.9, thickness=12), **common), row=1, col=3)
for c in (1, 2, 3):
    fig.update_xaxes(title_text="방위각 φ (°)", row=1, col=c, showgrid=False, zeroline=False)
    fig.update_yaxes(autorange="reversed", row=1, col=c, showgrid=False, zeroline=False)
fig.update_yaxes(title_text="극각 θ (°)", row=1, col=1)
fig.update_layout(
    title="DC 항 c₀·Y₀⁰ 은 함수의 구면 평균: f = (평균) + (평균 0인 고차 변동)",
    width=1500, height=460, template="plotly_white", margin=dict(l=60, r=40, t=90, b=60),
)
_show(fig)

png_path = os.path.join(HERE, "expy.png")
try:
    fig.write_image(png_path, scale=2)
    print("저장:", png_path)
except Exception as e:                                          # kaleido 미설치 등
    print("PNG 저장 실패:", e)
# 출력: 저장: .../7ca2e0fe-b3d3-4fb4-b0f3-fe480785718d/expy.png

# %% [markdown]
# ## 정리
#
# | 확인한 것 | 결과 |
# |---|---|
# | 1D: $a_0/2$ | 구간 평균과 일치 |
# | $\int(Y_0^0)^2d\Omega$ | $=1$, 따라서 $Y_0^0=1/(2\sqrt\pi)$, $(Y_0^0)^2=1/(4\pi)$ |
# | `coeffs[0]*C0` vs 격자 평균 | 상수·$z$·$z^2$·$\max(z,0)$·환경조명 모두 일치, 해석 참값($2,0,\tfrac13,\tfrac14$)과도 일치 |
# | $\ell\ge1$ 기저의 평균 | 모두 0 → 고차 항만의 지도는 평균 0, 양·음 상쇄 |
# | $c_0$ 변화 $\delta$ | 평균 변화 $=\delta\cdot C_0$ (선형), $\delta=1/C_0$ 이면 평균 +1 |
#
# **DC 항만으로 복원한 $c_0^0Y_0^0$은 함수의 구면 평균 $\overline f$** 다. 3DGS에서 `sh0 * C0 + 0.5` 가 Gaussian의 "방향 무관 기본색"인 이유.
