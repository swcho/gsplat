# %% [markdown]
# # SH 계수의 초기값은 어떻게 설정하는가?
#
# **결론**: DC(0차) 항만 SfM 색으로 $c_0 = (\text{rgb} - 0.5)/C_0$ 로 설정하고,
# 고차항 15개는 모두 0으로 둔다. 여기서 $C_0 = 1/(2\sqrt{\pi}) = 0.28209479177387814$.
#
# 이 스크립트는 gsplat을 import하지 않고 numpy/torch로 SH 기저와 색 평가를 직접
# 재구현해 다음을 단계적으로 확인한다.
#
# 1. $C_0$가 정규직교 조건 $\int_{S^2} (Y_0^0)^2 d\Omega = 1$ 에서 나온다는 것
# 2. `rgb_to_sh`가 렌더러 색 계산식 $\max(0,\ 0.5 + \sum_k c_k Y_k)$ 의 **역함수**라는 것
# 3. 고차항 0 $\Rightarrow$ 모든 방향에서 동일한 색(뷰 독립)이라는 것
# 4. 고차항을 랜덤 초기화하면 무엇이 망가지는지

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

C0_SRC = 0.28209479177387814  # gsplat/examples/utils.py 의 rgb_to_sh 하드코딩 값
C0_MATH = 1.0 / (2.0 * math.sqrt(math.pi))  # 정규직교 조건에서 유도되는 값

print(f"소스 하드코딩 C0 = {C0_SRC!r}")
print(f"1/(2*sqrt(pi))  = {C0_MATH!r}")
print(f"두 값이 같은가?  {C0_SRC == C0_MATH}")
print(f"참고: sqrt(pi) = {math.sqrt(math.pi):.10f}, 1/C0 = {1.0 / C0_SRC:.10f}")
# 출력: 소스 하드코딩 C0 = 0.28209479177387814
# 출력: 1/(2*sqrt(pi))  = 0.28209479177387814
# 출력: 두 값이 같은가?  True
# 출력: 참고: sqrt(pi) = 1.7724538509, 1/C0 = 3.5449077018

# %% [markdown]
# ## 1단계: $C_0$는 어디서 오는가 — 구면 위 정규직교 조건
#
# SH 기저는 구면 적분에 대해 정규직교하도록 스케일을 잡는다.
#
# $$\int_{S^2} Y_i(\mathbf d)\,Y_j(\mathbf d)\,d\Omega = \delta_{ij}$$
#
# $\ell = 0$ 기저는 상수함수 $Y_0^0 = C_0$ 하나뿐이고, 구 전체 입체각은 $4\pi$ 이므로
#
# $$C_0^2 \cdot 4\pi = 1 \quad\Longrightarrow\quad C_0 = \frac{1}{2\sqrt{\pi}}$$
#
# 구면 좌표 $(\theta, \phi)$ 격자에서 $d\Omega = \sin\theta\, d\theta\, d\phi$ 로 수치적분해 확인한다.

# %%
def sphere_quadrature(n_theta=400, n_phi=800, device="cpu"):
    """구면 위 균일 격자 방향과 그 입체각 가중치 dΩ = sinθ dθ dφ."""
    theta = (torch.arange(n_theta, device=device, dtype=torch.float64) + 0.5) * math.pi / n_theta
    phi = (torch.arange(n_phi, device=device, dtype=torch.float64) + 0.5) * 2 * math.pi / n_phi
    th, ph = torch.meshgrid(theta, phi, indexing="ij")
    dirs = torch.stack(
        [torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)], dim=-1
    )  # [n_theta, n_phi, 3] 단위벡터
    w = torch.sin(th) * (math.pi / n_theta) * (2 * math.pi / n_phi)  # [n_theta, n_phi]
    return dirs, w


dirs_q, w_q = sphere_quadrature()
print(f"방향 격자 shape = {tuple(dirs_q.shape)}, |d| 최대오차 = {(dirs_q.norm(dim=-1) - 1).abs().max():.2e}")
print(f"∫ dΩ (수치)  = {w_q.sum():.8f}")
print(f"4π (해석)     = {4 * math.pi:.8f}")

# Y00 = C0 (상수) 를 넣고 정규화 적분값 확인
norm_sq = (C0_MATH**2 * w_q).sum().item()
print(f"∫ (Y00)^2 dΩ = {norm_sq:.8f}  (1이어야 함)")
# 역산: 적분이 1이 되게 하는 상수 c
c_solved = (1.0 / w_q.sum().sqrt()).item()
print(f"조건에서 역산한 C0 = {c_solved!r}  (오차 {abs(c_solved - C0_SRC):.2e})")
# 출력: 방향 격자 shape = (400, 800, 3), |d| 최대오차 = 2.22e-16
# 출력: ∫ dΩ (수치)  = 12.56640291
# 출력: 4π (해석)     = 12.56637061
# 출력: ∫ (Y00)^2 dΩ = 1.00000257  (1이어야 함)
# 출력: 조건에서 역산한 C0 = 0.2820944292525708  (오차 3.63e-07)

# %% [markdown]
# 격자 이산화 오차 정도(1e-7)로 일치한다. **`0.2821`은 마법의 숫자가 아니라 $1/(2\sqrt\pi)$** 다.
#
# ## 2단계: SH 기저 16개 (degree 0..3) 재구현
#
# gsplat CUDA 커널(`SHCommon.h`)과 동일한 실수 SH 기저를 파이썬으로 옮긴다.
# 첫 번째 기저 `Y[..., 0]`이 상수 $C_0$ 인지 확인하는 것이 핵심이다.

# %%
_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
_C3 = [
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
    -0.4570457994644658, 1.445305721320277, -0.5900435899266435,
]


def sh_basis(dirs: torch.Tensor, degree: int = 3) -> torch.Tensor:
    """단위벡터 dirs [..., 3] -> SH 기저값 [..., (degree+1)^2]."""
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    out = [torch.full_like(x, _C0)]  # ℓ=0: 상수!
    if degree >= 1:
        out += [-_C1 * y, _C1 * z, -_C1 * x]
    if degree >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        out += [_C2[0] * xy, _C2[1] * yz, _C2[2] * (2 * zz - xx - yy), _C2[3] * xz, _C2[4] * (xx - yy)]
    if degree >= 3:
        out += [
            _C3[0] * y * (3 * xx - yy), _C3[1] * xy * z, _C3[2] * y * (4 * zz - xx - yy),
            _C3[3] * z * (2 * zz - 3 * xx - 3 * yy), _C3[4] * x * (4 * zz - xx - yy),
            _C3[5] * z * (xx - yy), _C3[6] * x * (xx - 3 * yy),
        ]
    return torch.stack(out, dim=-1)


Y_q = sh_basis(dirs_q, degree=3)  # [400, 800, 16]
print(f"기저 개수 = {Y_q.shape[-1]}  ((3+1)^2 = {(3 + 1) ** 2})")
print(f"Y[...,0] 최소 = {Y_q[..., 0].min():.17f}")
print(f"Y[...,0] 최대 = {Y_q[..., 0].max():.17f}  -> 방향과 무관한 상수")

# 정규직교성 수치 확인: G[i,j] = ∫ Yi Yj dΩ 가 단위행렬인가
G = torch.einsum("abi,abj,ab->ij", Y_q, Y_q, w_q)
print(f"대각 성분 범위 = [{G.diag().min():.6f}, {G.diag().max():.6f}]  (1이어야 함)")
off = G - torch.diag(G.diag())
print(f"비대각 최대 절댓값 = {off.abs().max():.2e}  (0이어야 함)")
# 출력: 기저 개수 = 16  ((3+1)^2 = 16)
# 출력: Y[...,0] 최소 = 0.28209479177387814
# 출력: Y[...,0] 최대 = 0.28209479177387814  -> 방향과 무관한 상수
# 출력: 대각 성분 범위 = [1.000000, 1.000018]  (1이어야 함)
# 출력: 비대각 최대 절댓값 = 1.18e-05  (0이어야 함)

# %% [markdown]
# ## 3단계: 렌더러의 색 계산식과 그 역함수
#
# gsplat `rendering.py:711-718` 이 하는 일:
#
# $$\text{rgb}(\mathbf d) = \max\!\Big(0,\ 0.5 + \sum_{k=0}^{15} \mathbf c_k\,Y_k(\mathbf d)\Big)$$
#
# 고차항을 0으로 두면 $Y_0 = C_0$ 만 남아 $\text{rgb} = 0.5 + c_0 C_0$ (방향 무관).
# 이를 $c_0$ 에 대해 푼 일차방정식이 곧 초기화 공식이다.
#
# $$c_0 = \frac{\text{rgb} - 0.5}{C_0}$$

# %%
def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """gsplat examples/utils.py:163 과 동일 — 렌더 식의 역함수."""
    return (rgb - 0.5) / _C0


def eval_sh_color(coeffs: torch.Tensor, dirs: torch.Tensor, degree: int, clamp: bool = True):
    """coeffs [N,K,3], dirs [N,3] -> rgb [N,3]. rendering.py 의 shift_relu 경로."""
    K = (degree + 1) ** 2
    feat = torch.einsum("nk,nkc->nc", sh_basis(dirs, degree), coeffs[:, :K, :])
    feat = feat + 0.5
    return torch.clamp_min(feat, 0.0) if clamp else feat


# 대표적인 색들에 대해 왕복(round-trip) 검증
names = ["검정", "중간회색", "흰색", "빨강", "SfM샘플"]
rgb_in = torch.tensor(
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.42, 0.61, 0.33]]
)
c0 = rgb_to_sh(rgb_in)  # [5,3]

coeffs = torch.zeros(len(rgb_in), 16, 3)  # 고차항 = 0
coeffs[:, 0, :] = c0  # DC만 채운다

d = torch.nn.functional.normalize(torch.randn(len(rgb_in), 3), dim=-1)  # 아무 시선 방향
rgb_out = eval_sh_color(coeffs, d, degree=3)

print(f"{'색':>9} | {'입력 rgb':>21} | {'c0 = (rgb-0.5)/C0':>27} | {'복원 rgb':>21}")
for n, a, b, c in zip(names, rgb_in, c0, rgb_out):
    print(f"{n:>8} | [{a[0]:5.2f} {a[1]:5.2f} {a[2]:5.2f}] | [{b[0]:8.4f} {b[1]:8.4f} {b[2]:8.4f}] | [{c[0]:5.2f} {c[1]:5.2f} {c[2]:5.2f}]")
print(f"왕복 최대오차 = {(rgb_out - rgb_in).abs().max():.3e}")
print(f"c0 이론 경계 ±sqrt(pi) = ±{math.sqrt(math.pi):.4f}, 실제 최대 절댓값 = {c0.abs().max():.4f}")
# 출력:         색 |                입력 rgb |           c0 = (rgb-0.5)/C0 |                복원 rgb
# 출력:       검정 | [ 0.00  0.00  0.00] | [ -1.7725  -1.7725  -1.7725] | [ 0.00  0.00  0.00]
# 출력:     중간회색 | [ 0.50  0.50  0.50] | [  0.0000   0.0000   0.0000] | [ 0.50  0.50  0.50]
# 출력:       흰색 | [ 1.00  1.00  1.00] | [  1.7725   1.7725   1.7725] | [ 1.00  1.00  1.00]
# 출력:       빨강 | [ 1.00  0.00  0.00] | [  1.7725  -1.7725  -1.7725] | [ 1.00  0.00  0.00]
# 출력:    SfM샘플 | [ 0.42  0.61  0.33] | [ -0.2836   0.3899  -0.6026] | [ 0.42  0.61  0.33]
# 출력: 왕복 최대오차 = 0.000e+00
# 출력: c0 이론 경계 ±sqrt(pi) = ±1.7725, 실제 최대 절댓값 = 1.7725

# %% [markdown]
# 초기 렌더는 SfM 색을 **정확히** 재현한다. 그리고 계수의 크기는 $\pm\sqrt{\pi} \approx \pm 1.77$
# 를 넘지 않는다 — 학습 중 `sh0`가 수십으로 튄다면 이상 신호라는 감각을 여기서 얻는다.
#
# ## 4단계: 고차항 0 $\Rightarrow$ 뷰 독립 (분산이 정확히 0)
#
# 여러 방향에서 같은 Gaussian을 평가해 색이 변하는지 본다.

# %%
n_dirs = 2000
dirs_many = torch.nn.functional.normalize(torch.randn(n_dirs, 3), dim=-1)
one_rgb = torch.tensor([[0.42, 0.61, 0.33]])
co_dc = torch.zeros(1, 16, 3)
co_dc[:, 0, :] = rgb_to_sh(one_rgb)

cols_dc = eval_sh_color(co_dc.expand(n_dirs, -1, -1), dirs_many, degree=3)
print("[DC만] 방향별 색 통계")
print(f"  평균     = {cols_dc.mean(0).tolist()}")
print(f"  표준편차 = {cols_dc.std(0).tolist()}")
print(f"  최대-최소 = {(cols_dc.max(0).values - cols_dc.min(0).values).abs().max():.3e}")

# 대비군: 고차항을 표준정규로 랜덤 초기화했다면?
co_rand = co_dc.clone().expand(n_dirs, -1, -1).clone()
co_rand[:, 1:, :] = torch.randn(1, 15, 3)  # 모든 방향에 대해 같은(고정) 랜덤 계수
cols_rand = eval_sh_color(co_rand, dirs_many, degree=3)
clamped = (torch.einsum("nk,nkc->nc", sh_basis(dirs_many, 3), co_rand) + 0.5 < 0).float().mean()
print("[고차항 랜덤] 방향별 색 통계")
print(f"  평균     = {cols_rand.mean(0).tolist()}")
print(f"  표준편차 = {cols_rand.std(0).tolist()}")
print(f"  clamp(0)에 걸린 채널 비율 = {clamped * 100:.1f}%  <- 이 채널은 gradient가 0")
# 출력: [DC만] 방향별 색 통계
# 출력:   평균     = [0.4200000464916229, 0.6100000143051147, 0.3299999535083771]
# 출력:   표준편차 = [0.0, 0.0, 0.0]
# 출력:   최대-최소 = 0.000e+00
# 출력: [고차항 랜덤] 방향별 색 통계
# 출력:   평균     = [0.717612087726593, 1.0309618711471558, 0.6656113862991333]
# 출력:   표준편차 = [0.8683663010597229, 0.9931728839874268, 0.8098597526550293]
# 출력:   clamp(0)에 걸린 채널 비율 = 35.7%  <- 이 채널은 gradient가 0

# %% [markdown]
# **표준편차가 정확히 0** — 고차항 0은 "모든 방향에서 같은 색"(램버시안 가정)을 뜻한다.
# 반면 고차항을 랜덤 초기화하면 색이 방향마다 요란하게 출렁이고, 약 1/4의 채널이
# $\max(0,\cdot)$ clamp에 걸려 **기울기가 0인 죽은 계수**가 된다.
#
# ## 5단계: DC 항은 "구면 평균 밝기"다
#
# 정규직교성 덕분에 계수는 내적으로 뽑힌다: $c_k = \int_{S^2} c(\mathbf d) Y_k(\mathbf d)\,d\Omega$.
# $k=0$ 에 넣으면 $Y_0 = C_0$ 가 상수이므로
#
# $$c_0 = C_0 \int_{S^2} c(\mathbf d)\,d\Omega = 4\pi C_0 \cdot \bar c$$
#
# 즉 DC는 방향 평균, 고차항은 그 평균으로부터의 편차다. 임의의 방향 의존 색함수에
# 이를 적용해 확인한다.

# %%
# 임의의 뷰 의존 색: 정면 하이라이트 + 완만한 배경 (오프셋 0.5 제외한 순수 신호)
ref_dir = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
cos_t = (dirs_q * ref_dir).sum(-1).clamp(-1, 1)
signal = 0.3 + 0.5 * cos_t.clamp_min(0) ** 8  # [400,800] 하이라이트가 있는 함수

mean_val = ((signal * w_q).sum() / w_q.sum()).item()  # 구면 평균
c0_proj = (signal * Y_q[..., 0] * w_q).sum().item()  # 내적으로 뽑은 c0
print(f"구면 평균  c̄            = {mean_val:.8f}")
print(f"내적 c0 = ∫ c·Y00 dΩ    = {c0_proj:.8f}")
print(f"4π·C0·c̄ (예측)          = {4 * math.pi * C0_MATH * mean_val:.8f}")
print(f"c0 / (4π·C0)            = {c0_proj / (4 * math.pi * C0_MATH):.8f}  <- c̄ 와 일치")

# 고차항 계수 크기: 방향 의존성이 강할수록 커진다
c_all = torch.einsum("ab,abk,ab->k", signal, Y_q, w_q)
print(f"|c0| = {abs(c_all[0]):.4f}, |c1..c15| 최대 = {c_all[1:].abs().max():.4f}")
# 출력: 구면 평균  c̄            = 0.32777835
# 출력: 내적 c0 = ∫ c·Y00 dΩ    = 1.16194698
# 출력: 4π·C0·c̄ (예측)          = 1.16194399
# 출력: c0 / (4π·C0)            = 0.32777919  <- c̄ 와 일치
# 출력: |c0| = 1.1619, |c1..c15| 최대 = 0.1601

# %% [markdown]
# ## 6단계: 실제 초기화 루틴 재현 (`sh0` / `shN` 분리)
#
# 워크스루의 `init_splats_with_optimizers`와 같은 방식으로 가짜 SfM 포인트 클라우드에서
# 초기화한다. gsplat은 계수를 `sh0` `[N,1,3]`(DC)와 `shN` `[N,15,3]`(고차)로 나눠 저장한다.

# %%
N, SH_DEGREE = 5000, 3
points_rgb = np.random.randint(0, 256, size=(N, 3), dtype=np.uint8)  # SfM 색 (0..255)
rgbs = torch.from_numpy(points_rgb / 255.0).float()

colors = torch.zeros(N, (SH_DEGREE + 1) ** 2, 3)  # [N,16,3] 전부 0
colors[:, 0, :] = (rgbs - 0.5) / _C0  # DC만 SfM 색으로

sh0 = torch.nn.Parameter(colors[:, :1, :].contiguous())  # .contiguous(): fused Adam 요구사항
shN = torch.nn.Parameter(colors[:, 1:, :].contiguous())

print(f"sh0 shape = {tuple(sh0.shape)}, shN shape = {tuple(shN.shape)}")
print(f"shN 이 전부 0인가? {bool((shN == 0).all())}")
print(f"sh0 범위 = [{sh0.min():.4f}, {sh0.max():.4f}]  (이론: ±{math.sqrt(math.pi):.4f})")
print(f"학습률: sh0 = {2.5e-3:.1e}, shN = {2.5e-3 / 20:.3e}  (고차 SH는 1/20 속도)")

# 초기 렌더가 SfM 색을 복원하는가 (degree 0 으로만 평가 — 워밍업 초기 상태)
cat = torch.cat([sh0, shN], dim=1)  # rasterize_splats 가 하는 concat
d_rand = torch.nn.functional.normalize(torch.randn(N, 3), dim=-1)
rec0 = eval_sh_color(cat, d_rand, degree=0)
rec3 = eval_sh_color(cat, d_rand, degree=3)
print(f"degree 0 복원 최대오차 = {(rec0 - rgbs).abs().max():.3e}")
print(f"degree 3 복원 최대오차 = {(rec3 - rgbs).abs().max():.3e}  (고차항 0이라 동일)")

# 차수 워밍업 스케줄
print("SH 차수 스케줄  min(step // 1000, 3):")
print("  " + ", ".join(f"step {s}->deg {min(s // 1000, SH_DEGREE)}" for s in [0, 500, 1000, 2000, 3000, 5000]))
# 출력: sh0 shape = (5000, 1, 3), shN shape = (5000, 15, 3)
# 출력: shN 이 전부 0인가? True
# 출력: sh0 범위 = [-1.7725, 1.7725]  (이론: ±1.7725)
# 출력: 학습률: sh0 = 2.5e-03, shN = 1.250e-04  (고차 SH는 1/20 속도)
# 출력: degree 0 복원 최대오차 = 5.960e-08
# 출력: degree 3 복원 최대오차 = 5.960e-08  (고차항 0이라 동일)
# 출력: SH 차수 스케줄  min(step // 1000, 3):
# 출력:   step 0->deg 0, step 500->deg 0, step 1000->deg 1, step 2000->deg 2, step 3000->deg 3, step 5000->deg 3

# %% [markdown]
# ## 7단계: 시각화
#
# 1. **rgb $\to c_0$ 선형사상**: 기울기 $1/C_0 = 2\sqrt\pi \approx 3.5449$, $\text{rgb}=0.5$ 에서 0을 지난다
# 2. **초기화된 SfM 색 vs 계수 분포**: 히스토그램이 $\pm\sqrt\pi$ 안에 갇힌다
# 3. **DC만 (올바른 초기화)**: 구면 위 색이 완전히 균일 — 뷰 독립
# 4. **고차항 랜덤 (잘못된 초기화)**: 방향마다 출렁이고 clamp에 걸린 검은 영역 발생

# %%
# 방향 격자: θ(극각) 0..180, φ(방위각) -180..180 을 단조 증가 좌표로 직접 만든다
# (heatmap 축은 단조여야 한다 — arctan2 로 φ를 되돌리면 ±180 에서 순서가 깨져 빈 그림이 나온다)
th_deg = np.linspace(1, 179, 90)
ph_deg = np.linspace(-179, 179, 180)
TH, PH = np.meshgrid(np.radians(th_deg), np.radians(ph_deg), indexing="ij")
grid_dirs = torch.from_numpy(
    np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1)
).float()  # [90, 180, 3]

base_rgb = torch.tensor([[0.42, 0.61, 0.33]])
co_a = torch.zeros(1, 16, 3)
co_a[:, 0, :] = rgb_to_sh(base_rgb)  # 올바른 초기화: DC만
co_b = co_a.clone()
co_b[:, 1:, :] = torch.randn(1, 15, 3)  # 잘못된 초기화: 고차항 랜덤

flat = grid_dirs.reshape(-1, 3)
lum_a = eval_sh_color(co_a.expand(len(flat), -1, -1), flat, 3).mean(-1).reshape(90, 180).numpy()
lum_b = eval_sh_color(co_b.expand(len(flat), -1, -1), flat, 3).mean(-1).reshape(90, 180).numpy()

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "① rgb → c₀ 선형사상 (기울기 1/C₀ = 2√π = 3.5449)",
        "② 초기화된 계수 분포: sh0(청록) vs shN(주황, 전부 0)",
        "③ DC만 (올바름): 구면 위 밝기가 완전 균일 = 뷰 독립",
        "④ 고차항 랜덤 (나쁨): 방향마다 출렁 + 검은 clamp 영역",
    ),
    vertical_spacing=0.15, horizontal_spacing=0.10,
)

# ① 선형사상
r = np.linspace(0, 1, 201)
fig.add_trace(go.Scatter(x=r, y=(r - 0.5) / _C0, mode="lines",
                         line=dict(color="#2563eb", width=3), showlegend=False), row=1, col=1)
fig.add_hline(y=math.sqrt(math.pi), line=dict(color="#dc2626", dash="dash", width=1), row=1, col=1,
              annotation_text="+√π = 1.7725", annotation_position="bottom right")
fig.add_hline(y=-math.sqrt(math.pi), line=dict(color="#dc2626", dash="dash", width=1), row=1, col=1,
              annotation_text="−√π = −1.7725", annotation_position="top right")
fig.add_trace(go.Scatter(x=[0.5], y=[0.0], mode="markers+text", text=["  rgb=0.5 → c₀=0 (중간 회색)"],
                         textposition="middle right", marker=dict(size=11, color="#dc2626"),
                         showlegend=False), row=1, col=1)
fig.update_xaxes(title_text="SfM rgb (0~1)", row=1, col=1)
fig.update_yaxes(title_text="DC 계수 c₀", range=[-2.4, 2.4], row=1, col=1)

# ② 계수 히스토그램 (범례 대신 소제목/주석으로 구분 — 범례가 소제목을 가린다)
fig.add_trace(go.Histogram(x=sh0.detach().reshape(-1).numpy(), nbinsx=60, opacity=0.8,
                           marker_color="#0d9488", showlegend=False), row=1, col=2)
fig.add_trace(go.Histogram(x=shN.detach().reshape(-1).numpy(), nbinsx=60, opacity=0.8,
                           marker_color="#f59e0b", showlegend=False), row=1, col=2)
fig.add_annotation(row=1, col=2, x=0.0, y=math.log10(shN.numel()), yanchor="bottom",
                   text=f"shN 전체 {shN.numel():,}개가<br>정확히 0 (델타)", showarrow=True,
                   arrowhead=2, ax=55, ay=-25, font=dict(color="#b45309", size=11))
fig.add_annotation(row=1, col=2, x=-1.35, y=math.log10(30), text="sh0 ∈ [−√π, √π]",
                   showarrow=False, font=dict(color="#0f766e", size=11))
fig.update_xaxes(title_text="계수값", range=[-2.4, 2.4], row=1, col=2)
fig.update_yaxes(title_text="개수 (log)", type="log", row=1, col=2)

# ③④ 구면 밝기 heatmap (공통 색범위)
zmax = float(max(lum_a.max(), lum_b.max()))
for col, lum in [(1, lum_a), (2, lum_b)]:
    fig.add_trace(go.Heatmap(z=lum, x=ph_deg, y=th_deg, colorscale="Viridis",
                             zmin=0.0, zmax=zmax, showscale=(col == 2),
                             colorbar=dict(title="밝기", len=0.36, y=0.17)), row=2, col=col)
    fig.update_xaxes(title_text="방위각 φ (deg)", row=2, col=col)
    fig.update_yaxes(title_text="극각 θ (deg)", autorange="reversed", row=2, col=col)
fig.add_annotation(row=2, col=1, x=0, y=90, text=f"모든 방향에서 밝기 = {lum_a.mean():.4f}<br>(표준편차 = 0)",
                   showarrow=False, font=dict(color="white", size=13))

fig.update_layout(
    title_text="SH 계수 초기화: DC = (rgb−0.5)/C₀,  고차항 = 0   (C₀ = 1/(2√π) = 0.2820947917738781)",
    height=840, width=1180, barmode="overlay", template="plotly_white",
    showlegend=False, font=dict(size=12), margin=dict(t=95, b=60),
)

_show(fig)
fig.write_image("expy.png", scale=2)
print("expy.png 저장 완료")
print(f"③ DC만 밝기 범위 = [{lum_a.min():.4f}, {lum_a.max():.4f}]  (폭 {lum_a.max() - lum_a.min():.1e})")
print(f"④ 랜덤 밝기 범위 = [{lum_b.min():.4f}, {lum_b.max():.4f}]  (폭 {lum_b.max() - lum_b.min():.4f})")
# 출력: expy.png 저장 완료
# 출력: ③ DC만 밝기 범위 = [0.4533, 0.4533]  (폭 0.0e+00)
# 출력: ④ 랜덤 밝기 범위 = [0.0000, 1.8041]  (폭 1.8041)

# %% [markdown]
# ## 정리
#
# | 항목 | 값 / 이유 |
# |---|---|
# | $C_0$ | $1/(2\sqrt\pi) = 0.28209479177387814$ — 정규직교 조건 $C_0^2 \cdot 4\pi = 1$ |
# | DC 초기값 | $c_0 = (\text{rgb}_{\text{SfM}} - 0.5)/C_0$ — 렌더 식 $\max(0, 0.5 + \sum c_k Y_k)$ 의 역함수 |
# | $-0.5$ 의 정체 | 렌더러가 더하는 $+0.5$ 오프셋의 상쇄 (계수 0 = 중간 회색) |
# | 고차항 초기값 | $0$ — 램버시안(뷰 독립) 가정; SfM은 방향 정보를 주지 않음 |
# | 계수 크기 감각 | $\text{rgb} \in [0,1] \Rightarrow c_0 \in [-\sqrt\pi, \sqrt\pi] \approx [-1.77, 1.77]$ |
# | 안전장치 | 차수 워밍업 `min(step//1000, 3)`, `shN` lr = `sh0` lr / 20 |
