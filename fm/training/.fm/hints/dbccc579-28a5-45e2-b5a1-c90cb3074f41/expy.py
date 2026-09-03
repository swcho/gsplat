# %% [markdown]
# # `SH_DEGREE = 3` → SH 계수 16개, 직접 세어 보기
#
# 질문: **`SH_DEGREE = 3`일 때 spherical harmonics 계수는 몇 개인가?**
# 답: $(3+1)^2 = 16$개. 일반적으로 최대 차수 $d$에 대해 $(d+1)^2$개.
#
# 이 스크립트는 그 숫자를 네 가지 방법으로 확인한다.
#
# 1. 차수별 개수 $2\ell+1$을 누적해 $(d+1)^2$이 되는지 (등차수열의 합)
# 2. gsplat과 동일한 SH 기저 함수를 numpy로 재구현해 **실제로 16개가 나오는지**
# 3. 그 16개가 구면 위에서 **직교정규(orthonormal)** 기저인지 몬테카를로 적분으로
# 4. 차수를 0→3으로 올릴 때 뷰 의존 색이 얼마나 세밀해지는지
#
# 필요 패키지: numpy, plotly, kaleido (torch는 shape 확인에만 사용)
# gsplat 자체는 import하지 않는다 (JIT 빌드가 30분 이상 걸림) — 토이 재구현으로 대체.

# %%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


np.random.seed(42)
SH_DEGREE = 3  # training_walkthrough.py:68 과 동일
print("SH_DEGREE =", SH_DEGREE)
# 출력: SH_DEGREE = 3

# %% [markdown]
# ## 1단계: 차수별 개수 $2\ell+1$을 누적하면 $(d+1)^2$
#
# 구면조화함수 $Y_\ell^m$은 두 지수를 가진다.
#
# - 차수 $\ell = 0, 1, 2, \dots$ — 각도 방향으로 얼마나 잘게 흔들리는지
# - $m = -\ell, \dots, 0, \dots, +\ell$ — 그 차수 안에서의 종류
#
# $m$이 $-\ell$부터 $+\ell$까지이므로 차수 $\ell$의 함수 개수는 $2\ell+1$개(홀수).
# 0차부터 $d$차까지 모두 더하면 **첫째항 1, 공차 2, 항 개수 $d+1$인 등차수열의 합**이다.
#
# $$\sum_{\ell=0}^{d}(2\ell+1) = \frac{(d+1)\big(1+(2d+1)\big)}{2} = (d+1)^2$$

# %%
rows = []
cumulative = 0
for ell in range(SH_DEGREE + 1):
    per_degree = 2 * ell + 1  # m = -l ... +l
    cumulative += per_degree
    closed_form = (ell + 1) ** 2  # 등차수열 합 공식
    assert cumulative == closed_form
    rows.append((ell, list(range(-ell, ell + 1)), per_degree, cumulative, closed_form))

print(f"{'l':>2} {'m 범위':>22} {'2l+1':>5} {'누적':>5} {'(l+1)^2':>8}")
for ell, ms, per, cum, cf in rows:
    print(f"{ell:>2} {str(ms):>22} {per:>5} {cum:>5} {cf:>8}")
print()
print("1 + 3 + 5 + 7 =", sum(2 * l + 1 for l in range(4)))
print(f"K = (SH_DEGREE + 1)**2 = ({SH_DEGREE}+1)**2 =", (SH_DEGREE + 1) ** 2)
# 출력:
#  l                   m 범위  2l+1    누적  (l+1)^2
#  0                    [0]     1     1        1
#  1             [-1, 0, 1]     3     4        4
#  2      [-2, -1, 0, 1, 2]     5     9        9
#  3 [-3, -2, -1, 0, 1, 2, 3]     7    16       16
#
# 1 + 3 + 5 + 7 = 16
# K = (SH_DEGREE + 1)**2 = (3+1)**2 = 16

# %% [markdown]
# 홀수를 처음부터 더하면 완전제곱수가 되는 이유는 점을 정사각형으로 쌓아 보면 눈에 보인다.
# $\ell=0$은 점 1개, $\ell=1$은 그 옆에 ㄱ자로 3개, $\ell=2$는 다시 ㄱ자로 5개 …
# $4\times4$ 정사각형이 정확히 채워진다. (아래 7단계 그림의 ② 패널 — 오른쪽 위)
#
# **혼동 주의:** $d=3$에서
#
# - $2d+1 = 7$ → "3차 항만"의 개수
# - $(d+1)^2 = 16$ → "0차부터 3차까지 전부"의 개수 ← 이게 답

# %% [markdown]
# ## 2단계: gsplat의 SH 기저를 numpy로 재구현
#
# `gsplat/cuda/_torch_impl.py`의 `_eval_sh_bases_fast()`와 같은 상수를 쓴다
# (Sloan, *Efficient Spherical Harmonic Evaluation*, JCGT 2013).
# `basis_dim`에 $(d+1)^2$을 넘기면 그만큼의 기저값이 돌아온다.

# %%
def eval_sh_bases(basis_dim: int, dirs: np.ndarray) -> np.ndarray:
    """단위 방향 dirs [..., 3] 에서 SH 기저 basis_dim 개를 평가 → [..., basis_dim].

    basis_dim 은 1, 4, 9, 16 (= (d+1)^2) 중 하나여야 의미가 있다.
    """
    result = np.zeros((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype)
    result[..., 0] = 0.2820947917738781  # Y_0^0 = 1/(2 sqrt(pi))
    if basis_dim <= 1:
        return result

    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

    fTmpA = -0.48860251190292  # l = 1  → 3개
    result[..., 1] = fTmpA * y
    result[..., 2] = -fTmpA * z
    result[..., 3] = fTmpA * x
    if basis_dim <= 4:
        return result

    z2 = z * z  # l = 2  → 5개
    fTmpB = -1.092548430592079 * z
    fTmpA = 0.5462742152960395
    fC1 = x * x - y * y
    fS1 = 2 * x * y
    result[..., 4] = fTmpA * fS1
    result[..., 5] = fTmpB * y
    result[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    result[..., 7] = fTmpB * x
    result[..., 8] = fTmpA * fC1
    if basis_dim <= 9:
        return result

    fTmpC = -2.285228997322329 * z2 + 0.4570457994644658  # l = 3 → 7개
    fTmpB = 1.445305721320277 * z
    fTmpA = -0.5900435899266435
    fC2 = x * fC1 - y * fS1
    fS2 = x * fS1 + y * fC1
    result[..., 9] = fTmpA * fS2
    result[..., 10] = fTmpB * fS1
    result[..., 11] = fTmpC * y
    result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    result[..., 13] = fTmpC * x
    result[..., 14] = fTmpB * fC1
    result[..., 15] = fTmpA * fC2
    return result


# 임의의 단위 방향 하나에서 차수별로 몇 개가 나오는지
d_test = np.array([0.3, -0.5, 0.8])
d_test = d_test / np.linalg.norm(d_test)
for deg in range(4):
    K = (deg + 1) ** 2
    vals = eval_sh_bases(K, d_test)
    print(f"degree {deg}: basis_dim = (d+1)^2 = {K:2d}, 기저값 개수 = {vals.shape[-1]:2d}")
print()
print("degree 3 기저값 16개:")
print(np.round(eval_sh_bases(16, d_test), 4))
# 출력:
# degree 0: basis_dim = (d+1)^2 =  1, 기저값 개수 =  1
# degree 1: basis_dim = (d+1)^2 =  4, 기저값 개수 =  4
# degree 2: basis_dim = (d+1)^2 =  9, 기저값 개수 =  9
# degree 3: basis_dim = (d+1)^2 = 16, 기저값 개수 = 16
#
# degree 3 기저값 16개:
# [ 0.2821  0.2468  0.3949 -0.1481 -0.1672  0.4459  0.3025 -0.2676 -0.0892
#   0.0061 -0.3575  0.5229  0.08   -0.3138 -0.1907  0.1204]

# %% [markdown]
# ## 3단계: 이 16개가 정말 "직교정규 기저"인가?
#
# 기저가 쓸모 있으려면 서로 겹치지 않아야(직교) 하고 크기가 1이어야(정규) 한다.
# 구면 위 내적은 적분으로 정의된다.
#
# $$\langle Y_i, Y_j \rangle = \int_{S^2} Y_i(\mathbf{d})\,Y_j(\mathbf{d})\; d\Omega = \delta_{ij}$$
#
# 균일 분포로 뽑은 방향 $M$개로 몬테카를로 추정하면
# $\langle Y_i, Y_j\rangle \approx \frac{4\pi}{M}\sum_k Y_i(\mathbf{d}_k) Y_j(\mathbf{d}_k)$.
# 결과가 $16\times16$ 단위행렬에 가까우면 확인 완료.

# %%
def sample_uniform_sphere(M: int) -> np.ndarray:
    """구면 위 균일 샘플 [M,3]. z ~ U(-1,1), phi ~ U(0,2pi) 이면 균일하다."""
    z = np.random.uniform(-1.0, 1.0, M)
    phi = np.random.uniform(0.0, 2.0 * np.pi, M)
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


M = 400_000
dirs_mc = sample_uniform_sphere(M)
Y = eval_sh_bases(16, dirs_mc)  # [M,16]
gram = (4.0 * np.pi / M) * (Y.T @ Y)  # [16,16] ≈ I

print("Gram 행렬 shape:", gram.shape)
print("대각 성분(=norm^2) 최소/최대:", round(gram.diagonal().min(), 4), "/", round(gram.diagonal().max(), 4))
off = gram - np.diag(gram.diagonal())
print("비대각 성분 절댓값 최대:", round(np.abs(off).max(), 4))
print("||Gram - I||_max:", round(np.abs(gram - np.eye(16)).max(), 4))
# 출력:
# Gram 행렬 shape: (16, 16)
# 대각 성분(=norm^2) 최소/최대: 0.9985 / 1.0018
# 비대각 성분 절댓값 최대: 0.0032
# ||Gram - I||_max: 0.0032

# %% [markdown]
# 단위행렬과의 최대 오차가 0.01 이하 → **서로 직교하는 16개의 독립적인 함수**가 맞다.
# 개수를 셀 때 중복이 없다는 뜻이기도 하다.

# %% [markdown]
# ## 4단계: gsplat 파라미터 텐서 모양 — `sh0` + `shN` = 16
#
# `training_walkthrough.py`의 초기화 코드가 공식을 그대로 쓴다.
#
# ```python
# C0 = 0.28209479177387814
# colors = torch.zeros(N, (sh_degree + 1) ** 2, 3)   # [N, 16, 3]
# colors[:, 0, :] = (rgbs - 0.5) / C0                # DC 만 SfM 색으로
# splats["sh0"] = colors[:, :1, :]    # [N, 1, 3]   l=0
# splats["shN"] = colors[:, 1:, :]    # [N,15, 3]   l=1,2,3
# ```

# %%
import torch

torch.manual_seed(42)  # 재현성 (numpy 쪽은 상단에서 seed(42))

N = 1000  # Gaussian 개수 (실제 garden 씬은 약 14만개)
K = (SH_DEGREE + 1) ** 2
C0 = 0.28209479177387814

rgbs = torch.rand(N, 3)
colors = torch.zeros(N, K, 3)
colors[:, 0, :] = (rgbs - 0.5) / C0  # 0차만 초기화, 고차항은 0

sh0 = colors[:, :1, :].contiguous()  # [N, 1,3]
shN = colors[:, 1:, :].contiguous()  # [N,15,3]

print("colors :", tuple(colors.shape), "  ← [N, K, 3], K = (d+1)^2 =", K)
print("sh0    :", tuple(sh0.shape), "    ← l=0        계수", sh0.shape[1], "개")
print("shN    :", tuple(shN.shape), "    ← l=1,2,3    계수", shN.shape[1], "개")
print("합계   :", sh0.shape[1], "+", shN.shape[1], "=", sh0.shape[1] + shN.shape[1])
print("Gaussian 하나가 색에 쓰는 실수 개수:", K, "x 3 =", K * 3)
print("나머지 파라미터(means 3 + scales 3 + quats 4 + opacity 1) =", 3 + 3 + 4 + 1)
print(f"→ 색이 차지하는 비율: {K*3 / (K*3 + 11):.1%}")
# 출력:
# colors : (1000, 16, 3)   ← [N, K, 3], K = (d+1)^2 = 16
# sh0    : (1000, 1, 3)     ← l=0        계수 1 개
# shN    : (1000, 15, 3)     ← l=1,2,3    계수 15 개
# 합계   : 1 + 15 = 16
# Gaussian 하나가 색에 쓰는 실수 개수: 16 x 3 = 48
# 나머지 파라미터(means 3 + scales 3 + quats 4 + opacity 1) = 11
# → 색이 차지하는 비율: 81.4%

# %% [markdown]
# `C0 = 0.28209479177387814`은 $Y_0^0 = \dfrac{1}{2\sqrt{\pi}}$이다. 확인해 보자.
# gsplat은 최종 색을 $\mathrm{RGB} = \max\!\big(f(\mathbf d) + 0.5,\ 0\big)$로 만들기 때문에
# ($\texttt{rendering.py}$의 `clamp_min(features + 0.5, 0.0)`),
# 초기 색을 심으려면 $c_0 = (\mathrm{rgb} - 0.5)/C_0$로 역산해야 한다.

# %%
print("1/(2*sqrt(pi))      =", 1.0 / (2.0 * np.sqrt(np.pi)))
print("코드의 C0            =", C0)

# 0차 계수만 있는 상태에서 색을 복원하면 원래 rgb 가 나와야 한다 (방향과 무관하게)
dirs = sample_uniform_sphere(5)
Y1 = eval_sh_bases(K, dirs)  # [5,16]
#  [5,1,16] @ [5,16,3] → [5,1,3]  (기저 · 계수 = 색)
recon = (torch.from_numpy(Y1).float()[:, None, :] @ colors[:5]).squeeze(1)
recon = torch.clamp_min(recon + 0.5, 0.0)
print("원래 rgb[0]          =", np.round(rgbs[0].numpy(), 5))
print("SH 복원 rgb[0]       =", np.round(recon[0].numpy(), 5))
print("최대 오차            =", float((recon - rgbs[:5]).abs().max()))
# 출력:
# 1/(2*sqrt(pi))      = 0.28209479177387814
# 코드의 C0            = 0.28209479177387814
# 원래 rgb[0]          = [0.88227 0.915   0.38286]
# SH 복원 rgb[0]       = [0.88227 0.915   0.38286]
# 최대 오차            = 0.0

# %% [markdown]
# ## 5단계: 차수를 올리면 뷰 의존 색이 얼마나 세밀해지나
#
# 계수를 학습된 splat처럼 (고차항은 작게) 채워 놓고, 카메라가 Gaussian 주위를
# 한 바퀴 도는 방향 궤적에서 색을 평가한다. 활성 차수 $\ell_{\text{use}}$를 0→3으로
# 올리면 사용하는 계수가 $1 \to 4 \to 9 \to 16$개로 늘어난다.

# %%
K_full = 16
coef = np.zeros((K_full, 3))
coef[0] = (np.array([0.55, 0.40, 0.30]) - 0.5) / C0  # 기본색(흙빛)
for ell in range(1, 4):  # 고차항: 차수가 오를수록 진폭을 줄인다
    lo, hi = ell**2, (ell + 1) ** 2
    coef[lo:hi] = np.random.randn(hi - lo, 3) * (0.9 * 0.45**ell)

# 카메라가 고도 20도에서 방위각으로 한 바퀴 도는 궤적
azim = np.linspace(0, 2 * np.pi, 360)
elev = np.deg2rad(20.0)
traj = np.stack(
    [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.full_like(azim, np.sin(elev))],
    axis=-1,
)

curves = {}
for deg in range(4):
    n_used = (deg + 1) ** 2
    B = np.zeros((traj.shape[0], K_full))
    B[:, :n_used] = eval_sh_bases(n_used, traj)  # 활성 차수까지만 채우고 나머지는 0
    rgb = np.clip(B @ coef + 0.5, 0.0, 1.0)  # [360,3]
    curves[deg] = rgb
    print(
        f"degree {deg}: 사용 계수 {n_used:2d}개  "
        f"R 채널 범위 [{rgb[:,0].min():.3f}, {rgb[:,0].max():.3f}]  "
        f"R 변동폭(표준편차) {rgb[:, 0].std():.4f}"
    )
# 출력:
# degree 0: 사용 계수  1개  R 채널 범위 [0.550, 0.550]  R 변동폭(표준편차) 0.0000
# degree 1: 사용 계수  4개  R 채널 범위 [0.519, 0.600]  R 변동폭(표준편차) 0.0286
# degree 2: 사용 계수  9개  R 채널 범위 [0.421, 0.682]  R 변동폭(표준편차) 0.0865
# degree 3: 사용 계수 16개  R 채널 범위 [0.360, 0.754]  R 변동폭(표준편차) 0.1132

# %% [markdown]
# degree 0은 **완전히 평평한 직선**(방향과 무관한 상수색), 차수가 오를수록 곡선이
# 더 잘게 꺾인다. 이것이 반사·광택 같은 뷰 의존 효과를 표현하는 능력이다.

# %% [markdown]
# ## 6단계: 학습 중 차수 스케줄 — 16개를 처음부터 다 쓰지는 않는다
#
# ```python
# sh_degree_to_use = min(step // 1000, SH_DEGREE)
# ```
#
# **16은 "저장 공간"이고, 매 스텝 실제로 평가하는 개수는 $(\ell_{\text{use}}+1)^2$로
# 서서히 늘어난다.** 형상을 먼저 잡고 나중에 세부 반사를 허락하는 coarse-to-fine 전략.

# %%
print(f"{'step':>6} {'sh_degree_to_use':>17} {'활성 계수':>10} {'저장 계수':>10}")
for step in [0, 500, 1000, 1500, 2000, 2999, 3000, 5000, 30000]:
    deg = min(step // 1000, SH_DEGREE)
    print(f"{step:>6} {deg:>17} {(deg+1)**2:>10} {K:>10}")

# rendering.py 의 단언문: (sh_degree + 1)**2 <= K
for deg in [3, 4]:
    ok = (deg + 1) ** 2 <= K
    print(f"sh_degree={deg}: (d+1)^2={(deg+1)**2:3d} <= K={K} → {'통과' if ok else 'AssertionError'}")
# 출력:
#   step  sh_degree_to_use      활성 계수      저장 계수
#      0                 0          1         16
#    500                 0          1         16
#   1000                 1          4         16
#   1500                 1          4         16
#   2000                 2          9         16
#   2999                 2          9         16
#   3000                 3         16         16
#   5000                 3         16         16
#  30000                 3         16         16
# sh_degree=3: (d+1)^2= 16 <= K=16 → 통과
# sh_degree=4: (d+1)^2= 25 <= K=16 → AssertionError

# %% [markdown]
# ## 7단계: 시각화 (`expy.png`)
#
# 네 패널로 정리한다.
#
# 1. 차수별 개수 $2\ell+1$과 누적 $(d+1)^2$
# 2. 홀수의 합 = 완전제곱수: $1+3+5+7=16=4^2$ 점 쌓기
# 3. 활성 차수별 뷰 의존 R 채널 색 곡선
# 4. $16\times16$ Gram 행렬 $\approx I$ (직교정규 확인)

# %%
DEG_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "① 차수별 개수 2ℓ+1 과 누적 (d+1)²",
        "② 1+3+5+7 = 16 = 4²  (홀수의 합 = 완전제곱수)",
        "③ 활성 차수별 뷰 의존 색 (R 채널)",
        "④ Gram 행렬 ≈ I  (16개가 직교정규)",
    ),
    specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "heatmap"}]],
    vertical_spacing=0.14,
    horizontal_spacing=0.11,
)

# ① 막대: 차수별 2l+1, 선: 누적 (d+1)^2
ells = list(range(SH_DEGREE + 1))
fig.add_trace(
    go.Bar(
        x=ells,
        y=[2 * l + 1 for l in ells],
        marker_color=DEG_COLORS,
        hovertemplate="ℓ=%{x}: 2ℓ+1=%{y}<extra></extra>",
        name="차수별 2ℓ+1",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=ells,
        y=[(l + 1) ** 2 for l in ells],
        mode="lines+markers+text",
        text=[f" 누적 {(l+1)**2}" for l in ells],
        textposition="middle right",
        line=dict(color="#333", dash="dash"),
        marker=dict(size=9, symbol="diamond"),
        name="누적 (d+1)²",
    ),
    row=1,
    col=1,
)

# ② 4x4 정사각형 점 쌓기: 각 점을 자신이 속한 차수 색으로
sq_x, sq_y, sq_c, sq_t = [], [], [], []
for i in range(4):
    for j in range(4):
        ell = max(i, j)  # ㄱ자 껍질 = 차수
        sq_x.append(j)
        sq_y.append(3 - i)
        sq_c.append(DEG_COLORS[ell])
        sq_t.append(f"ℓ={ell}")
fig.add_trace(
    go.Scatter(
        x=sq_x,
        y=sq_y,
        mode="markers",
        marker=dict(size=34, color=sq_c, line=dict(color="white", width=2), symbol="square"),
        text=sq_t,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ),
    row=1,
    col=2,
)
for ell in ells:
    fig.add_annotation(
        x=3.62,
        y=3 - ell,
        xref="x2",
        yref="y2",
        text=f"  ℓ={ell}: +{2*ell+1}",
        showarrow=False,
        xanchor="left",
        font=dict(size=11, color=DEG_COLORS[ell]),
        row=1,
        col=2,
    )

# ③ 뷰 의존 색 곡선
for deg in range(4):
    fig.add_trace(
        go.Scatter(
            x=np.rad2deg(azim),
            y=curves[deg][:, 0],
            mode="lines",
            line=dict(color=DEG_COLORS[deg], width=2.2),
            name=f"ℓ_use={deg} ({(deg+1)**2}개)",
        ),
        row=2,
        col=1,
    )

# ④ Gram 행렬 히트맵
fig.add_trace(
    go.Heatmap(
        z=gram,
        zmin=-0.15,
        zmax=1.0,
        colorscale="Blues",
        showscale=True,
        colorbar=dict(len=0.36, y=0.16, thickness=12),
        hovertemplate="⟨Y_i,Y_j⟩=%{z:.3f}<extra></extra>",
    ),
    row=2,
    col=2,
)
# 차수 경계선 (1 / 4 / 9 / 16)
for b in [0.5, 3.5, 8.5]:
    fig.add_shape(type="line", x0=b, x1=b, y0=-0.5, y1=15.5, line=dict(color="#E45756", width=1.4), row=2, col=2)
    fig.add_shape(type="line", x0=-0.5, x1=15.5, y0=b, y1=b, line=dict(color="#E45756", width=1.4), row=2, col=2)

fig.update_xaxes(
    title_text="최대 차수 d (= ℓ)",
    tickvals=ells,
    ticktext=[f"{l}<br><span style='color:{DEG_COLORS[l]}'>2ℓ+1={2*l+1}</span>" for l in ells],
    row=1,
    col=1,
)
fig.update_yaxes(title_text="계수 개수", range=[0, 19], row=1, col=1)
fig.update_xaxes(visible=False, range=[-0.7, 5.3], row=1, col=2)
fig.update_yaxes(visible=False, scaleanchor="x2", scaleratio=1, range=[-0.7, 3.7], row=1, col=2)
fig.update_xaxes(title_text="카메라 방위각 (도)", dtick=90, row=2, col=1)
fig.update_yaxes(title_text="R 채널 값", range=[-0.03, 1.03], row=2, col=1)
fig.update_xaxes(title_text="기저 인덱스 j", dtick=5, row=2, col=2)
fig.update_yaxes(title_text="기저 인덱스 i", dtick=5, autorange="reversed", row=2, col=2)

fig.update_layout(
    title_text="SH_DEGREE = 3 → 계수 (3+1)² = 16개",
    height=880,
    width=1180,
    template="plotly_white",
    legend=dict(orientation="h", y=-0.06, x=0.0, font=dict(size=11)),
    font=dict(size=12),
    bargap=0.45,
)

_show(fig)
fig.write_image("expy.png", scale=2)  # kaleido 필요
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 정리
#
# | 확인 방법 | 결과 |
# |---|---|
# | 등차수열 합 $\sum_{\ell=0}^{3}(2\ell+1)$ | $1+3+5+7 = 16 = (3+1)^2$ |
# | SH 기저 재구현 `eval_sh_bases(16, d)` | 기저값 정확히 16개 |
# | 몬테카를로 직교성 검사 | $16\times16$ Gram $\approx I$ (오차 < 0.01) |
# | gsplat 텐서 모양 | `colors [N,16,3]` = `sh0 [N,1,3]` + `shN [N,15,3]` |
#
# - 차수 $\ell$ 하나에는 $m=-\ell,\dots,\ell$의 **$2\ell+1$개**
# - 0차부터 $d$차까지 누적하면 **$(d+1)^2$개**
# - `SH_DEGREE = 3` → **16개**, RGB 채널까지 곱하면 Gaussian당 48개 실수
# - 학습 중에는 `min(step // 1000, 3)`으로 활성 차수를 올려 $1 \to 4 \to 9 \to 16$개를 점진적으로 사용
