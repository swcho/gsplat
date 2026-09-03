# %% [markdown]
# # SH 계수를 관측으로부터 구하는 두 방법 — A: 최소제곱(닫힌 해) vs B: Adam 경사하강
#
# 정답 함수 $f^\star(\mathbf d)$(확산 기본색 + 광택 로브)를 $n_{\text{view}}$개 방향에서 관측하고,
# 3차 SH(16개 기저) 계수 $\mathbf c\in\mathbb R^{16\times 3}$를 두 방식으로 맞춘 뒤 비교한다.
#
# - **방법 A** — 최소제곱: $\min_{\mathbf c}\sum_j\|Y(\mathbf d_j)\mathbf c-f^\star(\mathbf d_j)\|^2$.
#   행렬 $A=[Y_k(\mathbf d_j)]\in\mathbb R^{n\times 16}$로 쓰면 $\min_{\mathbf c}\|A\mathbf c-\mathbf y\|^2$이고
#   정규방정식 $(A^\top A)\mathbf c=A^\top\mathbf y$의 해로 **한 번에** 구해진다.
# - **방법 B** — Adam: 3DGS 학습과 같은 설정(sh0/shN 학습률 분리, 차수 램프업, `+0.5`와 `clamp`, L1 손실)으로 반복 갱신.
#
# 필요 패키지: torch, numpy, plotly, kaleido

# %%
import math
import os

import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEVICE = "cpu"
torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# %% [markdown]
# ## 1. 준비 — SH 기저, 구면 격자, 정답 함수 (노트북 4.2와 동일)

# %%
C0 = 0.28209479177387814  # Y₀⁰ = 1/(2√π)
LM = [(l, m) for l in range(4) for m in range(-l, l + 1)]


def sh_bases(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """단위 방향 dirs[..., 3] → SH 기저값 [..., (degree+1)²] (gsplat 순서·부호)."""
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


def sphere_grid(n_theta=128, n_phi=256, device=DEVICE):
    """구면 (θ, φ) 격자 방향 d[nθ,nφ,3]와 적분 가중치 w = sinθ dθ dφ (Σw ≈ 4π)."""
    theta = (torch.arange(n_theta, device=device) + 0.5) * math.pi / n_theta
    phi = (torch.arange(n_phi, device=device) + 0.5) * 2 * math.pi / n_phi - math.pi
    th, ph = torch.meshgrid(theta, phi, indexing="ij")
    d = torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], dim=-1)
    w = th.sin() * (math.pi / n_theta) * (2 * math.pi / n_phi)
    return d, w


def f_star(d: torch.Tensor) -> torch.Tensor:
    """정답 시점 의존 색: 기본색 + 한 방향으로의 넓은 광택 로브."""
    h = F.normalize(torch.tensor([-0.4, 0.6, 0.7], device=d.device), dim=0)
    spec = torch.clamp_min(d @ h, 0) ** 8
    base = torch.tensor([0.6, 0.35, 0.2], device=d.device)
    return base + 0.6 * spec[..., None]


dirs, w = sphere_grid()
B = sh_bases(dirs, 3)          # [128, 256, 16]
f_true = f_star(dirs)          # [128, 256, 3]


def sphere_mse(coeffs, degree=3):
    rec = torch.einsum("abk,kc->abc", B[..., :(degree + 1) ** 2], coeffs)
    return (((rec - f_true) ** 2) * w[..., None]).sum() / w.sum()


print("기저 텐서:", tuple(B.shape), " Σw =", round(w.sum().item(), 5), "(4π =", round(4 * math.pi, 5), ")")
# 출력: 기저 텐서: (128, 256, 16)  Σw = 12.56669 (4π = 12.56637 )

# %% [markdown]
# ## 2. 관측 60개 생성
#
# 카메라 60대가 서로 다른 방향에서 이 Gaussian을 본다고 가정한다. 관측은 $(\mathbf d_j, f^\star(\mathbf d_j))$ 쌍이다.
# $A$의 shape은 `[n_view, K]` = `[60, 16]`, 관측 행렬 $\mathbf y$는 `[60, 3]`이다.

# %%
n_view = 60
obs_d = F.normalize(torch.randn(n_view, 3, device=DEVICE), dim=-1)   # [60, 3]
obs_c = f_star(obs_d)                                                # [60, 3]
A = sh_bases(obs_d, 3)                                               # [60, 16]
print("A:", tuple(A.shape), " y:", tuple(obs_c.shape), " rank(A) =", torch.linalg.matrix_rank(A).item(),
      " cond(A) =", round(torch.linalg.cond(A).item(), 3))
# 출력: A: (60, 16)  y: (60, 3)  rank(A) = 16  cond(A) = 3.822

# %% [markdown]
# ## 3. 방법 A — 최소제곱 닫힌 해
#
# $$
# \hat{\mathbf c}_A=\arg\min_{\mathbf c}\|A\mathbf c-\mathbf y\|_F^2
# \;\;\Longleftrightarrow\;\;
# (A^\top A)\,\hat{\mathbf c}_A = A^\top\mathbf y
# $$
#
# $A^\top A$가 가역(rank $K$)이면 해는 유일하다. 두 가지로 풀어 같은지 확인한다:
# 1. 정규방정식을 직접 풀기 (`torch.linalg.solve`)
# 2. `torch.linalg.lstsq` — 내부적으로 QR(기본 driver `gelsy`) 또는 SVD(`gelsd`)를 써서 $A^\top A$를 만들지 않아 수치적으로 더 안정

# %%
AtA, Aty = A.T @ A, A.T @ obs_c
c_normal = torch.linalg.solve(AtA, Aty)                              # [16, 3]
c_lstsq = torch.linalg.lstsq(A, obs_c).solution                      # [16, 3]
print("정규방정식 vs lstsq 최대 차이:", f"{(c_normal - c_lstsq).abs().max().item():.2e}")
print("cond(A) =", f"{torch.linalg.cond(A).item():.3f}", "  cond(AᵀA) = cond(A)² =", f"{torch.linalg.cond(AtA).item():.3f}")
print("잔차 ‖Ac−y‖² (관측점) =", f"{((A @ c_lstsq - obs_c) ** 2).sum().item():.5f}",
      "  전 구면 MSE =", f"{sphere_mse(c_lstsq).item():.5f}")
# 출력: 정규방정식 vs lstsq 최대 차이: 6.18e-07
# 출력: cond(A) = 3.822   cond(AᵀA) = cond(A)² = 14.607
# 출력: 잔차 ‖Ac−y‖² (관측점) = 0.37959   전 구면 MSE = 0.01363

# %% [markdown]
# ## 4. 방법 B — Adam 경사하강 (3DGS 학습 방식 축소판)
#
# 노트북 셀과 같은 설정:
# - 파라미터를 `sh0`(DC, `[1,3]`)와 `shN`(나머지 15개, `[15,3]`)으로 나눠 학습률 `2.5e-2`, `2.5e-2/20`
# - 500스텝마다 사용 차수 +1 (3DGS의 `sh_degree_interval` 흉내) → 저차부터 먼저 수렴
# - 렌더러와 같은 `+0.5` 오프셋과 `clamp_min(0)`
# - **L1 손실** (3DGS의 광도 손실은 L1 + D-SSIM)
#
# Adam 한 스텝: $m\leftarrow\beta_1 m+(1-\beta_1)g,\; v\leftarrow\beta_2 v+(1-\beta_2)g^2,\;
# \theta\leftarrow\theta-\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$ — 계수별로 스텝 크기가 정규화되어
# 크기 차이가 큰 SH 계수(DC ≫ 고차)에도 잘 맞는다.

# %%
import time

torch.manual_seed(1)
sh0 = torch.zeros(1, 3, device=DEVICE, requires_grad=True)
shN = torch.zeros(15, 3, device=DEVICE, requires_grad=True)
opt = torch.optim.Adam([{"params": [sh0], "lr": 2.5e-2}, {"params": [shN], "lr": 2.5e-2 / 20}])
hist = []
t0 = time.time()
N_STEPS = 4000
for step in range(N_STEPS):
    degree_to_use = min(step // 500, 3)
    Kd = (degree_to_use + 1) ** 2
    coeffs = torch.cat([sh0, shN], dim=0)                            # [16, 3]
    pred = torch.clamp_min(A[:, :Kd] @ coeffs[:Kd] + 0.5, 0.0)
    loss = F.l1_loss(pred, obs_c)
    opt.zero_grad(); loss.backward(); opt.step()
    hist.append(loss.item())

c_adam = torch.cat([sh0, shN], dim=0).detach().clone()
c_adam[0] += 0.5 / C0          # +0.5 오프셋을 DC 계수로 흡수 → f*와 직접 비교 가능 (0.5 = (0.5/C0)·Y₀⁰)
print(f"Adam {N_STEPS} 스텝, {time.time() - t0:.1f}s   L1 loss: 처음 {hist[0]:.4f} → 마지막 {hist[-1]:.5f}")
print("차수 전환 시점의 loss:", {s: round(hist[s], 5) for s in [499, 999, 1499, 1999, 2999, 3999]})
# 출력: Adam 4000 스텝, 1.6s   L1 loss: 처음 0.1928 → 마지막 0.03104
# 출력: 차수 전환 시점의 loss: {499: 0.06381, 999: 0.061, 1499: 0.04662, 1999: 0.03103, 2999: 0.03099, 3999: 0.03104}

# %% [markdown]
# ## 5. A vs B — 계수 벡터 차이와 전 구면 MSE
#
# 같은 60개 관측을 썼으니 B가 충분히 수렴하면 A와 가까워야 한다. 그러나 **정확히 같지는 않다**:
# A는 L2, B는 L1을 최소화하고(다른 목적함수 → 다른 최적점), B에는 `clamp`와 차수 램프업(고차 계수는
# 1500스텝 이후에야 학습)이 있다. 아래 손실 기록을 보면 B는 2000스텝 이후 0.031에서 평탄해져 L1 최적점에 수렴했으므로,
# 남는 차이는 "수렴 부족"이 아니라 주로 **L1 vs L2**에서 온다.

# %%
diff = c_adam - c_lstsq
print(f"‖c_A − c_B‖₂ = {diff.norm().item():.4f}   ‖c_A‖₂ = {c_lstsq.norm().item():.4f}   상대 오차 = {diff.norm().item() / c_lstsq.norm().item():.3%}")
print(f"전 구면 MSE   최소제곱(A): {sphere_mse(c_lstsq).item():.5f}   Adam(B): {sphere_mse(c_adam).item():.5f}")
print(f"관측점 L1     최소제곱(A): {F.l1_loss(A @ c_lstsq, obs_c).item():.5f}   Adam(B): {F.l1_loss(A @ c_adam, obs_c).item():.5f}")
print()
print(" k  (ℓ, m)      A(R)      B(R)     차이(R)")
for k, (l, m) in enumerate(LM):
    print(f"{k:2d}  ({l},{m:+d})  {c_lstsq[k, 0].item():9.4f} {c_adam[k, 0].item():9.4f} {diff[k, 0].item():9.4f}")
# 출력: ‖c_A − c_B‖₂ = 0.1915   ‖c_A‖₂ = 2.8642   상대 오차 = 6.685%
# 출력: 전 구면 MSE   최소제곱(A): 0.01363   Adam(B): 0.01054
# 출력: 관측점 L1     최소제곱(A): 0.03844   Adam(B): 0.03106
# 출력:
# 출력:  k  (ℓ, m)      A(R)      B(R)     차이(R)
# 출력:  0  (0,+0)     2.2578    2.2363   -0.0215
# 출력:  1  (1,-1)    -0.0852   -0.0934   -0.0082
# 출력:  2  (1,+0)     0.1078    0.1137    0.0059
# 출력:  3  (1,+1)     0.0450    0.0109   -0.0341
# 출력:  4  (2,-2)    -0.0942   -0.0559    0.0384
# 출력:  5  (2,-1)    -0.1808   -0.1418    0.0391
# 출력:  6  (2,+0)     0.0673    0.0379   -0.0294
# 출력:  7  (2,+1)     0.1785    0.1631   -0.0154
# 출력:  8  (2,+2)    -0.0031    0.0157    0.0187
# 출력:  9  (3,-3)     0.0107    0.0055   -0.0052
# 출력: 10  (3,-2)    -0.1910   -0.1504    0.0406
# 출력: 11  (3,-1)    -0.0589   -0.0446    0.0143
# 출력: 12  (3,+0)    -0.0806   -0.0777    0.0028
# 출력: 13  (3,+1)     0.0149    0.0030   -0.0119
# 출력: 14  (3,+2)    -0.1098   -0.0635    0.0463
# 출력: 15  (3,+3)    -0.1245   -0.0808    0.0436
# 해석: 관측점 L1은 B가 낮다(B는 L1을 직접 최소화). 이 시드에서는 전 구면 MSE도 B가 약간 낮은데,
#       A는 60개 관측점의 L2 오차만 최소화하다 고차 계수(k≥9)를 더 크게 잡아 관측 사이에서 과적합했고,
#       L1은 큰 잔차에 덜 끌려가 고차 계수가 작게 남았기 때문. 계수 크기 자체는 두 방법이 6.7% 차이로 근접한다.

# %% [markdown]
# ## 6. 관측 수와 조건수 — n = 8 / 20 / 60 / 300
#
# 관측이 $K=16$개보다 적으면 $A$는 rank $<16$ → $A^\top A$가 특이(singular)하고 최소제곱 해가 **유일하지 않다**
# (`lstsq`는 최소 노름 해를 돌려준다). 관측이 16개를 조금 넘는 수준이면 해는 유일하지만
# 조건수 $\kappa(A)=\sigma_{\max}/\sigma_{\min}$가 커져 노이즈에 민감하고, 관측 사이에서 고차 항이 요동(과적합)한다.
# 관측이 구면에 고르게 많이 퍼질수록 $A^\top A\to n\cdot\frac{1}{4\pi}I$에 가까워져(정규직교성) $\kappa\to 1$이다.

# %%
torch.manual_seed(2)
print("  n    rank  cond(A)     MSE(L=0)  MSE(L=1)  MSE(L=2)  MSE(L=3)")
for n in [8, 20, 60, 300]:
    d_n = F.normalize(torch.randn(n, 3, device=DEVICE), dim=-1)
    y_n = f_star(d_n)
    A_n = sh_bases(d_n, 3)
    cond = torch.linalg.cond(A_n).item()
    rank = torch.linalg.matrix_rank(A_n).item()
    mses = [sphere_mse(torch.linalg.lstsq(A_n[:, :(L + 1) ** 2], y_n).solution, L).item() for L in range(4)]
    print(f"{n:4d}  {rank:5d}  {cond:9.3f}    " + "  ".join(f"{v:.5f}" for v in mses))
# 출력:   n    rank  cond(A)     MSE(L=0)  MSE(L=1)  MSE(L=2)  MSE(L=3)
# 출력:    8      8      4.064    0.03176  0.03176  0.20677  0.37703
# 출력:   20     16     38.115    0.02874  0.02154  0.01425  0.02969
# 출력:   60     16      2.806    0.02855  0.02114  0.01355  0.00665
# 출력:  300     16      1.563    0.02859  0.02091  0.01204  0.00545
# 해석: n=8 → rank 8 < 16: 해가 유일하지 않고(최소 노름 해) L=2,3에서 MSE가 폭발(과적합). 이때 cond(A)는 8개의
#       0이 아닌 특이값만으로 계산되어 작아 보이므로 rank를 함께 봐야 한다.
#       n=20 → rank는 16이지만 cond=38로 나빠 L=3이 L=2보다 오히려 나쁘다(고차 과적합).
#       n=60, 300 → cond가 1에 가까워지며(직교성 회복) 차수가 높을수록 MSE가 단조 감소.

# %% [markdown]
# ## 7. 시각화 — 왼쪽: Adam 손실 곡선(log), 오른쪽: R채널 16개 계수 A vs B

# %%
fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
                    subplot_titles=("방법 B: Adam L1 loss (500스텝마다 차수 +1)", "R채널 SH 계수 — A(최소제곱) vs B(Adam)"))
fig.add_trace(go.Scatter(y=hist, mode="lines", name="L1 loss", line=dict(color="#4C6EF5", width=1.5)), row=1, col=1)
for s in [500, 1000, 1500]:
    fig.add_vline(x=s, line=dict(color="#999", dash="dot", width=1), row=1, col=1)
fig.update_yaxes(type="log", title_text="L1 loss", row=1, col=1)
fig.update_xaxes(title_text="step", row=1, col=1)

labels = [f"k={k}<br>({l},{m:+d})" for k, (l, m) in enumerate(LM)]
fig.add_trace(go.Bar(x=labels, y=c_lstsq[:, 0].tolist(), name="A: 최소제곱", marker_color="#4C6EF5"), row=1, col=2)
fig.add_trace(go.Bar(x=labels, y=c_adam[:, 0].tolist(), name="B: Adam", marker_color="#F59F00"), row=1, col=2)
fig.update_yaxes(title_text="계수 값 (R)", row=1, col=2)
fig.update_layout(barmode="group", width=1300, height=460, template="plotly_white",
                  legend=dict(orientation="h", y=-0.18, x=0.55),
                  title_text=f"n_view=60, L=3 — 전 구면 MSE  A: {sphere_mse(c_lstsq).item():.5f}   B: {sphere_mse(c_adam).item():.5f}   "
                             f"‖c_A−c_B‖₂ = {diff.norm().item():.4f}")
_show(fig)
png_path = os.path.join(HERE, "expy.png")
fig.write_image(png_path, scale=2)
print("저장:", png_path)
# 출력: 저장: /home/sungwoo/projects/swcho/gsplat/fm/sh/.fm/hints/0088e583-f26b-454a-8e24-9dfbc92044c7/expy.png

# %% [markdown]
# ## 정리
#
# | | 방법 A: 최소제곱 | 방법 B: Adam |
# |---|---|---|
# | 문제 | $\min\|A\mathbf c-\mathbf y\|^2$ — 선형·볼록 | $\min \text{L1}(\max(0, A\mathbf c+0.5), \mathbf y)$ — 비선형(clamp) |
# | 풀이 | 정규방정식/QR/SVD, 한 번에 | 수천 스텝 반복 |
# | 유일성 | rank$(A)=K$이면 유일 | 국소해 가능(여기선 볼록에 가까워 A와 근접) |
# | 3DGS에서 | 불가능 — 렌더러(스플래팅·알파 합성)가 비선형이고 픽셀 하나에 여러 Gaussian이 섞임 | 실제 사용 — 위치·크기·불투명도와 **공동** 최적화, 수백만 Gaussian을 미니배치 SGD로 |
# | 관측 부족 시 | rank 부족 → 해 불유일, 고차 과적합 | 마찬가지로 과적합(정규화 없음); 차수 램프업이 약한 완화 |
