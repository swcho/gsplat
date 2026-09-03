# %% [markdown]
# # `eps2d=0.3` — 최소 블러가 없으면 작은 Gaussian은 사라진다
#
# 투영 단계에서 gsplat은 2D 공분산에 항상 상수 대각 행렬을 더한다:
#
# $$\Sigma_{2D} = J\,\Sigma_c\,J^\top + \epsilon I,\qquad \epsilon = \texttt{eps2d} = 0.3\ \text{px}^2$$
#
# 이 노트북은 왜 그게 필요한지, 그리고 그 대가가 무엇인지를 숫자로 보여준다.
#
# 래스터화기가 픽셀 하나에서 계산하는 것은 (walkthrough의 ⑦ 블렌딩과 동일):
#
# $$\sigma_i = \tfrac12 \mathbf{d}^\top \Sigma_{2D}^{-1} \mathbf{d},\qquad
#   \alpha_i = \min\!\big(0.99,\; o_i e^{-\sigma_i}\big),\qquad
#   \alpha_i < \tfrac{1}{255} \Rightarrow \text{skip}$$
#
# 여기서 $\mathbf{d}$는 **픽셀 중심**(정수 + 0.5)과 Gaussian 중심의 차이다.
# 즉 Gaussian은 픽셀 중심이라는 **점 하나에서만 샘플링**된다 — 픽셀 면적에 대한 적분이 아니다.
# 그래서 Gaussian이 1px보다 작아지면 "중심들 사이로 빠져나가" 사라진다.
#
# 필요 패키지: numpy, plotly, kaleido

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


ALPHA_THRESHOLD = 1.0 / 255.0  # gsplat Config.h
MAX_ALPHA = 0.99  # gsplat Config.h
EPS2D = 0.3  # gsplat 기본값 (Inria 3DGS의 cov[0][0] += 0.3f 유래)

print(f"ALPHA_THRESHOLD = {ALPHA_THRESHOLD:.6f}, MAX_ALPHA = {MAX_ALPHA}, eps2d = {EPS2D}")
# 출력: ALPHA_THRESHOLD = 0.003922, MAX_ALPHA = 0.99, eps2d = 0.3

# %% [markdown]
# ## 1. 픽셀 중심 샘플링 모델
#
# 등방 Gaussian $\Sigma = \sigma^2 I$를 화면에 놓고, `eps2d`를 더한 뒤(또는 안 더한 뒤)
# 주변 픽셀 중심들에서 $\alpha$를 계산한다.
# 한 Gaussian이 화면에 남기는 총 에너지를 $\sum_p \alpha_p$ ("총 렌더 기여량")로 재겠다.

# %%
def pixel_alphas(mu, sigma_px, opacity=1.0, eps2d=0.0, half=8, raw=False):
    """mu 주변 (2*half+1)^2 픽셀 중심에서의 alpha. Σ = sigma_px^2 I + eps2d I.

    raw=True면 1/255 임계를 적용하기 전 값을 돌려준다.
    """
    cov = np.eye(2) * (sigma_px**2) + np.eye(2) * eps2d
    conic = np.linalg.inv(cov)
    c0 = np.floor(mu).astype(int)
    ii = np.arange(c0[0] - half, c0[0] + half + 1) + 0.5  # 픽셀 중심 = 정수 + 0.5
    jj = np.arange(c0[1] - half, c0[1] + half + 1) + 0.5
    X, Y = np.meshgrid(ii, jj, indexing="ij")
    dx, dy = X - mu[0], Y - mu[1]
    sig = 0.5 * (conic[0, 0] * dx**2 + conic[1, 1] * dy**2) + conic[0, 1] * dx * dy
    a = np.minimum(MAX_ALPHA, opacity * np.exp(-sig))
    if raw:
        return a
    return np.where(a >= ALPHA_THRESHOLD, a, 0.0)  # 임계 미만은 건너뛴다


sigma = 0.2  # px — 1픽셀보다 훨씬 작은 Gaussian

for name, mu in [("픽셀 중심 위 (10.5, 10.5)", np.array([10.5, 10.5])),
                 ("픽셀 경계 위 (10.0, 10.0)", np.array([10.0, 10.0]))]:
    a_off = pixel_alphas(mu, sigma, eps2d=0.0)
    a_on = pixel_alphas(mu, sigma, eps2d=EPS2D)
    r_off = pixel_alphas(mu, sigma, eps2d=0.0, raw=True)
    print(f"{name}:")
    print(f"   eps2d=0.0  → 임계 전 max α = {r_off.max():.6f}, 살아남은 픽셀 {int((a_off>0).sum())}개, Σα = {a_off.sum():.4f}")
    print(f"   eps2d=0.3  → 임계 전 max α = {a_on.max():.6f}, 살아남은 픽셀 {int((a_on>0).sum())}개, Σα = {a_on.sum():.4f}")
# 출력: 픽셀 중심 위 (10.5, 10.5):
# 출력:    eps2d=0.0  → 임계 전 max α = 0.990000, 살아남은 픽셀 1개, Σα = 0.9900
# 출력:    eps2d=0.3  → 임계 전 max α = 0.990000, 살아남은 픽셀 9개, Σα = 2.1204
# 출력: 픽셀 경계 위 (10.0, 10.0):
# 출력:    eps2d=0.0  → 임계 전 max α = 0.001930, 살아남은 픽셀 0개, Σα = 0.0000
# 출력:    eps2d=0.3  → 임계 전 max α = 0.479364, 살아남은 픽셀 12개, Σα = 2.1200

# %% [markdown]
# **σ=0.2px Gaussian이 픽셀 경계에 있으면 `eps2d=0`에서는 완전히 사라진다.**
# 가장 가까운 픽셀 중심까지 거리가 $(0.5, 0.5)$이므로
# $\sigma_i = \tfrac{0.5^2+0.5^2}{2\cdot 0.2^2} = 6.25$, $\alpha = e^{-6.25} = 0.00193 < 1/255$.
#
# `eps2d=0.3`을 더하면 유효 분산이 $0.04 + 0.3 = 0.34$가 되어 $\alpha = 0.479$로 살아난다.
# 즉 $\epsilon = 0.3\ \text{px}^2 \approx (0.55\,\text{px})^2$ 폭의 **저역통과(low-pass) 필터**를 강제로 씌우는 것이고,
# 이는 픽셀 격자의 나이퀴스트 한계(주기 1px → 차단 주파수 0.5 cycle/px)에 맞춘 재구성 필터에 해당한다.

# %%
# 위치 스윕: Gaussian 중심을 픽셀 안에서 0 → 1px 이동시키며 총 기여량을 본다
offsets = np.linspace(0.0, 1.0, 201)
sigmas = [0.2, 0.5, 1.0]
sweep = {}
for s in sigmas:
    for eps in (0.0, EPS2D):
        sweep[(s, eps)] = np.array(
            [pixel_alphas(np.array([10.0 + t, 10.0 + t]), s, eps2d=eps).sum() for t in offsets]
        )

for s in sigmas:
    a0, a1 = sweep[(s, 0.0)], sweep[(s, EPS2D)]
    rng = lambda v: (v.max() - v.min()) / max(v.mean(), 1e-9)
    print(f"σ={s}px  eps2d=0: Σα ∈ [{a0.min():.3f}, {a0.max():.3f}] 변동폭 {rng(a0)*100:5.1f}% | "
          f"eps2d=0.3: Σα ∈ [{a1.min():.3f}, {a1.max():.3f}] 변동폭 {rng(a1)*100:4.1f}%")
# 출력: σ=0.2px  eps2d=0: Σα ∈ [0.000, 0.990] 변동폭 281.1% | eps2d=0.3: Σα ∈ [2.120, 2.137] 변동폭  0.8%
# 출력: σ=0.5px  eps2d=0: Σα ∈ [1.522, 1.612] 변동폭   5.8% | eps2d=0.3: Σα ∈ [3.429, 3.451] 변동폭  0.6%
# 출력: σ=1.0px  eps2d=0: Σα ∈ [6.250, 6.266] 변동폭   0.3% | eps2d=0.3: Σα ∈ [8.126, 8.145] 변동폭  0.2%

# %% [markdown]
# `eps2d=0`이면 σ=0.2px Gaussian의 총 기여량이 위치에 따라 0 ↔ 0.99로 **깜빡인다**(변동폭 281%).
# 카메라가 서브픽셀만큼만 움직여도 점이 켜졌다 꺼졌다 하는 전형적인 시간적 에일리어싱이다.
# `eps2d=0.3`이면 변동폭이 1% 미만으로 안정적이다. σ=1px 이상이면 격자가 충분히 조밀하므로 둘 다 안정적이다.
#
# ## 2. 대가: 작은 Gaussian이 실제보다 밝고 두꺼워진다
#
# $\Sigma \to \Sigma + \epsilon I$는 면적을 키우므로 총 기여량이 늘어난다.
# 3D 크기가 고정된 Gaussian을 줌아웃하면 화면상 $\sigma$가 줄어드는데, 그때 총 에너지는
# 줄어들다 말고 $\epsilon$이 만든 바닥에 걸려 **밝기가 팽창**한다.
#
# Mip-Splatting의 `rasterize_mode="antialiased"`는 불투명도에 보정 계수를 곱해 이를 되돌린다:
#
# $$\rho = \sqrt{\frac{\det \Sigma}{\det (\Sigma + \epsilon I)}},\qquad o \leftarrow \rho\, o$$
#
# (등방 $\Sigma=\sigma^2 I$이면 $\rho = \sigma^2/(\sigma^2+\epsilon)$.)
# gsplat에서는 `_fully_fused_projection(..., calc_compensations=True)`가 `det_orig/det_blur`로 계산하고
# (`gsplat/cuda/_torch_impl.py`), `rendering.py`에서 `opacities = opacities * compensations`로 적용된다.
# 하한은 `MIN_COMPENSATION = 0.005`.

# %%
MIN_COMPENSATION = 0.005  # gsplat/cuda/_constants.py


def compensation(sigma_px, eps2d=EPS2D):
    cov = np.eye(2) * sigma_px**2
    det_orig = np.linalg.det(cov)
    det_blur = np.linalg.det(cov + np.eye(2) * eps2d)
    return np.sqrt(max(det_orig / det_blur, MIN_COMPENSATION**2))


scan = np.geomspace(0.05, 4.0, 120)
mu_c = np.array([10.5, 10.5])  # 최선의 경우(픽셀 중심)에서 비교
E_ideal = np.array([pixel_alphas(mu_c, s, eps2d=0.0).sum() for s in scan])
E_blur = np.array([pixel_alphas(mu_c, s, eps2d=EPS2D).sum() for s in scan])
rho = np.array([compensation(s) for s in scan])
E_aa = np.array([pixel_alphas(mu_c, s, opacity=compensation(s), eps2d=EPS2D).sum() for s in scan])

for s in (0.1, 0.3, 0.55, 1.0, 2.0):
    i = int(np.argmin(abs(scan - s)))
    print(f"σ={scan[i]:.2f}px  ρ={rho[i]:.4f}  Σα: eps2d없음 {E_ideal[i]:.3f} / "
          f"classic {E_blur[i]:.3f} / antialiased {E_aa[i]:.3f}")
# 출력: σ=0.10px  ρ=0.0327  Σα: eps2d없음 0.990 / classic 1.947 / antialiased 0.059
# 출력: σ=0.30px  ρ=0.2353  Σα: eps2d없음 1.008 / classic 2.445 / antialiased 0.572
# 출력: σ=0.55px  ρ=0.4999  Σα: eps2d없음 1.887 / classic 3.750 / antialiased 1.880
# 출력: σ=0.99px  ρ=0.7646  Σα: eps2d없음 6.099 / classic 7.971 / antialiased 6.102
# 출력: σ=1.99px  ρ=0.9294  Σα: eps2d없음 24.702 / classic 26.570 / antialiased 24.703

# %% [markdown]
# 읽는 법:
#
# - $\sigma \gtrsim 0.55$px에서는 antialiased(1.880 / 6.102 / 24.703)가 "eps2d 없음"(1.887 / 6.099 / 24.702)과
#   거의 정확히 일치한다 — 블러가 늘린 면적만큼 불투명도를 깎아 **에너지를 보존**한다.
# - $\sigma \to 0$에서 classic은 1.947이라는 **바닥값**에 걸린다. 3D 크기가 고정된 Gaussian을
#   아무리 멀리 밀어도(줌아웃해도) 화면에서 이만큼 밝은 얼룩이 남는다는 뜻이다 → 줌아웃 시 밝기 팽창.
# - antialiased는 같은 구간에서 0.059까지 매끄럽게 줄어든다. 서브픽셀 크기의 점은 **점점 옅어져야**
#   물리적으로 맞다("eps2d 없음"이 0.99인 것은 픽셀 중심에 정확히 얹힌 최선의 경우만 본 값이고,
#   위치를 조금만 옮기면 0으로 튀는 에일리어싱 그 자체다).

# %%
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "① σ=0.2px, 중심을 픽셀 경계에 둔 경우 — 픽셀 중심에서의 α",
        "② 서브픽셀 위치 스윕 → 총 기여량 Σα (점멸 여부)",
        "③ 화면 크기 σ에 따른 총 기여량 (줌아웃 = 왼쪽)",
        "④ Mip-Splatting 보정 계수 ρ = √(detΣ / det(Σ+εI))",
    ),
    vertical_spacing=0.13, horizontal_spacing=0.09,
)

# ① 픽셀 중심에서의 α 프로파일 (중심에서 가장 가까운 픽셀 행의 1D 단면, 임계 적용 전 값)
xs = np.arange(7, 14) + 0.5
mu_b = np.array([10.0, 10.0])
for eps, color in ((0.0, "#d62728"), (EPS2D, "#1f77b4")):
    cov = np.eye(2) * (0.2**2 + eps)
    conic = np.linalg.inv(cov)
    d, dy = xs - mu_b[0], 0.5  # 가장 가까운 픽셀 행까지 0.5px
    sg = 0.5 * (conic[0, 0] * d**2 + conic[1, 1] * dy**2)
    a = np.maximum(np.minimum(MAX_ALPHA, np.exp(-sg)), 1e-6)
    fig.add_trace(go.Bar(x=xs, y=a, name=f"① eps2d={eps}", marker_color=color,
                         text=[f"{v:.4f}" if v > 3e-5 else "" for v in a],
                         textposition="outside", textfont=dict(size=9)), row=1, col=1)
fig.add_hline(y=ALPHA_THRESHOLD, line=dict(color="gray", dash="dot"),
              annotation_text="α = 1/255 → 이 아래는 skip", annotation_position="top left",
              row=1, col=1)

# ② 위치 스윕
for s, color in zip(sigmas, ("#d62728", "#ff7f0e", "#2ca02c")):
    fig.add_trace(go.Scatter(x=offsets, y=sweep[(s, 0.0)], mode="lines",
                             line=dict(color=color, dash="dash"),
                             name=f"② σ={s} eps2d=0"), row=1, col=2)
    fig.add_trace(go.Scatter(x=offsets, y=sweep[(s, EPS2D)], mode="lines",
                             line=dict(color=color),
                             name=f"② σ={s} eps2d=0.3"), row=1, col=2)

# ③ σ에 따른 총 기여량
m = scan <= 1.5
fig.add_trace(go.Scatter(x=scan[m], y=E_blur[m], mode="lines", name="③ classic (Σ+0.3I)",
                         line=dict(color="#d62728", width=2.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=scan[m], y=E_aa[m], mode="lines", name="③ antialiased (ρ·o)",
                         line=dict(color="#1f77b4", width=5)), row=2, col=1)
fig.add_trace(go.Scatter(x=scan[m], y=E_ideal[m], mode="lines", name="③ eps2d 없음(픽셀 중심)",
                         line=dict(color="#333333", dash="dot", width=2)), row=2, col=1)
fig.add_annotation(x=np.log10(0.09), y=E_blur[0], text="바닥값 ≈1.95<br>(줌아웃해도 안 사라짐)",
                   showarrow=True, arrowhead=2, ax=55, ay=-30,
                   font=dict(size=10, color="#d62728"), row=2, col=1)

# ④ 보정 계수
fig.add_trace(go.Scatter(x=scan[m], y=rho[m], mode="lines", name="④ ρ",
                         line=dict(color="#9467bd")), row=2, col=2)
fig.add_vline(x=np.sqrt(EPS2D), line=dict(color="gray", dash="dot"),
              annotation_text="σ = √0.3 ≈ 0.55px → ρ = 0.5", row=2, col=2)

fig.update_xaxes(title_text="픽셀 중심 x 좌표", row=1, col=1)
fig.update_yaxes(title_text="α (log)", type="log", range=[-3.4, 0.15], row=1, col=1)
fig.update_xaxes(title_text="서브픽셀 오프셋 (px)", row=1, col=2)
fig.update_yaxes(title_text="Σα", row=1, col=2)
fig.update_xaxes(title_text="화면상 σ (px, log)", type="log",
                 range=[np.log10(0.05), np.log10(1.5)], row=2, col=1)
fig.update_yaxes(title_text="Σα", row=2, col=1)
fig.update_xaxes(title_text="화면상 σ (px, log)", type="log",
                 range=[np.log10(0.05), np.log10(1.5)], row=2, col=2)
fig.update_yaxes(title_text="ρ", range=[0, 1], row=2, col=2)
fig.update_layout(
    height=820, width=1180, template="plotly_white",
    title_text="eps2d=0.3 — 최소 블러의 효과와 부작용",
    legend=dict(font=dict(size=10)),
)

_show(fig)
fig.write_image("expy.png", scale=2)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 3. gsplat에서의 위치
#
# - `gsplat/rendering.py`: `rasterization(..., eps2d: float = 0.3, rasterize_mode="classic"|"antialiased")`
# - `gsplat/cuda/_torch_impl.py::_fully_fused_projection`:
#   `covars2d = covars2d + torch.eye(2) * eps2d` → `det_orig/det`로 `compensations` 계산
# - `gsplat/rendering.py`: `if compensations is not None: opacities = opacities * compensations`
# - CUDA 커널 쪽에는 0.3이 하드코딩된 경로가 있어 `assert eps2d == 0.3`을 거는 래퍼도 있다.
#
# 요약: `eps2d`는 **에일리어싱을 막는 저역통과 필터**이고, `antialiased` 모드는 그 필터가 빼앗은
# 밝기를 불투명도로 되돌려주는 보정이다.
