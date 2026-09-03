# %% [markdown]
# # `rasterize_mode="antialiased"` — eps2d 블러 보정 실험
#
# gsplat의 `rasterize_mode="antialiased"`가 하는 일은 딱 두 줄이다.
#
# - `_torch_impl.py` / `_fully_fused_projection`:
#   `compensations = sqrt(clamp(det_orig / det, min=MIN_COMPENSATION**2))`
# - `rendering.py`: `opacities = opacities * compensations`
#
# 여기서
#
# $$\det{}_0 = \det(\Sigma),\qquad \det = \det(\Sigma + \epsilon I),\qquad
#   \rho = \sqrt{\frac{\det_0}{\det}} \le 1$$
#
# 이 노트북은 gsplat을 **import 하지 않고**(그러면 30분짜리 JIT CUDA 빌드가 돈다)
# numpy만으로 같은 계산을 재현해서, 이 $\rho$ 가 왜 필요한지 눈으로 확인한다.
#
# 확인할 것:
# 1. splat 하나를 픽셀 격자에 렌더해 **총 밝기**(α 합)를 재는 함수
# 2. σ를 4px → 0.1px로 줄이며(= 카메라가 멀어지며) classic 모드의 총 밝기가 **바닥값에 걸리는** 현상
# 3. antialiased 보정 후 총 밝기가 $\sigma^2$ 에 비례해 **정상적으로 감소**하는 것
# 4. $\rho$ 를 σ의 함수로 그리기
# 5. 3개 σ에서 두 모드의 splat 이미지 비교 히트맵

# %%
# 필요 패키지: numpy, plotly, kaleido (expy.png 저장용)
# gsplat은 절대 import 하지 않는다 (import 시 30분 이상 걸리는 JIT CUDA 빌드가 발생).
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
EPS2D = 0.3              # gsplat 기본값 (px^2)
MIN_COMPENSATION = 0.005  # gsplat/cuda/_constants.py
np.set_printoptions(precision=4, suppress=True)
print(f"eps2d = {EPS2D}, MIN_COMPENSATION = {MIN_COMPENSATION}")
# 출력: eps2d = 0.3, MIN_COMPENSATION = 0.005

# %% [markdown]
# ## 1. splat 하나를 픽셀 격자에 렌더하기
#
# gsplat의 알파는 **정규화 상수가 없다**. 중심에서의 값은 언제나 정확히 $o$ 다.
#
# $$\alpha(\mathbf{x}) = o \cdot \exp\!\left(-\tfrac12 (\mathbf{x}-\boldsymbol\mu)^\top \Sigma^{-1} (\mathbf{x}-\boldsymbol\mu)\right)$$
#
# 그래서 화면에 뿌려지는 **총량**은 고정이 아니라 $\Sigma$ 에 따라 변한다:
#
# $$\iint \alpha\,dx\,dy = o \cdot 2\pi\sqrt{\det\Sigma}$$
#
# $\sqrt{\det\Sigma}$ 가 "splat이 덮는 타원의 넓이 배율"이다.
# 등방 $\Sigma=\sigma^2 I$ 이면 $\sqrt{\det\Sigma}=\sigma^2$ 이므로 총량 $=2\pi o\sigma^2$.

# %%
def render_splat(sigma, opacity=1.0, eps2d=EPS2D, antialiased=False, half=12, sub=1):
    """등방 splat 하나를 (2*half x 2*half) 픽셀 격자에 렌더하고 (이미지, 총 alpha 합)을 반환.

    sub>1이면 픽셀당 sub x sub 서브샘플의 평균을 써서 '픽셀 = 면적 적분'에 가깝게 만든다.
    (sub=1이면 픽셀 중심 point sampling — 실제 3DGS 래스터라이저와 동일)
    """
    cov0 = np.array([[sigma**2, 0.0], [0.0, sigma**2]])   # 블러 전 Σ
    det0 = np.linalg.det(cov0)
    cov = cov0 + eps2d * np.eye(2)                        # 블러 후 Σ + εI
    det = max(np.linalg.det(cov), 1e-10)

    o = opacity
    if antialiased:
        rho = np.sqrt(max(det0 / det, MIN_COMPENSATION**2))
        o = o * rho

    conic = np.linalg.inv(cov)                            # gsplat의 (a, b, c)
    # 픽셀 중심 좌표 (0.5 오프셋), splat 중심은 (0, 0)
    c = (np.arange(-half, half) + 0.5)
    off = (np.arange(sub) + 0.5) / sub - 0.5
    xs = (c[:, None] + off[None, :]).ravel()
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    q = conic[0, 0] * X**2 + 2 * conic[0, 1] * X * Y + conic[1, 1] * Y**2
    a = o * np.exp(-0.5 * q)
    if sub > 1:                                           # 서브샘플 평균 → 픽셀값
        n = 2 * half
        a = a.reshape(n, sub, n, sub).mean(axis=(1, 3))
    return a, a.sum()


# 이론값과 맞는지 확인: 총합 ≈ o * 2π * sqrt(det(Σ+εI))
for s in [4.0, 1.0, 0.3]:
    img, tot = render_splat(s, sub=8)
    det = (s**2 + EPS2D) ** 2
    print(f"sigma={s:4.1f}  측정 총합={tot:9.4f}   이론 2*pi*sqrt(det)={2*np.pi*np.sqrt(det):9.4f}")
# 출력: sigma= 4.0  측정 총합= 101.8115   이론 2*pi*sqrt(det)= 102.4159
# 출력: sigma= 1.0  측정 총합=   8.1681   이론 2*pi*sqrt(det)=   8.1681
# 출력: sigma= 0.3  측정 총합=   2.4504   이론 2*pi*sqrt(det)=   2.4504

# %% [markdown]
# (σ=4에서 살짝 어긋나는 건 12px 격자가 splat 꼬리를 다 담지 못해서일 뿐, 원리는 정확히 맞는다.)
#
# ## 2. 카메라가 멀어진다 = σ가 줄어든다
#
# 투영 Jacobian의 성분이 $f/z$ 이므로 화면상 크기는 깊이에 반비례한다:
#
# $$\sigma \propto \frac{1}{z}$$
#
# 그런데 $\epsilon = 0.3\,\text{px}^2$ 은 **고정**이다. 그래서 멀어질수록 $\epsilon$ 의 상대 비중이 커진다.
#
# - **이상적(블러 없음)** 총 밝기: $2\pi o \sigma^2 \propto 1/z^2$ — 멀어지면 정직하게 어두워져야 한다.
# - **classic** 총 밝기: $2\pi o (\sigma^2+\epsilon)$ — σ→0에서 $2\pi o \epsilon$ 이라는 **바닥값에 걸린다**.
# - **antialiased** 총 밝기: $2\pi o (\sigma^2+\epsilon)\cdot\dfrac{\sigma^2}{\sigma^2+\epsilon} = 2\pi o\sigma^2$ — 이상값과 정확히 일치.

# %%
sigmas = np.geomspace(4.0, 0.1, 60)

tot_classic = np.array([render_splat(s, antialiased=False, sub=8)[1] for s in sigmas])
tot_aa = np.array([render_splat(s, antialiased=True, sub=8)[1] for s in sigmas])
tot_ideal = 2 * np.pi * sigmas**2                     # 블러가 전혀 없었다면
rho = sigmas**2 / (sigmas**2 + EPS2D)                 # 등방일 때 sqrt(det0/det)

for s in [4.0, 2.0, 1.0, 0.5, 0.2, 0.1]:
    d0, d = s**4, (s**2 + EPS2D) ** 2
    r = np.sqrt(d0 / d)
    print(f"sigma={s:4.2f}  det0={d0:9.5f}  det={d:9.5f}  rho=sqrt(det0/det)={r:.4f}"
          f"   classic/ideal={(s**2+EPS2D)/s**2:8.2f}x")
# 출력: sigma=4.00  det0=256.00000  det=265.69000  rho=sqrt(det0/det)=0.9816   classic/ideal=    1.02x
# 출력: sigma=2.00  det0= 16.00000  det= 18.49000  rho=sqrt(det0/det)=0.9302   classic/ideal=    1.07x
# 출력: sigma=1.00  det0=  1.00000  det=  1.69000  rho=sqrt(det0/det)=0.7692   classic/ideal=    1.30x
# 출력: sigma=0.50  det0=  0.06250  det=  0.30250  rho=sqrt(det0/det)=0.4545   classic/ideal=    2.20x
# 출력: sigma=0.20  det0=  0.00160  det=  0.11560  rho=sqrt(det0/det)=0.1176   classic/ideal=    8.50x
# 출력: sigma=0.10  det0=  0.00010  det=  0.09610  rho=sqrt(det0/det)=0.0323   classic/ideal=   31.00x

# %% [markdown]
# **σ=0.1px 서브픽셀 splat에서 classic 모드는 이상값보다 31배 밝다.**
# 멀어질수록 물체가 점점 밝아지고 뭉개지는 아티팩트의 정체가 이것이다.
#
# ## 3. 두 모드의 splat 이미지 비교
#
# classic은 σ가 아무리 작아져도 알파의 **연속 봉우리 높이가 $o=1$ 로 고정**이지만,
# antialiased는 $o\cdot\rho$ 로 낮아진다. 두 모드의 픽셀값 비는 어디서나 정확히 $\rho$ 다.
# (아래 `peak`은 픽셀 면적 평균값이라 σ가 작으면 1보다 작게 나오지만, 두 모드의 비는 그대로 ρ)

# %%
SIGMAS_VIS = [4.0, 1.0, 0.2]
imgs = {}
for s in SIGMAS_VIS:
    imgs[("classic", s)] = render_splat(s, antialiased=False, sub=4)[0]
    imgs[("antialiased", s)] = render_splat(s, antialiased=True, sub=4)[0]
    print(f"sigma={s:4.1f}  classic peak={imgs[('classic', s)].max():.4f} 합={imgs[('classic', s)].sum():8.3f}"
          f"  |  aa peak={imgs[('antialiased', s)].max():.4f} 합={imgs[('antialiased', s)].sum():8.3f}")
# 출력: sigma= 4.0  classic peak=0.9801 합= 101.812  |  aa peak=0.9621 합=  99.938
# 출력: sigma= 1.0  classic peak=0.7862 합=   8.168  |  aa peak=0.6048 합=   6.283
# 출력: sigma= 0.2  classic peak=0.4482 합=   2.136  |  aa peak=0.0527 합=   0.251

# %% [markdown]
# ## 4. 종합 그래프 (`expy.png`로도 저장)
#
# - **1행**: 총 밝기 곡선 / compensation $\rho$ / 이상값 대비 배율
# - **2행**: classic 모드 splat (σ=4, 1, 0.2) — 작아져도 중심이 계속 밝다
# - **3행**: antialiased 모드 splat — σ가 작아지면 정직하게 흐려진다
#
# 2·3행 히트맵은 같은 색 스케일(0~1)을 쓴다.

# %%
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=(
        "총 밝기 (α 합) vs σ",
        "compensation ρ = √(det₀/det)",
        "이상값 대비 배율",
        *[f"classic σ={s}px — 합 {imgs[('classic', s)].sum():.2f}" for s in SIGMAS_VIS],
        *[f"antialiased σ={s}px — 합 {imgs[('antialiased', s)].sum():.2f}" for s in SIGMAS_VIS],
    ),
    vertical_spacing=0.10, horizontal_spacing=0.07,
    row_heights=[0.42, 0.29, 0.29],
)

# --- (1,1) 총 밝기 곡선 ---
fig.add_trace(go.Scatter(x=sigmas, y=tot_classic, name="classic", mode="lines",
                         line=dict(color="#d1495b", width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=sigmas, y=tot_aa, name="antialiased", mode="lines",
                         line=dict(color="#0f8b8d", width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=sigmas, y=tot_ideal, name="이상값 2πσ²", mode="lines",
                         line=dict(color="#333333", width=2, dash="dot")), row=1, col=1)
fig.add_hline(y=2 * np.pi * EPS2D, line=dict(color="#d1495b", width=1, dash="dash"),
              row=1, col=1)
fig.add_annotation(x=np.log10(0.6), y=np.log10(2 * np.pi * EPS2D), xref="x", yref="y",
                   text="바닥값 2πε — classic은 여기서 멈춘다", showarrow=False,
                   yshift=-14, font=dict(size=11, color="#d1495b"), row=1, col=1)
fig.update_xaxes(type="log", autorange="reversed", title_text="σ (px) — 오른쪽이 '멀어짐'",
                 row=1, col=1)
fig.update_yaxes(type="log", title_text="총 α 합", row=1, col=1)

# --- (1,2) compensation ---
fig.add_trace(go.Scatter(x=sigmas, y=rho, name="ρ", mode="lines", showlegend=False,
                         line=dict(color="#0f8b8d", width=3)), row=1, col=2)
for s in [4.0, 1.0, 0.2]:
    r = s**2 / (s**2 + EPS2D)
    fig.add_trace(go.Scatter(x=[s], y=[r], mode="markers+text", showlegend=False,
                             text=[f"σ={s}<br>ρ={r:.3f}"], textposition="bottom right",
                             textfont_size=10, marker=dict(color="#333", size=8)), row=1, col=2)
fig.update_xaxes(type="log", autorange="reversed", title_text="σ (px)", row=1, col=2)
fig.update_yaxes(title_text="ρ (≤1)", range=[0, 1.08], row=1, col=2)

# --- (1,3) 이상값 대비 배율 ---
fig.add_trace(go.Scatter(x=sigmas, y=tot_classic / tot_ideal, name="classic/ideal",
                         mode="lines", showlegend=False,
                         line=dict(color="#d1495b", width=3)), row=1, col=3)
fig.add_trace(go.Scatter(x=sigmas, y=tot_aa / tot_ideal, name="aa/ideal",
                         mode="lines", showlegend=False,
                         line=dict(color="#0f8b8d", width=3)), row=1, col=3)
fig.add_hline(y=1.0, line=dict(color="#333", width=1, dash="dot"), row=1, col=3)
fig.update_xaxes(type="log", autorange="reversed", title_text="σ (px)", row=1, col=3)
fig.update_yaxes(type="log", title_text="총 밝기 / 이상값", row=1, col=3)

# --- (2,*) / (3,*) 히트맵 ---
for j, s in enumerate(SIGMAS_VIS, start=1):
    for i, mode in enumerate(["classic", "antialiased"], start=2):
        img = imgs[(mode, s)]
        fig.add_trace(go.Heatmap(z=img, zmin=0.0, zmax=1.0, colorscale="Inferno",
                                 showscale=(i == 2 and j == 3),
                                 colorbar=dict(len=0.55, y=0.27, thickness=12, title="α")),
                      row=i, col=j)
        fig.update_xaxes(visible=False, row=i, col=j)
        fig.update_yaxes(visible=False, scaleanchor=f"x{(i-1)*3+j}", row=i, col=j)

fig.update_layout(
    height=1050, width=1400, template="plotly_white",
    title_text=("rasterize_mode='antialiased' — eps2d=0.3 블러가 부풀린 밝기를 "
                "ρ=√(det₀/det)로 되돌린다"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    font=dict(size=12),
)
for ann in fig.layout.annotations[:9]:
    ann.font.size = 13

_show(fig)
fig.write_image(os.path.join(HERE, "expy.png"), scale=2)
print("saved:", os.path.join(HERE, "expy.png"))
# 출력: saved: .../expy.png

# %% [markdown]
# ## 5. 결론
#
# | | classic | antialiased |
# |---|---|---|
# | 불투명도 | $o$ | $o \cdot \sqrt{\det_0/\det}$ |
# | σ=4px (가까움) | 기준 | ρ=0.981 → 사실상 동일 |
# | σ=0.1px (멀리) | 이상값의 **31배** 밝음 | 이상값과 일치 |
# | 줌아웃 | 점점 밝아지고 뭉개짐 | 밝기 보존 |
#
# 한 문장: **정규화하지 않는 splat에 eps2d 블러를 더하면 총 밝기가 $\sqrt{\det/\det_0}$ 배로 부풀므로,
# 그 역수 $\rho=\sqrt{\det_0/\det}$ 를 불투명도에 곱해 되돌린다.** 이것이 Mip-Splatting의 2D Mip filter다.
#
# 실제 gsplat에서는 이것이 `calc_compensations=True` → `opacities = opacities * compensations` 두 줄이고,
# ρ에는 `MIN_COMPENSATION = 0.005` 하한이 걸려 있어 극단적으로 작은 splat도 완전히 0이 되지는 않는다.
