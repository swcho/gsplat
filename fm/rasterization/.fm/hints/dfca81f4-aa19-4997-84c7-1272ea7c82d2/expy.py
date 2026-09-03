# %% [markdown]
# # 불투명도 인지(opacity-aware) 반경 — 실행 가능한 예제
#
# 알파 $\alpha(r) = o\,e^{-r^2/(2\sigma^2)}$ 가 $1/255$ 아래로 떨어지는 반경을
# 직접 풀어서, gsplat의 `GAUSSIAN_EXTEND = 3.33f` 상수의 정체와
# 불투명도가 낮을 때의 타일 절감량을 확인한다.
#
# 필요 패키지: numpy, plotly, kaleido(정적 png 저장용)
# **gsplat은 import 하지 않는다** (JIT CUDA 빌드가 30분 이상 걸림). 상수만 그대로 옮겨 쓴다.

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


# gsplat/cuda/include/Common.h 에 있는 값 그대로
ALPHA_THRESHOLD = 1.0 / 255.0   # #define ALPHA_THRESHOLD (1.f / 255.f)
GAUSSIAN_EXTEND = 3.33          # #define GAUSSIAN_EXTEND 3.33f

print(f"ALPHA_THRESHOLD = {ALPHA_THRESHOLD:.8f}")
print(f"GAUSSIAN_EXTEND = {GAUSSIAN_EXTEND}")
# 출력: ALPHA_THRESHOLD = 0.00392157
# 출력: GAUSSIAN_EXTEND = 3.33

# %% [markdown]
# ## 1. 3.33의 정체 — $\sqrt{2\ln 255}$
#
# $$o\,e^{-r^2/(2\sigma^2)} = \frac{1}{255}
#   \;\Longrightarrow\; \frac{r^2}{2\sigma^2} = \ln(255\,o)
#   \;\Longrightarrow\; r = \sigma\sqrt{2\ln(255\,o)}$$
#
# $o = 1$을 넣으면 $\sqrt{2\ln 255}$가 나온다.

# %%
k1 = np.sqrt(2.0 * np.log(255.0))
print(f"sqrt(2*ln(255))        = {k1:.6f}")
print(f"소수 둘째 자리 반올림  = {round(k1, 2)}")
print(f"GAUSSIAN_EXTEND 와 일치: {round(k1, 2) == GAUSSIAN_EXTEND}")
# 출력: sqrt(2*ln(255))        = 3.329043
# 출력: 소수 둘째 자리 반올림  = 3.33
# 출력: GAUSSIAN_EXTEND 와 일치: True

# 역검증: r = 3.329043*sigma 에서의 알파가 정말 1/255 인가?
sigma = 4.0
r = k1 * sigma
alpha_at_r = 1.0 * np.exp(-r**2 / (2 * sigma**2))
print(f"\nsigma={sigma}, r={r:.4f} 에서 alpha = {alpha_at_r:.8f}  (1/255 = {ALPHA_THRESHOLD:.8f})")
# 출력:
# 출력: sigma=4.0, r=13.3162 에서 alpha = 0.00392157  (1/255 = 0.00392157)

# %% [markdown]
# ## 2. 확장 계수 $k(o) = \min\big(3.33,\ \sqrt{2\ln(o/(1/255))}\big)$
#
# CUDA 커널(`ProjectionEWA3DGSFused.cu`)의 해당 부분:
#
# ```c
# float extend = GAUSSIAN_EXTEND;
# if (opacities != nullptr) {
#     float opacity = opacities[bid * N + gid];
#     if (opacity < ALPHA_THRESHOLD) { /* radii = 0 → 완전 컬링 */ return; }
#     extend = min(GAUSSIAN_EXTEND, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
# }
# float radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
# float radius_y = ceilf(extend * sqrtf(covar2d[1][1]));
# ```

# %%
def extend_factor(o):
    """불투명도 o에 대한 sigma 배수. o < 1/255 이면 0 (컬링)."""
    o = np.asarray(o, dtype=np.float64)
    inside = 2.0 * np.log(np.maximum(o, 1e-30) / ALPHA_THRESHOLD)
    k = np.sqrt(np.maximum(inside, 0.0))          # ln이 음수면 해 없음 → 0
    k = np.minimum(GAUSSIAN_EXTEND, k)            # 3.33 상한 (절대 늘리지 않는다)
    return np.where(o < ALPHA_THRESHOLD, 0.0, k)


print(" o      255*o    ln(255o)   k(o)    면적비 (k/3.33)^2")
for o in [1.0, 0.5, 0.3, 0.1, 0.05, 0.01, ALPHA_THRESHOLD, 0.002]:
    k = float(extend_factor(o))
    ln = np.log(255.0 * o)
    print(f"{o:7.5f} {255*o:8.3f} {ln:9.3f}  {k:5.3f}   {(k/GAUSSIAN_EXTEND)**2:6.3f}")
# 출력:  o      255*o    ln(255o)   k(o)    면적비 (k/3.33)^2
# 출력: 1.00000  255.000     5.541  3.329    0.999
# 출력: 0.50000  127.500     4.848  3.114    0.874
# 출력: 0.30000   76.500     4.337  2.945    0.782
# 출력: 0.10000   25.500     3.239  2.545    0.584
# 출력: 0.05000   12.750     2.546  2.256    0.459
# 출력: 0.01000    2.550     0.936  1.368    0.169
# 출력: 0.00392    1.000     0.000  0.000    0.000
# 출력: 0.00200    0.510    -0.673  0.000    0.000
#
# → o가 1/255(0.00392) 이하면 ln이 0 또는 음수 → 반경 0 → Gaussian 통째로 컬링.

# %% [markdown]
# ## 3. 토이 씬: 64×48 이미지, 16px 타일
#
# 축정렬 반경 사각형이 겹치는 타일 개수를 3.33σ 고정 방식과 불투명도 인지 방식으로 비교한다.
# (gsplat의 AABB 타일 교차와 같은 규칙: `radii` 사각형이 걸치는 타일 전부)

# %%
W, H, TILE = 64, 48, 16
TW, TH = W // TILE, H // TILE          # 4 x 3 = 12 타일
print(f"이미지 {W}x{H}, 타일 {TILE}px → 격자 {TW}x{TH} = {TW*TH} 타일")
# 출력: 이미지 64x48, 타일 16px → 격자 4x3 = 12 타일


def tile_count(cx, cy, rx, ry):
    """중심 (cx,cy), 반경 (rx,ry) 사각형이 겹치는 타일 수 (이미지 밖은 클램프)."""
    if rx <= 0 or ry <= 0:
        return 0
    x0 = max(int(np.floor((cx - rx) / TILE)), 0)
    x1 = min(int(np.floor((cx + rx - 1e-9) / TILE)), TW - 1)
    y0 = max(int(np.floor((cy - ry) / TILE)), 0)
    y1 = min(int(np.floor((cy + ry - 1e-9) / TILE)), TH - 1)
    if x1 < x0 or y1 < y0:
        return 0
    return (x1 - x0 + 1) * (y1 - y0 + 1)


# (이름, cx, cy, sigma_x, sigma_y, opacity)
scene = [
    ("g0 불투명", 16.0, 12.0, 4.0, 4.0, 1.00),
    ("g1 중간",   32.0, 24.0, 4.0, 3.0, 0.30),
    ("g2 옅음",   48.0, 12.0, 5.0, 5.0, 0.05),
    ("g3 큼",     32.0, 36.0, 7.0, 5.0, 0.60),
    ("g4 유령",   50.0, 36.0, 6.0, 6.0, 0.002),   # < 1/255 → 완전 컬링
]

rows, tot_fixed, tot_aware = [], 0, 0
print(f"{'gauss':10s} {'o':>6s} {'k(o)':>5s} {'radii(3.33σ)':>14s} {'radii(aware)':>14s} {'tiles':>12s}")
for name, cx, cy, sx, sy, o in scene:
    kf, ka = GAUSSIAN_EXTEND, float(extend_factor(o))
    rfx, rfy = np.ceil(kf * sx), np.ceil(kf * sy)
    rax, ray = np.ceil(ka * sx), np.ceil(ka * sy)
    tf, ta = tile_count(cx, cy, rfx, rfy), tile_count(cx, cy, rax, ray)
    tot_fixed += tf
    tot_aware += ta
    rows.append((name, o, ka, tf, ta))
    print(f"{name:10s} {o:6.3f} {ka:5.2f} {f'({rfx:.0f},{rfy:.0f})':>14s} "
          f"{f'({rax:.0f},{ray:.0f})':>14s} {f'{tf} → {ta}':>12s}")

print(f"\n총 타일 교차: 3.33σ 고정 = {tot_fixed},  불투명도 인지 = {tot_aware} "
      f"({100*(1-tot_aware/tot_fixed):.1f}% 절감)")
# 출력: gauss           o  k(o)   radii(3.33σ)   radii(aware)        tiles
# 출력: g0 불투명      1.000  3.33        (14,14)        (14,14)        4 → 4
# 출력: g1 중간       0.300  2.95        (14,10)         (12,9)        6 → 6
# 출력: g2 옅음       0.050  2.26        (17,17)        (12,12)        6 → 4
# 출력: g3 큼        0.600  3.17        (24,17)        (23,16)        8 → 8
# 출력: g4 유령       0.002  0.00        (20,20)          (0,0)        6 → 0
# 출력:
# 출력: 총 타일 교차: 3.33σ 고정 = 30,  불투명도 인지 = 22 (26.7% 절감)
#
# → 타일 교차 개수 = 정렬해야 할 키 개수 = 블렌딩 루프 길이. 30개 → 22개.
#   g4처럼 o < 1/255 인 Gaussian은 통째로 사라진다(6 → 0).
#   g1/g3처럼 반경이 줄어도 같은 타일 경계 안에 머무르면 타일 수는 그대로다 —
#   절감은 반경이 타일 경계를 넘느냐에 달려 있어, 실제 씬(수십만 Gaussian)에서 통계적으로 나타난다.

# %% [markdown]
# ## 4. 시각화 (3패널)
#
# 1. 확장 계수 곡선 $k(o)$ — $o$를 0.002~1로 스윕
# 2. $\sigma$가 같은 세 Gaussian($o = 1.0/0.3/0.05$)의 알파 프로파일과 잘리는 지점
# 3. 토이 씬의 Gaussian별 타일 수: 3.33σ 고정 vs 불투명도 인지

# %%
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("① 확장 계수 k(o) = min(3.33, √(2 ln(255o)))",
                    "② 알파 프로파일 (σ=4 공통)",
                    "③ 타일 교차 수 (64×48, 16px 타일)"),
)

# --- ① k(o) 곡선 ---
o_sweep = np.linspace(0.002, 1.0, 800)
fig.add_trace(go.Scatter(x=o_sweep, y=extend_factor(o_sweep), mode="lines",
                         name="k(o)", line=dict(color="#2563eb", width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=[0.002, 1.0], y=[3.33, 3.33], mode="lines", name="3.33 (고정)",
                         line=dict(color="#dc2626", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=[ALPHA_THRESHOLD, ALPHA_THRESHOLD], y=[0, 3.5], mode="lines",
                         name="o = 1/255 (컬링 경계)",
                         line=dict(color="#9ca3af", dash="dot")), row=1, col=1)
fig.update_xaxes(title_text="불투명도 o", row=1, col=1)
fig.update_yaxes(title_text="σ 배수", range=[0, 3.6], row=1, col=1)

# --- ② 알파 프로파일 ---
sig = 4.0
rr = np.linspace(0, 16, 600)
for o, col in [(1.0, "#111827"), (0.3, "#0891b2"), (0.05, "#ea580c")]:
    a = o * np.exp(-rr**2 / (2 * sig**2))
    fig.add_trace(go.Scatter(x=rr, y=a, mode="lines", name=f"o={o}",
                             line=dict(color=col, width=2.5)), row=1, col=2)
    rc = float(extend_factor(o)) * sig                      # 잘리는 반경
    fig.add_trace(go.Scatter(x=[rc], y=[ALPHA_THRESHOLD], mode="markers+text",
                             text=[f"{rc:.1f}px"], textposition="top right",
                             marker=dict(color=col, size=10, symbol="x"),
                             showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[0, 16], y=[ALPHA_THRESHOLD, ALPHA_THRESHOLD], mode="lines",
                         name="1/255", line=dict(color="#dc2626", dash="dash")), row=1, col=2)
fig.update_xaxes(title_text="중심으로부터 거리 r (px)", row=1, col=2)
fig.update_yaxes(title_text="알파", type="log", range=[-4, 0.05], row=1, col=2)

# --- ③ 타일 수 막대 ---
names = [f"{r[0]}<br>o={r[1]:g}" for r in rows]
fig.add_trace(go.Bar(x=names, y=[r[3] for r in rows], name="3.33σ 고정",
                     marker_color="#dc2626"), row=1, col=3)
fig.add_trace(go.Bar(x=names, y=[r[4] for r in rows], name="불투명도 인지",
                     marker_color="#2563eb"), row=1, col=3)
fig.update_yaxes(title_text="겹치는 타일 수", row=1, col=3)

fig.update_layout(
    height=460, width=1500, barmode="group",
    title_text=f"불투명도 인지 반경 — 토이 씬 타일 교차 {tot_fixed} → {tot_aware} "
               f"({100*(1-tot_aware/tot_fixed):.0f}% 절감)",
    legend=dict(orientation="h", y=-0.22),
    template="plotly_white",
)
_show(fig)

import os
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expy.png")
fig.write_image(_out, scale=2)
print("저장:", _out)
# 출력: 저장: .../hints/dfca81f4-aa19-4997-84c7-1272ea7c82d2/expy.png

# %% [markdown]
# ## 5. 정리
#
# - $3.33$은 상수가 아니라 $\sqrt{2\ln 255} = 3.3291\ldots$의 반올림값 — "$o=1$일 때 알파가
#   $1/255$로 떨어지는 반경"이다.
# - 불투명도를 알면 그 지점을 $\sigma\sqrt{2\ln(255\,o)}$로 정확히 다시 계산할 수 있고,
#   $o<1$이면 항상 더 작다.
# - $o < 1/255$면 $\ln$이 음수 → 실수 해 없음 → 중심에서도 안 보임 → 반경 0으로 완전 컬링.
# - $\min(3.33,\cdot)$은 "이 상자는 줄이기만 한다"는 계약을 지키는 안전장치(+ 분기 없는 GPU 코드).
# - 잘라낸 영역의 기여는 정의상 $1/255$ 미만 → 8비트 출력에서 사라짐 → **화질 손실 없이**
#   정렬 키 개수와 블렌딩 루프 길이만 줄어든다.
