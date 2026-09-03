# %% [markdown]
# # backward 커널의 투과율 $T$ 되감기
#
# forward 알파 블렌딩:
#
# $$C = \sum_{i} c_i\,\alpha_i\,T_i,\qquad T_0 = 1,\quad T_{i+1} = T_i(1-\alpha_i)$$
#
# forward는 **최종 $T$** 와 **마지막 인덱스**만 저장한다. backward는 같은 목록을
# 뒤→앞으로 훑으며 `T /= (1 - alpha)` 로 중간 $T_i$ 를 복원한다.
#
# 이 스크립트에서 확인할 것:
# 1. 되감기가 forward $T$ 를 정확히 복원 (float64 ~1e-16, float32 ~1e-6)
# 2. 복원한 $T_i$ 와 뒤쪽 누적 색 $S_{i+1}$ 로 $\partial C/\partial\alpha_i$ 계산 → autograd와 일치
# 3. $\alpha=0.999$ / $\alpha=1.0$ 에서 되감기가 깨지는 모습 (MAX_ALPHA=0.99 의 존재 이유)
# 4. forward $T$ vs 복원 $T$ 겹쳐 그리기 + float32 오차 막대 (plotly → expy.png)

# %%
# 필요 패키지: numpy, torch, plotly, kaleido (expy.png 저장용)
# 주의: gsplat 은 import 하지 않는다 (JIT CUDA 빌드가 30분 이상 걸림)
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# gsplat/cuda/_constants.py, gsplat/cuda/include/Common.h 의 실제 값
MAX_ALPHA = 0.99
TRANSMITTANCE_THRESHOLD = 1e-4
MIN_ONE_MINUS_ALPHA = 1e-6

np.set_printoptions(precision=8, suppress=False)
print("MAX_ALPHA =", MAX_ALPHA, "| MIN_ONE_MINUS_ALPHA =", MIN_ONE_MINUS_ALPHA)
# 출력: MAX_ALPHA = 0.99 | MIN_ONE_MINUS_ALPHA = 1e-06

# %% [markdown]
# ## 1. 손으로 확인 — $\alpha = (0.5,\ 0.2,\ 0.8)$
#
# 앞으로 곱하고, 마지막 $T$ 에서 나눗셈으로 되감으면 같은 값이 나온다.
# $T_{i+1} = T_i(1-\alpha_i)$ 는 (공비가 매번 바뀌는) 등비수열이므로
# $T_i = T_{i+1}/(1-\alpha_i)$ 로 정확히 한 칸 뒤로 갈 수 있다.

# %%
a3 = np.array([0.5, 0.2, 0.8])

T_fwd3 = [1.0]
for a in a3:
    T_fwd3.append(float(T_fwd3[-1] * (1.0 - a)))
print("forward T:", T_fwd3)          # T_0..T_3
# 출력: forward T: [1.0, 0.5, 0.4, 0.07999999999999999]

T = T_fwd3[-1]                        # forward 가 저장한 유일한 값
T_bwd3 = [T]
for a in a3[::-1]:                    # 뒤 → 앞
    T = float(T / (1.0 - a))
    T_bwd3.append(T)
T_bwd3 = T_bwd3[::-1]
print("복원  T:", T_bwd3)
# 출력: 복원  T: [1.0, 0.5, 0.4, 0.07999999999999999]
print("최대 오차:", max(abs(f - b) for f, b in zip(T_fwd3, T_bwd3)))
# 출력: 최대 오차: 0.0
#   (0.4 -> 0.08 -> 0.4 처럼 곱했다 나누면 float 에서도 대개 정확히 되돌아온다)

# %% [markdown]
# ## 2. 픽셀 하나, Gaussian 8개 — float64 / float32 되감기
#
# 실제 커널처럼 forward 는 `T_final` 과 `last_id` 만 남기고, backward 는
# 같은 목록을 역순으로 걸으며 $T$ 를 복원한다.

# %%
rng = np.random.default_rng(0)
N = 8
alphas64 = np.minimum(rng.uniform(0.05, 0.95, size=N), MAX_ALPHA)
colors64 = rng.uniform(0.0, 1.0, size=(N, 3))
print("alpha:", alphas64)
# 출력: alpha: [0.62326552 0.29280804 0.08687617 0.06487487 0.78194322 0.87148002
#                0.5959722  0.7065469 ]


def forward(alphas, colors, dtype=np.float64):
    """앞→뒤 블렌딩. 커널이 저장하는 것은 T_final(=1-render_alpha)과 last_id 뿐."""
    alphas, colors = alphas.astype(dtype), colors.astype(dtype)
    T = dtype(1.0)
    C = np.zeros(colors.shape[1], dtype=dtype)
    T_traj = [T]                      # 이건 '검증용'으로만 남긴다 (커널은 저장 안 함)
    last_id = len(alphas)
    for i, a in enumerate(alphas):
        if T <= TRANSMITTANCE_THRESHOLD:   # 조기 종료: 이 Gaussian은 제외
            last_id = i
            break
        C = C + colors[i] * a * T
        T = T * (dtype(1.0) - a)
        T_traj.append(T)
    return C, T, last_id, np.array(T_traj)


def backward_restore_T(alphas, T_final, last_id, dtype=np.float64):
    """뒤→앞으로 T /= (1-alpha).  커널의 `float ra = 1/max(MIN, 1-alpha); T *= ra;`"""
    alphas = alphas.astype(dtype)
    T = dtype(T_final)
    T_rest = np.empty(last_id + 1, dtype=dtype)
    T_rest[last_id] = T
    for i in range(last_id - 1, -1, -1):
        ra = dtype(1.0) / max(dtype(MIN_ONE_MINUS_ALPHA), dtype(1.0) - alphas[i])
        T = T * ra                    # T_{i+1} -> T_i
        T_rest[i] = T
    return T_rest


C64, Tf64, last64, Ttraj64 = forward(alphas64, colors64, np.float64)
Trest64 = backward_restore_T(alphas64, Tf64, last64, np.float64)
print("last_id =", last64, "| T_final =", Tf64)
# 출력: last_id = 8 | T_final = 0.0007558964193787512
print("forward T :", Ttraj64)
# 출력: forward T : [1.00000000e+00 3.76734481e-01 2.66423595e-01 2.43277733e-01
#                   2.27495122e-01 4.96068548e-02 6.37547200e-03 2.57586794e-03
#                   7.55896419e-04]
print("복원  T   :", Trest64)
# 출력: 복원  T   : (위와 육안으로 동일)
print("float64 최대 오차:", np.abs(Ttraj64[:last64 + 1] - Trest64).max())
# 출력: float64 최대 오차: 5.551115123125783e-17

C32, Tf32, last32, Ttraj32 = forward(alphas64, colors64, np.float32)
Trest32 = backward_restore_T(alphas64, Tf32, last32, np.float32)
err32 = np.abs(Ttraj32[:last32 + 1] - Trest32)
rel32 = err32 / np.maximum(Ttraj32[:last32 + 1], 1e-30)
print("float32 최대 절대오차:", err32.max(), "| 최대 상대오차:", rel32.max())
# 출력: float32 최대 절대오차: 1.1920929e-07 | 최대 상대오차: 1.1920929e-07
#   float32 machine eps(~6e-8)의 두 배 수준. gradient 스케일에서 무시 가능.

# %% [markdown]
# 되감기 길이에 따른 float32 오차 누적. 연산당 상대오차 $\varepsilon\approx 6\times10^{-8}$ 이
# 대략 $n\varepsilon$ 로 쌓인다(실제로는 부호가 섞여 $\sqrt{n}\,\varepsilon$ 에 가깝다).
# 픽셀당 Gaussian 수를 늘리려면 $\alpha$ 가 작아야 하므로(안 그러면 $T$ 가 1e-4 밑으로
# 떨어져 조기 종료) 여기서는 $\alpha \sim U(0.001, 0.03)$ 을 쓴다.

# %%
print(f"{'n':>5} {'되감기 길이':>10} {'최대 상대오차(f32)':>20}")
for n in [8, 64, 512]:
    a = np.minimum(rng.uniform(0.001, 0.03, size=n), MAX_ALPHA)
    c = rng.uniform(0.0, 1.0, size=(n, 3))
    _, Tf, li, Tt = forward(a, c, np.float32)
    Tr = backward_restore_T(a, Tf, li, np.float32)
    rel = np.abs(Tt[:li + 1] - Tr) / Tt[:li + 1]
    print(f"{n:5d} {li:10d} {rel.max():20.3e}")
# 출력:     n     되감기 길이     최대 상대오차(f32)
# 출력:     8          8            1.258e-07
# 출력:    64         64            1.856e-07
# 출력:   512        512            2.061e-06
#   n이 64배 늘어도 오차는 ~16배. 500개를 되감아도 2e-6 수준.

# %% [markdown]
# ## 3. 되감은 $T_i$ 와 $S_{i+1}$ 로 gradient 만들기
#
# $$\frac{\partial C}{\partial \alpha_i} = c_i T_i - \frac{1}{1-\alpha_i}\sum_{j>i} c_j\alpha_j T_j
#  = T_i\,(c_i - S_{i+1}),\qquad S_{i+1}\equiv\frac{1}{T_{i+1}}\sum_{j>i}c_j\alpha_j T_j$$
#
# 커널은 $\sum_{j>i}c_j\alpha_j T_j$ 를 `buffer[k]` 에 뒤→앞으로 누적하고
# `v_alpha += (rgb*T - buffer*ra) * v_render_c` 로 이 식을 그대로 계산한다.
# 아래에서 `torch.autograd` 결과와 비교한다.

# %%
def backward_kernel(alphas, colors, v_C, T_final, last_id):
    """뒤→앞 한 번의 패스로 T 와 buffer 를 동시에 되감아 v_alpha, v_rgb 를 얻는다."""
    T = float(T_final)
    buffer = np.zeros(colors.shape[1])       # sum_{j>i} c_j alpha_j T_j
    v_alpha = np.zeros(last_id)
    v_rgb = np.zeros_like(colors)
    for i in range(last_id - 1, -1, -1):
        ra = 1.0 / max(MIN_ONE_MINUS_ALPHA, 1.0 - alphas[i])
        T = T * ra                            # <-- T_{i+1} -> T_i  (커널의 `T *= ra`)
        fac = alphas[i] * T                   # alpha_i * T_i
        v_rgb[i] = fac * v_C                  # v_rgb_local[k] = fac * v_render_c[k]
        v_alpha[i] = np.dot(colors[i] * T - buffer * ra, v_C)
        buffer += colors[i] * fac             # 다음(=앞쪽) 스텝을 위해 누적
    return v_alpha, v_rgb


v_C = np.array([1.0, -0.5, 0.25])            # 상류 gradient dL/dC
va_ker, vrgb_ker = backward_kernel(alphas64, colors64, v_C, Tf64, last64)

# torch.autograd 로 같은 것 계산
ta = torch.tensor(alphas64, requires_grad=True)
tc = torch.tensor(colors64, requires_grad=True)
T_t = torch.ones((), dtype=torch.float64)
C_t = torch.zeros(3, dtype=torch.float64)
for i in range(last64):
    C_t = C_t + tc[i] * ta[i] * T_t
    T_t = T_t * (1.0 - ta[i])
(C_t * torch.tensor(v_C)).sum().backward()

print("kernel  v_alpha:", va_ker)
# 출력: kernel  v_alpha: [ 0.20684728 -0.26144356  0.16958249  0.07184689 -0.07106252 -0.00416371
#                         0.00250626  0.00103928]
print("autograd v_alpha:", ta.grad.numpy())
# 출력: autograd v_alpha: (위와 동일)
print("v_alpha 최대 오차:", np.abs(va_ker - ta.grad.numpy()).max())
# 출력: v_alpha 최대 오차: 4.163336342344337e-17
print("v_rgb   최대 오차:", np.abs(vrgb_ker - tc.grad.numpy()).max())
# 출력: v_rgb   최대 오차: 2.7755575615628914e-17

# %% [markdown]
# ### $\partial C/\partial\alpha_i = T_i(c_i - S_{i+1})$ 형태로도 확인
#
# 되감기로 얻은 $T_i$ 와 $S_{i+1}$ 만으로 같은 값이 나오는지 본다.

# %%
S = np.zeros((last64 + 1, 3))                # S_{i} : i 이후(뒤쪽)만 렌더링한 색
for i in range(last64 - 1, -1, -1):
    S[i] = colors64[i] * alphas64[i] + (1 - alphas64[i]) * S[i + 1]
va_form = np.array([Trest64[i] * np.dot(colors64[i] - S[i + 1], v_C) for i in range(last64)])
print("T_i (c_i - S_{i+1}) 형태와의 오차:", np.abs(va_form - ta.grad.numpy()).max())
# 출력: T_i (c_i - S_{i+1}) 형태와의 오차: 2.7755575615628914e-17

# %% [markdown]
# ## 4. 왜 $\alpha$ 를 0.99로 자르는가
#
# 되감기의 유일한 전제는 $1-\alpha_i \ne 0$. $\alpha=1$ 이면 forward 에서 $T=0$ 이 되고
# $0/0$ 이라 **복원할 정보 자체가 사라진다**. $\alpha$ 가 1에 가까울수록 나눗셈 계수
# $1/(1-\alpha)$ 가 커져 float32 오차를 증폭한다.

# %%
# (a) alpha = 1 이면 T_final = 0. 되감기 자체가 불가능하고, floor 가 없으면 NaN/Inf.
a_bad = np.array([0.3, 1.0, 0.4])
_, Tf_bad, li_bad, Tt_bad = forward(a_bad, rng.uniform(0, 1, (3, 3)), np.float32)
print("alpha에 1.0 포함 -> forward T:", Tt_bad, "| T_final:", Tf_bad)
# 출력: alpha에 1.0 포함 -> forward T: [1.  0.7 0. ] | T_final: 0.0
#   (T=0 이 되어 다음 스텝에서 조기 종료, last_id=2)
print("  복원 시도:", backward_restore_T(a_bad, Tf_bad, li_bad, np.float32))
# 출력:   복원 시도: [0. 0. 0.]   <- T_0=1 을 못 되살린다 (정보 소실)
print("  floor 없이 0/(1-1.0) =", np.float32(0.0) / np.float32(0.0))
# 출력:   floor 없이 0/(1-1.0) = nan
print("  floor 없이 1e-3/(1-1.0) =", np.float32(1e-3) / np.float32(0.0))
# 출력:   floor 없이 1e-3/(1-1.0) = inf
print("  MIN_ONE_MINUS_ALPHA 적용:",
      np.float32(1e-3) / max(np.float32(MIN_ONE_MINUS_ALPHA), np.float32(0.0)))
# 출력:   MIN_ONE_MINUS_ALPHA 적용: 1000.00006
#   NaN 은 atomicAdd 로 전체 파라미터에 퍼지므로 '유한한 큰 값'이 훨씬 낫다.

# %% [markdown]
# ### (b) 진짜 위험은 저장 포맷에 있다
#
# forward 는 $T_\text{final}$ 을 그대로 두지 않고 `render_alphas = 1 - T_final` 로 저장하고,
# backward 는 `T_final = 1 - render_alphas[pix_id]` 로 되살린다
# (`RasterizeToPixels3DGSSerialBatchBwd.cu:135`).
# $T_\text{final}$ 이 작을수록 $1-T_\text{final}$ 은 1에 붙고, float32 는 1 근처에서
# 절대 분해능이 $\sim 6\times10^{-8}$ 뿐이라 **작은 $T$ 가 통째로 갈려 나간다**.
# 이 상대오차는 되감기 나눗셈을 타고 모든 $T_i$ 에 그대로 전파된다.
# $\text{TRANSMITTANCE\_THRESHOLD} = (1-\text{MAX\_ALPHA})^2 = 10^{-4}$ 가 $T_\text{final}$ 의
# 하한을 지켜 주는 이유다.

# %%
print(f"{'T_final':>10} {'1-T (f32)':>14} {'복원 T_final':>14} {'상대오차':>12}")
for T_true in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
    ra_stored = np.float32(1.0) - np.float32(T_true)      # forward 저장
    T_back = np.float32(1.0) - ra_stored                  # backward 복원
    rel = abs(float(T_back) - T_true) / T_true
    mark = "  <- TRANSMITTANCE_THRESHOLD" if T_true == 1e-4 else ""
    print(f"{T_true:10.0e} {ra_stored:14.9f} {T_back:14.6e} {rel:12.3e}{mark}")
# 출력:    T_final      1-T (f32)     복원 T_final         상대오차
# 출력:      1e-02    0.990000010   9.999990e-03    9.537e-07
# 출력:      1e-03    0.999000013   9.999871e-04    1.287e-05
# 출력:      1e-04    0.999899983   1.000166e-04    1.659e-04  <- TRANSMITTANCE_THRESHOLD
# 출력:      1e-05    0.999989986   1.001358e-05    1.358e-03
# 출력:      1e-06    0.999998987   1.013279e-06    1.328e-02
# 출력:      1e-08    1.000000000   0.000000e+00    1.000e+00
#   T_final 이 1e-4 아래로 내려가면 저장 왕복만으로 유효숫자가 무너진다.
#   alpha 를 0.99 로 자르고 T <= 1e-4 에서 조기 종료하는 것이 바로 이 영역을 피하는 장치.

# %% [markdown]
# 그 오차가 되감기를 타고 어떻게 퍼지는지 확인한다.
# $\alpha$ 상한만 바꿔서 $T_\text{final}$ 을 작게 만든 뒤, 저장 왕복을 거친 값으로 되감아 본다.

# %%
print(f"{'alpha 상한':>10} {'T_final':>12} {'복원 T_0':>12} {'T_0 상대오차':>14}")
for a_cap in [0.99, 0.999, 0.9999, 0.99999]:
    a = np.array([0.5, a_cap, 0.4])
    Tf = np.float32(np.prod(1.0 - a))
    Tf_roundtrip = np.float32(1.0) - (np.float32(1.0) - Tf)   # render_alpha 저장 왕복
    T0 = backward_restore_T(a, Tf_roundtrip, len(a), np.float32)[0]
    print(f"{a_cap:10.5f} {Tf:12.3e} {T0:12.6f} {abs(float(T0) - 1.0):14.3e}")
# 출력:   alpha 상한      T_final       복원 T_0       T_0 상대오차
# 출력:    0.99000    3.000e-03     1.000008      7.987e-06
# 출력:    0.99900    3.000e-04     0.999980      1.991e-05
# 출력:    0.99990    3.000e-05     0.999205      7.946e-04
# 출력:    0.99999    3.000e-06     0.992063      7.937e-03
#   T_0 는 정의상 정확히 1 이어야 한다. alpha 가 1에 가까울수록 복원값이 1에서 멀어진다.

# %% [markdown]
# ## 5. 시각화 — forward $T$ vs 복원 $T$, 그리고 float32 오차

# %%
fig = make_subplots(
    rows=1, cols=2, subplot_titles=("forward T (앞→뒤) vs 복원 T (뒤→앞)",
                                    "float32 되감기 상대오차"),
    horizontal_spacing=0.12,
)
idx = np.arange(last64 + 1)
fig.add_trace(go.Scatter(x=idx, y=Ttraj64[:last64 + 1], mode="lines+markers",
                         name="forward T (저장 안 함)", line=dict(width=3, color="#1f77b4")),
              row=1, col=1)
fig.add_trace(go.Scatter(x=idx, y=Trest64, mode="markers", name="backward 복원 T",
                         marker=dict(size=12, symbol="circle-open",
                                     line=dict(width=3, color="#d62728"))),
              row=1, col=1)
fig.add_trace(go.Scatter(x=[last64], y=[Tf64], mode="markers+text",
                         name="forward가 저장한 T_final", text=["저장된 값"],
                         textposition="top center",
                         marker=dict(size=16, symbol="star", color="#2ca02c")),
              row=1, col=1)
fig.add_trace(go.Bar(x=np.arange(last32 + 1), y=rel32, name="float32 상대오차",
                     marker_color="#ff7f0e"), row=1, col=2)
fig.update_xaxes(title_text="Gaussian 인덱스 i", row=1, col=1)
fig.update_xaxes(title_text="Gaussian 인덱스 i", row=1, col=2)
fig.update_yaxes(title_text="T_i", type="log", row=1, col=1)
fig.update_yaxes(title_text="|T_fwd - T_rest| / T_fwd", tickformat=".1e",
                 rangemode="tozero", row=1, col=2)   # 선형축: 오차 0인 항목도 보이게
fig.update_layout(title="backward 는 최종 T 하나에서 T /= (1-α) 로 모든 중간 T 를 되살린다",
                  height=460, width=1100, template="plotly_white")
_show(fig)
fig.write_image("expy.png", scale=2)         # kaleido 필요
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 정리
#
# - $T_{i+1}=T_i(1-\alpha_i)$ 는 곱셈 점화식이므로 나눗셈으로 정확히 되감긴다.
# - forward 는 `T_final`(=1-render_alpha)과 `last_ids` 만 저장 → 메모리 $O(\text{픽셀})$.
# - backward 는 같은 타일 목록을 뒤→앞으로 훑으며 `T /= (1-α)` 로 $T_i$ 를,
#   `buffer` 로 $S_{i+1}$ 을 되감아 $\partial C/\partial\alpha_i = T_i(c_i - S_{i+1})$ 을 만든다.
# - 이것이 성립하려면 $\alpha<1$ 이어야 하고, 그래서 `MAX_ALPHA = 0.99` 와
#   `MIN_ONE_MINUS_ALPHA = 1e-6` 이 존재한다.
