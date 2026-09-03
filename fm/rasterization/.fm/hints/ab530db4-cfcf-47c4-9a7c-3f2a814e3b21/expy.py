# %% [markdown]
# # 알파 블렌딩 누적 공식 — 앞→뒤 순회를 손으로 굴려 보기
#
# $$\alpha_i = \min(0.99,\ o_i e^{-\sigma_i}),\qquad
#   C_p = \sum_i c_i\,\alpha_i\,T_i,\qquad
#   T_{i+1} = T_i(1-\alpha_i),\qquad
#   \text{render\_alpha} = 1 - T_{end}$$
#
# 확인할 것:
#
# 1. 한 픽셀에 Gaussian 5개가 앞→뒤로 놓였을 때 $T$, 기여 $\alpha_i T_i$, 누적 색을 표로
# 2. $\sum_i \alpha_i T_i = 1 - T_{end}$ (망원급수)
# 3. **순서를 뒤집으면 색은 달라지지만 alpha는 같다**
# 4. `MAX_ALPHA=0.99` / `ALPHA_THRESHOLD=1/255` / `TRANSMITTANCE_THRESHOLD=1e-4`의 효과
# 5. 1D 단면에서 Gaussian 여러 개가 겹친 "픽셀 행"을 렌더 → 컬러 띠 + $T$ 감쇠 곡선
#
# 필요 패키지: numpy, plotly (+ 정적 이미지 저장용 kaleido). gsplat은 import하지 않는다.

# %%
# 필요 패키지: numpy, plotly, kaleido
import os

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


# gsplat/cuda/_constants.py 와 gsplat/cuda/include/Common.h 에 같은 값으로 정의된 상수
ALPHA_THRESHOLD = 1.0 / 255.0  # 이보다 작은 α는 건너뜀 (8bit 반올림하면 0)
MAX_ALPHA = 0.99  # α 상한 (backward의 1/(1-α) 보호)
TRANSMITTANCE_THRESHOLD = 1e-4  # T가 이 이하면 그 픽셀 조기 종료. = (1-MAX_ALPHA)^2

np.set_printoptions(precision=4, suppress=True)
print(f"ALPHA_THRESHOLD        = {ALPHA_THRESHOLD:.6f}")
print(f"MAX_ALPHA              = {MAX_ALPHA}")
print(f"TRANSMITTANCE_THRESHOLD= {TRANSMITTANCE_THRESHOLD}   == (1-MAX_ALPHA)^2 = {(1-MAX_ALPHA)**2}")
# 출력: ALPHA_THRESHOLD        = 0.003922
# 출력: MAX_ALPHA              = 0.99
# 출력: TRANSMITTANCE_THRESHOLD= 0.0001   == (1-MAX_ALPHA)^2 = 0.00010000000000000018
# 출력:   ↑ float에서 (1-0.99)^2 가 1e-4 보다 아주 조금 크다. §5에서 이게 눈에 보인다.

# %% [markdown]
# ## 1. 한 픽셀, Gaussian 5개 — 루프를 그대로 굴린다
#
# 커널 안쪽 두 줄이 전부다.
#
# ```
# C += c_i * alpha_i * T ;  T *= (1 - alpha_i)
# ```

# %%
# 앞(카메라에 가까움) → 뒤 순서. 색은 RGB.
names = ["빨강", "초록", "파랑", "노랑", "회색"]
colors5 = np.array(
    [
        [1.0, 0.1, 0.1],  # 빨강
        [0.1, 0.9, 0.2],  # 초록
        [0.2, 0.3, 1.0],  # 파랑
        [1.0, 0.9, 0.1],  # 노랑
        [0.5, 0.5, 0.5],  # 회색
    ]
)
alphas5 = np.array([0.30, 0.50, 0.40, 0.70, 0.60])


def blend(colors, alphas, verbose=True):
    """앞→뒤 알파 블렌딩. 커널 루프 그대로."""
    T = 1.0
    C = np.zeros(3)
    rows = []
    for i, (c, a) in enumerate(zip(colors, alphas)):
        w = a * T  # 기여 가중치 α_i T_i
        C = C + c * w
        T_next = T * (1.0 - a)
        rows.append((i, a, T, w, T_next, C.copy()))
        T = T_next
    if verbose:
        print(f"{'i':>2} {'α_i':>6} {'T_i':>8} {'α_i·T_i':>9} {'T_{i+1}':>9}   누적 색 C")
        for i, a, Ti, w, Tn, Cc in rows:
            print(f"{i:>2} {a:>6.2f} {Ti:>8.4f} {w:>9.4f} {Tn:>9.4f}   {Cc}")
    return C, T, rows


C5, T5, rows5 = blend(colors5, alphas5)
print(f"\n최종 색      C_p = {C5}")
print(f"최종 투과율  T_end = {T5:.6f}")
print(f"render_alpha = 1 - T = {1 - T5:.6f}")
# 출력:  i    α_i      T_i   α_i·T_i   T_{i+1}   누적 색 C
# 출력:  0   0.30   1.0000    0.3000    0.7000   [0.3  0.03 0.03]
# 출력:  1   0.50   0.7000    0.3500    0.3500   [0.335 0.345 0.1  ]
# 출력:  2   0.40   0.3500    0.1400    0.2100   [0.363 0.387 0.24 ]
# 출력:  3   0.70   0.2100    0.1470    0.0630   [0.51   0.5193 0.2547]
# 출력:  4   0.60   0.0630    0.0378    0.0252   [0.5289 0.5382 0.2736]
# 출력:
# 출력: 최종 색      C_p = [0.5289 0.5382 0.2736]
# 출력: 최종 투과율  T_end = 0.025200
# 출력: render_alpha = 1 - T = 0.974800

# %% [markdown]
# ## 2. $\sum_i \alpha_i T_i = 1 - T_{end}$ — 망원급수 확인
#
# $\alpha_i T_i = T_i - T_{i+1}$ 이므로 합이 접힌다:
#
# $$\sum_{i=1}^{n}\alpha_i T_i = (T_1 - T_2) + (T_2 - T_3) + \cdots + (T_n - T_{n+1}) = 1 - T_{n+1}$$

# %%
w_sum = sum(r[3] for r in rows5)  # Σ α_i T_i
prod = np.prod(1.0 - alphas5)  # Π (1-α_i) = T_end
print(f"Σ α_i·T_i        = {w_sum:.10f}")
print(f"1 - Π(1-α_i)     = {1 - prod:.10f}")
print(f"1 - T_end        = {1 - T5:.10f}")
print(f"일치 여부        : {np.allclose([w_sum, 1 - prod], 1 - T5)}")

# 항별로 T_i - T_{i+1} 인지도 확인
tele = [abs(r[3] - (r[2] - r[4])) for r in rows5]
print(f"max |α_i T_i - (T_i - T_(i+1))| = {max(tele):.3e}")
# 출력: Σ α_i·T_i        = 0.9748000000
# 출력: 1 - Π(1-α_i)     = 0.9748000000
# 출력: 1 - T_end        = 0.9748000000
# 출력: 일치 여부        : True
# 출력: max |α_i T_i - (T_i - T_(i+1))| = 5.551e-17

# %% [markdown]
# ## 3. 순서를 뒤집으면? — 색은 달라지고 alpha는 그대로
#
# $C_p = \sum c_i\alpha_i T_i$ 는 순서 의존이지만
# $1 - T_{end} = 1 - \prod(1-\alpha_i)$ 는 곱이라 **순서 무관**이다.
# 그래서 깊이 정렬은 색을 위해 필요하다.

# %%
C_rev, T_rev, _ = blend(colors5[::-1], alphas5[::-1], verbose=False)
print(f"앞→뒤 원래 순서 : C = {C5}   alpha = {1 - T5:.6f}")
print(f"뒤집은 순서     : C = {C_rev}   alpha = {1 - T_rev:.6f}")
print(f"색 차이 (L∞)    : {np.abs(C5 - C_rev).max():.4f}")
print(f"alpha 차이      : {abs(T5 - T_rev):.3e}   ← 0")

# 극단 예: 2장, α=0.8, 빨강 vs 파랑
c2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
a2 = np.array([0.8, 0.8])
Ca, Ta, _ = blend(c2, a2, verbose=False)
Cb, Tb, _ = blend(c2[::-1], a2, verbose=False)
print(f"\n빨강이 앞 : C = {Ca}  alpha = {1-Ta:.4f}")
print(f"파랑이 앞 : C = {Cb}  alpha = {1-Tb:.4f}")
print("→ 같은 두 판인데 색은 5배 차이, alpha는 동일")
# 출력: 앞→뒤 원래 순서 : C = [0.5289 0.5382 0.2736]   alpha = 0.974800
# 출력: 뒤집은 순서     : C = [0.604  0.5999 0.3843]   alpha = 0.974800
# 출력: 색 차이 (L∞)    : 0.1107
# 출력: alpha 차이      : 0.000e+00   ← 0
# 출력:
# 출력: 빨강이 앞 : C = [0.8  0.   0.16]  alpha = 0.9600
# 출력: 파랑이 앞 : C = [0.16 0.   0.8 ]  alpha = 0.9600
# 출력: → 같은 두 판인데 색은 5배 차이, alpha는 동일

# %% [markdown]
# ## 4. σ → α: `MAX_ALPHA` 상한과 `ALPHA_THRESHOLD` 하한
#
# $\sigma_i = \tfrac12(a\,dx^2 + c\,dy^2) + b\,dx\,dy$ 는 타원 모양으로 잰 거리의 제곱의 절반.
# $\alpha_i = \min(0.99,\ o_i e^{-\sigma_i})$.
#
# - $\sigma$가 작으면(중심 근처) $o_i e^{-\sigma}$가 $o_i$에 육박 → **0.99로 잘린다**
# - $\sigma$가 크면(가장자리) $\alpha < 1/255$ → **건너뛴다** = Gaussian이 유한 반경으로 잘림

# %%
def eval_alpha(sigma, opac):
    """gsplat eval_gaussian_weight 와 동일: vis, alpha, valid."""
    vis = np.exp(-sigma)
    alpha = np.minimum(MAX_ALPHA, opac * vis)
    valid = (sigma >= 0) & (alpha >= ALPHA_THRESHOLD)
    return vis, alpha, valid


opac = 0.999
sig_grid = np.array([0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 5.5, 6.0, 8.0])
_, a_grid, v_grid = eval_alpha(sig_grid, opac)
print(f"o_i = {opac}")
print(f"{'σ':>6} {'o·e^-σ':>10} {'α=min(.99,·)':>13} {'clamp?':>7} {'valid?':>7}")
for s, a in zip(sig_grid, a_grid):
    raw = opac * np.exp(-s)
    print(f"{s:>6.3f} {raw:>10.6f} {a:>13.6f} {str(raw > MAX_ALPHA):>7} {str(a >= ALPHA_THRESHOLD):>7}")

# α가 1/255로 떨어지는 σ (= Gaussian의 잘림 반경)
sig_cut = -np.log(ALPHA_THRESHOLD / opac)
print(f"\nα = 1/255 가 되는 σ = {sig_cut:.4f}  →  마할라노비스 반경 sqrt(2σ) = {np.sqrt(2*sig_cut):.3f}σ")
print(f"(참고: GAUSSIAN_EXTEND = 3.33 과 같은 자릿수)")
print(f"1/(1-MAX_ALPHA) = {1/(1-MAX_ALPHA):.1f}   ← backward의 T /= (1-α) 되감기가 유한해짐")
# 출력: o_i = 0.999
# 출력:      σ     o·e^-σ  α=min(.99,·)  clamp?  valid?
# 출력:  0.000   0.999000      0.990000    True    True
# 출력:  0.001   0.998001      0.990000    True    True
# 출력:  0.010   0.989060      0.989060   False    True
# 출력:  0.100   0.903933      0.903933   False    True
# 출력:  0.500   0.605924      0.605924   False    True
# 출력:  1.000   0.367512      0.367512   False    True
# 출력:  2.000   0.135200      0.135200   False    True
# 출력:  4.000   0.018297      0.018297   False    True
# 출력:  5.500   0.004083      0.004083   False    True
# 출력:  6.000   0.002476      0.002476   False   False
# 출력:  8.000   0.000335      0.000335   False   False
# 출력:
# 출력: α = 1/255 가 되는 σ = 5.5403  →  마할라노비스 반경 sqrt(2σ) = 3.329σ
# 출력: (참고: GAUSSIAN_EXTEND = 3.33 과 같은 자릿수)
# 출력: 1/(1-MAX_ALPHA) = 100.0   ← backward의 T /= (1-α) 되감기가 유한해짐

# %% [markdown]
# ## 5. `TRANSMITTANCE_THRESHOLD` — 어느 인덱스에서 조기 종료되는가
#
# 커널은 $T_{i+1} \le 10^{-4}$가 되는 Gaussian을 **더하지 않고 배제한 채(exclusive)** 종료한다.
#
# ```cuda
# const float next_T = T[p] * (1.0f - alpha);
# if (next_T <= TRANSMITTANCE_THRESHOLD) { done_mask |= (1u<<p); continue; }
# ```
#
# 상수 설계 의도: $10^{-4} = (1-0.99)^2$ — "최대 불투명 Gaussian 두 장이 겹쳐야 겨우 포화".

# %%
def blend_kernel(colors, alphas_raw, opac_scale=1.0):
    """gsplat 커널과 동일한 게이트 3종을 모두 적용한 앞→뒤 블렌딩."""
    T, C = 1.0, np.zeros(3)
    n_blend, n_skip, stop_at = 0, 0, None
    for i, (c, a_raw) in enumerate(zip(colors, alphas_raw)):
        alpha = min(MAX_ALPHA, a_raw * opac_scale)  # MAX_ALPHA 상한
        if alpha < ALPHA_THRESHOLD:  # ALPHA_THRESHOLD 하한 → skip
            n_skip += 1
            continue
        next_T = T * (1.0 - alpha)
        if next_T <= TRANSMITTANCE_THRESHOLD:  # 포화 → 이 Gaussian 제외하고 종료
            stop_at = i
            break
        C = C + c * (alpha * T)
        T = next_T
        n_blend += 1
    return C, T, n_blend, n_skip, stop_at


# 불투명한 Gaussian 20개가 줄줄이 놓인 픽셀
rng = np.random.default_rng(0)
N = 20
col_seq = rng.random((N, 3))
alpha_seq_raw = np.full(N, 0.999)  # 전부 최대 불투명 → MAX_ALPHA=0.99 로 clamp
C, T, nb, ns, stop = blend_kernel(col_seq, alpha_seq_raw)
print(f"α=0.99 짜리 {N}개:  블렌딩된 개수 = {nb}, 조기 종료 인덱스 = {stop}, T_end = {T:.3e}")
print(f"  T 추이: 1 → {(1-MAX_ALPHA):.2f} → {(1-MAX_ALPHA)**2!r} → {(1-MAX_ALPHA)**3:.0e}")
print(f"  설계 의도대로 '최대 불투명 2장이면 포화'. 다만 float에서 (1-0.99)^2 = "
      f"{(1-MAX_ALPHA)**2:.17g} 가 1e-4 보다 미세하게 커서")
print(f"  2장째는 통과하고 3장째(i={stop})에서 next_T <= 1e-4 → 그 Gaussian은 배제하고 종료. "
      f"뒤의 {N-1-stop}개는 아예 안 봄.")

# 좀 더 현실적인 α들
for a in [0.9, 0.5, 0.2, 0.05, 0.003]:
    C, T, nb, ns, stop = blend_kernel(col_seq, np.full(N, a))
    print(f"α={a:<5}: blended={nb:>2} skipped={ns:>2} stop_at={str(stop):>4} T_end={T:.3e} alpha={1-T:.6f}")
# 출력: α=0.99 짜리 20개:  블렌딩된 개수 = 2, 조기 종료 인덱스 = 2, T_end = 1.000e-04
# 출력:   T 추이: 1 → 0.01 → 0.00010000000000000018 → 1e-06
# 출력:   설계 의도대로 '최대 불투명 2장이면 포화'. 다만 float에서 (1-0.99)^2 = 0.00010000000000000018 가 1e-4 보다 미세하게 커서
# 출력:   2장째는 통과하고 3장째(i=2)에서 next_T <= 1e-4 → 그 Gaussian은 배제하고 종료. 뒤의 17개는 아예 안 봄.
# 출력: α=0.9  : blended= 3 skipped= 0 stop_at=   3 T_end=1.000e-03 alpha=0.999000
# 출력: α=0.5  : blended=13 skipped= 0 stop_at=  13 T_end=1.221e-04 alpha=0.999878
# 출력: α=0.2  : blended=20 skipped= 0 stop_at=None T_end=1.153e-02 alpha=0.988471
# 출력: α=0.05 : blended=20 skipped= 0 stop_at=None T_end=3.585e-01 alpha=0.641514
# 출력: α=0.003: blended= 0 skipped=20 stop_at=None T_end=1.000e+00 alpha=0.000000
# 출력:   ↑ α=0.9 는 4장째(i=3)에서 next_T=1e-4 <= 1e-4 → 3개만 블렌딩. α<1/255 면 전부 skip.

# %% [markdown]
# 마지막 줄: `α = 0.003 < 1/255` 이면 **20개 전부 skip** → $T_{end} = 1$, render_alpha $= 0$.
# Gaussian이 20개나 있어도 이 픽셀에서는 "아무것도 없는 것"과 완전히 같다.

# %%
C, T, nb, ns, stop = blend_kernel(col_seq, np.full(N, 0.003))
print(f"α=0.003: T_end = {T}, render_alpha = 1 - T = {1 - T}, 색 = {C}")
# 출력: α=0.003: T_end = 1.0, render_alpha = 1 - T = 0.0, 색 = [0. 0. 0.]

# %% [markdown]
# ## 6. 1D 단면 렌더 — 픽셀 한 행에 Gaussian 6개
#
# 화면의 한 줄(가로 200픽셀)에 1D Gaussian 6개를 깊이순으로 놓고 앞→뒤 블렌딩한다.
# 각 픽셀 x에서
#
# $$\sigma(x) = \tfrac{(x-\mu_i)^2}{2s_i^2},\quad
#   \alpha_i(x) = \min(0.99,\ o_i e^{-\sigma_i(x)})$$
#
# 결과: 컬러 띠(픽셀 행), 픽셀별 $T$ 감쇠, render_alpha, 각 Gaussian의 기여 $\alpha_i T_i$.

# %%
W = 200
xs = np.arange(W) + 0.5

# 앞(가까움) → 뒤 순서로 6개
gauss = [
    # (중심, 표준편차, opacity, RGB)
    (40.0, 12.0, 0.95, (0.90, 0.20, 0.25)),  # 빨강 (제일 앞, 진함)
    (95.0, 18.0, 0.80, (0.20, 0.65, 0.90)),  # 하늘
    (150.0, 10.0, 0.90, (0.95, 0.75, 0.15)),  # 노랑
    (70.0, 30.0, 0.55, (0.30, 0.80, 0.35)),  # 초록 (넓고 옅음)
    (120.0, 25.0, 0.70, (0.65, 0.35, 0.85)),  # 보라
    (100.0, 90.0, 0.60, (0.35, 0.35, 0.40)),  # 회색 배경판 (제일 뒤, 아주 넓음)
]


def render_row(gauss, xs):
    T = np.ones_like(xs)
    C = np.zeros((len(xs), 3))
    contribs, Ts = [], [T.copy()]
    for mu, s, o, col in gauss:
        sigma = (xs - mu) ** 2 / (2.0 * s * s)
        alpha = np.minimum(MAX_ALPHA, o * np.exp(-sigma))
        alpha = np.where(alpha >= ALPHA_THRESHOLD, alpha, 0.0)  # 하한 게이트
        w = alpha * T  # α_i T_i
        C += w[:, None] * np.array(col)
        T = T * (1.0 - alpha)
        contribs.append(w)
        Ts.append(T.copy())
    return C, T, np.array(contribs), np.array(Ts)


C_row, T_row, contribs, Ts = render_row(gauss, xs)
alpha_row = 1.0 - T_row

print(f"Σ_i α_i T_i  vs  1 - T_end :  max|diff| = {np.abs(contribs.sum(0) - alpha_row).max():.3e}")
print(f"render_alpha 범위 : [{alpha_row.min():.4f}, {alpha_row.max():.4f}]")
print(f"T_end        범위 : [{T_row.min():.4f}, {T_row.max():.4f}]")
print(f"색 범위 (모든 채널): [{C_row.min():.4f}, {C_row.max():.4f}]   ← alpha 이하로 유지됨")
print(f"C <= alpha (볼록결합) 성립: {bool((C_row.max(1) <= alpha_row + 1e-9).all())}")
# 출력: Σ_i α_i T_i  vs  1 - T_end :  max|diff| = 2.220e-16
# 출력: render_alpha 범위 : [0.3287, 0.9828]
# 출력: T_end        범위 : [0.0172, 0.6713]
# 출력: 색 범위 (모든 채널): [0.1150, 0.8831]   ← alpha 이하로 유지됨
# 출력: C <= alpha (볼록결합) 성립: True

# %%
# 순서를 뒤집어 렌더 → 색은 달라지고 alpha는 그대로
C_rev_row, T_rev_row, _, _ = render_row(gauss[::-1], xs)
print(f"색  max|diff| (원래 vs 역순) = {np.abs(C_row - C_rev_row).max():.4f}")
print(f"alpha max|diff|              = {np.abs(T_row - T_rev_row).max():.3e}  ← 0")
# 출력: 색  max|diff| (원래 vs 역순) = 0.3554
# 출력: alpha max|diff|              = 1.110e-16  ← 0

# %% [markdown]
# ## 7. 시각화
#
# 4개 패널:
# 1. 렌더된 픽셀 행(앞→뒤) / 역순 렌더 행 — 같은 Gaussian, 다른 색
# 2. 각 Gaussian의 기여 $\alpha_i T_i$ (누적 면적) + `render_alpha` 곡선 — 합이 정확히 $1-T$
# 3. 투과율 $T$가 Gaussian을 하나씩 지날 때마다 계단식으로 떨어지는 모습
# 4. 5개 판 예제의 $T_i$, $\alpha_i T_i$, 누적 $\sum\alpha_iT_i$ vs $1-T_{i+1}$

# %%
def rgb_str(c):
    r, g, b = (np.clip(np.asarray(c), 0, 1) * 255).astype(int)
    return f"rgb({r},{g},{b})"


fig = make_subplots(
    rows=4,
    cols=1,
    row_heights=[0.16, 0.29, 0.26, 0.29],
    vertical_spacing=0.075,
    subplot_titles=(
        "① 렌더된 픽셀 행 — 위: 앞→뒤 (정답) / 아래: 순서 뒤집음 (색이 달라진다)",
        "② 각 Gaussian의 기여 α_i·T_i (누적) — 총합 = render_alpha = 1 − T",
        "③ 투과율 T: Gaussian을 지날 때마다 (1−α)씩 곱해져 감소 — 이 예제는 1e-4 선에 한참 못 미쳐 조기 종료 없음",
        "④ 5개 판 예제: α_i·T_i = T_i − T_{i+1} 이라 합이 1 − T_end 로 접힌다",
    ),
)

# ① 두 줄짜리 컬러 띠 (heatmap 대신 RGB를 그대로 쓰려고 bar로 채움)
for row_i, (Crow, ylab) in enumerate([(C_row, "앞→뒤"), (C_rev_row, "역순")]):
    fig.add_trace(
        go.Bar(
            x=xs,
            y=np.ones(W),
            base=np.full(W, -row_i - 1.0),
            width=1.0,
            marker=dict(color=[rgb_str(c) for c in Crow], line=dict(width=0)),
            showlegend=False,
            hovertemplate="x=%{x:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

# ② 기여 α_i T_i 를 누적 면적으로
for i, (mu, s, o, col) in enumerate(gauss):
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=contribs[i],
            name=f"G{i} (μ={mu:.0f}, o={o})",
            mode="lines",
            line=dict(width=0.5, color=rgb_str(col)),
            fillcolor=rgb_str(col),
            stackgroup="contrib",
            legendgroup=f"g{i}",
        ),
        row=2,
        col=1,
    )
fig.add_trace(
    go.Scatter(
        x=xs,
        y=alpha_row,
        name="render_alpha = 1 − T",
        mode="lines",
        line=dict(width=2.5, color="black", dash="dot"),
    ),
    row=2,
    col=1,
)

# ③ T 감쇠
for i in range(len(gauss) + 1):
    lbl = "T₁ = 1 (시작)" if i == 0 else f"T after G{i-1}"
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=Ts[i],
            name=lbl,
            mode="lines",
            line=dict(width=1.8, color=f"rgba(20,40,90,{0.25 + 0.75*i/len(gauss)})"),
            showlegend=False,
            hovertemplate=lbl + "<br>x=%{x:.0f}<br>T=%{y:.4f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
fig.add_hline(
    y=TRANSMITTANCE_THRESHOLD,
    line=dict(color="crimson", width=1, dash="dash"),
    annotation_text="TRANSMITTANCE_THRESHOLD = 1e-4 (이 아래면 조기 종료)",
    annotation_font_size=10,
    row=3,
    col=1,
)

# ④ 5개 판 예제
idx = np.arange(len(alphas5))
Ti = np.array([r[2] for r in rows5])
wi = np.array([r[3] for r in rows5])
cum = np.cumsum(wi)
fig.add_trace(
    go.Bar(x=idx, y=Ti, name="T_i (도달률)", marker_color="rgb(120,150,200)", width=0.32, offset=-0.34),
    row=4,
    col=1,
)
fig.add_trace(
    go.Bar(x=idx, y=wi, name="α_i·T_i (기여)", marker_color="rgb(220,120,80)", width=0.32, offset=0.02),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(x=idx, y=cum, name="Σ α_i·T_i (누적)", mode="lines+markers", line=dict(color="black", width=2)),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=idx,
        y=1.0 - np.array([r[4] for r in rows5]),
        name="1 − T_{i+1}",
        mode="markers",
        marker=dict(symbol="circle-open", size=13, color="crimson", line=dict(width=2.5)),
    ),
    row=4,
    col=1,
)

fig.update_yaxes(visible=False, range=[-2.05, 0.05], row=1, col=1)
fig.update_xaxes(range=[0, W], row=1, col=1)
fig.update_yaxes(title_text="기여 / alpha", range=[0, 1.05], row=2, col=1)
fig.update_yaxes(title_text="T (log)", type="log", row=3, col=1)
fig.update_xaxes(title_text="픽셀 x", row=3, col=1)
fig.update_xaxes(title_text="Gaussian 인덱스 i (앞 → 뒤)", tickvals=idx, row=4, col=1)
fig.update_yaxes(title_text="값", range=[0, 1.08], row=4, col=1)
fig.update_layout(
    title=dict(
        text="<b>알파 블렌딩 앞→뒤 누적</b>  ·  α_i = min(0.99, o_i e^(−σ_i)),  C = Σ c_i α_i T_i,  T_(i+1) = T_i(1−α_i),  alpha = 1 − T",
        x=0.5,
        font=dict(size=15),
    ),
    height=1180,
    width=1180,
    barmode="overlay",
    bargap=0,
    template="plotly_white",
    legend=dict(orientation="v", x=1.005, y=1.0, font=dict(size=10)),
    margin=dict(l=70, r=230, t=90, b=55),
)
for a in fig.layout.annotations:
    a.font.size = 12

_show(fig)

out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expy.png")
try:
    fig.write_image(out_png, scale=2)
    print(f"저장: {out_png}")
except Exception as e:  # kaleido 미설치 등
    print(f"PNG 저장 실패(무시 가능): {type(e).__name__}: {e}")
# 출력: 저장: .../ab530db4-cfcf-47c4-9a7c-3f2a814e3b21/expy.png

# %% [markdown]
# ## 정리
#
# | 확인한 것 | 결과 |
# |---|---|
# | $T_{i+1} = T_i(1-\alpha_i)$ | 도달률이 곱으로 누적 |
# | $\alpha_i T_i = T_i - T_{i+1}$ | 각 항이 이웃 $T$의 차 (망원급수) |
# | $\sum_i \alpha_i T_i = 1 - T_{end}$ | 오차 $10^{-16}$ 수준으로 정확히 성립 → `render_alpha = 1 - T` |
# | 순서 뒤집기 | **색은 바뀌고(1D 행에서 L∞ 0.355) alpha는 그대로(1e-16)** → 깊이 정렬은 색을 위한 것 |
# | `MAX_ALPHA = 0.99` | $1/(1-\alpha) \le 100$ 보장 → backward의 $T /= (1-\alpha)$ 되감기 안전 |
# | `ALPHA_THRESHOLD = 1/255` | $o=0.999$일 때 $\sigma > 5.54$, 즉 $3.33\sigma$ 밖은 잘림 (`GAUSSIAN_EXTEND`) |
# | `TRANSMITTANCE_THRESHOLD = 1e-4` | $(1-0.99)^2$ — 최대 불투명 2장이면 포화. α=0.9면 3개, α=0.5면 13개 블렌딩 후 종료 |
