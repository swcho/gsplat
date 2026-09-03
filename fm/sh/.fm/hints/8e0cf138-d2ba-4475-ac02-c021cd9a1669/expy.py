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
# # SH 계수 인덱스 $k$ ↔ $(\ell, m)$ 대응 규칙: $k=\ell^2+\ell+m$
#
# 3DGS/gsplat은 Gaussian 하나의 색을 SH 계수 `[16, 3]` 텐서로 저장한다. 첫 축 인덱스 $k\in\{0,\dots,15\}$가
# 어떤 기저 $Y_\ell^m$에 대응하는지가 이 노트의 주제다.
#
# **규칙**: 차수 $\ell$ 블록을 낮은 차수부터 이어 붙이고, 블록 안에서는 $m=-\ell,\dots,\ell$ 순으로 둔다. 그러면
#
# $$
# k \;=\; \underbrace{\sum_{j<\ell}(2j+1)}_{=\;\ell^2\ (\text{앞 블록 크기 합})} \;+\; \underbrace{(m+\ell)}_{\text{블록 안 위치}}
# \;=\; \ell^2+\ell+m .
# $$
#
# 역변환은 $\ell=\lfloor\sqrt{k}\rfloor,\quad m=k-\ell^2-\ell$ 이다.
#
# 이 노트는 (1) 규칙과 역변환을 표로 확인하고, (2) 블록 크기 $2\ell+1$·누적 $(L+1)^2$을 확인하고,
# (3) `sh_walkthrough.py`의 `sh_bases` 출력 순서가 실제로 이 규칙을 따르는지 수치로 검증한 뒤, (4) 격자 그림으로 정리한다.

# %%
# 필요 패키지: numpy, torch, plotly, kaleido(정적 PNG 저장), scipy(선택: 교차 검증)
import math
import os

import numpy as np
import torch


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

# %% [markdown]
# ## 1. 정방향: $(\ell, m)$ 리스트를 만들고 $k=\ell^2+\ell+m$ 이 곧 리스트 인덱스임을 확인
#
# `sh_walkthrough.py`의 `LM = [(l, m) for l in range(4) for m in range(-l, l + 1)]` 는
# "낮은 차수부터, 블록 안에서는 $m$ 오름차순"이라는 규약을 그대로 코드로 옮긴 것이다.
# 리스트의 위치 `k`와 공식 $\ell^2+\ell+m$ 이 모두 일치해야 한다.

# %%
L_MAX = 3
LM = [(l, m) for l in range(L_MAX + 1) for m in range(-l, l + 1)]     # k번째 기저의 (ℓ, m)


def k_of(l: int, m: int) -> int:
    """(ℓ, m) → 계수 배열 인덱스 k."""
    assert -l <= m <= l
    return l * l + l + m


print(f"{'k':>2} | {'ℓ':>2} {'m':>3} | ℓ²+ℓ+m")
print("-" * 22)
for k, (l, m) in enumerate(LM):
    assert k_of(l, m) == k
    print(f"{k:>2} | {l:>2} {m:>+3} | {k_of(l, m):>6}")
print("총 개수:", len(LM), "=", f"(L+1)² = {(L_MAX + 1) ** 2}")
# 출력:
#  k |  ℓ   m | ℓ²+ℓ+m
# ----------------------
#  0 |  0  +0 |      0
#  1 |  1  -1 |      1
#  2 |  1  +0 |      2
#  3 |  1  +1 |      3
#  4 |  2  -2 |      4
#  5 |  2  -1 |      5
#  6 |  2  +0 |      6
#  7 |  2  +1 |      7
#  8 |  2  +2 |      8
#  9 |  3  -3 |      9
# 10 |  3  -2 |     10
# 11 |  3  -1 |     11
# 12 |  3  +0 |     12
# 13 |  3  +1 |     13
# 14 |  3  +2 |     14
# 15 |  3  +3 |     15
# 총 개수: 16 = (L+1)² = 16

# %% [markdown]
# ## 2. 역방향: $\ell=\lfloor\sqrt{k}\rfloor$, $m=k-\ell^2-\ell$
#
# 블록 $\ell$은 $k\in[\ell^2,\,(\ell+1)^2)$ 를 차지하므로 $\ell^2\le k<(\ell+1)^2 \iff \ell\le\sqrt k<\ell+1$, 즉 $\ell=\lfloor\sqrt k\rfloor$.
# 그 다음 $m=k-\ell^2-\ell$ 은 자동으로 $[-\ell,\ell]$ 안에 떨어진다.
# gsplat이 4차(25개)까지 지원하므로 $k=0\sim24$ 로 넓혀서 확인한다.

# %%
def lm_of(k: int) -> tuple[int, int]:
    """계수 배열 인덱스 k → (ℓ, m)."""
    l = math.isqrt(k)            # ⌊√k⌋ (정수 제곱근이라 부동소수 오차 없음)
    m = k - l * l - l
    return l, m


LM25 = [(l, m) for l in range(5) for m in range(-l, l + 1)]
ok = all(lm_of(k) == lm for k, lm in enumerate(LM25))
ok_roundtrip = all(k_of(*lm_of(k)) == k for k in range(25))
print("k=0..24 역변환 == LM25 :", ok, " / k → (ℓ,m) → k 왕복 :", ok_roundtrip)
print("차수 블록의 시작 k (=ℓ²) :", [l * l for l in range(5)])
print("각 블록의 m=0 위치 (=ℓ²+ℓ):", [l * l + l for l in range(5)])
print("k=9~15 →", [lm_of(k) for k in range(9, 16)])
# 출력:
# k=0..24 역변환 == LM25 : True  / k → (ℓ,m) → k 왕복 : True
# 차수 블록의 시작 k (=ℓ²) : [0, 1, 4, 9, 16]
# 각 블록의 m=0 위치 (=ℓ²+ℓ): [0, 2, 6, 12, 20]
# k=9~15 → [(3, -3), (3, -2), (3, -1), (3, 0), (3, 1), (3, 2), (3, 3)]

# %% [markdown]
# ## 3. 블록 크기 $2\ell+1$ 과 누적 $(L+1)^2$ — 차수 활성화 슬라이싱과의 관계
#
# 3DGS 학습은 `sh_degree_interval` 스텝마다 사용 차수를 하나씩 올린다. 이때 계수 텐서를 `coeffs[:(L+1)**2]`로 자르기만 하면
# 정확히 $\ell\le L$ 인 계수만 남는다 — 블록이 차수 순으로 연속해서 놓여 있기 때문이다.
# 또 `sh0 = coeffs[:1]` (DC, $\ell=0$), `shN = coeffs[1:]` (나머지 15개) 분리도 같은 이유로 슬라이스 한 번이다.

# %%
sizes = [2 * l + 1 for l in range(L_MAX + 1)]
cum = np.cumsum(sizes).tolist()
print("블록 크기 2ℓ+1      :", sizes)
print("누적 합 Σ(2j+1)     :", cum)
print("(L+1)²              :", [(L + 1) ** 2 for L in range(L_MAX + 1)])
for L in range(L_MAX + 1):
    K = (L + 1) ** 2
    degs = sorted({l for l, _ in LM[:K]})
    print(f"L={L}: coeffs[:{K:>2}] 에 포함된 차수 = {degs}")
# 출력:
# 블록 크기 2ℓ+1      : [1, 3, 5, 7]
# 누적 합 Σ(2j+1)     : [1, 4, 9, 16]
# (L+1)²              : [1, 4, 9, 16]
# L=0: coeffs[: 1] 에 포함된 차수 = [0]
# L=1: coeffs[: 4] 에 포함된 차수 = [0, 1]
# L=2: coeffs[: 9] 에 포함된 차수 = [0, 1, 2]
# L=3: coeffs[:16] 에 포함된 차수 = [0, 1, 2, 3]

# %% [markdown]
# ## 4. `sh_bases`의 출력 순서가 정말 $k=\ell^2+\ell+m$ 인지 수치 검증
#
# 노트북의 `sh_bases`(gsplat `_eval_sh_bases_fast`와 같은 값·순서)를 그대로 가져와, 각 열 $k$가 주장하는 $(\ell,m)$과
# 실제 함수 성질이 맞는지 세 가지로 확인한다.
#
# 1. **$\ell$ 확인 — Unsöld 정리**: 차수 $\ell$ 블록 전체의 제곱합은 방향과 무관한 상수다.
#    $$\sum_{m=-\ell}^{\ell}\big[Y_\ell^m(\mathbf d)\big]^2=\frac{2\ell+1}{4\pi}$$
#    블록 경계 $k\in[\ell^2,(\ell+1)^2)$ 를 잘못 잡으면 이 합이 방향에 따라 흔들린다.
# 2. **$|m|$ 확인 — 방위각 주파수**: 고정 극각 $\theta$의 원 위에서 $Y_\ell^m$은 $\cos(m\varphi)$ 또는 $\sin(|m|\varphi)$에 비례하므로
#    FFT의 지배 주파수가 $|m|$이다.
# 3. **$m$의 부호 확인 — $y$ 반전 대칭**: $m<0$ 항은 $\sin(|m|\varphi)$ 라서 $y\to-y$ ($\varphi\to-\varphi$)에 **홀**함수,
#    $m\ge0$ 항은 $\cos(m\varphi)$ 라서 **짝**함수다.
#
# 마지막으로 scipy가 있으면 복소 SH `sph_harm_y`로 만든 실수 SH와 직접 비교한다.

# %%
C0 = 0.28209479177387814                                       # Y₀⁰ = 1/(2√π)


def sh_bases(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """단위 방향 dirs[..., 3] → SH 기저값 [..., (degree+1)²]. (sh_walkthrough.py에서 그대로 가져옴)"""
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


# (1) ℓ 확인: 무작위 방향 2000개에서 블록별 제곱합이 상수 (2ℓ+1)/4π 인지
d = torch.nn.functional.normalize(torch.randn(2000, 3, dtype=torch.float64), dim=-1)
Bv = sh_bases(d, L_MAX)                                          # [2000, 16]
print("블록별 Σ_m Y² (평균 ± 표준편차)  vs  (2ℓ+1)/4π")
for l in range(L_MAX + 1):
    blk = Bv[:, l * l:(l + 1) ** 2]
    s = (blk ** 2).sum(-1)
    print(f"ℓ={l}: k∈[{l*l:>2},{(l+1)**2:>2})  {s.mean():.6f} ± {s.std():.1e}   이론 {(2*l+1)/(4*math.pi):.6f}")
# 출력:
# 블록별 Σ_m Y² (평균 ± 표준편차)  vs  (2ℓ+1)/4π
# ℓ=0: k∈[ 0, 1)  0.079577 ± 1.4e-17   이론 0.079577
# ℓ=1: k∈[ 1, 4)  0.238732 ± 6.7e-17   이론 0.238732
# ℓ=2: k∈[ 4, 9)  0.397887 ± 1.2e-16   이론 0.397887
# ℓ=3: k∈[ 9,16)  0.557042 ± 2.6e-16   이론 0.557042

# %%
# (2) |m| 확인: 고정 θ의 원 위에서 φ에 대한 FFT 지배 주파수  /  (3) m 부호 확인: y → −y 대칭
theta = 1.0                                                     # 특별한 영점(적도·극)을 피한 임의의 극각
phi = torch.arange(64, dtype=torch.float64) * 2 * math.pi / 64
ring = torch.stack([math.sin(theta) * phi.cos(), math.sin(theta) * phi.sin(),
                    torch.full_like(phi, math.cos(theta))], dim=-1)          # [64, 3]
Br = sh_bases(ring, L_MAX)                                                  # [64, 16]
spec = torch.fft.rfft(Br, dim=0).abs()                                      # [33, 16]
freq = spec.argmax(dim=0).tolist()                                          # 열마다 지배 주파수

flip = d * torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)              # y 반전
Bf = sh_bases(flip, L_MAX)
parity = torch.where((Bf + Bv).abs().amax(0) < 1e-12, -1, torch.where((Bf - Bv).abs().amax(0) < 1e-12, 1, 0))

print(f"{'k':>2} {'(ℓ,m) 주장':>10} | FFT |m| | y-반전 | 판정")
all_ok = True
for k, (l, m) in enumerate(LM):
    p = int(parity[k])
    sign_ok = (m < 0 and p == -1) or (m >= 0 and p == 1)
    good = (freq[k] == abs(m)) and sign_ok
    all_ok &= good
    print(f"{k:>2} {str((l, m)):>10} | {freq[k]:>7} | {'홀' if p == -1 else '짝' if p == 1 else '?':>5} | {'OK' if good else 'FAIL'}")
print("모든 열이 k=ℓ²+ℓ+m 규칙과 일치:", bool(all_ok))
# 출력:
#  k  (ℓ,m) 주장 | FFT |m| | y-반전 | 판정
#  0     (0, 0) |       0 |     짝 | OK
#  1    (1, -1) |       1 |     홀 | OK
#  2     (1, 0) |       0 |     짝 | OK
#  3     (1, 1) |       1 |     짝 | OK
#  4    (2, -2) |       2 |     홀 | OK
#  5    (2, -1) |       1 |     홀 | OK
#  6     (2, 0) |       0 |     짝 | OK
#  7     (2, 1) |       1 |     짝 | OK
#  8     (2, 2) |       2 |     짝 | OK
#  9    (3, -3) |       3 |     홀 | OK
# 10    (3, -2) |       2 |     홀 | OK
# 11    (3, -1) |       1 |     홀 | OK
# 12     (3, 0) |       0 |     짝 | OK
# 13     (3, 1) |       1 |     짝 | OK
# 14     (3, 2) |       2 |     짝 | OK
# 15     (3, 3) |       3 |     짝 | OK
# 모든 열이 k=ℓ²+ℓ+m 규칙과 일치: True

# %%
# (선택) scipy 교차 검증: 복소 SH → 실수 SH  (m>0: √2·Re Y_ℓ^m, m<0: √2·Im Y_ℓ^{|m|}, m=0: Y_ℓ^0; Condon–Shortley 위상 포함)
try:
    from scipy.special import sph_harm_y                       # scipy ≥ 1.15: sph_harm_y(n, m, θ극각, φ방위각)

    th = torch.acos(d[:, 2].clamp(-1, 1)).numpy()
    ph = torch.atan2(d[:, 1], d[:, 0]).numpy()
    ref = np.zeros((d.shape[0], len(LM)))
    for k, (l, m) in enumerate(LM):
        Y = sph_harm_y(l, abs(m), th, ph)
        ref[:, k] = Y.real if m == 0 else math.sqrt(2) * (Y.real if m > 0 else Y.imag)
    diff = np.abs(ref - Bv.numpy()).max(axis=0)
    print("열별 |scipy 실수 SH − sh_bases| 최대:", np.array2string(diff, precision=1))
    print("전체 최대 차이:", f"{diff.max():.2e}", "→ 순서·부호 모두 일치" if diff.max() < 1e-12 else "→ 불일치!")
except ImportError:
    print("scipy 없음 — 교차 검증 생략")
# 출력:
# 열별 |scipy 실수 SH − sh_bases| 최대: [0.0e+00 1.4e-15 1.7e-16 4.5e-15 4.4e-16 3.1e-15 4.4e-16 1.0e-14 3.3e-16
#  7.6e-16 5.0e-16 5.3e-15 7.8e-16 1.7e-14 6.1e-16 7.2e-16]
# 전체 최대 차이: 1.68e-14 → 순서·부호 모두 일치

# %% [markdown]
# ## 5. 그림: $\ell$ 행 × $m$ 열 격자에 $k$ 번호 표시
#
# 각 행이 한 차수 블록이고, 왼쪽($m=-\ell$)에서 오른쪽($m=+\ell$)으로 $k$가 1씩 증가한다.
# 행이 바뀔 때 $k$는 $\ell^2$에서 다시 시작하며, 행 길이가 $2\ell+1$이라 피라미드 모양이 된다.
# 가운데 열($m=0$)의 $k=\ell^2+\ell$ 을 기준으로 좌우 $\pm m$ 만큼 이동한다고 읽으면 된다.

# %%
import plotly.graph_objects as go

ms = list(range(-L_MAX, L_MAX + 1))
ls = list(range(L_MAX + 1))
Z = np.full((len(ls), len(ms)), np.nan)
text = [["" for _ in ms] for _ in ls]
for k, (l, m) in enumerate(LM):
    Z[l, m + L_MAX] = k
    text[l][m + L_MAX] = f"<b>k={k}</b><br>Y({l},{m:+d})"

fig = go.Figure(go.Heatmap(
    z=Z, x=[f"m={m:+d}" for m in ms], y=[f"ℓ={l}" for l in ls],
    text=text, texttemplate="%{text}", textfont=dict(size=13),
    colorscale=[[0.0, "#dbe7f3"], [1.0, "#1f4e79"]],            # 단일 색상 순차 램프(밝음 → 어두움 = k 증가)
    xgap=3, ygap=3, showscale=False,
    hovertemplate="ℓ=%{y}, %{x}<br>k=%{z}<extra></extra>",
))
fig.update_layout(
    title=dict(text="SH 계수 인덱스 k = ℓ² + ℓ + m  (행: 차수 ℓ, 열: m; 빈 칸은 |m|>ℓ)", x=0.5, font=dict(size=16)),
    xaxis=dict(side="top", showgrid=False, zeroline=False),
    yaxis=dict(autorange="reversed", showgrid=False, zeroline=False),
    width=820, height=460, margin=dict(l=60, r=30, t=90, b=30),
    paper_bgcolor="white", plot_bgcolor="white",
    annotations=[dict(text="행 시작 k = ℓ² : 0, 1, 4, 9   |   행 길이 2ℓ+1 : 1, 3, 5, 7   |   가운데(m=0) k = ℓ²+ℓ : 0, 2, 6, 12",
                      xref="paper", yref="paper", x=0.5, y=-0.06, showarrow=False, font=dict(size=12, color="#555"))],
)
_show(fig)
png_path = os.path.join(HERE, "expy.png")
fig.write_image(png_path, scale=2)
print("저장:", png_path)
# 출력:
# 저장: /home/sungwoo/projects/swcho/gsplat/fm/sh/.fm/hints/8e0cf138-d2ba-4475-ac02-c021cd9a1669/expy.png

# %% [markdown]
# ## 정리
#
# | 방향 | 공식 | 예 |
# |---|---|---|
# | $(\ell,m)\to k$ | $k=\ell^2+\ell+m$ | $(2,-1)\to 5$, $(3,0)\to 12$ |
# | $k\to(\ell,m)$ | $\ell=\lfloor\sqrt k\rfloor,\ m=k-\ell^2-\ell$ | $7\to(2,+1)$, $9\to(3,-3)$ |
# | 블록 범위 | $k\in[\ell^2,(\ell+1)^2)$, 크기 $2\ell+1$ | $\ell=3$: $9\sim15$ |
# | 차수 $L$까지 | 앞에서 $(L+1)^2$개 → `coeffs[:(L+1)**2]` | $L=3$: 16개 |
#
# 따라서 $k=0\to(0,0)$, $k=1,2,3\to(1,-1),(1,0),(1,1)$, $k=4\sim8\to$ 2차, $k=9\sim15\to$ 3차 순서이며,
# `sh_bases`(= gsplat `_eval_sh_bases_fast`)의 열 순서도 이와 정확히 일치한다.
