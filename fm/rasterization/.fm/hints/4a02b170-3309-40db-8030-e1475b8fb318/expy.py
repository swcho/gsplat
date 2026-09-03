# %% [markdown]
# # float32 비트를 정수 정렬 키로 그냥 써도 되는 이유
#
# gsplat의 `isect_tiles`는 (Gaussian, 타일) 쌍마다 64비트 키를 만든다:
#
# ```
# [ image_id | tile_id | float32(depth) 비트 ]     ← 상위 → 하위
# ```
#
# 여기서 depth는 **아무 변환 없이** 비트 패턴 그대로 하위 32비트에 들어간다
# (`__float_as_uint(depth_f)` / `struct.pack("f", d)`). 그런데도 이 64비트 키를
# 그냥 정수 radix sort 하면 "같은 타일 안에서 가까운 것부터" 순서가 정확히 나온다.
#
# 이유는 IEEE 754 single의 비트 배치가 **음이 아닌 값에 대해 단조(monotonic)** 이기 때문이다.
#
# $$ 0 \le a \le b \quad\Longleftrightarrow\quad \mathrm{bits}(a) \le \mathrm{bits}(b) \quad(\text{unsigned 비교}) $$
#
# 이 노트북은 그 성질을 직접 확인하고, 음수에서 어떻게 깨지는지, 표준 해법이 무엇인지 본다.
#
# 필요 패키지: numpy, plotly, kaleido

# %%
# 필요 패키지: numpy, plotly, kaleido  (gsplat은 import하지 않는다)
import struct

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


HERE = __file__.rsplit("/", 1)[0] if "__file__" in dir() else "."
print("numpy", np.__version__)
# 출력: numpy 2.4.6

# %% [markdown]
# ## 1. IEEE 754 single 레이아웃
#
# 32비트를 세 조각으로 나눈다.
#
# | 필드 | 비트 수 | 위치(상위→하위) |
# |---|---|---|
# | 부호 $s$ | 1 | bit 31 |
# | 지수 $e$ (biased) | 8 | bit 30..23 |
# | 가수 $m$ (fraction) | 23 | bit 22..0 |
#
# 정규 수(normal, $1 \le e \le 254$)의 값은
#
# $$ x = (-1)^s \times 2^{\,e-127} \times \left(1 + \frac{m}{2^{23}}\right) $$
#
# 핵심은 지수가 **biased(offset binary)** 로 저장된다는 점이다. 실제 지수 $-126 \ldots 127$을
# $+127$ 해서 $1 \ldots 254$의 **부호 없는 정수**로 넣는다. 그래서 "지수가 크다 = 저장된
# 8비트 필드가 크다"가 그대로 성립한다. (2의 보수였다면 음의 지수가 최상위 비트를 세워서 뒤집혔을 것이다.)

# %%
def bits_of(x) -> int:
    """float32의 비트 패턴을 uint32로."""
    return int(np.float32(x).view(np.uint32))


# gsplat의 PyTorch 참조 구현(_torch_impl._isect_tiles)이 쓰는 struct 방식과 동일하다:
#   depth_id = struct.unpack("i", struct.pack("f", depth_f32))[0] & 0xFFFFFFFF
assert bits_of(3.25) == struct.unpack("I", struct.pack("f", 3.25))[0]
print("np.view(uint32) == struct.pack/unpack :", bits_of(3.25))
# 출력: np.view(uint32) == struct.pack/unpack : 1078984704


def split(x):
    b = bits_of(x)
    s = b >> 31
    e = (b >> 23) & 0xFF
    m = b & 0x7FFFFF
    return b, s, e, m


print(f"{'value':>12} {'uint32':>11}  {'s':>1} {'exp':>3}(=2^{'k':>4}) {'mantissa':>8}  bits")
for v in [0.0, 1e-40, 1.17549435e-38, 0.5, 1.0, 1.5, 1.9999999, 2.0, 3.0, 100.0, np.inf]:
    b, s, e, m = split(v)
    k = "-" if e in (0, 255) else f"{e - 127:>4}"
    print(f"{v:>12.6g} {b:>11d}  {s} {e:>3}(=2^{k}) {m:>8d}  {b:032b}")
# 출력:
#        value      uint32  s exp(=2^   k) mantissa  bits
#            0           0  0   0(=2^-)        0  00000000000000000000000000000000
#        1e-40       71362  0   0(=2^-)    71362  00000000000000010001011011000010
#  1.17549e-38     8388608  0   1(=2^-126)        0  00000000100000000000000000000000
#          0.5  1056964608  0 126(=2^  -1)        0  00111111000000000000000000000000
#            1  1065353216  0 127(=2^   0)        0  00111111100000000000000000000000
#          1.5  1069547520  0 127(=2^   0)  4194304  00111111110000000000000000000000
#            2  1073741823  0 127(=2^   0)  8388607  00111111111111111111111111111111   <- 1.9999999
#            2  1073741824  0 128(=2^   1)        0  01000000000000000000000000000000
#            3  1077936128  0 128(=2^   1)  4194304  01000000010000000000000000000000
#          100  1120403456  0 133(=2^   6)  4718592  01000010110010000000000000000000
#          inf  2139095040  0 255(=2^-)        0  01111111100000000000000000000000

# %% [markdown]
# 위 표에서 uint32 열이 값과 **같은 방향으로 증가**하는 것에 주목.
#
# 단조성이 성립하는 논리는 두 층이다.
#
# 1. **바깥 층(지수)**: 지수 필드가 상위에 있고 biased라서, 큰 값 → 큰 지수 필드 → 큰 정수.
#    (denormal은 $e=0$으로 가장 작은 구간을 차지하고, 그 안에서도 가수가 곧 값의 크기라 자연스럽게 이어진다.)
# 2. **안쪽 층(가수)**: 같은 지수 구간 $[2^k, 2^{k+1})$ 안에서 값은 $m$에 **선형**이므로,
#    23비트 가수를 사전식(lexicographic)으로 비교한 것이 곧 값 비교다.
#
# 즉 상위 필드가 먼저 갈리고, 같으면 하위 필드로 넘어가는 — **radix/사전식 비교와 정확히 같은 구조**다.
# radix sort는 애초에 비트를 자리별로 훑는 알고리즘이라, 이 구조에 그대로 얹힌다.

# %%
# 지수 경계에서 인접한 값들: 비트를 +1 하면 "다음 표현 가능한 float"가 된다 (nextafter)
print("연속된 uint32 ↔ 연속된 float (지수 경계를 넘어도 이어진다)")
for base in [1.0, 2.0, 1.9999999]:
    b = bits_of(base)
    for d in (-1, 0, 1):
        u = np.uint32(b + d)
        print(f"  uint32 {int(u):>11d} -> {float(u.view(np.float32)):.9g}")
    print()
# 출력:
# 연속된 uint32 ↔ 연속된 float (지수 경계를 넘어도 이어진다)
#   uint32  1065353215 -> 0.99999994
#   uint32  1065353216 -> 1
#   uint32  1065353217 -> 1.00000012
#
#   uint32  1073741823 -> 1.99999988
#   uint32  1073741824 -> 2
#   uint32  1073741825 -> 2.00000024
#
#   uint32  1073741822 -> 1.99999976
#   uint32  1073741823 -> 1.99999988
#   uint32  1073741824 -> 2

# %% [markdown]
# ## 2. 양수 10만 개로 검증: 정수 정렬 순서 == float 정렬 순서
#
# gsplat의 depth는 near-plane 컬링을 통과한 카메라 좌표 $z$라서 **항상 양수**다.
# 그 조건에서 아래 assert가 성립한다.

# %%
rng = np.random.default_rng(0)

# 렌더링 depth와 비슷한 스케일 + 극단값(아주 작은 값, 큰 값)까지 섞는다
d = np.concatenate(
    [
        rng.uniform(0.01, 100.0, 60_000),
        np.exp(rng.uniform(np.log(1e-30), np.log(1e30), 40_000)),
    ]
).astype(np.float32)
d = d[d > 0]  # 양수만
print("n =", d.size, " min =", d.min(), " max =", d.max())
# 출력: n = 100000  min = 1.0004825e-30  max = 9.902657e+29

u = d.view(np.uint32)  # 비트 재해석 (값 변환이 아님!)
order_float = np.argsort(d, kind="stable")
order_uint = np.argsort(u, kind="stable")

assert np.array_equal(order_float, order_uint), "순서가 다르다!"
assert np.all(np.diff(u[order_uint]) >= 0)
print("양수 float: 정수 뷰 정렬 == float 정렬  ->  OK")
# 출력: 양수 float: 정수 뷰 정렬 == float 정렬  ->  OK

# int32로 봐도(부호 비트가 0이라 항상 비음수) 동일 — gsplat torch 참조 구현이 쓰는 방식
i32 = d.view(np.int32)
assert np.array_equal(np.argsort(i32, kind="stable"), order_float)
print("int32 뷰로 봐도 동일 (부호 비트가 0이므로)")
# 출력: int32 뷰로 봐도 동일 (부호 비트가 0이므로)

# %% [markdown]
# ## 3. gsplat 64비트 키 재현
#
# ```cuda
# // IntersectTile.cu
# float depth_f = static_cast<float>(depths[idx]);
# assert(depth_f >= 0.f);                       // 이 불변식이 전부다
# depth_id_enc = __float_as_uint(depth_f);      // 비트 재해석, 하위 32비트로 zero-extend
# ...
# isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
# ```
#
# 키의 배치가 `[image_id | tile_id | depth_bits]`이므로, 64비트 정수 하나를 정렬하면
#
# 1. 이미지별로 뭉치고,
# 2. 그 안에서 타일별로 뭉치고,
# 3. 그 안에서 **depth 오름차순(가까운 것부터)**
#
# 이 세 단계가 한 번의 radix sort로 동시에 끝난다. depth를 위해 별도 변환·재정렬이 없다.

# %%
TILE_N_BITS = 22  # e.g. image id (10 bits) | tile id (22 bits) | depth (32 bits)


def make_key(image_id: int, tile_id: int, depth: float) -> int:
    depth_bits = int(np.float32(depth).view(np.uint32))  # == __float_as_uint
    return (image_id << (32 + TILE_N_BITS)) | (tile_id << 32) | depth_bits


items = [  # (image, tile, depth, gaussian)
    (0, 5, 12.5, "g0"),
    (0, 5, 3.25, "g1"),
    (0, 5, 0.001, "g2"),
    (0, 3, 99.0, "g3"),
    (1, 5, 0.5, "g4"),
    (0, 3, 7.0, "g5"),
]
keys = [(make_key(i, t, z), i, t, z, g) for i, t, z, g in items]
for k, i, t, z, g in sorted(keys):
    print(f"  key=0x{k:016x}  image={i} tile={t:>2} depth={z:>8.3f}  {g}")
# 출력:
#   key=0x0000000340e00000  image=0 tile= 3 depth=   7.000  g5
#   key=0x0000000342c60000  image=0 tile= 3 depth=  99.000  g3
#   key=0x000000053a83126f  image=0 tile= 5 depth=   0.001  g2
#   key=0x0000000540500000  image=0 tile= 5 depth=   3.250  g1
#   key=0x0000000541480000  image=0 tile= 5 depth=  12.500  g0
#   key=0x004000053f000000  image=1 tile= 5 depth=   0.500  g4

# 정렬 결과가 (image, tile, depth) 사전식과 정확히 일치하는지 검증
by_key = [(g) for _, _, _, _, g in sorted(keys)]
by_tuple = [g for _, _, _, g in sorted(items, key=lambda r: (r[0], r[1], r[2]))]
assert by_key == by_tuple, (by_key, by_tuple)
print("키 정렬 == (image, tile, depth) 사전식 정렬  ->  OK :", by_key)
# 출력: 키 정렬 == (image, tile, depth) 사전식 정렬  ->  OK : ['g5', 'g3', 'g2', 'g1', 'g0', 'g4']

# %% [markdown]
# ## 4. 음수를 섞으면 깨진다
#
# 두 가지 이유로 깨진다.
#
# 1. **부호 비트가 최상위(bit 31)** 다. unsigned 비교에서 음수는 전부 $\ge 2^{31}$이 되어
#    **모든 양수보다 크게** 취급된다 → 순서가 완전히 뒤집힌 블록이 생긴다.
# 2. 음수 영역 **내부에서도 반전**된다. $-1.0$과 $-2.0$은 크기(magnitude)만 비트에 담기므로
#    $\mathrm{bits}(-2.0) > \mathrm{bits}(-1.0)$인데, 실제 값은 $-2.0 < -1.0$이다.
#
# gsplat이 `assert(depth_f >= 0.f)`를 박아둔 이유가 이것이다 — near plane 컬링 뒤라
# depth는 항상 양수지만, `isect_tiles`는 독립적으로 호출될 수 있는 op이므로 불변식을 못박아 둔다.

# %%
mix = np.array([-100.0, -2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 100.0], dtype=np.float32)
mu = mix.view(np.uint32)
print(f"{'value':>8} {'uint32':>11}")
for v, b in zip(mix, mu):
    print(f"{v:>8.1f} {int(b):>11d}")
print("\n정수 뷰 오름차순 정렬 결과:", mix[np.argsort(mu, kind="stable")])
print("실제 float 정렬 결과      :", np.sort(mix))
# 출력:
#    value      uint32
#  -100.0  3267887104
#    -2.0  3221225472
#    -1.0  3212836864
#    -0.0  2147483648
#     0.0           0
#     1.0  1065353216
#     2.0  1073741824
#   100.0  1120403456
#
# 정수 뷰 오름차순 정렬 결과: [   0.    1.    2.  100.   -0.   -1.   -2. -100.]
# 실제 float 정렬 결과      : [-100.   -2.   -1.   -0.    0.    1.    2.  100.]

neg_mixed = rng.normal(0.0, 10.0, 50_000).astype(np.float32)
ok = np.array_equal(
    np.argsort(neg_mixed.view(np.uint32), kind="stable"),
    np.argsort(neg_mixed, kind="stable"),
)
print("음수 섞인 배열에서 정수 뷰 정렬이 맞는가?:", ok)
# 출력: 음수 섞인 배열에서 정수 뷰 정렬이 맞는가?: False

# %% [markdown]
# ## 5. 표준 해법: 부호 비트 flip / XOR 트릭
#
# 음수까지 다루려면 비트 패턴을 "순서 보존 부호 없는 정수(order-preserving key)"로 바꾼다.
#
# $$
# \mathrm{key}(u) =
# \begin{cases}
# u \oplus \mathtt{0x80000000}, & \text{부호 비트가 } 0 \ (\text{양수}) \\[2pt]
# \sim u \ (= u \oplus \mathtt{0xFFFFFFFF}), & \text{부호 비트가 } 1 \ (\text{음수})
# \end{cases}
# $$
#
# 브랜치 없이 한 줄로도 쓴다:
#
# ```c
# uint32_t mask = (uint32_t)(-(int32_t)(u >> 31)) | 0x80000000u;
# key = u ^ mask;
# ```
#
# - 양수: 최상위 비트만 세워 음수 블록보다 뒤로 보낸다.
# - 음수: 전 비트를 뒤집어 (a) 앞으로 보내고 (b) 내부 반전도 되돌린다.
#
# 역변환(`key -> u`)도 같은 형태라 정렬 후 원래 float를 복원할 수 있다.
# gsplat은 depth가 양수임이 보장되므로 **이 단계를 아예 생략**한다 — 커널에서 아낀 몇 개의 명령이자,
# 무엇보다 키 인코딩이 곧 `__float_as_uint` 한 줄이 된다.

# %%
def float_to_ordered_u32(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    mask = np.where(u >> 31 != 0, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
    return u ^ mask


def ordered_u32_to_float(k: np.ndarray) -> np.ndarray:
    mask = np.where(k >> 31 != 0, np.uint32(0x80000000), np.uint32(0xFFFFFFFF))
    return (k ^ mask).view(np.float32)


k = float_to_ordered_u32(mix)
print("flip 후 정렬:", mix[np.argsort(k, kind="stable")])
# 출력: flip 후 정렬: [-100.   -2.   -1.   -0.    0.    1.    2.  100.]

kk = float_to_ordered_u32(neg_mixed)
assert np.array_equal(
    np.argsort(kk, kind="stable"), np.argsort(neg_mixed, kind="stable")
)
assert np.array_equal(ordered_u32_to_float(kk), neg_mixed)  # 역변환도 정확
print("음수 5만 개 포함: flip 트릭으로 정렬 순서 일치 + 역변환 정확  ->  OK")
# 출력: 음수 5만 개 포함: flip 트릭으로 정렬 순서 일치 + 역변환 정확  ->  OK

# %% [markdown]
# ## 6. 예외 케이스 (짧게)
#
# | 케이스 | 비트 | 정수 비교에서 |
# |---|---|---|
# | `+0.0` | `0x00000000` | 가장 작은 키. 문제 없음 |
# | `-0.0` | `0x80000000` | `+0.0 == -0.0`인데 비트는 다르다 → 정수 비교에선 `-0.0 > +0.0`. 순서만 정하는 용도라 무해하지만 "동일 값" 가정은 깨진다 |
# | denormal (`e=0`) | `0x000001`~`0x7FFFFF` | 값도 $m$에 선형 → 단조성 유지 |
# | `+inf` | `0x7F800000` | 모든 유한 양수보다 큼 → 올바르게 맨 뒤 |
# | `NaN` | `e=255, m≠0` | `0x7F800001` 이상 → inf보다 뒤로 정렬된다. 값 비교로는 무의미하지만 "터지진" 않는다. depth에 NaN이 들어오면 애초에 상류가 잘못된 것 |
#
# gsplat 파이프라인에서 depth는 near-plane 컬링을 통과한 카메라 $z$이므로
# `+0.0` 이상의 유한한 정상 값이고, 위 예외들은 실질적으로 발생하지 않는다.
#
# 참고로 **원 3DGS 구현도 완전히 같은 방식**이다:
#
# ```cpp
# // diff-gaussian-rasterization
# gaussian_keys_unsorted[off] = key;                       // (tile << 32)
# gaussian_keys_unsorted[off] |= *((uint32_t*)&depths[idx]);
# ```

# %%
special = np.array([0.0, -0.0, 1e-45, 1.4e-45, 3.4e38, np.inf, np.nan], dtype=np.float32)
print(f"{'value':>14} {'uint32':>11}  bits")
for v in special:
    b = int(np.float32(v).view(np.uint32))
    print(f"{v:>14.6g} {b:>11d}  {b:032b}")
# 출력:
#          value      uint32  bits
#              0           0  00000000000000000000000000000000
#             -0  2147483648  10000000000000000000000000000000
#     1.4013e-45           1  00000000000000000000000000000001
#     1.4013e-45           1  00000000000000000000000000000001
#        3.4e+38  2139081118  01111111011111111100100110011110
#            inf  2139095040  01111111100000000000000000000000
#            nan  2143289344  01111111110000000000000000000000

# %% [markdown]
# ## 7. 시각화: float 값 vs 정수 뷰
#
# 왼쪽: $x$축 float 값, $y$축 uint32 비트값. **단조 증가**하지만 지수 구간마다 기울기가
# 절반씩 꺾이는 계단(piecewise-linear) 모양이다 — 지수가 1 오를 때마다 값의 폭은 2배가 되는데
# 정수 키는 항상 $2^{23}$씩만 늘기 때문.
#
# 오른쪽: 같은 데이터를 $x$축 로그 스케일로. 로그 축에서는 거의 **직선**이 되는데,
# $\mathrm{bits}(x) \approx 2^{23}\left(\log_2 x + 127\right)$ 이기 때문이다.
# 즉 float 비트 패턴은 값의 로그에 대한 (구간별 선형) 근사 정수다.
#
# 음수를 포함한 빨간 곡선은 부호 비트 때문에 $x<0$에서 $\ge 2^{31}$로 튀어 오르며
# 단조성이 깨지는 모습을 보여준다.

# %%
xs_pos = np.concatenate(
    [np.linspace(0.25, 16.0, 4000), np.array([0.5, 1.0, 2.0, 4.0, 8.0])]
).astype(np.float32)
xs_pos.sort()
ys_pos = xs_pos.view(np.uint32).astype(np.float64)

xs_all = np.linspace(-16.0, 16.0, 8000).astype(np.float32)
ys_all = xs_all.view(np.uint32).astype(np.float64)
ys_fix = float_to_ordered_u32(xs_all).astype(np.float64)

xs_log = np.exp(np.linspace(np.log(1e-6), np.log(1e6), 4000)).astype(np.float32)
ys_log = xs_log.view(np.uint32).astype(np.float64)

fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=(
        "양수: 단조 증가 (지수마다 기울기 꺾임)",
        "로그 축: 거의 직선 ≈ 2²³(log₂x + 127)",
        "음수 포함: 그대로 쓰면 깨짐 / flip 하면 복구",
    ),
)

fig.add_trace(
    go.Scatter(x=xs_pos, y=ys_pos, mode="lines", name="uint32(bits)",
               line=dict(color="#2563eb", width=2)),
    row=1, col=1,
)
for bx in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
    fig.add_vline(x=bx, line=dict(color="#94a3b8", width=1, dash="dot"), row=1, col=1)

fig.add_trace(
    go.Scatter(x=xs_log, y=ys_log, mode="lines", name="uint32 (log x)",
               line=dict(color="#059669", width=2), showlegend=True),
    row=1, col=2,
)

fig.add_trace(
    go.Scatter(x=xs_all, y=ys_all, mode="lines", name="raw bits (깨짐)",
               line=dict(color="#dc2626", width=2)),
    row=1, col=3,
)
fig.add_trace(
    go.Scatter(x=xs_all, y=ys_fix, mode="lines", name="sign-flip key (정상)",
               line=dict(color="#7c3aed", width=2, dash="dash")),
    row=1, col=3,
)

fig.update_xaxes(title_text="float 값", row=1, col=1)
fig.update_xaxes(title_text="float 값 (log)", type="log", row=1, col=2)
fig.update_xaxes(title_text="float 값", row=1, col=3)
fig.update_yaxes(title_text="정수 뷰(uint32)", row=1, col=1)
fig.update_layout(
    title="float32 비트 패턴 ↔ 정수 키: 양수에서만 단조",
    width=1500,
    height=520,
    template="plotly_white",
    legend=dict(orientation="h", y=-0.18),
)

_show(fig)
fig.write_image(f"{HERE}/expy.png", scale=2)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 정리
#
# - IEEE 754 single은 `[부호 1 | 지수 8(biased) | 가수 23]` 순서이고, 이 배치가
#   **음이 아닌 값에 대해 순서 보존(monotonic) 사상**을 만든다.
# - 지수가 offset binary라 "큰 지수 = 큰 정수", 같은 지수 안에서는 값이 가수에 선형 →
#   상위 필드부터 비교하는 **사전식 = radix 비교**가 곧 크기 비교가 된다.
# - radix sort는 비트를 자리별로 훑으므로 float 비트를 **변환 없이** 그대로 먹인다.
# - 음수는 (a) 부호 비트가 최상위라 양수보다 커지고 (b) 내부 순서도 반전되어 깨진다 →
#   `u ^ (u>>31 ? 0xFFFFFFFF : 0x80000000)` 로 고친다.
# - gsplat은 near-plane 컬링 후 depth가 항상 양수라 이 트릭이 필요 없고,
#   `__float_as_uint(depth)`를 64비트 키 하위 32비트에 그대로 OR 한다.
