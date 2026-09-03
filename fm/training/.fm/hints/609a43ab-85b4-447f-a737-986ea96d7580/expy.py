# %% [markdown]
# # `isect_tiles` — 타일 교차 단계를 손으로 재현하기
#
# gsplat `rasterization()`의 3번째 커널 단계인 **타일 교차**(`isect_tiles`)가
# 실제로 무엇을 계산하는지, numpy만으로 같은 결과를 만들어 본다.
#
# 흐름:
# 1. 화면을 `tile_size`(=16) 픽셀 격자로 자른다 → `tile_width`, `tile_height`
# 2. 각 Gaussian의 화면공간 AABB로 걸치는 타일 범위를 구한다 → `tiles_per_gauss`
# 3. 교차마다 64비트 키 `image_id | tile_id | depth`를 만든다 → `isect_ids`
# 4. 키를 정렬한다 → 타일별로 모여 있고, 타일 안에서는 깊이 오름차순
# 5. 정렬된 배열을 타일 경계 오프셋으로 인코딩한다 → `isect_offsets` (`isect_offset_encode`)
#
# 참고 소스: `gsplat/cuda/_wrapper.py:1196` (`isect_tiles`),
# `gsplat/cuda/csrc/IntersectTile.cu` (`intersect_tile_kernel`, `intersect_offset_kernel`),
# `gsplat/rendering.py:886`.

# %%
# 필요 패키지: numpy, plotly, kaleido (모두 gsplat 환경에 설치돼 있음)
# gsplat 자체는 import하지 않는다 (JIT 빌드가 매우 오래 걸림) — 순수 numpy 재구현.
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


np.set_printoptions(precision=3, suppress=True)

# gsplat/cuda/include/Common.h 의 상수
ALPHA_THRESHOLD = 1.0 / 255.0
GAUSSIAN_EXTEND = 3.33

# %% [markdown]
# ## 1. 화면을 타일로 나눈다
#
# `rendering.py`는 다음처럼 격자 크기를 정한다.
#
# $$\text{tile\_width} = \left\lceil \frac{W}{\text{tile\_size}} \right\rceil,\qquad
#   \text{tile\_height} = \left\lceil \frac{H}{\text{tile\_size}} \right\rceil$$
#
# 3DGS 커널은 `TILE_SIZE=16`으로만 컴파일된다(`_resolve_tile_size()`).
# 타일 하나 = CUDA 블록 하나 = 16×16 = 256 픽셀 → 스레드 1개가 픽셀 1개를 담당.

# %%
W, H = 64, 48  # 장난감 해상도 (실제로는 1080p 등)
TILE = 16
tile_width = int(np.ceil(W / TILE))
tile_height = int(np.ceil(H / TILE))
n_tiles = tile_width * tile_height
print(f"이미지 {W}x{H}, tile_size={TILE} -> 격자 {tile_width}x{tile_height} = {n_tiles} 타일")
# 출력: 이미지 64x48, tile_size=16 -> 격자 4x3 = 12 타일

# 투영(fully_fused_projection) 결과라고 가정하는 입력들
# means2d: 화면 픽셀 좌표 [N,2], radii: 축별 반경(픽셀) [N,2], depths: 카메라 z [N]
means2d = np.array(
    [
        [8.0, 8.0],  # g0: 좌상단 타일 안에 완전히 들어감
        [32.0, 24.0],  # g1: 4개 타일 경계에 걸침 (큰 반경)
        [50.0, 10.0],  # g2: 중간 크기
        [32.0, 24.0],  # g3: g1과 같은 위치, 더 앞쪽 깊이
        [-5.0, 40.0],  # g4: 화면 왼쪽 밖으로 삐져나감 → 클램프
    ],
    dtype=np.float32,
)
radii = np.array([[5, 5], [20, 14], [9, 9], [6, 6], [10, 10]], dtype=np.int32)
depths = np.array([3.5, 8.0, 1.2, 2.0, 5.0], dtype=np.float32)
N = len(depths)
I = 1  # 이미지(카메라) 개수
print("N =", N)
# 출력: N = 5

# %% [markdown]
# ## 2. 1차 패스: Gaussian별 타일 개수 (`tiles_per_gauss`)
#
# `IntersectTile.cu`의 AABB 경로를 그대로 옮기면
#
# ```
# tile_min.x = clamp(floor(mean.x/T - r_x/T), 0, tile_width)   # 포함
# tile_max.x = clamp(ceil (mean.x/T + r_x/T), 0, tile_width)   # 배타
# ```
#
# 커널은 이 계산을 **두 번** 한다. 1차 패스는 개수만 세어 `tiles_per_gauss`를 채우고,
# 그 누적합(`cub::DeviceScan::InclusiveSum`)으로 각 Gaussian이 쓸 출력 슬롯을
# 미리 확보한 뒤, 2차 패스에서 같은 루프를 돌며 그 슬롯에 키를 써 넣는다.
# 출력 배열 크기 `n_isects`를 미리 알 수 없기 때문에 필요한 구조다.
#
# `radii <= 0`이면(투영 단계에서 컬링된 Gaussian) 0개 타일로 즉시 반환한다.

# %%
def tile_bbox(mean, r):
    """IntersectTile.cu AABB 경로: 타일 격자 좌표의 [min, max) 범위."""
    tmin_x = min(max(0, int(np.floor(mean[0] / TILE - r[0] / TILE))), tile_width)
    tmin_y = min(max(0, int(np.floor(mean[1] / TILE - r[1] / TILE))), tile_height)
    tmax_x = min(max(0, int(np.ceil(mean[0] / TILE + r[0] / TILE))), tile_width)
    tmax_y = min(max(0, int(np.ceil(mean[1] / TILE + r[1] / TILE))), tile_height)
    return (tmin_x, tmin_y), (tmax_x, tmax_y)


bboxes = [tile_bbox(means2d[i], radii[i]) for i in range(N)]
tiles_per_gauss = np.array(
    [max(0, b[1][0] - b[0][0]) * max(0, b[1][1] - b[0][1]) for b in bboxes], dtype=np.int32
)
for i, (b, c) in enumerate(zip(bboxes, tiles_per_gauss)):
    print(f"g{i}: tile_min={b[0]} tile_max={b[1]} -> {c} 타일")
print("tiles_per_gauss =", tiles_per_gauss)

cum = np.cumsum(tiles_per_gauss.astype(np.int64))  # InclusiveSum
n_isects = int(cum[-1])
print("cum_tiles_per_gauss =", cum, " n_isects =", n_isects)
# 출력: g0: tile_min=(0, 0) tile_max=(1, 1) -> 1 타일
# 출력: g1: tile_min=(0, 0) tile_max=(4, 3) -> 12 타일
# 출력: g2: tile_min=(2, 0) tile_max=(4, 2) -> 4 타일
# 출력: g3: tile_min=(1, 1) tile_max=(3, 2) -> 2 타일
# 출력: g4: tile_min=(0, 1) tile_max=(1, 3) -> 2 타일  (x가 음수 → 0으로 클램프)
# 출력: tiles_per_gauss = [ 1 12  4  2  2]
# 출력: cum_tiles_per_gauss = [ 1 13 17 19 21]  n_isects = 21

# %% [markdown]
# ## 3. 64비트 교차 키 인코딩
#
# 키 레이아웃(`intersect_tile_kernel`):
#
# ```
# [ image_id : image_n_bits ][ tile_id : tile_n_bits ][ depth : 32 bits ]
#  <---------------- 상위 32비트 ---------------->    <--- 하위 32비트 --->
# ```
#
# - `image_n_bits + tile_n_bits <= 32`이어야 한다(초과 시 `TORCH_CHECK` 실패).
# - 깊이는 float32 **비트를 그대로** 하위 32비트에 넣는다: `__float_as_uint(depth)`.
#   IEEE-754 양의 float은 비트 패턴의 부호 없는 정수 순서가 값의 순서와 같기 때문에,
#   부동소수 비교 없이 정수 radix sort 한 번으로 깊이 정렬이 된다.
#   음수 깊이면 부호 비트가 순서를 뒤집으므로 near-plane 컬링 뒤의 $d \ge 0$이 전제다.
# - 한 Gaussian이 걸친 타일마다 **키 하나**가 생긴다 → 같은 Gaussian이 여러 번 등장.
#   그래서 짝으로 `flatten_ids`(원래 Gaussian 인덱스)를 함께 들고 다닌다.

# %%
def bits_for_count(count):
    """MathUtils.h bits_for_count: count-1을 담는 데 필요한 비트 수."""
    return 0 if count <= 1 else int(count - 1).bit_length()


image_n_bits = bits_for_count(I)
tile_n_bits = bits_for_count(n_tiles)
print(f"image_n_bits={image_n_bits}, tile_n_bits={tile_n_bits} (합 {image_n_bits + tile_n_bits} <= 32)")
# 출력: image_n_bits=0, tile_n_bits=4 (합 4 <= 32)


def depth_key(d):
    """__float_as_uint(float(d)) — float32 비트를 uint32로 재해석."""
    return int(np.float32(d).view(np.uint32))


# 단조성 확인: 값 순서 == 비트 패턴 순서 (음수가 아닌 한)
probe = np.array([0.0, 1e-6, 0.5, 1.2, 2.0, 3.5, 8.0, 1e6], dtype=np.float32)
keys = [depth_key(x) for x in probe]
print("깊이 오름차순 -> 키도 오름차순?", all(a < b for a, b in zip(keys, keys[1:])))
print("예: depth=1.2 ->", hex(depth_key(1.2)), " depth=8.0 ->", hex(depth_key(8.0)))
# 출력: 깊이 오름차순 -> 키도 오름차순? True
# 출력: 예: depth=1.2 -> 0x3f99999a  depth=8.0 -> 0x41000000

# 2차 패스: 슬롯에 키를 써 넣는다
isect_ids = np.zeros(n_isects, dtype=np.int64)
flatten_ids = np.zeros(n_isects, dtype=np.int32)
cursor = 0
for gi in range(N):
    (tx0, ty0), (tx1, ty1) = bboxes[gi]
    iid = 0  # image_id (packed=False에서는 idx // N)
    iid_enc = iid << (32 + tile_n_bits)
    d_enc = depth_key(depths[gi])
    for ty in range(ty0, ty1):
        for tx in range(tx0, tx1):
            tile_id = ty * tile_width + tx
            isect_ids[cursor] = iid_enc | (tile_id << 32) | d_enc
            flatten_ids[cursor] = gi
            cursor += 1
assert cursor == n_isects
print("정렬 전 앞 6개 (tile_id, gauss, depth):")
for k in range(6):
    tid = (isect_ids[k] >> 32) & ((1 << tile_n_bits) - 1)
    print(f"  tile={tid:2d} g{flatten_ids[k]} depth={depths[flatten_ids[k]]:.2f}")
# 출력: 정렬 전 앞 6개 (tile_id, gauss, depth):
# 출력:   tile= 0 g0 depth=3.50
# 출력:   tile= 0 g1 depth=8.00
# 출력:   tile= 1 g1 depth=8.00
# 출력:   tile= 2 g1 depth=8.00
# 출력:   tile= 3 g1 depth=8.00
# 출력:   tile= 4 g1 depth=8.00

# %% [markdown]
# ## 4. 정렬 = 이 단계의 핵심
#
# `cub::DeviceRadixSort::SortPairs(keys=isect_ids, values=flatten_ids)`를
# 하위 `32 + tile_n_bits + image_n_bits` 비트만 대상으로 돌린다
# (`radix_sort_double_buffer`). 상위 비트가 image·tile이므로 정렬 결과는
#
# - 같은 이미지의 같은 타일에 속한 교차들이 **연속 구간**으로 모이고,
# - 그 구간 안에서는 **깊이 오름차순**(앞→뒤)이 된다.
#
# 즉 정렬 한 번으로 "타일별 깊이순 리스트"가 공짜로 만들어진다.
# 이게 4단계 `rasterize_to_pixels`의 알파 블렌딩이 요구하는 정확한 순서다.

# %%
order = np.argsort(isect_ids, kind="stable")
isect_ids_sorted = isect_ids[order]
flatten_ids_sorted = flatten_ids[order]

tid_sorted = ((isect_ids_sorted >> 32) & ((1 << tile_n_bits) - 1)).astype(np.int64)
depth_sorted = depths[flatten_ids_sorted]
print("idx  tile  gauss  depth")
for k in range(n_isects):
    print(f"{k:3d}  {tid_sorted[k]:4d}  g{flatten_ids_sorted[k]}     {depth_sorted[k]:.2f}")
# 출력: idx  tile  gauss  depth
# 출력:   0     0  g0     3.50
# 출력:   1     0  g1     8.00
# 출력:   2     1  g1     8.00
# 출력:   3     2  g2     1.20
# 출력:   4     2  g1     8.00
# 출력:   5     3  g2     1.20
# 출력:   6     3  g1     8.00
# 출력:   7     4  g4     5.00
# 출력:   8     4  g1     8.00
# 출력:   9     5  g3     2.00
# 출력:  10     5  g1     8.00
# 출력:  11     6  g2     1.20
# 출력:  12     6  g3     2.00
# 출력:  13     6  g1     8.00
# 출력:  14     7  g2     1.20
# 출력:  15     7  g1     8.00
# 출력:  16     8  g4     5.00
# 출력:  17     8  g1     8.00
# 출력:  18     9  g1     8.00
# 출력:  19    10  g1     8.00
# 출력:  20    11  g1     8.00
# (g1은 12개 타일 전부에 등장한다 — 큰 Gaussian 하나가 교차 수를 지배하는 모습)

# 각 타일 구간이 정말 깊이 오름차순인지 검증
ok = True
for t in range(n_tiles):
    seg = depth_sorted[tid_sorted == t]
    ok &= bool(np.all(np.diff(seg) >= 0))
print("모든 타일 구간이 깊이 오름차순:", ok)
# 출력: 모든 타일 구간이 깊이 오름차순: True

# %% [markdown]
# ## 5. `isect_offset_encode` — 타일 → 구간 시작 인덱스
#
# 정렬된 배열만으로는 래스터라이저가 "내 타일의 교차가 어디서 시작하는가"를 모른다.
# `isect_offset_encode()`가 정렬된 `isect_ids`를 훑어 타일 경계가 바뀌는 지점을 찾아
# `[I, tile_height, tile_width]` 크기의 오프셋 텐서를 만든다.
#
# 래스터라이저 커널(`RasterizeToPixels3DGSSerialBatchFwd.cu:164`)은 이렇게 읽는다.
#
# ```
# range_start = isect_offsets[tile_id]
# range_end   = (마지막 타일이면 n_isects) else isect_offsets[tile_id + 1]
# ```
#
# 교차가 하나도 없는 타일은 `range_start == range_end`가 되어 즉시 배경으로 끝난다.
# (커널 주석의 예시: ids `[1,1,1,3,3]`, n_tiles=6 → offsets `[0,0,3,3,5,5]`)

# %%
def isect_offset_encode(sorted_tile_ids, n_tiles_total):
    """정렬된 tile_id 배열 -> 타일별 시작 오프셋 [n_tiles]."""
    return np.searchsorted(sorted_tile_ids, np.arange(n_tiles_total), side="left").astype(np.int32)


isect_offsets = isect_offset_encode(tid_sorted, n_tiles)
print("isect_offsets =", isect_offsets)
print("(2D 격자)\n", isect_offsets.reshape(tile_height, tile_width))
# 출력: isect_offsets = [ 0  2  3  5  7  9 11 14 16 18 19 20]
# 출력: (2D 격자)
# 출력:  [[ 0  2  3  5]
# 출력:  [ 7  9 11 14]
# 출력:  [16 18 19 20]]

print("\n타일별 앞→뒤 처리 순서:")
for t in range(n_tiles):
    s = int(isect_offsets[t])
    e = int(isect_offsets[t + 1]) if t + 1 < n_tiles else n_isects
    ty, tx = divmod(t, tile_width)
    lst = [f"g{g}({d:.1f})" for g, d in zip(flatten_ids_sorted[s:e], depth_sorted[s:e])]
    print(f"  tile {t:2d} (tx={tx},ty={ty}) [{s:2d},{e:2d}) : {' -> '.join(lst) if lst else '(비어있음)'}")
# 출력: 타일별 앞→뒤 처리 순서:
# 출력:   tile  0 (tx=0,ty=0) [ 0, 2) : g0(3.5) -> g1(8.0)
# 출력:   tile  1 (tx=1,ty=0) [ 2, 3) : g1(8.0)
# 출력:   tile  2 (tx=2,ty=0) [ 3, 5) : g2(1.2) -> g1(8.0)
# 출력:   tile  3 (tx=3,ty=0) [ 5, 7) : g2(1.2) -> g1(8.0)
# 출력:   tile  4 (tx=0,ty=1) [ 7, 9) : g4(5.0) -> g1(8.0)
# 출력:   tile  5 (tx=1,ty=1) [ 9,11) : g3(2.0) -> g1(8.0)
# 출력:   tile  6 (tx=2,ty=1) [11,14) : g2(1.2) -> g3(2.0) -> g1(8.0)
# 출력:   tile  7 (tx=3,ty=1) [14,16) : g2(1.2) -> g1(8.0)
# 출력:   tile  8 (tx=0,ty=2) [16,18) : g4(5.0) -> g1(8.0)
# 출력:   tile  9 (tx=1,ty=2) [18,19) : g1(8.0)
# 출력:   tile 10 (tx=2,ty=2) [19,20) : g1(8.0)
# 출력:   tile 11 (tx=3,ty=2) [20,21) : g1(8.0)
# tile 6은 g2 -> g3 -> g1 순서: 깊이 1.2 -> 2.0 -> 8.0, 즉 앞→뒤.

# %% [markdown]
# ## 6. 왜 이 단계가 성능을 좌우하는가 — AccuTile(SNUGBOX) 가지치기
#
# AABB 경로는 타원을 **축 정렬 사각형**으로 근사하므로, 비스듬히 길쭉한 Gaussian은
# 실제로 스치지도 않는 타일까지 교차로 잡는다. 그 타일의 256개 스레드는 이 Gaussian을
# shared memory로 올려놓고 전부 $\alpha < 1/255$로 버린다 — 순수 낭비다.
#
# gsplat은 `conics`, `opacities`가 주어지면(3DGS 경로) 커널이 AccuTile/SNUGBOX
# (SpeedySplat, arXiv 2412.00578)로 전환해, 불투명도 임계값을 넘는 등고선 타원만으로
# 보수적 교차를 판정한다. 임계 등고선 레벨은
#
# $$\alpha = o\,e^{-q/2} \ge \tfrac{1}{255}
#   \;\Longrightarrow\; q \le t = \min\!\left(\text{EXTEND}^2,\; 2\ln\frac{o}{1/255}\right)$$
#
# 이고, $q = a\,dx^2 + 2b\,dx\,dy + c\,dy^2$ (conic = $\Sigma^{-1}$의 상삼각).
# 아래에서 45도 기울어진 길쭉한 Gaussian으로 AABB 개수와 실제 교차 개수를 비교한다.

# %%
# 45도 회전한 길쭉한 2D Gaussian
theta = np.deg2rad(45.0)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
S = np.diag([18.0**2, 2.0**2])  # 주축 18px, 부축 2px
Sigma = R @ S @ R.T
conic = np.linalg.inv(Sigma)
mean_t = np.array([32.0, 24.0])
opacity = 0.9
t_level = min(GAUSSIAN_EXTEND**2, 2.0 * np.log(opacity / ALPHA_THRESHOLD))
print("Sigma =\n", Sigma, "\nconic =\n", conic, "\nt_level =", round(t_level, 3))
# 출력: Sigma =
# 출력:  [[164.  160.]
# 출력:  [160.  164.]]
# 출력: conic =
# 출력:  [[ 0.127 -0.123]
# 출력:  [-0.123  0.127]]
# 출력: t_level = 10.872

# AABB 경로가 잡는 타일 (gsplat과 동일하게 축별 표준편차 * EXTEND)
r_aabb = np.ceil(GAUSSIAN_EXTEND * np.sqrt(np.diag(Sigma))).astype(np.int32)
(bx0, by0), (bx1, by1) = tile_bbox(mean_t, r_aabb)
aabb_tiles = {(tx, ty) for ty in range(by0, by1) for tx in range(bx0, bx1)}
print("radii(AABB) =", r_aabb, " AABB 타일 수 =", len(aabb_tiles))
# 출력: radii(AABB) = [43 43]  AABB 타일 수 = 12

# 실제로 알파 임계를 넘는 픽셀이 있는 타일만 (AccuTile이 근사하려는 정답)
yy, xx = np.mgrid[0:H, 0:W]
dx = (xx + 0.5) - mean_t[0]
dy = (yy + 0.5) - mean_t[1]
q = conic[0, 0] * dx * dx + 2 * conic[0, 1] * dx * dy + conic[1, 1] * dy * dy
alpha_map = opacity * np.exp(-0.5 * q)
hit = alpha_map >= ALPHA_THRESHOLD
true_tiles = set()
for ty in range(tile_height):
    for tx in range(tile_width):
        if hit[ty * TILE : (ty + 1) * TILE, tx * TILE : (tx + 1) * TILE].any():
            true_tiles.add((tx, ty))
print("실제 교차 타일 수 =", len(true_tiles))
print("AABB가 헛되게 잡은 타일 =", sorted(aabb_tiles - true_tiles))
# 출력: 실제 교차 타일 수 = 8
# 출력: AABB가 헛되게 잡은 타일 = [(0, 1), (0, 2), (3, 0), (3, 1)]
# → 12개 중 4개(33%)가 헛교차. 그 타일의 256개 스레드는 이 Gaussian을 shared memory에
#   올려놓고 전부 버린다. AccuTile은 이 4개를 애초에 키로 만들지 않는다.

# %% [markdown]
# ## 7. 시각화
#
# - 왼쪽: 타일 격자 위의 Gaussian AABB(점선)와 타일별 교차 개수(색). 숫자는 `isect_offsets`.
# - 오른쪽: 정렬된 `isect_ids` 배열. x=배열 인덱스, y=tile_id, 색=깊이.
#   타일이 계단처럼 단조 증가하고, 같은 타일 안에서는 깊이가 커지는 것이 정렬의 결과다.

# %%
counts = np.zeros((tile_height, tile_width), dtype=np.int32)
for t in range(n_tiles):
    s = int(isect_offsets[t])
    e = int(isect_offsets[t + 1]) if t + 1 < n_tiles else n_isects
    counts[t // tile_width, t % tile_width] = e - s

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("타일 격자와 Gaussian AABB (색=타일별 교차 수)", "정렬된 isect_ids (색=깊이)"),
    column_widths=[0.52, 0.48],
)

fig.add_trace(
    go.Heatmap(
        z=counts,
        x=[(i + 0.5) * TILE for i in range(tile_width)],
        y=[(j + 0.5) * TILE for j in range(tile_height)],
        colorscale="Blues",
        showscale=False,
        zmin=0,
        hovertemplate="tile 교차 수=%{z}<extra></extra>",
    ),
    row=1,
    col=1,
)
for t in range(n_tiles):
    ty, tx = divmod(t, tile_width)
    fig.add_annotation(
        x=(tx + 0.5) * TILE,
        y=(ty + 0.5) * TILE,
        text=f"t{t}<br>off={isect_offsets[t]}",
        showarrow=False,
        font=dict(size=9, color="#fff" if counts[ty, tx] >= 3 else "#333"),
        row=1,
        col=1,
    )
palette = ["#e45756", "#4c78a8", "#54a24b", "#f58518", "#b279a2"]
for gi in range(N):
    (tx0, ty0), (tx1, ty1) = bboxes[gi]
    x0, y0 = means2d[gi] - radii[gi]
    x1, y1 = means2d[gi] + radii[gi]
    fig.add_shape(
        type="rect",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        line=dict(color=palette[gi], width=2, dash="dot"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[means2d[gi][0]],
            y=[means2d[gi][1]],
            mode="markers+text",
            marker=dict(size=9, color=palette[gi]),
            text=[f"g{gi} d={depths[gi]:.1f}"],
            textposition="bottom center" if gi == 3 else "top center",
            textfont=dict(size=9),
            name=f"g{gi} ({tiles_per_gauss[gi]} tiles)",
        ),
        row=1,
        col=1,
    )
for i in range(tile_width + 1):
    fig.add_shape(type="line", x0=i * TILE, y0=0, x1=i * TILE, y1=H, line=dict(color="#888", width=1), row=1, col=1)
for j in range(tile_height + 1):
    fig.add_shape(type="line", x0=0, y0=j * TILE, x1=W, y1=j * TILE, line=dict(color="#888", width=1), row=1, col=1)

fig.add_trace(
    go.Scatter(
        x=np.arange(n_isects),
        y=tid_sorted,
        mode="markers+lines",
        line=dict(color="#bbb", width=1),
        marker=dict(
            size=11,
            color=depth_sorted,
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(title="depth", x=1.01, len=0.8),
        ),
        text=[f"g{g} depth={d:.2f}" for g, d in zip(flatten_ids_sorted, depth_sorted)],
        hovertemplate="idx=%{x}<br>tile=%{y}<br>%{text}<extra></extra>",
        name="isect (sorted)",
    ),
    row=1,
    col=2,
)
for t in range(n_tiles):
    s = int(isect_offsets[t])
    fig.add_shape(type="line", x0=s - 0.5, y0=-0.5, x1=s - 0.5, y1=n_tiles - 0.5,
                  line=dict(color="#ddd", width=1), row=1, col=2)

fig.update_xaxes(title="pixel x", range=[-12, W + 4], row=1, col=1)
fig.update_yaxes(title="pixel y", range=[H + 4, -12], scaleanchor="x", row=1, col=1)
fig.update_xaxes(title="정렬된 배열 인덱스 (isect_offsets가 구간 경계)", row=1, col=2)
fig.update_yaxes(title="tile_id", dtick=1, row=1, col=2)
fig.update_layout(
    title=f"isect_tiles: {N} Gaussians -> {n_isects} 교차, {n_tiles} 타일 (tile_size={TILE})",
    height=520,
    width=1180,
    showlegend=True,
    legend=dict(orientation="h", y=-0.18),
    template="plotly_white",
)

fig.write_image("expy.png", scale=2)
print("expy.png 저장 완료")
# 출력: expy.png 저장 완료
_show(fig)

# %% [markdown]
# ## 정리
#
# | 산출물 | 모양 | 의미 |
# |---|---|---|
# | `tiles_per_gauss` | int32 `[..., N]` | Gaussian이 걸친 타일 수 (밀도화/프로파일링에 유용) |
# | `isect_ids` | int64 `[n_isects]` | `image_id \| tile_id \| depth` 정렬 키 |
# | `flatten_ids` | int32 `[n_isects]` | 각 교차의 원래 Gaussian 인덱스 (`[I*N]` 평탄 인덱스) |
# | `isect_offsets` | int32 `[I, th, tw]` | 정렬된 배열에서 타일별 구간 시작 (`isect_offset_encode`) |
#
# 핵심: `isect_tiles`는 **픽셀 값을 하나도 계산하지 않는다**. 오직
# "어느 타일이 어떤 Gaussian들을, 어떤 순서로 처리해야 하는가"라는 **작업 스케줄**을
# 만드는 준비 단계다. 이 스케줄이 있어야 다음 단계 `rasterize_to_pixels`가
# 타일=CUDA 블록 단위로 앞→뒤 알파 블렌딩을 하고, 투과율이 임계값 아래로 떨어지는
# 즉시 조기 종료할 수 있다.
#
# 주의: `n_isects`는 Gaussian 수 $N$이 아니라 **면적에 비례**한다. 크고 흐릿한
# Gaussian 하나가 수천 개의 키를 만들 수 있고, 정렬 비용과 메모리는 여기서 나온다.
# 그래서 밀도화 전략이 큰 Gaussian을 쪼개고(`split`), AccuTile이 헛교차를 잘라내며,
# `_resolve_tile_size()`가 해상도에 따라 타일 크기를 고르는 것이다.
