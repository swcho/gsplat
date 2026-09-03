# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # `isect_offset_encode`는 무엇을 하는가?
#
# 한 줄 답: **정렬된 (image, tile, depth) 키 배열에서 타일이 바뀌는 지점을 찾아, 타일별 시작 오프셋 배열을 만든다.**
# 빈 타일은 앞 타일의 끝 위치를 그대로 물려받으므로 `[start, end)` 구간의 길이가 0이 된다.
#
# ## 왜 필요한가
#
# ⑤ 단계(`isect_tiles`)가 만든 것은 두 배열뿐이다.
#
# - `isect_ids[n_isects]` — 64비트 키. 상위→하위로 `[ image_id | tile_id | float32(depth) 비트 ]`
# - `flatten_ids[n_isects]` — 같은 순서의 Gaussian 인덱스
#
# 키를 radix sort 했으므로 **같은 타일의 교차가 연속 구간으로 모여 있고, 그 안에서는 가까운 것부터** 놓여 있다.
# ⑦ 단계에서 CUDA 블록 하나가 타일 하나를 맡을 때 필요한 것은 "내 타일의 구간이 어디부터 어디까지냐"뿐이다.
# 즉 각 타일의 **시작 인덱스**만 알면 된다.
#
# $$
# \text{tile } t \text{ 의 Gaussian 목록} \;=\; \texttt{flatten\_ids}\bigl[\,\text{offsets}[t] \;:\; \text{offsets}[t{+}1]\,\bigr]
# $$
#
# (마지막 타일의 끝은 `n_isects`.) `isect_offset_encode`가 만드는 것이 바로 이 `offsets`이고,
# 모양은 `[C, tile_h, tile_w]`(int32)이다.
#
# ## 세 가지 동치 표현
#
# 타일 $t$ 의 교차 개수를 $c_t$ 라 하면
#
# $$
# \text{offsets}[t] \;=\; \sum_{u < t} c_u \;=\; \underbrace{\text{cumsum}(c)[t]}_{\text{포함 누적합}} - c_t
# $$
#
# 이고, 정렬된 타일 id 배열 $K$ 에 대해서는
#
# $$
# \text{offsets}[t] \;=\; \bigl|\{\,i : K_i < t \,\}\bigr| \;=\; \texttt{searchsorted}(K,\; t,\; \text{side=left})
# $$
#
# 이다. CUDA 커널은 세 번째 표현 — **"이웃과 비교해서 달라지는 곳에만 쓴다"** — 을 쓴다.
# 아래에서 셋이 모두 같은 배열을 내는 것을 확인한다.

# %%
# 필요 패키지: numpy, torch, plotly, kaleido  (gsplat 은 import 하지 않는다 — JIT CUDA 빌드가 30분 이상 걸림)
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.set_printoptions(linewidth=140)
torch.set_printoptions(linewidth=140)


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


HERE = __file__.rsplit("/", 1)[0] if "__file__" in dir() else "."
print("setup ok")
# 출력: setup ok


# %% [markdown]
# ## 1. 키 포장 규칙 — `bits_for_count`
#
# 타일이 `n_tiles` 개이면 tile_id 는 `0 .. n_tiles-1` 을 담아야 하므로 필요한 비트 수는
# $\lceil \log_2 n_{\text{tiles}} \rceil$ 이다. gsplat 은 이것을 `(n_tiles - 1).bit_length()` 로 계산한다
# (`count.bit_length()` 를 쓰면 2의 거듭제곱에서 1비트 과다: 8 → 4비트가 되어버린다).
#
# ```
#   63                    32+B          32                        0
#    +----------------------+------------+-------------------------+
#    |       image_id       |  tile_id   |   float32(depth) bits   |
#    +----------------------+------------+-------------------------+
#                              B = tile_n_bits
# ```
#
# 하위 32비트(depth)를 버리는 `isect_ids >> 32` 가 곧 **(image, tile) 쌍의 id** 이고,
# 여기에 `image_id * n_tiles + tile_id` 로 다시 펴면 `[C, tile_h, tile_w]` 를 평탄화한 **전역 타일 번호**가 된다.

# %%
def bits_for_count(count: int) -> int:
    """0..count-1 을 담는 데 필요한 비트 수 (gsplat::bits_for_count 와 동일)."""
    return (count - 1).bit_length() if count > 1 else 0


TILE_W, TILE_H = 4, 3  # 워크스루의 토이 씬: 64x48 이미지 / 16px 타일 → 4x3 타일
N_TILES = TILE_W * TILE_H
TILE_N_BITS = bits_for_count(N_TILES)

print(f"tiles {TILE_W}x{TILE_H} = {N_TILES},  tile_n_bits = {TILE_N_BITS}")
print("naive bit_length() 였다면:", N_TILES.bit_length(), "(2의 거듭제곱일 때 과다 계산)")
# 출력: tiles 4x3 = 12,  tile_n_bits = 4
# 출력: naive bit_length() 였다면: 4 (2의 거듭제곱일 때 과다 계산)


# %%
def pack_keys(image_ids, tile_ids, depths, tile_n_bits):
    """[image_id | tile_id | float32(depth) 비트] 로 64비트 키를 만든다."""
    d = np.asarray(depths, dtype=np.float32).view(np.int32).astype(np.int64) & 0xFFFFFFFF
    iid = np.asarray(image_ids, dtype=np.int64)
    tid = np.asarray(tile_ids, dtype=np.int64)
    return (iid << (32 + tile_n_bits)) | (tid << 32) | d


# 손으로 만든 정렬된 교차 목록 (C=1). 중간에 빈 타일이 일부러 들어 있다.
#   점유 타일 : 0(3개), 2(2개), 5(4개), 9(1개), 11(1개)
#   빈 타일   : 1, 3, 4, 6, 7, 8, 10
tile_seq = np.array([0, 0, 0, 2, 2, 5, 5, 5, 5, 9, 11])
depth_seq = np.array([1.0, 2.5, 4.0, 0.7, 3.3, 1.1, 1.9, 2.2, 5.0, 0.4, 2.8], dtype=np.float32)
gauss_seq = np.array([3, 0, 1, 3, 2, 0, 3, 1, 2, 2, 1])  # flatten_ids 에 해당

isect_ids = torch.from_numpy(pack_keys(np.zeros_like(tile_seq), tile_seq, depth_seq, TILE_N_BITS))
flatten_ids = torch.from_numpy(gauss_seq.astype(np.int32))
n_isects = isect_ids.numel()

print("n_isects =", n_isects)
print("정렬 확인 (키가 단조 증가?):", bool(torch.all(isect_ids[1:] >= isect_ids[:-1])))
print(" idx  tile  depth  gauss")
for i in range(n_isects):
    print(f" {i:3d}  {tile_seq[i]:4d}  {depth_seq[i]:5.2f}   g{gauss_seq[i]}")
# 출력: n_isects = 11
# 출력: 정렬 확인 (키가 단조 증가?): True
# 출력:  idx  tile  depth  gauss
# 출력:    0     0   1.00   g3
# 출력:    1     0   2.50   g0
# 출력:    2     0   4.00   g1
# 출력:    3     2   0.70   g3
# 출력:    4     2   3.30   g2
# 출력:    5     5   1.10   g0
# 출력:    6     5   1.90   g3
# 출력:    7     5   2.20   g1
# 출력:    8     5   5.00   g2
# 출력:    9     9   0.40   g2
# 출력:   10    11   2.80   g1


# %% [markdown]
# ## 2. CUDA 커널 방식 — "이전 원소와 비교, 달라지면 그 사이를 메운다"
#
# `IntersectTile.cu` 의 `intersect_offset_kernel` 은 **교차 하나당 스레드 하나**를 띄운다.
# 각 스레드 `idx` 는 자기 키와 바로 앞 키의 `(image, tile)` 부분만 비교한다.
#
# ```c
# int64_t isect_id_curr = isect_ids[idx] >> 32;              // depth 를 잘라낸다
# int64_t id_curr = iid_curr * n_tiles + tid_curr;           // 전역 타일 번호
#
# if (idx == 0)                                              // 맨 앞: 첫 점유 타일까지 0 으로
#     for (i = 0; i < id_curr + 1; ++i) offsets[i] = idx;
#
# if (idx == n_isects - 1)                                   // 맨 뒤: 남은 꼬리를 n_isects 로
#     for (i = id_curr + 1; i < I * n_tiles; ++i) offsets[i] = n_isects;
#
# if (idx > 0) {
#     int64_t isect_id_prev = isect_ids[idx - 1] >> 32;
#     if (isect_id_prev == isect_id_curr) return;            // 경계가 아니면 할 일 없음
#     // 경계다: 이전 타일 다음부터 현재 타일까지 전부 idx 를 쓴다  ← 빈 타일이 메워지는 지점
#     for (i = id_prev + 1; i < id_curr + 1; ++i) offsets[i] = idx;
# }
# ```
#
# 핵심은 마지막 루프다. 이전 타일이 3이고 현재 타일이 7이면 `offsets[4..7]` 에 모두 같은 `idx` 를 쓴다.
# 4, 5, 6 은 **빈 타일**이고 이들의 시작 = 끝 = 7의 시작이 되어 길이 0이 된다.
# 대부분의 스레드는 `isect_id_prev == isect_id_curr` 로 즉시 return 하므로, 실제 쓰기는 타일 경계 수만큼만 일어난다.

# %%
def offsets_cuda_style(isect_ids, I, tile_width, tile_height):
    """intersect_offset_kernel 의 스레드별 로직을 Python 루프로 재현."""
    n_tiles = tile_width * tile_height
    tile_n_bits = bits_for_count(n_tiles)
    n_isects = isect_ids.numel()
    offsets = np.full(I * n_tiles, -1, dtype=np.int64)  # -1 = 아직 아무도 안 씀
    trace = []

    if n_isects == 0:  # launch_intersect_offset_kernel 의 조기 반환
        return np.zeros(I * n_tiles, dtype=np.int32), trace

    ids = (isect_ids.numpy() >> 32)  # depth 제거
    for idx in range(n_isects):  # ← CUDA 에서는 이 루프가 병렬 스레드
        iid_curr = ids[idx] >> tile_n_bits
        tid_curr = ids[idx] & ((1 << tile_n_bits) - 1)
        id_curr = iid_curr * n_tiles + tid_curr

        if idx == 0:
            offsets[0 : id_curr + 1] = idx
            trace.append((idx, "head", f"offsets[0:{id_curr + 1}] = 0"))
        if idx == n_isects - 1:
            offsets[id_curr + 1 : I * n_tiles] = n_isects
            trace.append((idx, "tail", f"offsets[{id_curr + 1}:{I * n_tiles}] = {n_isects}"))
        if idx > 0:
            if ids[idx - 1] == ids[idx]:
                continue  # 같은 타일 안 → 경계 아님 → 아무것도 안 함
            iid_prev = ids[idx - 1] >> tile_n_bits
            tid_prev = ids[idx - 1] & ((1 << tile_n_bits) - 1)
            id_prev = iid_prev * n_tiles + tid_prev
            offsets[id_prev + 1 : id_curr + 1] = idx
            trace.append((idx, "boundary", f"offsets[{id_prev + 1}:{id_curr + 1}] = {idx}"))

    assert (offsets >= 0).all(), "빠진 칸이 있으면 안 된다"
    return offsets.astype(np.int32), trace


off_cuda, trace = offsets_cuda_style(isect_ids, 1, TILE_W, TILE_H)
print("실제로 쓰기가 일어난 스레드 (나머지는 즉시 return):")
for idx, kind, what in trace:
    print(f"  idx={idx:2d}  {kind:9s}  {what}")
print("\noffsets =", off_cuda)
# 출력: 실제로 쓰기가 일어난 스레드 (나머지는 즉시 return):
# 출력:   idx= 0  head       offsets[0:1] = 0
# 출력:   idx= 3  boundary   offsets[1:3] = 3
# 출력:   idx= 5  boundary   offsets[3:6] = 5
# 출력:   idx= 9  boundary   offsets[6:10] = 9
# 출력:   idx=10  tail       offsets[12:12] = 11
# 출력:   idx=10  boundary   offsets[10:12] = 10
# 출력:
# 출력: offsets = [ 0  3  3  5  5  5  9  9  9  9 10 10]


# %% [markdown]
# ## 3. PyTorch 참조 구현 — 벡터화 (`unique_consecutive` + `cumsum`)
#
# `_torch_impl._isect_offset_encode` 는 같은 결과를 이웃 비교 없이 만든다.
#
# 1. `unique_consecutive(isect_ids >> 32, return_counts=True)` — 정렬돼 있으니 연속 중복만 묶으면 타일별 개수 $c_t$
# 2. 그 개수를 `[C, tile_h, tile_w]` 격자에 흩뿌린다 (안 나온 타일은 0으로 남는다 = 빈 타일)
# 3. 포함 누적합에서 자기 개수를 뺀다: $\text{offsets} = \text{cumsum}(c) - c$ (배타적 누적합)
#
# 2단계에서 빈 타일의 칸이 **0으로 남는 것**이 전부다. 0을 더해도 누적합이 안 늘어나므로
# 그 타일의 시작은 앞 타일의 끝과 같아지고, 자연히 길이 0 구간이 된다.

# %%
def isect_offset_encode_torch(isect_ids, I, tile_width, tile_height):
    """gsplat/cuda/_torch_impl.py 의 _isect_offset_encode 와 동일한 알고리즘."""
    tile_n_bits = bits_for_count(tile_width * tile_height)
    tile_counts = torch.zeros((I, tile_height, tile_width), dtype=torch.int64)

    ids_uq, counts = torch.unique_consecutive(isect_ids >> 32, return_counts=True)
    image_ids_uq = ids_uq >> tile_n_bits
    tile_ids_uq = ids_uq & ((1 << tile_n_bits) - 1)
    tile_counts[image_ids_uq, tile_ids_uq // tile_width, tile_ids_uq % tile_width] = counts

    cum = torch.cumsum(tile_counts.flatten(), dim=0).reshape_as(tile_counts)
    return (cum - tile_counts).int()


off_torch = isect_offset_encode_torch(isect_ids, 1, TILE_W, TILE_H)
print("shape:", tuple(off_torch.shape))
print("offsets [tile_h, tile_w]:\n", off_torch[0].numpy())
print("flatten:", off_torch.flatten().numpy())
print("CUDA 방식과 일치:", np.array_equal(off_torch.flatten().numpy(), off_cuda))
# 출력: shape: (1, 3, 4)
# 출력: offsets [tile_h, tile_w]:
# 출력:  [[ 0  3  3  5]
# 출력:  [ 5  5  9  9]
# 출력:  [ 9  9 10 10]]
# 출력: flatten: [ 0  3  3  5  5  5  9  9  9  9 10 10]
# 출력: CUDA 방식과 일치: True


# %% [markdown]
# ## 4. 한 줄짜리 동치 — `searchsorted`
#
# 정렬된 전역 타일 id 배열 $K$ 에서 $t$ 의 **왼쪽 삽입 위치**가 곧 $t$ 의 시작 오프셋이다.
#
# $$\text{offsets}[t] = \texttt{searchsorted}(K,\,t,\,\text{left}) = \bigl|\{ i : K_i < t \}\bigr|$$
#
# 빈 타일 $t$ 에서는 $K$ 에 $t$ 가 없으므로 삽입 위치가 "$t$ 보다 큰 첫 원소의 위치" = 앞 타일의 끝이 된다.
# 이것이 "**빈 타일이 이전 오프셋을 물려받는다**"의 정확한 의미다.

# %%
def isect_offset_encode_searchsorted(isect_ids, I, tile_width, tile_height):
    n_tiles = tile_width * tile_height
    tile_n_bits = bits_for_count(n_tiles)
    ids = isect_ids >> 32
    global_tile = (ids >> tile_n_bits) * n_tiles + (ids & ((1 << tile_n_bits) - 1))
    t = torch.arange(I * n_tiles, dtype=global_tile.dtype)
    return torch.searchsorted(global_tile.contiguous(), t, right=False).reshape(I, tile_height, tile_width).int()


off_ss = isect_offset_encode_searchsorted(isect_ids, 1, TILE_W, TILE_H)
print("searchsorted:", off_ss.flatten().numpy())
print("세 방식 모두 일치:", torch.equal(off_ss, off_torch) and np.array_equal(off_ss.flatten().numpy(), off_cuda))
# 출력: searchsorted: [ 0  3  3  5  5  5  9  9  9  9 10 10]
# 출력: 세 방식 모두 일치: True


# %% [markdown]
# ## 5. 빈 타일이 길이 0이 되는 것을 표로
#
# `offsets` 뒤에 `n_isects` 를 한 칸 덧붙이면 인접 차분이 곧 **타일별 Gaussian 수**다.
#
# $$c_t = \text{offsets}[t{+}1] - \text{offsets}[t], \qquad \text{offsets}[N_{\text{tiles}}] \equiv n_{\text{isects}}$$

# %%
flat = torch.cat([off_torch.flatten(), off_torch.new_tensor([n_isects])])
per_tile = (flat[1:] - flat[:-1]).numpy()

print("tile  (ty,tx)   start   end   count   flatten_ids[start:end]")
for t in range(N_TILES):
    s, e = int(flat[t]), int(flat[t + 1])
    ty, tx = divmod(t, TILE_W)
    mark = "  <- 빈 타일 (start == end)" if s == e else ""
    print(f" {t:3d}   ({ty},{tx})   {s:5d} {e:5d} {e - s:6d}   {flatten_ids[s:e].tolist()}{mark}")
print("\n타일별 Gaussian 수 [tile_h, tile_w]:\n", per_tile.reshape(TILE_H, TILE_W))
print("합계 =", per_tile.sum(), "== n_isects =", n_isects)
# 출력: tile  (ty,tx)   start   end   count   flatten_ids[start:end]
# 출력:   0   (0,0)       0     3      3   [3, 0, 1]
# 출력:   1   (0,1)       3     3      0   []  <- 빈 타일 (start == end)
# 출력:   2   (0,2)       3     5      2   [3, 2]
# 출력:   3   (0,3)       5     5      0   []  <- 빈 타일 (start == end)
# 출력:   4   (1,0)       5     5      0   []  <- 빈 타일 (start == end)
# 출력:   5   (1,1)       5     9      4   [0, 3, 1, 2]
# 출력:   6   (1,2)       9     9      0   []  <- 빈 타일 (start == end)
# 출력:   7   (1,3)       9     9      0   []  <- 빈 타일 (start == end)
# 출력:   8   (2,0)       9     9      0   []  <- 빈 타일 (start == end)
# 출력:   9   (2,1)       9    10      1   [2]
# 출력:  10   (2,2)      10    10      0   []  <- 빈 타일 (start == end)
# 출력:  11   (2,3)      10    11      1   [1]
# 출력:
# 출력: 타일별 Gaussian 수 [tile_h, tile_w]:
# 출력:  [[3 0 2 0]
# 출력:  [0 4 0 0]
# 출력:  [0 1 0 1]]
# 출력: 합계 = 11 == n_isects = 11


# %% [markdown]
# ## 6. 여러 카메라 (C=2) — 오프셋은 이미지 경계를 넘어 **이어서** 증가한다
#
# `offsets` 는 `[C, tile_h, tile_w]` 이지만 누적합은 **평탄화한 전체**에 대해 한 번만 돈다.
# 즉 카메라 1의 오프셋은 0에서 다시 시작하지 않고 카메라 0의 마지막 값에서 이어진다
# (`flatten_ids` 가 이미지별로 나뉘어 있지 않은 하나의 긴 배열이기 때문).
#
# 이 때문에 키의 최상위 필드가 `image_id` 여야 한다 — 정렬 후 이미지끼리도 연속 블록이 된다.

# %%
tile_seq2 = np.array([1, 1, 6, 6, 6, 10, 0, 0, 4, 4, 4, 4, 11])
img_seq2 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
depth2 = np.linspace(0.5, 6.0, len(tile_seq2)).astype(np.float32)

isect_ids2 = torch.from_numpy(pack_keys(img_seq2, tile_seq2, depth2, TILE_N_BITS))
assert bool(torch.all(isect_ids2[1:] >= isect_ids2[:-1])), "키가 정렬돼 있어야 한다"

off2_torch = isect_offset_encode_torch(isect_ids2, 2, TILE_W, TILE_H)
off2_cuda, _ = offsets_cuda_style(isect_ids2, 2, TILE_W, TILE_H)
off2_ss = isect_offset_encode_searchsorted(isect_ids2, 2, TILE_W, TILE_H)

print("shape:", tuple(off2_torch.shape), " n_isects =", isect_ids2.numel())
print("camera 0 offsets:\n", off2_torch[0].numpy())
print("camera 1 offsets:\n", off2_torch[1].numpy())
print("CUDA/searchsorted 와 일치:",
      np.array_equal(off2_torch.flatten().numpy(), off2_cuda) and torch.equal(off2_torch, off2_ss))

flat2 = torch.cat([off2_torch.flatten(), off2_torch.new_tensor([isect_ids2.numel()])])
per_tile2 = (flat2[1:] - flat2[:-1]).numpy().reshape(2, TILE_H, TILE_W)
print("타일별 개수 cam0:\n", per_tile2[0], "\n타일별 개수 cam1:\n", per_tile2[1])
# 출력: shape: (2, 3, 4)  n_isects = 13
# 출력: camera 0 offsets:
# 출력:  [[0 0 2 2]
# 출력:  [2 2 2 5]
# 출력:  [5 5 5 6]]
# 출력: camera 1 offsets:
# 출력:  [[ 6  8  8  8]
# 출력:  [ 8 12 12 12]
# 출력:  [12 12 12 12]]
# 출력: CUDA/searchsorted 와 일치: True
# 출력: 타일별 개수 cam0:
# 출력:  [[0 2 0 0]
# 출력:  [0 0 3 0]
# 출력:  [0 0 1 0]]
# 출력: 타일별 개수 cam1:
# 출력:  [[2 0 0 0]
# 출력:  [4 0 0 0]
# 출력:  [0 0 0 1]]


# %% [markdown]
# ## 7. 엣지 케이스
#
# - `n_isects == 0`: CUDA 런처가 커널을 띄우지 않고 `offsets.fill_(0)` — 모든 타일이 `[0,0)` 로 빈 구간
# - 앞쪽 타일들이 전부 비어 있음: `idx == 0` 스레드가 `offsets[0 .. id_curr]` 를 전부 0으로
# - 뒤쪽 타일들이 전부 비어 있음: `idx == n_isects-1` 스레드가 남은 꼬리를 전부 `n_isects` 로

# %%
empty = torch.zeros(0, dtype=torch.int64)
print("n_isects=0 →", isect_offset_encode_torch(empty, 1, TILE_W, TILE_H).flatten().numpy())

# 첫 타일과 마지막 타일이 비어 있는 경우: 점유 타일 = 4, 7 뿐
edge_ids = torch.from_numpy(pack_keys([0, 0, 0], [4, 4, 7], np.array([1.0, 2.0, 3.0], np.float32), TILE_N_BITS))
e_t = isect_offset_encode_torch(edge_ids, 1, TILE_W, TILE_H).flatten().numpy()
e_c, _ = offsets_cuda_style(edge_ids, 1, TILE_W, TILE_H)
print("앞뒤가 비었을 때 →", e_t, " CUDA 일치:", np.array_equal(e_t, e_c))
print("  앞쪽 0..3 은 모두 0(길이 0), 뒤쪽 8..11 은 모두 3(=n_isects, 길이 0)")
# 출력: n_isects=0 → [0 0 0 0 0 0 0 0 0 0 0 0]
# 출력: 앞뒤가 비었을 때 → [0 0 0 0 0 2 2 2 3 3 3 3]  CUDA 일치: True
# 출력:   앞쪽 0..3 은 모두 0(길이 0), 뒤쪽 8..11 은 모두 3(=n_isects, 길이 0)


# %% [markdown]
# ## 8. 시각화
#
# 왼쪽: 오프셋 계단 그래프. 수평 구간(계단의 평평한 부분)이 **빈 타일**이고, 수직 점프의 높이가 그 타일의 교차 수다.
# `offsets` 는 정의상 **단조 비감소**이며, 마지막에 `n_isects` 를 붙이면 전체가 하나의 배타적 누적합이 된다.
#
# 오른쪽: 차분으로 얻은 타일별 Gaussian 수 히트맵 (`[tile_h, tile_w]` 격자, 0인 칸이 빈 타일).

# %%
tiles_axis = np.arange(N_TILES + 1)
offs_plot = flat.numpy()

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("offsets 계단 (평평한 구간 = 빈 타일)", "타일별 Gaussian 수 = offsets 차분"),
    column_widths=[0.58, 0.42],
    horizontal_spacing=0.12,
)

fig.add_trace(
    go.Scatter(
        x=tiles_axis, y=offs_plot, mode="lines+markers", line_shape="hv",
        name="offsets", line=dict(color="#2563eb", width=2), marker=dict(size=7),
        hovertemplate="tile %{x}<br>offset %{y}<extra></extra>",
    ),
    row=1, col=1,
)
empty_t = np.where(per_tile == 0)[0]
fig.add_trace(
    go.Scatter(
        x=empty_t, y=offs_plot[empty_t], mode="markers", name="빈 타일",
        marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(color="#dc2626", width=2)),
        hovertemplate="tile %{x} 은 비어 있음 (start == end)<extra></extra>",
    ),
    row=1, col=1,
)
fig.add_hline(y=n_isects, line=dict(color="#9ca3af", dash="dot"), row=1, col=1)

fig.add_trace(
    go.Heatmap(
        z=per_tile.reshape(TILE_H, TILE_W), colorscale="Magma",
        text=per_tile.reshape(TILE_H, TILE_W), texttemplate="%{text}",
        textfont=dict(size=15), showscale=False,
        hovertemplate="tile (%{y},%{x})<br>%{z} 개<extra></extra>",
    ),
    row=1, col=2,
)

fig.update_xaxes(title_text="전역 타일 번호 t", dtick=1, row=1, col=1)
fig.update_yaxes(title_text="flatten_ids 인덱스", dtick=1, row=1, col=1)
fig.update_xaxes(title_text="tile x", dtick=1, row=1, col=2)
fig.update_yaxes(title_text="tile y", dtick=1, autorange="reversed", scaleanchor="x2", row=1, col=2)
fig.update_layout(
    title_text=f"isect_offset_encode — {TILE_W}x{TILE_H} 타일, n_isects={n_isects}",
    width=1150, height=520, template="plotly_white",
    margin=dict(t=90, b=70),
    # 범례는 좌측 그래프의 비어 있는 좌상단 안쪽에 둔다 (제목과 겹치지 않게)
    legend=dict(x=0.03, y=0.98, xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.85)", bordercolor="#d1d5db", borderwidth=1),
)

_show(fig)
fig.write_image(f"{HERE}/expy.png", scale=2)
print("saved:", f"{HERE}/expy.png")
# 출력: saved: <hint dir>/expy.png


# %% [markdown]
# ## 정리
#
# | 관점 | 내용 |
# |---|---|
# | 입력 | 정렬된 `isect_ids[n_isects]` (상위 = image\|tile, 하위 32비트 = depth) |
# | 출력 | `offsets[C, tile_h, tile_w]` int32 — 타일별 **시작 인덱스** |
# | 사용법 | 타일 t 의 Gaussian = `flatten_ids[offsets[t] : offsets[t+1]]`, 마지막은 `n_isects` |
# | 알고리즘(CUDA) | 교차당 스레드 1개, 앞 원소와 `id >> 32` 비교 → 다르면 `offsets[id_prev+1 .. id_curr]` 에 `idx` 기록 |
# | 알고리즘(PyTorch) | `unique_consecutive` 로 타일별 개수 → 격자에 흩뿌림 → `cumsum - count` (배타적 누적합) |
# | 빈 타일 | 개수가 0이라 누적합이 안 늘어남 → `start == end` → 길이 0 구간, ⑦ 단계에서 블록이 즉시 종료 |
# | 다중 카메라 | 누적합이 평탄화 전체에 걸쳐 한 번 — 카메라 1의 오프셋은 0이 아니라 카메라 0의 끝에서 이어짐 |
