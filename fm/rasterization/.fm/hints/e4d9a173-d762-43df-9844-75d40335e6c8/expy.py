# %% [markdown]
# # 정렬된 64비트 `isect_ids` 키를 다시 해독하기
#
# 필요 패키지: `torch`, `numpy`, `plotly`, `kaleido` (PNG 저장용). **gsplat은 import하지 않는다.**
#
# gsplat의 타일 교차 단계(`isect_tiles`)는 (Gaussian, 타일) 쌍마다 **64비트 정수 키** 하나를 만든다.
# 상위 → 하위 순서로 세 필드가 이어 붙어 있다:
#
# $$
# \texttt{isect\_id} \;=\; \underbrace{\texttt{image\_id}}_{\text{상위}} \ll (32 + b)
# \;\;|\;\; \underbrace{\texttt{tile\_id}}_{b\ \text{비트}} \ll 32
# \;\;|\;\; \underbrace{\texttt{float32}(\texttt{depth})\ \text{비트}}_{\text{하위 }32\text{비트}}
# $$
#
# 여기서 $b = \texttt{tile\_n\_bits} = (\texttt{tile\_w}\cdot\texttt{tile\_h} - 1).\texttt{bit\_length()}$ 이다.
#
# 이 키를 그냥 **정수로 오름차순 정렬**하면 사전식으로
# $(\texttt{image\_id},\ \texttt{tile\_id},\ \texttt{depth})$ 순서가 된다.
# 양수 float32의 IEEE-754 비트 패턴은 부호 없는 정수 비교와 순서가 같기 때문에
# 깊이도 자동으로 "가까운 것부터"가 된다.
#
# 정렬이 끝난 뒤 **다시 세 필드를 꺼내는 법**이 이 카드의 주제다:
#
# ```python
# tile_n_bits = (tile_w * tile_h - 1).bit_length()
# depth_key = (isect_ids & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
# tile_id   = (isect_ids >> 32) & ((1 << tile_n_bits) - 1)
# image_id  =  isect_ids >> (32 + tile_n_bits)
# ```
#
# 아래에서 C=2 카메라, 4×3 타일짜리 토이 씬으로 인코드 → 정렬 → 디코드를 직접 돌려본다.

# %%
import numpy as np
import torch


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


torch.manual_seed(0)
rng = np.random.default_rng(0)

C = 2  # 카메라(이미지) 수
TILE_W, TILE_H = 4, 3  # 타일 격자 (16px 타일이면 64×48 픽셀 이미지)
N_TILES = TILE_W * TILE_H  # 12개 타일

tile_n_bits = (TILE_W * TILE_H - 1).bit_length()
image_n_bits = (C - 1).bit_length()
print(f"tile_n_bits  = {tile_n_bits}   (타일 {N_TILES}개를 0..{N_TILES - 1}로 세려면)")
print(f"image_n_bits = {image_n_bits}   (이미지 {C}개)")
print(f"총 사용 비트 = {image_n_bits} + {tile_n_bits} + 32 = {image_n_bits + tile_n_bits + 32} <= 64  OK")
# 출력:
# tile_n_bits  = 4   (타일 12개를 0..11로 세려면)
# image_n_bits = 1   (이미지 2개)
# 총 사용 비트 = 1 + 4 + 32 = 37 <= 64  OK

# %% [markdown]
# ## 1. 인코딩: (image_id, tile_id, depth) → 64비트 키
#
# `_torch_impl._isect_tiles`가 하는 일과 같다. depth는 float32 비트 패턴을 **하위 32비트에 zero-extend**해
# 넣는다(부호 확장이 되면 상위 필드를 오염시키므로 `& 0xFFFFFFFF`가 필수).

# %%
def encode(image_ids, tile_ids, depths, tile_n_bits):
    """(image_id, tile_id, depth) 삼중 → int64 키 [image | tile | float32(depth)]"""
    image_ids = torch.as_tensor(image_ids, dtype=torch.int64)
    tile_ids = torch.as_tensor(tile_ids, dtype=torch.int64)
    depths = torch.as_tensor(depths, dtype=torch.float32)
    # float32 비트 → int32 → int64로 zero-extend (부호 확장 방지)
    depth_bits = depths.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return (image_ids << (32 + tile_n_bits)) | (tile_ids << 32) | depth_bits


# 토이 씬: (Gaussian, 타일) 교차 쌍 14개를 무작위로 만든다
n_isects = 14
src_image = rng.integers(0, C, n_isects)
src_tile = rng.integers(0, N_TILES, n_isects)
src_depth = np.round(rng.uniform(0.5, 9.5, n_isects), 2).astype(np.float32)

isect_ids_unsorted = encode(src_image, src_tile, src_depth, tile_n_bits)
print("생성 순서(정렬 전):")
for i in range(n_isects):
    print(f"  image={src_image[i]}  tile={src_tile[i]:2d}  depth={src_depth[i]:.2f}  key={isect_ids_unsorted[i].item()}")
# 출력:
# 생성 순서(정렬 전):
#   image=1  tile=11  depth=7.07  key=117052685681
#   image=1  tile= 8  depth=2.08  key=104153292472
#   image=1  tile= 7  depth=8.27  key=99875049964
#   image=0  tile= 6  depth=5.37  key=26854807306
#   image=0  tile= 6  depth=3.20  key=26848578765
#   image=0  tile=11  depth=4.30  key=48327399834
#   image=0  tile= 3  depth=0.75  key=13946060800
#   image=0  tile= 9  depth=1.62  key=39725259817
#   image=0  tile= 8  depth=6.54  key=35447195566
#   image=1  tile= 0  depth=6.32  key=69806472561
#   image=1  tile= 4  depth=6.04  key=86985754542
#   image=1  tile=10  depth=3.95  key=112751070413
#   image=1  tile= 6  depth=9.47  key=95581340959
#   image=1  tile= 0  depth=9.33  key=69811390382

# %% [markdown]
# ## 2. 비트 문자열로 필드 경계 눈으로 확인
#
# 64비트를 `[상위 27비트 미사용][image 1비트][tile 4비트][depth 32비트]`로 잘라서 찍어 본다.

# %%
def bitstr(key, tile_n_bits, image_n_bits):
    b = format(key & ((1 << 64) - 1), "064b")
    lo, mid, hi = b[64 - 32 :], b[64 - 32 - tile_n_bits : 64 - 32], b[: 64 - 32 - tile_n_bits]
    pad = hi[:-image_n_bits] if image_n_bits else hi
    img = hi[-image_n_bits:] if image_n_bits else ""
    return f"{pad}|{img}|{mid}|{lo}"


print(f"{'':>21}{'unused':<27}|img|tile|{'float32(depth)':^32}")
for i in range(5):
    k = isect_ids_unsorted[i].item()
    print(f"img={src_image[i]} tile={src_tile[i]:2d} d={src_depth[i]:.2f}  {bitstr(k, tile_n_bits, image_n_bits)}")
# 출력:
#                      unused                     |img|tile|         float32(depth)
# img=1 tile=11 d=7.07  000000000000000000000000000|1|1011|01000000111000100011110101110001
# img=1 tile= 8 d=2.08  000000000000000000000000000|1|1000|01000000000001010001111010111000
# img=1 tile= 7 d=8.27  000000000000000000000000000|1|0111|01000001000001000101000111101100
# img=0 tile= 6 d=5.37  000000000000000000000000000|0|0110|01000000101010111101011100001010
# img=0 tile= 6 d=3.20  000000000000000000000000000|0|0110|01000000010011001100110011001101

# %% [markdown]
# ## 3. 정렬 → 답변의 세 줄로 디코딩
#
# 키를 단순 정수 정렬하면 $(\texttt{image},\ \texttt{tile},\ \texttt{depth})$ 사전식 정렬이 된다.
# CUDA에서는 이 자리에 CUB radix sort가 들어간다.

# %%
isect_ids, order = torch.sort(isect_ids_unsorted)  # ← radix sort 자리

# ▼▼ 답변의 핵심 3줄 ▼▼
tile_n_bits = (TILE_W * TILE_H - 1).bit_length()
depth_key = (isect_ids & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
tile_id = (isect_ids >> 32) & ((1 << tile_n_bits) - 1)
image_id = isect_ids >> (32 + tile_n_bits)
# ▲▲▲▲

print(" idx  image  tile(y,x)   depth")
for i in range(n_isects):
    ty, tx = divmod(tile_id[i].item(), TILE_W)
    print(f"  {i:2d}     {image_id[i].item()}    {tile_id[i].item():2d} ({ty},{tx})   {depth_key[i].item():5.2f}")
# 출력:
#  idx  image  tile(y,x)   depth
#   0     0     3 (0,3)    0.75
#   1     0     6 (1,2)    3.20
#   2     0     6 (1,2)    5.37
#   3     0     8 (2,0)    6.54
#   4     0     9 (2,1)    1.62
#   5     0    11 (2,3)    4.30
#   6     1     0 (0,0)    6.32
#   7     1     0 (0,0)    9.33
#   8     1     4 (1,0)    6.04
#   9     1     6 (1,2)    9.47
#  10     1     7 (1,3)    8.27
#  11     1     8 (2,0)    2.08
#  12     1    10 (2,2)    3.95
#  13     1    11 (2,3)    7.07

# %%
# 원본과 일치하는지 검증 (order로 원본 순서를 되돌려 비교)
assert torch.equal(image_id, torch.as_tensor(src_image, dtype=torch.int64)[order])
assert torch.equal(tile_id, torch.as_tensor(src_tile, dtype=torch.int64)[order])
assert torch.equal(depth_key, torch.as_tensor(src_depth, dtype=torch.float32)[order])
# 정렬이 실제로 (image, tile, depth) 사전식인지도 확인
keys = list(zip(image_id.tolist(), tile_id.tolist(), depth_key.tolist()))
assert keys == sorted(keys)
print("round-trip OK — 세 필드 모두 원본과 정확히 일치, 정렬 순서도 (image, tile, depth) 사전식")
# 출력: round-trip OK — 세 필드 모두 원본과 정확히 일치, 정렬 순서도 (image, tile, depth) 사전식

# %% [markdown]
# ### depth는 왜 `.view(float32)`인가?
#
# 하위 32비트는 depth의 **IEEE-754 비트 패턴 그 자체**다. 값을 정수로 캐스팅한 게 아니므로
# `.to(torch.float32)`(값 변환)가 아니라 `.view(torch.float32)`(비트 재해석)를 써야 한다.
# 중간에 `.to(torch.int32)`를 거치는 이유는 int64 텐서를 바로 float32로 `view`할 수 없어서다
# (원소 크기가 다르면 마지막 축이 갈라진다).

# %%
sample = isect_ids[:3]
print("틀린 방법 .to(float32) :", (sample & 0xFFFFFFFF).to(torch.float32).tolist())
print("맞는 방법 .view(float32):", (sample & 0xFFFFFFFF).to(torch.int32).view(torch.float32).tolist())
# 출력:
# 틀린 방법 .to(float32) : [1061158912.0, 1078775040.0, 1085003520.0]
# 맞는 방법 .view(float32): [0.75, 3.200000047683716, 5.369999885559082]

# %% [markdown]
# ## 4. `tile_n_bits`는 타일 수에 따라 어떻게 변하나
#
# $b = \lceil \log_2 T \rceil$ 을 정수 연산으로 쓴 것이 `(T - 1).bit_length()` 다.
# **`-1`이 중요하다**: $T$가 2의 거듭제곱일 때 `T.bit_length()`는 한 비트를 더 센다
# (예: $T=256$ → 9비트로 과대계산, 실제로는 8비트면 $0..255$를 담는다).

# %%
grids = [(1, 1), (2, 2), (4, 3), (8, 4), (16, 16), (32, 32), (120, 68), (240, 136)]
print(f"{'tile_w×tile_h':>14} {'T=타일수':>9} {'(T-1).bit_length()':>19} {'T.bit_length()':>15} {'image 여유비트':>13}")
for w, h in grids:
    T = w * h
    b = (T - 1).bit_length() if T > 1 else 0
    print(f"{f'{w}×{h}':>14} {T:>9} {b:>19} {T.bit_length():>15} {64 - 32 - b:>13}")
# 출력:
#  tile_w×tile_h    T=타일수  (T-1).bit_length()  T.bit_length()   image 여유비트
#            1×1         1                   0               1            32
#            2×2         4                   2               3            30
#            4×3        12                   4               4            28
#            8×4        32                   5               6            27
#          16×16       256                   8               9            24
#          32×32      1024                  10              11            22
#         120×68      8160                  13              13            19
#        240×136     32640                  15              15            17

# %% [markdown]
# 1920×1080 이미지를 16px 타일로 나누면 $120 \times 68 = 8160$ 타일 → $b = 13$비트.
# 남는 상위 $64 - 32 - 13 = 19$비트에 image_id가 들어가므로 배치 이미지를 50만 장까지 담을 수 있다.

# %% [markdown]
# ## 5. 마스크 `(1 << tile_n_bits) - 1`은 왜 필요한가
#
# `isect_ids >> 32`를 하면 depth 32비트는 날아가지만 **그 위에 얹힌 image_id 비트가 남는다**.
# 마스크로 하위 $b$비트만 남겨야 순수한 tile_id가 된다.
#
# $$\texttt{tile\_id} = (\texttt{isect\_id} \gg 32)\ \&\ (2^{b}-1)$$

# %%
raw = isect_ids >> 32  # 마스크 없음: [image | tile]이 붙어 있다
print("마스크 없이 >>32 :", raw.tolist())
print("마스크 적용      :", tile_id.tolist())
print("원본 tile        :", torch.as_tensor(src_tile)[order].tolist())
print("차이 = image_id * 2**b :", (raw - tile_id).tolist())
# 출력:
# 마스크 없이 >>32 : [3, 6, 6, 8, 9, 11, 16, 16, 20, 22, 23, 24, 26, 27]
# 마스크 적용      : [3, 6, 6, 8, 9, 11, 0, 0, 4, 6, 7, 8, 10, 11]
# 원본 tile        : [3, 6, 6, 8, 9, 11, 0, 0, 4, 6, 7, 8, 10, 11]
# 차이 = image_id * 2**b : [0, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16, 16]

# %% [markdown]
# image=0인 앞쪽 6개는 마스크가 없어도 우연히 맞지만, image=1인 뒤쪽 8개는 전부 $+2^4 = +16$만큼
# 어긋난다. 즉 **카메라가 하나뿐이면 버그가 드러나지 않는다** — C≥2에서만 터지는 종류의 실수다.

# %% [markdown]
# ## 6. 잘못된 `tile_n_bits`를 쓰면 어떻게 깨지나

# %%
def decode(isect_ids, b):
    d = (isect_ids & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    t = (isect_ids >> 32) & ((1 << b) - 1)
    im = isect_ids >> (32 + b)
    return im, t, d


for b_wrong, why in [(4, "정답"), (5, "과대(T.bit_length() 실수)"), (3, "과소"), (0, "타일 필드 없음")]:
    im, t, _ = decode(isect_ids, b_wrong)
    ok_i = torch.equal(im, image_id)
    ok_t = torch.equal(t, tile_id)
    print(f"b={b_wrong} ({why:<22}) image_id 맞음={str(ok_i):<5} tile_id 맞음={str(ok_t):<5}")
    print(f"          image={im.tolist()}")
    print(f"          tile ={t.tolist()}")
# 출력:
# b=4 (정답                    ) image_id 맞음=True  tile_id 맞음=True
#           image=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
#           tile =[3, 6, 6, 8, 9, 11, 0, 0, 4, 6, 7, 8, 10, 11]
# b=5 (과대(T.bit_length() 실수) ) image_id 맞음=False tile_id 맞음=False
#           image=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#           tile =[3, 6, 6, 8, 9, 11, 16, 16, 20, 22, 23, 24, 26, 27]
# b=3 (과소                    ) image_id 맞음=False tile_id 맞음=False
#           image=[0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
#           tile =[3, 6, 6, 0, 1, 3, 0, 0, 4, 6, 7, 0, 2, 3]
# b=0 (타일 필드 없음              ) image_id 맞음=False tile_id 맞음=False
#           image=[3, 6, 6, 8, 9, 11, 16, 16, 20, 22, 23, 24, 26, 27]
#           tile =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# %% [markdown]
# - **b가 크면**(5): tile 필드가 image_id 비트를 빨아들여 image_id가 전부 0이 되고,
#   tile_id는 image=1 쪽에서 $+16$만큼 부풀어 존재하지 않는 타일 16, 20, 22, ...를 가리킨다.
#   → `isect_offset_encode`가 범위 밖 타일에 쓰려다 out-of-bounds.
# - **b가 작으면**(3): tile_id의 최상위 비트가 image_id로 새어 나가, 타일 8~11이 잘려
#   서로 다른 타일이 같은 id로 뭉개진다(8→0, 9→1, 11→3). 렌더링은 조용히 틀린 그림을 낸다.
# - depth 필드(하위 32비트)는 b와 무관하므로 어떤 경우에도 멀쩡하다 — 그래서 디버깅이 더 어렵다.

# %% [markdown]
# ## 7. 정렬된 키를 (image, tile) 축으로 시각화
#
# 정렬 결과가 (image → tile → depth) 계단 구조라는 것을 그림으로 확인한다.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

im_l, ti_l, dp_l = image_id.tolist(), tile_id.tolist(), depth_key.tolist()
idx = list(range(n_isects))
palette = ["#3b6fd4", "#d4693b"]

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "정렬 순서 vs 전역 타일 (image·N_TILES + tile) — 단조 증가 계단",
        "정렬 순서 vs depth — 타일이 바뀔 때만 리셋되는 톱니",
        "타일별 교차 개수 (isect_offsets가 세는 값)",
        "타일 격자 크기에 따른 tile_n_bits = (T-1).bit_length()",
    ),
    vertical_spacing=0.16,
    horizontal_spacing=0.10,
)

# (1,1) 전역 타일 계단
for c in range(C):
    m = [i for i in idx if im_l[i] == c]
    fig.add_trace(
        go.Scatter(
            x=m,
            y=[im_l[i] * N_TILES + ti_l[i] for i in m],
            mode="markers+lines",
            name=f"image {c}",
            legendgroup=f"img{c}",
            marker=dict(size=11, color=palette[c], line=dict(width=1, color="white")),
            line=dict(color=palette[c], width=1, dash="dot"),
            text=[f"tile {ti_l[i]}, d={dp_l[i]:.2f}" for i in m],
            hovertemplate="idx %{x}<br>global tile %{y}<br>%{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )
fig.add_hline(y=N_TILES - 0.5, line=dict(color="#888", dash="dash"), row=1, col=1)

# (1,2) depth 톱니
for c in range(C):
    m = [i for i in idx if im_l[i] == c]
    fig.add_trace(
        go.Scatter(
            x=m,
            y=[dp_l[i] for i in m],
            mode="markers+lines",
            name=f"image {c}",
            legendgroup=f"img{c}",
            showlegend=False,
            marker=dict(size=11, color=palette[c], line=dict(width=1, color="white")),
            line=dict(color=palette[c], width=1, dash="dot"),
            text=[f"tile {ti_l[i]}" for i in m],
            hovertemplate="idx %{x}<br>depth %{y:.2f}<br>%{text}<extra></extra>",
        ),
        row=1,
        col=2,
    )
# 타일 경계에 세로선
for i in range(1, n_isects):
    if (im_l[i], ti_l[i]) != (im_l[i - 1], ti_l[i - 1]):
        fig.add_vline(x=i - 0.5, line=dict(color="#bbb", width=1), row=1, col=2)

# (2,1) 타일별 개수
labels = [f"i{c}/t{t}" for c in range(C) for t in range(N_TILES)]
counts = [sum(1 for i in idx if im_l[i] == c and ti_l[i] == t) for c in range(C) for t in range(N_TILES)]
fig.add_trace(
    go.Bar(
        x=labels,
        y=counts,
        marker_color=[palette[0]] * N_TILES + [palette[1]] * N_TILES,
        showlegend=False,
        hovertemplate="%{x}<br>%{y} isects<extra></extra>",
    ),
    row=2,
    col=1,
)

# (2,2) tile_n_bits 표
gl = [f"{w}×{h}" for w, h in grids]
gb = [((w * h - 1).bit_length() if w * h > 1 else 0) for w, h in grids]
fig.add_trace(
    go.Bar(
        x=gl,
        y=gb,
        marker_color=["#6aa84f" if g != "4×3" else "#d4693b" for g in gl],
        text=gb,
        textposition="outside",
        showlegend=False,
        hovertemplate="%{x}: %{y} bits<extra></extra>",
    ),
    row=2,
    col=2,
)

fig.update_xaxes(title_text="정렬된 인덱스", row=1, col=1, dtick=1)
fig.update_yaxes(title_text="image·12 + tile", row=1, col=1)
fig.update_xaxes(title_text="정렬된 인덱스", row=1, col=2, dtick=1)
fig.update_yaxes(title_text="depth (가까울수록 작음)", row=1, col=2)
fig.update_xaxes(title_text="(image, tile)", row=2, col=1, tickangle=-60)
fig.update_yaxes(title_text="교차 개수", row=2, col=1, dtick=1)
fig.update_xaxes(title_text="tile_w × tile_h", row=2, col=2)
fig.update_yaxes(title_text="tile_n_bits", row=2, col=2, range=[0, 17])
fig.update_layout(
    title_text="64비트 isect_ids 키: 정렬 결과와 필드 폭 (C=2, 4×3 타일)",
    height=820,
    width=1180,
    template="plotly_white",
    margin=dict(t=100),
    legend=dict(
        x=0.01,
        y=0.93,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="#ccc",
        borderwidth=1,
    ),
)

_show(fig)

import pathlib

out = pathlib.Path(__file__).with_name("expy.png") if "__file__" in globals() else pathlib.Path("expy.png")
fig.write_image(str(out), scale=2)
print("saved:", out)
# 출력: saved: .../expy.png

# %% [markdown]
# ## 정리
#
# | 필드 | 위치 | 꺼내는 식 |
# |---|---|---|
# | `depth` | 하위 32비트 | `(isect_ids & 0xFFFFFFFF).to(torch.int32).view(torch.float32)` |
# | `tile_id` | 32번째 비트부터 $b$비트 | `(isect_ids >> 32) & ((1 << tile_n_bits) - 1)` |
# | `image_id` | 그 위 전부 | `isect_ids >> (32 + tile_n_bits)` |
#
# - $b = \texttt{tile\_n\_bits} = (\texttt{tile\_w} \cdot \texttt{tile\_h} - 1).\texttt{bit\_length()}$ — `-1`을 빼먹으면 2의 거듭제곱 타일 수에서 한 비트 과대계산.
# - depth는 값 캐스팅(`.to`)이 아니라 **비트 재해석(`.view`)** 이다.
# - tile 마스크를 빼먹으면 image_id가 tile_id에 섞여 들어간다 (C=1에서는 증상이 없다).
# - 이 세 줄은 `isect_offset_encode`가 타일 경계를 찾는 근거이자, `flatten_ids`와 짝을 맞춰
#   "타일 t를 그릴 때 어떤 Gaussian을 앞→뒤로 훑을지"를 결정한다.
