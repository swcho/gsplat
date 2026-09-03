# %% [markdown]
# # `meta["isect_offsets"]` — 타일별 시작 오프셋 (CSR 인덱스)
#
# 질문: **`meta["isect_offsets"]`의 모양과 역할은?**
#
# 답: **`[C, tile_h, tile_w]`** 모양의 int32 텐서로, 각 타일이 `flatten_ids`에서
# **시작하는 위치**를 담는다. 타일 `t`의 Gaussian 목록은
# `flatten_ids[offsets[t] : offsets[t+1]]` 이다.
#
# 이건 희소 행렬의 **CSR(Compressed Sparse Row) 포맷**과 정확히 같은 구조다.
#
# | CSR | gsplat |
# |---|---|
# | `indptr` (행 시작 위치) | `isect_offsets` — 단, 마지막 `n_isects` 항목이 **없다** |
# | `indices` (열 인덱스) | `flatten_ids` — (이미지, Gaussian) 평탄화 인덱스 |
# | 행 | 타일 |
#
# 수식으로 쓰면, 타일을 행 우선(row-major)으로 평탄화한 선형 id를
# $t = c \cdot (H_t W_t) + y \cdot W_t + x$ 라 할 때
#
# $$\texttt{offsets}[t] \;=\; \sum_{t' < t} \texttt{count}[t'] \qquad (\text{배타적 prefix sum})$$
#
# $$\texttt{count}[t] \;=\; \texttt{offsets}[t{+}1] - \texttt{offsets}[t],\qquad
#   \texttt{offsets}[T] \equiv n_{\text{isects}}$$
#
# 핵심 성질 두 가지:
# 1. **마지막 원소가 없다.** 길이가 `T+1`이 아니라 `T`라서, 마지막 타일의 끝을 알려면
#    `n_isects = flatten_ids.numel()`을 직접 이어 붙여야 한다.
# 2. **빈 타일은 앞 타일의 끝 오프셋을 그대로 물려받는다** → `offsets[t] == offsets[t+1]`,
#    즉 슬라이스 길이가 0이 된다. (별도의 "빈 타일" 표식이 필요 없다.)
#
# 아래에서 4개 Gaussian · 64×48 이미지 · 16px 타일(= 4×3 타일) 토이 씬으로 직접 만들어 본다.
# gsplat은 **import하지 않고**(CUDA JIT 빌드가 오래 걸림) numpy로 동일 로직을 재현한다.

# %%
# 필요 패키지: numpy, plotly, kaleido  (torch/gsplat 불필요 — gsplat은 import 금지)
import math

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


np.set_printoptions(linewidth=120)

# %% [markdown]
# ## 1. 토이 씬 → 화면 위의 2D Gaussian
#
# walkthrough와 같은 장면: 카메라는 원점에서 +z를 보고(`viewmat = I`), 이미지는 64×48,
# 타일은 16px이라 **4×3 = 12개 타일**. 4번째 Gaussian은 카메라 뒤(z<0)라 컬링된다.
#
# 투영은 원 walkthrough의 `fully_fused_projection`을 numpy로 간략 재현한 것이다:
# $\Sigma_{3D} = R S S^\top R^\top$, 아핀 야코비안 $J$로 $\Sigma_{2D} = J\Sigma_{3D}J^\top$,
# 반지름은 축별 $3\sigma$.

# %%
W, H, TILE = 64, 48, 16
tile_w, tile_h = math.ceil(W / TILE), math.ceil(H / TILE)  # 4, 3
C = 1  # 카메라(이미지) 수
fx = fy = 60.0
cx, cy = 32.0, 24.0

means3d = np.array([[0.0, 0.0, 3.0], [0.6, 0.3, 4.0], [-0.5, -0.2, 2.5], [0.0, 0.0, -2.0]])
scales = np.array([[0.30, 0.12, 0.10], [0.25, 0.25, 0.25], [0.08, 0.30, 0.10], [0.2, 0.2, 0.2]])
deg = np.array([30.0, 0.0, -20.0, 0.0])  # z축 회전
N = means3d.shape[0]


def rot_z(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


means2d = np.zeros((C, N, 2))
radii = np.zeros((C, N, 2), dtype=np.int32)  # [rx, ry] — 0이면 컬링
depths = np.zeros((C, N), dtype=np.float32)

for i in range(N):
    x, y, z = means3d[i]
    depths[0, i] = z
    if z <= 0.2:  # near-plane 컬링 → radii = 0 유지
        continue
    means2d[0, i] = [fx * x / z + cx, fy * y / z + cy]
    M = rot_z(deg[i]) @ np.diag(scales[i])
    cov3d = M @ M.T
    J = np.array([[fx / z, 0.0, -fx * x / z**2], [0.0, fy / z, -fy * y / z**2]])
    cov2d = J @ cov3d @ J.T  # (원 구현의 안티에일리어싱 blur는 생략)
    radii[0, i] = np.ceil(3.0 * np.sqrt(np.diag(cov2d))).astype(np.int32)

print("means2d:\n", means2d[0])
print("radii  :\n", radii[0])
print("depths :", depths[0])
# 출력:
# means2d:
#  [[32.  24. ]
#   [41.  28.5]
#   [20.  19.2]
#   [ 0.   0. ]]
# radii  :
#  [[16 11]
#   [12 12]
#   [10 21]
#   [ 0  0]]
# depths : [ 3.  4.  2.5 -2. ]

# %% [markdown]
# ## 2. (Gaussian, 타일) 쌍 만들기 — `isect_tiles`
#
# 각 Gaussian의 화면 AABB `[mean ± radii]`를 타일 격자에 올려 겹치는 타일을 모두 열거한다.
# 쌍마다 64비트 키를 붙인다:
#
# ```
# [ image_id | tile_id | float32(depth)의 비트 ]     상위 → 하위
# ```
#
# 양수 float32는 비트 패턴 그대로 정수로 비교해도 순서가 보존되므로, 이 키 하나만
# 정렬하면 **같은 타일끼리 연속으로 모이고, 그 안에서는 가까운 것부터** 줄을 선다.

# %%
tile_n_bits = max((tile_w * tile_h - 1).bit_length(), 1)  # 12개 타일 → 4비트

pairs = []  # (image_id, tile_id, depth, gauss_id)
tiles_per_gauss = np.zeros((C, N), dtype=np.int32)
for c in range(C):
    for i in range(N):
        rx, ry = radii[c, i]
        if rx <= 0 or ry <= 0:
            continue  # 컬링된 Gaussian은 아예 쌍을 만들지 않는다
        mx, my = means2d[c, i]
        x0 = int(np.clip(math.floor((mx - rx) / TILE), 0, tile_w))
        x1 = int(np.clip(math.ceil((mx + rx) / TILE), 0, tile_w))
        y0 = int(np.clip(math.floor((my - ry) / TILE), 0, tile_h))
        y1 = int(np.clip(math.ceil((my + ry) / TILE), 0, tile_h))
        tiles_per_gauss[c, i] = (x1 - x0) * (y1 - y0)
        for ty in range(y0, y1):
            for tx in range(x0, x1):
                pairs.append((c, ty * tile_w + tx, float(depths[c, i]), c * N + i))

n_isects = len(pairs)
print("tiles_per_gauss:", tiles_per_gauss[0].tolist())
print("n_isects =", n_isects)
# 출력:
# tiles_per_gauss: [6, 6, 6, 0]
# n_isects = 18

# %%
# 64비트 키 조립 → 정렬 → isect_ids / flatten_ids
def depth_bits(d: float) -> int:
    return int(np.float32(d).view(np.uint32))  # 양수 float은 비트 비교로도 순서 보존


keys = np.array(
    [(c << (tile_n_bits + 32)) | (t << 32) | depth_bits(d) for c, t, d, _ in pairs],
    dtype=np.uint64,
)
gids = np.array([g for *_, g in pairs], dtype=np.int32)

order = np.argsort(keys, kind="stable")
isect_ids = keys[order]  # [n_isects] 정렬된 키
flatten_ids = gids[order]  # [n_isects] 대응 Gaussian 인덱스

# 정렬된 키 해독해서 눈으로 확인
dec_tile = ((isect_ids >> np.uint64(32)) & np.uint64((1 << tile_n_bits) - 1)).astype(int)
dec_depth = (isect_ids & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.float32)
print(" idx  tile(y,x)  depth  gaussian")
for k in range(n_isects):
    ty, tx = divmod(dec_tile[k], tile_w)
    print(f" {k:3d}    ({ty},{tx})    {dec_depth[k]:5.2f}   g{flatten_ids[k] % N}")
# 출력:
#  idx  tile(y,x)  depth  gaussian
#    0    (0,0)     2.50   g2
#    1    (0,1)     2.50   g2
#    2    (0,1)     3.00   g0
#    3    (0,2)     3.00   g0
#    4    (1,0)     2.50   g2
#    5    (1,1)     2.50   g2
#    6    (1,1)     3.00   g0
#    7    (1,1)     4.00   g1
#    8    (1,2)     3.00   g0
#    9    (1,2)     4.00   g1
#   10    (1,3)     4.00   g1
#   11    (2,0)     2.50   g2
#   12    (2,1)     2.50   g2
#   13    (2,1)     3.00   g0
#   14    (2,1)     4.00   g1
#   15    (2,2)     3.00   g0
#   16    (2,2)     4.00   g1
#   17    (2,3)     4.00   g1
# → 같은 타일이 연속으로 모이고(예: idx 5~7 이 모두 타일 (1,1)),
#   그 안에서는 depth 오름차순(가까운 것부터)이다. 타일 (0,3)은 아무도 덮지 않아 등장하지 않는다.

# %% [markdown]
# ## 3. `isect_offset_encode` — 타일별 시작 오프셋
#
# 정렬된 키에서 **타일이 바뀌는 지점**만 찾으면 된다. 구현은 두 줄이다:
#
# 1. 타일별 개수 `count[t]`를 센다 (연속된 같은 키를 세는 `unique_consecutive`).
# 2. 배타적 prefix sum: `offsets = cumsum(count) - count`.
#
# 결과를 `[C, tile_h, tile_w]`로 reshape한 것이 `meta["isect_offsets"]`.

# %%
tile_counts = np.zeros((C, tile_h, tile_w), dtype=np.int64)
hi = (isect_ids >> np.uint64(32)).astype(np.int64)  # image_id | tile_id
uq, cnt = np.unique(hi, return_counts=True)  # 정렬돼 있으니 unique_consecutive와 동일
img_uq = uq >> tile_n_bits
t_uq = uq & ((1 << tile_n_bits) - 1)
tile_counts[img_uq, t_uq // tile_w, t_uq % tile_w] = cnt

cum = np.cumsum(tile_counts.reshape(-1)).reshape(tile_counts.shape)
isect_offsets = (cum - tile_counts).astype(np.int32)  # ★ [C, tile_h, tile_w]

print("isect_offsets.shape =", isect_offsets.shape, " (C, tile_h, tile_w)")
print("isect_offsets[0]:\n", isect_offsets[0])
print("타일별 개수 count[0]:\n", tile_counts[0])
# 출력:
# isect_offsets.shape = (1, 3, 4)  (C, tile_h, tile_w)
# isect_offsets[0]:
#  [[ 0  1  3  4]
#   [ 4  5  8 10]
#   [11 12 15 17]]
# 타일별 개수 count[0]:
#  [[1 2 1 0]
#   [1 3 2 1]
#   [1 3 2 1]]
# → 타일 (0,3)의 오프셋 4는 그 앞 타일 (0,2)의 끝값 4를 그대로 물려받은 것 (count 0)

# %% [markdown]
# ## 4. 슬라이싱으로 각 타일의 Gaussian 목록 복원
#
# 커널이 실제로 하는 일이 이것이다. CUDA 래스터라이저에서 **블록 = 타일 하나**이고,
# 그 블록은 `flatten_ids[offsets[t] : offsets[t+1]]` 구간만 앞에서부터(가까운 것부터)
# 훑으며 알파 블렌딩한다.
#
# `offsets`에는 마지막 끝값이 없으므로 **`n_isects`를 직접 이어 붙여** 길이 `T+1`로 만든다.

# %%
flat = np.concatenate([isect_offsets.reshape(-1), [n_isects]])  # 길이 T+1
print("flat (T+1):", flat.tolist())

for c in range(C):
    for ty in range(tile_h):
        for tx in range(tile_w):
            t = c * tile_h * tile_w + ty * tile_w + tx
            lo, hi_ = flat[t], flat[t + 1]
            gl = [f"g{g % N}" for g in flatten_ids[lo:hi_]]
            mark = "  ← 빈 타일 (lo == hi)" if lo == hi_ else ""
            print(f"  tile(c={c}, y={ty}, x={tx}) t={t:2d}  [{lo:2d}:{hi_:2d}] -> {gl}{mark}")
# 출력:
# flat (T+1): [0, 1, 3, 4, 4, 5, 8, 10, 11, 12, 15, 17, 18]
#   tile(c=0, y=0, x=0) t= 0  [ 0: 1] -> ['g2']
#   tile(c=0, y=0, x=1) t= 1  [ 1: 3] -> ['g2', 'g0']
#   tile(c=0, y=0, x=2) t= 2  [ 3: 4] -> ['g0']
#   tile(c=0, y=0, x=3) t= 3  [ 4: 4] -> []  ← 빈 타일 (lo == hi)
#   tile(c=0, y=1, x=0) t= 4  [ 4: 5] -> ['g2']
#   tile(c=0, y=1, x=1) t= 5  [ 5: 8] -> ['g2', 'g0', 'g1']
#   tile(c=0, y=1, x=2) t= 6  [ 8:10] -> ['g0', 'g1']
#   tile(c=0, y=1, x=3) t= 7  [10:11] -> ['g1']
#   tile(c=0, y=2, x=0) t= 8  [11:12] -> ['g2']
#   tile(c=0, y=2, x=1) t= 9  [12:15] -> ['g2', 'g0', 'g1']
#   tile(c=0, y=2, x=2) t=10  [15:17] -> ['g0', 'g1']
#   tile(c=0, y=2, x=3) t=11  [17:18] -> ['g1']

# %% [markdown]
# ### 불변식 검증
#
# - `offsets[t+1] - offsets[t] == count[t]`
# - 모든 타일의 슬라이스를 이어 붙이면 `flatten_ids` 전체가 정확히 한 번씩 복원된다
# - 빈 타일은 `offsets[t] == offsets[t+1]` (앞 오프셋을 그대로 물려받음)

# %%
diff = (flat[1:] - flat[:-1]).reshape(C, tile_h, tile_w)
assert np.array_equal(diff, tile_counts), "차분 == count 여야 한다"

recon = np.concatenate([flatten_ids[flat[t] : flat[t + 1]] for t in range(flat.size - 1)])
assert np.array_equal(recon, flatten_ids), "슬라이스를 이어 붙이면 flatten_ids 전체"

empty = [t for t in range(flat.size - 1) if flat[t] == flat[t + 1]]
print("차분 == count      :", np.array_equal(diff, tile_counts))
print("슬라이스 재조립 일치:", np.array_equal(recon, flatten_ids))
print("빈 타일 t 목록      :", empty, "→ 모두 길이 0")
print("offsets 은 단조 비감소:", bool(np.all(np.diff(flat) >= 0)))
print("sum(count) == n_isects:", int(tile_counts.sum()) == n_isects)
# 출력:
# 차분 == count      : True
# 슬라이스 재조립 일치: True
# 빈 타일 t 목록      : [3] → 모두 길이 0
# offsets 은 단조 비감소: True
# sum(count) == n_isects: True

# %% [markdown]
# ## 5. 흔한 함정
#
# - **길이가 `T+1`이 아니다.** `isect_offsets`는 `[C, tile_h, tile_w]`뿐이라
#   마지막 타일의 끝은 `n_isects`를 이어 붙여야 얻는다.
# - **`C` 차원도 평탄화에 포함된다.** 다중 카메라면 오프셋은 이미지 경계를 넘어
#   **이어서** 증가한다 (이미지 c의 첫 타일 오프셋 = 이미지 c-1까지의 총 교차 수).
#   따라서 `isect_offsets[c]`만 떼어내면 0부터 시작하지 않는다.
# - **`flatten_ids`는 Gaussian id가 아니라 `c * N + i` 평탄화 인덱스**다.
#   단일 이미지가 아니면 `% N` 으로 Gaussian 인덱스를 뽑아야 한다.

# %%
# C=2 로 확장해 "이미지 경계를 넘어 이어서 증가"를 확인 (같은 장면을 두 번 복제)
pairs2 = [(c, t, d, c * N + (g % N)) for c in range(2) for (_, t, d, g) in pairs]
keys2 = np.array(
    [(c << (tile_n_bits + 32)) | (t << 32) | depth_bits(d) for c, t, d, _ in pairs2],
    dtype=np.uint64,
)
gids2 = np.array([g for *_, g in pairs2], dtype=np.int32)
o2 = np.argsort(keys2, kind="stable")
ids2, fids2 = keys2[o2], gids2[o2]

tc2 = np.zeros((2, tile_h, tile_w), dtype=np.int64)
hi2 = (ids2 >> np.uint64(32)).astype(np.int64)
u2, c2 = np.unique(hi2, return_counts=True)
tc2[u2 >> tile_n_bits, (u2 & ((1 << tile_n_bits) - 1)) // tile_w, (u2 & ((1 << tile_n_bits) - 1)) % tile_w] = c2
off2 = (np.cumsum(tc2.reshape(-1)).reshape(tc2.shape) - tc2).astype(np.int32)

print("C=2 offsets.shape =", off2.shape)
print("image 0:\n", off2[0])
print(f"image 1:  (0이 아니라 {n_isects}부터 시작한다)\n", off2[1])
# 출력:
# C=2 offsets.shape = (2, 3, 4)
# image 0:
#  [[ 0  1  3  4]
#   [ 4  5  8 10]
#   [11 12 15 17]]
# image 1:  (0이 아니라 18부터 시작한다)
#  [[18 19 21 22]
#   [22 23 26 28]
#   [29 30 33 35]]

# %% [markdown]
# ## 6. 시각화 — 오프셋 / 타일별 개수 / CSR 구간
#
# 왼쪽: 각 타일의 시작 오프셋 `isect_offsets[0]` (셀 안에 `[lo:hi]` 표기).
# 가운데: 타일별 Gaussian 수 = 오프셋 차분 (0인 셀이 빈 타일).
# 오른쪽: `flatten_ids` 한 줄을 타일 구간으로 잘라 놓은 CSR 띠 — 경계가 곧 오프셋이다.

# %%
counts0 = tile_counts[0]
off0 = isect_offsets[0]

fig = make_subplots(
    rows=1,
    cols=3,
    column_widths=[0.3, 0.3, 0.4],
    subplot_titles=(
        "isect_offsets[0] (시작 위치)",
        "타일별 개수 = 오프셋 차분",
        "flatten_ids 를 타일 구간으로 자르기",
    ),
    horizontal_spacing=0.09,
)

txt = np.array(
    [[f"{off0[y, x]}<br>[{flat[y*tile_w+x]}:{flat[y*tile_w+x+1]}]" for x in range(tile_w)] for y in range(tile_h)]
)
fig.add_trace(
    go.Heatmap(z=off0, text=txt, texttemplate="%{text}", colorscale="Blues", showscale=False, zmin=0, zmax=n_isects),
    row=1,
    col=1,
)
fig.add_trace(
    go.Heatmap(
        z=counts0,
        text=counts0.astype(str),
        texttemplate="%{text}",
        colorscale="Magma",
        showscale=False,
        zmin=0,
        zmax=int(counts0.max()),
    ),
    row=1,
    col=2,
)

# CSR 띠: k = flatten_ids 인덱스, 색 = Gaussian id, 타일 경계에 세로선
gcolor = ["#e45756", "#54a24b", "#4c78a8", "#b279a2"]
for k in range(n_isects):
    g = int(flatten_ids[k] % N)
    fig.add_shape(
        type="rect",
        x0=k,
        x1=k + 1,
        y0=0,
        y1=1,
        fillcolor=gcolor[g],
        line=dict(color="white", width=1),
        row=1,
        col=3,
    )
    fig.add_annotation(x=k + 0.5, y=0.5, text=f"g{g}", showarrow=False, font=dict(size=9, color="white"), row=1, col=3)
for t in range(flat.size):
    fig.add_shape(type="line", x0=flat[t], x1=flat[t], y0=-0.25, y1=1.25, line=dict(color="#333", width=2), row=1, col=3)
for t in range(flat.size - 1):
    ty, tx = divmod(t, tile_w)
    if flat[t] < flat[t + 1]:  # 정상 타일: 구간 가운데에 (y,x)
        fig.add_annotation(
            x=(flat[t] + flat[t + 1]) / 2, y=1.35, text=f"({ty},{tx})", showarrow=False, font=dict(size=9), row=1, col=3
        )
    else:  # 빈 타일: 폭 0 이므로 위쪽에 화살표로 따로 표시
        fig.add_annotation(
            x=flat[t],
            y=1.72,
            ax=0,
            ay=-18,
            text=f"∅ ({ty},{tx}) 빈 타일",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowcolor="#d62728",
            font=dict(size=10, color="#d62728"),
            row=1,
            col=3,
        )
    fig.add_annotation(x=flat[t], y=-0.45, text=str(flat[t]), showarrow=False, font=dict(size=9), row=1, col=3)

for col in (1, 2):
    fig.update_yaxes(autorange="reversed", title="tile y", dtick=1, row=1, col=col)
    fig.update_xaxes(title="tile x", dtick=1, row=1, col=col)
fig.update_xaxes(title="flatten_ids 인덱스 k", range=[-0.5, n_isects + 0.5], row=1, col=3)
fig.update_yaxes(visible=False, range=[-0.8, 2.1], row=1, col=3)
fig.update_layout(
    title=f"isect_offsets [C={C}, tile_h={tile_h}, tile_w={tile_w}] — n_isects={n_isects} (빈 타일 = ∅)",
    height=420,
    width=1400,
    margin=dict(t=90, b=60),
)

fig.write_image("expy.png", scale=2)
_show(fig)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 요약
#
# | 항목 | 값 |
# |---|---|
# | 모양 | `[C, tile_h, tile_w]`, int32 |
# | 값 | 타일 `t`가 `flatten_ids`에서 시작하는 인덱스 (배타적 prefix sum) |
# | 사용 | `flatten_ids[offsets[t] : offsets[t+1]]` — 마지막 타일은 `n_isects`까지 |
# | 빈 타일 | `offsets[t] == offsets[t+1]` → 길이 0 |
# | 만드는 함수 | `isect_offset_encode` (참조 구현: `_torch_impl._isect_offset_encode`) |
