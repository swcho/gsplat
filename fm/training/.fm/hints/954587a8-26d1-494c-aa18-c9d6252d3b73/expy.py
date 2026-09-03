# 필요 패키지: numpy, torch, plotly, kaleido
# gsplat 자체는 import하지 않는다 (JIT 빌드가 오래 걸림). CUDA 커널의 수식을
# numpy/torch로 그대로 재구현해 카드의 두 식을 검증한다.
#
#   C = sum_i c_i * alpha_i * prod_{j<i} (1 - alpha_j)
#   alpha_i = o_i * exp(-1/2 * Delta^T Sigma'^{-1} Delta)

# %% [markdown]
# # `rasterize_to_pixels`의 알파 블렌딩 공식 해부
#
# 카드의 식은 두 부분이다.
#
# $$C = \sum_i c_i\,\alpha_i \prod_{j<i}(1-\alpha_j) \qquad\text{(겹친 반투명 층 합성)}$$
#
# $$\alpha_i = o_i \exp\!\left(-\tfrac12 \Delta^\top \Sigma'^{-1} \Delta\right) \qquad\text{(픽셀별 알파)}$$
#
# 아래에서 순서대로 만들어 본다.
#
# 1. 반투명 유리판 겹치기 → $\prod_{j<i}(1-\alpha_j)$의 정체
# 2. 재귀식 $T \leftarrow T(1-\alpha)$ 로 바꾸기 (커널이 실제로 하는 것)
# 3. 순서를 바꾸면 결과가 달라진다 → 깊이순 정렬이 필요한 이유
# 4. $\Delta^\top \Sigma'^{-1}\Delta$ = 타원 → `conics`
# 5. 미니 타일 래스터라이저 (조기 종료 포함)
# 6. 벡터화 torch 구현(`cumprod`)과 일치 확인
# 7. autograd: 가려진 Gaussian은 gradient가 작다

# %%
import struct

import numpy as np
import torch

np.set_printoptions(precision=4, suppress=True)
torch.manual_seed(0)


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# gsplat/cuda/include/Common.h, gsplat/cuda/_constants.py 의 실제 상수
ALPHA_THRESHOLD = 1.0 / 255.0
MAX_ALPHA = 0.99
TRANSMITTANCE_THRESHOLD = 1e-4
GAUSSIAN_EXTEND = 3.33
TILE_SIZE = 16

print(f"ALPHA_THRESHOLD         = {ALPHA_THRESHOLD:.6f}   (= 1/255)")
print(f"MAX_ALPHA               = {MAX_ALPHA}")
print(f"TRANSMITTANCE_THRESHOLD = {TRANSMITTANCE_THRESHOLD}  == (1-MAX_ALPHA)^2 = {(1-MAX_ALPHA)**2:.1e}")
# 출력: ALPHA_THRESHOLD         = 0.003922   (= 1/255)
# 출력: MAX_ALPHA               = 0.99
# 출력: TRANSMITTANCE_THRESHOLD = 0.0001  == (1-MAX_ALPHA)^2 = 1.0e-04

# %% [markdown]
# ## 1. 반투명 유리판 겹치기
#
# 1채널로 시작한다. 앞에서부터 유리판 3장, 색 $c=(1.0,\;0.0,\;0.5)$, 알파 $\alpha=(0.5,\;0.5,\;0.5)$.
#
# $$C = c_1\alpha_1 + c_2\alpha_2(1-\alpha_1) + c_3\alpha_3(1-\alpha_1)(1-\alpha_2)$$
#
# 손으로 쓴 식과 $\prod$ 기호 구현이 같은지 본다.

# %%
c = np.array([1.0, 0.0, 0.5])   # 앞→뒤 순서의 색
a = np.array([0.5, 0.5, 0.5])   # 앞→뒤 순서의 알파

hand = c[0] * a[0] + c[1] * a[1] * (1 - a[0]) + c[2] * a[2] * (1 - a[0]) * (1 - a[1])


def blend_prod(colors, alphas):
    """카드 식을 문자 그대로 구현: 매번 prod_{j<i}(1-alpha_j)를 다시 계산."""
    C = 0.0
    for i in range(len(alphas)):
        T_i = np.prod(1.0 - alphas[:i])   # prod_{j<i}(1-alpha_j); i=0이면 빈 곱 = 1
        C += colors[i] * alphas[i] * T_i
    return C


print(f"손계산      C = {hand:.6f}")
print(f"prod 구현   C = {blend_prod(c, a):.6f}")
print(f"각 층의 T_i = {[float(np.prod(1.0 - a[:i])) for i in range(len(a))]}")
print(f"각 층의 w_i = {[float(a[i] * np.prod(1.0 - a[:i])) for i in range(len(a))]}")
# 출력: 손계산      C = 0.562500
# 출력: prod 구현   C = 0.562500
# 출력: 각 층의 T_i = [1.0, 0.5, 0.25]
# 출력: 각 층의 w_i = [0.5, 0.25, 0.125]

# %% [markdown]
# ## 2. 재귀식으로 바꾸기 — 커널이 실제로 하는 계산
#
# $\prod$을 매번 다시 계산할 필요가 없다. 투과율 $T$ 하나만 들고 앞→뒤로 훑는다.
#
# $$T\leftarrow 1,\ C\leftarrow 0;\qquad C\mathrel{+}= c_i\alpha_i T,\quad T\leftarrow T(1-\alpha_i)$$
#
# CUDA 커널(`RasterizeToPixels3DGSSerialBatchFwd.cu:245`)의
# `next_T = T*(1-alpha); vis = alpha*T; pix_out += c*vis; T = next_T;` 와 같다.

# %%
def blend_recursive(colors, alphas, background=None, verbose=False):
    """커널과 동일한 앞→뒤 1-pass. 조기 종료와 상한/하한 클램프 포함."""
    T = 1.0
    C = 0.0
    n_processed = 0
    for i in range(len(alphas)):
        alpha = min(MAX_ALPHA, alphas[i])
        if alpha < ALPHA_THRESHOLD:              # 기여가 8비트 1LSB 미만 → 스킵
            continue
        next_T = T * (1.0 - alpha)
        if next_T <= TRANSMITTANCE_THRESHOLD:    # 포화 → 이 층은 제외하고 종료
            break
        w = alpha * T                            # 커널의 vis
        C += colors[i] * w
        T = next_T
        n_processed += 1
        if verbose:
            print(f"  i={i}  alpha={alpha:.4f}  w=alpha*T={w:.4f}  C={C:.4f}  T={T:.4f}")
    if background is not None:
        C = C + T * background                   # 남은 투과율만큼 배경이 섞인다
    return C, T, n_processed


C_rec, T_rec, n = blend_recursive(c, a, verbose=True)
sum_w = float(np.sum([a[i] * np.prod(1 - a[:i]) for i in range(3)]))
print(f"재귀 구현   C = {C_rec:.6f},  최종 T = {T_rec:.6f}")
print(f"누적 alpha = 1 - T = {1 - T_rec:.6f}   (= sum w_i = {sum_w:.6f})")
print(f"prod 구현과 일치? {np.isclose(C_rec, blend_prod(c, a))}")
C_bg, _, _ = blend_recursive(c, a, background=1.0)
print(f"배경 흰색(1.0) 합성: C = {C_bg:.6f}  (= {C_rec:.4f} + {T_rec:.4f}*1.0)")
# 출력:   i=0  alpha=0.5000  w=alpha*T=0.5000  C=0.5000  T=0.5000
# 출력:   i=1  alpha=0.5000  w=alpha*T=0.2500  C=0.5000  T=0.2500
# 출력:   i=2  alpha=0.5000  w=alpha*T=0.1250  C=0.5625  T=0.1250
# 출력: 재귀 구현   C = 0.562500,  최종 T = 0.125000
# 출력: 누적 alpha = 1 - T = 0.875000   (= sum w_i = 0.875000)
# 출력: prod 구현과 일치? True
# 출력: 배경 흰색(1.0) 합성: C = 0.687500  (= 0.5625 + 0.1250*1.0)

# %% [markdown]
# ### 투과율은 단조 감소하고, 뒤쪽 층은 기하급수적으로 묻힌다
#
# 알파가 모두 같으면 $T_i=(1-\alpha)^{i}$, $w_i=\alpha(1-\alpha)^i$ — **등비수열**이다.
# 그래서 앞→뒤로 훑다가 $T$가 $10^{-4}$ 아래로 떨어지는 순간 나머지는 볼 필요가 없다.

# %%
for alpha in (0.1, 0.3, 0.5, 0.9, MAX_ALPHA):
    k = int(np.ceil(np.log(TRANSMITTANCE_THRESHOLD) / np.log(1 - alpha)))
    print(f"alpha={alpha:5.2f} -> (1-alpha)^k <= 1e-4 가 되는 k = {k:4d}  (그 뒤 층은 조기 종료)")
# 출력: alpha= 0.10 -> (1-alpha)^k <= 1e-4 가 되는 k =   88  (그 뒤 층은 조기 종료)
# 출력: alpha= 0.30 -> (1-alpha)^k <= 1e-4 가 되는 k =   26  (그 뒤 층은 조기 종료)
# 출력: alpha= 0.50 -> (1-alpha)^k <= 1e-4 가 되는 k =   14  (그 뒤 층은 조기 종료)
# 출력: alpha= 0.90 -> (1-alpha)^k <= 1e-4 가 되는 k =    4  (그 뒤 층은 조기 종료)
# 출력: alpha= 0.99 -> (1-alpha)^k <= 1e-4 가 되는 k =    2  (그 뒤 층은 조기 종료)

# %% [markdown]
# ## 3. 순서가 결과를 바꾼다 → 깊이순 정렬이 필수
#
# 알파 블렌딩은 **교환법칙이 성립하지 않는다.** $\prod_{j<i}$의 $j<i$가 순서를 요구한다.

# %%
c2 = np.array([1.0, 0.0])   # 빨강(1.0) 앞, 파랑(0.0) 뒤
a2 = np.array([0.5, 0.5])
front_red = blend_prod(c2, a2)
front_blue = blend_prod(c2[::-1], a2[::-1])
print(f"빨강 앞: C = {front_red:.4f}")
print(f"파랑 앞: C = {front_blue:.4f}")
print(f"비율 = {front_red / front_blue:.1f}배  (같은 두 얼룩인데 순서만 바꿨다)")
# 출력: 빨강 앞: C = 0.5000
# 출력: 파랑 앞: C = 0.2500
# 출력: 비율 = 2.0배  (같은 두 얼룩인데 순서만 바꿨다)

# %% [markdown]
# ### `isect_tiles`의 64비트 정렬 키
#
# 정렬은 `(image_id, tile_id, depth)`를 64비트 정수 하나로 묶어서 한다.
# 양수 float32는 **비트 패턴을 그대로 정수로 읽어도 크기 순서가 보존**되므로,
# 실수 정렬을 정수 radix sort 한 번으로 처리할 수 있다
# (`gsplat/cuda/_torch_impl.py:418`).

# %%
depths_demo = np.array([0.5, 1.0, 2.0, 3.7, 100.0], dtype=np.float32)
bits = [struct.unpack("I", struct.pack("f", float(d)))[0] for d in depths_demo]
print("depth  ->  float32 비트 패턴(uint32)")
for d, b in zip(depths_demo, bits):
    print(f"{d:8.2f} -> {b:12d}")
print(f"비트 패턴이 depth와 같은 순서인가? {bits == sorted(bits)}")


def make_key(image_id, tile_id, depth_f32, tile_n_bits=8):
    depth_bits = struct.unpack("I", struct.pack("f", float(depth_f32)))[0]
    return (((image_id << tile_n_bits) | tile_id) << 32) | depth_bits


pairs = [(0, 2.0), (0, 0.5), (1, 2.0), (1, 0.5)]   # (tile_id, depth)
keys = [make_key(0, t, d) for t, d in pairs]
srt = sorted(range(4), key=lambda i: keys[i])
print(f"\n입력 (tile_id, depth) = {pairs}")
print(f"키 정렬 후 순서        = {[pairs[i] for i in srt]}  -> 타일별로 뭉치고, 타일 안에서 앞→뒤")
# 출력: depth  ->  float32 비트 패턴(uint32)
# 출력:     0.50 ->   1056964608
# 출력:     1.00 ->   1065353216
# 출력:     2.00 ->   1073741824
# 출력:     3.70 ->   1080872141
# 출력:   100.00 ->   1120403456
# 출력: 비트 패턴이 depth와 같은 순서인가? True
# 출력:
# 출력: 입력 (tile_id, depth) = [(0, 2.0), (0, 0.5), (1, 2.0), (1, 0.5)]
# 출력: 키 정렬 후 순서        = [(0, 0.5), (0, 2.0), (1, 0.5), (1, 2.0)]  -> 타일별로 뭉치고, 타일 안에서 앞→뒤

# %% [markdown]
# ## 4. $\alpha_i$의 안쪽: $\Delta^\top \Sigma'^{-1} \Delta$ 는 타원
#
# $\Sigma'^{-1} = \begin{pmatrix}a&b\\b&c\end{pmatrix}$ (= `conics`, 대칭이라 3원소)라 하면
#
# $$\Delta^\top \Sigma'^{-1}\Delta = a\,\Delta x^2 + 2b\,\Delta x\Delta y + c\,\Delta y^2$$
#
# 이고 이 값이 상수인 곳이 **회전된 타원**이다. 커널은 $\tfrac12$을 미리 흡수해서
#
# ```cpp
# sigma = 0.5f*(conic.x*dx*dx + conic.z*dy*dy) + conic.y*dx*dy;  // = 1/2 Delta^T Sigma'^-1 Delta
# alpha = min(MAX_ALPHA, opac * __expf(-sigma));
# ```
# 로 계산한다. $2b\Delta x\Delta y$의 2와 $\tfrac12$이 약분되어 교차항에 계수가 없다.

# %%
def make_conic(sx, sy, theta_deg):
    """반경 (sx, sy), theta 회전한 2D Gaussian의 Sigma' 와 conic (a,b,c) = Sigma'^{-1}."""
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    S = np.diag([sx**2, sy**2])
    cov = R @ S @ R.T                                  # Sigma' (2x2, 대칭 양의 정부호)
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
    conic = np.array([cov[1, 1] / det, -cov[0, 1] / det, cov[0, 0] / det])  # (a, b, c)
    return cov, conic


cov, conic = make_conic(6.0, 2.0, 30.0)
Sinv = np.array([[conic[0], conic[1]], [conic[1], conic[2]]])
print("Sigma' =\n", cov)
print("conic (a,b,c) =", conic)
print("Sigma' @ Sigma'^-1 = I ?\n", cov @ Sinv)


def sigma_kernel(conic, dx, dy):
    """커널 한 줄과 동일: 0.5*Delta^T Sigma'^-1 Delta."""
    return 0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy


def sigma_matrix(Sinv, dx, dy):
    """행렬식으로 직접: 0.5 * Delta^T Sigma'^-1 Delta."""
    d = np.array([dx, dy])
    return 0.5 * d @ Sinv @ d


for dx, dy in [(0.0, 0.0), (1.0, 0.0), (0.0, 3.0), (2.0, -1.5)]:
    sk, sm = sigma_kernel(conic, dx, dy), sigma_matrix(Sinv, dx, dy)
    print(f"Delta=({dx:5.1f},{dy:5.1f})  커널식={sk:9.5f}  행렬식={sm:9.5f}  일치={np.isclose(sk, sm)}")
# 출력: Sigma' =
# 출력:  [[28.     13.8564]
# 출력:  [13.8564 12.    ]]
# 출력: conic (a,b,c) = [ 0.0833 -0.0962  0.1944]
# 출력: Sigma' @ Sigma'^-1 = I ?
# 출력:  [[ 1. -0.]
# 출력:  [ 0.  1.]]
# 출력: Delta=(  0.0,  0.0)  커널식=  0.00000  행렬식=  0.00000  일치=True
# 출력: Delta=(  1.0,  0.0)  커널식=  0.04167  행렬식=  0.04167  일치=True
# 출력: Delta=(  0.0,  3.0)  커널식=  0.87500  행렬식=  0.87500  일치=True
# 출력: Delta=(  2.0, -1.5)  커널식=  0.67409  행렬식=  0.67409  일치=True

# %% [markdown]
# ### 부호는 상관없다 / 중심에서 값은 정확히 $o_i$
#
# 커널은 `dx = mean - pixel`, PyTorch 참조 구현은 `pixel - mean`을 쓴다.
# 이차형식이라 $\Delta \to -\Delta$에 불변이므로 동일하다.
# 또 정규화 상수 $\frac{1}{2\pi\sqrt{\det\Sigma'}}$가 **없어서** 중심에서 지수항이 정확히 1,
# 즉 $\alpha_\text{max} = o_i$다. (확률밀도가 아니라 "가리는 비율"이므로.)

# %%
o_i = 0.7
print(f"sigma(+Delta)={sigma_kernel(conic, 2.0, -1.5):.5f}  sigma(-Delta)={sigma_kernel(conic, -2.0, 1.5):.5f}")
print(f"중심 alpha = o * exp(-0) = {o_i * np.exp(-sigma_kernel(conic, 0.0, 0.0)):.4f}  (== o_i)")
r = GAUSSIAN_EXTEND    # radii 계산에 쓰이는 컷오프
print(f"{r} sigma 지점의 감쇠 = exp(-0.5*{r}^2) = {np.exp(-0.5*r*r):.6f} ~ 1/256 = {1/256:.6f}")
print(f"o=0.7일 때 alpha가 ALPHA_THRESHOLD로 떨어지는 마할라노비스 거리 = "
      f"{np.sqrt(2*np.log(o_i/ALPHA_THRESHOLD)):.3f} sigma")
# 출력: sigma(+Delta)=0.67409  sigma(-Delta)=0.67409
# 출력: 중심 alpha = o * exp(-0) = 0.7000  (== o_i)
# 출력: 3.33 sigma 지점의 감쇠 = exp(-0.5*3.33^2) = 0.003909 ~ 1/256 = 0.003906
# 출력: o=0.7일 때 alpha가 ALPHA_THRESHOLD로 떨어지는 마할라노비스 거리 = 3.220 sigma

# %% [markdown]
# ## 5. 미니 타일 래스터라이저
#
# 64x64 이미지, 16x16 타일 → 4x4 = 16 타일. 화면 중앙에서 서로 심하게 겹치는
# 진한 Gaussian 10개를 깊이순으로 정렬해 타일별로 앞→뒤 블렌딩한다.
# 커널과 같은 조기 종료를 넣고, 픽셀별로 **실제로 몇 개를 처리했는지** 센다.

# %%
W = H = 64

# (mean_x, mean_y, sx, sy, theta, opacity, depth, r, g, b)   — 일부러 겹침을 심하게
GAUSSIANS = [
    (26.0, 26.0, 11.0,  5.0,  25.0, 0.95, 1.0, 1.00, 0.25, 0.25),  # 앞: 빨강
    (34.0, 30.0, 12.0, 12.0,   0.0, 0.95, 1.5, 0.25, 0.45, 1.00),  # 파랑
    (30.0, 38.0,  7.0, 15.0, -35.0, 0.95, 2.0, 0.20, 0.90, 0.40),  # 초록
    (38.0, 36.0, 14.0,  6.0,  15.0, 0.95, 2.5, 1.00, 0.85, 0.20),  # 노랑
    (48.0, 20.0,  9.0,  9.0,   0.0, 0.90, 3.0, 0.90, 0.30, 0.90),  # 자홍
    (16.0, 46.0,  8.0,  8.0,   0.0, 0.90, 3.5, 0.20, 0.85, 0.90),  # 시안
    (52.0, 50.0, 10.0,  4.0, -20.0, 0.90, 4.0, 1.00, 0.55, 0.10),  # 주황
    (12.0, 14.0,  6.0, 12.0,  40.0, 0.90, 4.5, 0.60, 0.30, 1.00),  # 보라
    (30.0, 30.0, 16.0, 16.0,   0.0, 0.98, 5.0, 0.55, 0.55, 0.60),  # 중앙 겹침용 막 1
    (34.0, 34.0, 16.0, 16.0,   0.0, 0.98, 5.5, 0.45, 0.50, 0.55),  # 중앙 겹침용 막 2
    (32.0, 32.0, 18.0, 18.0,   0.0, 0.95, 6.0, 0.35, 0.35, 0.45),  # 넓은 회색막
    (32.0, 32.0, 45.0, 45.0,   0.0, 0.95, 9.0, 0.10, 0.10, 0.15),  # 맨 뒤 어두운 배경막
]


def prepare(gaussians):
    order = np.argsort([g[6] for g in gaussians])       # depth 오름차순 = 앞→뒤
    means, conics, opas, cols, deps = [], [], [], [], []
    for i in order:
        mx, my, sx, sy, th, o, d, cr, cg, cb = gaussians[i]
        _, cn = make_conic(sx, sy, th)
        means.append([mx, my]); conics.append(cn); opas.append(o)
        cols.append([cr, cg, cb]); deps.append(d)
    return (np.array(means), np.array(conics), np.array(opas),
            np.array(cols), np.array(deps), order)


means2d, conics_all, opacities, colors_all, depths_all, order = prepare(GAUSSIANS)
print("깊이순(앞→뒤) 정렬 후 원본 인덱스:", order.tolist())
print("정렬된 depth:", depths_all)
# 출력: 깊이순(앞→뒤) 정렬 후 원본 인덱스: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# 출력: 정렬된 depth: [1.  1.5 2.  2.5 3.  3.5 4.  4.5 5.  5.5 6.  9. ]


def screen_radii(conic):
    """3.33 sigma 컷오프 반경 (isect_tiles가 쓰는 축 정렬 바운딩박스)."""
    a, b, cc = conic
    det = a * cc - b * b
    return GAUSSIAN_EXTEND * np.sqrt(cc / det), GAUSSIAN_EXTEND * np.sqrt(a / det)


def rasterize_to_pixels_toy(means2d, conics_all, opacities, colors_all,
                            W, H, tile_size=TILE_SIZE, early_stop=True):
    """타일별 앞→뒤 알파 블렌딩. 커널의 스칼라 루프를 파이썬으로 그대로 옮긴 것."""
    render = np.zeros((H, W, 3))
    render_alpha = np.zeros((H, W))
    n_processed = np.zeros((H, W), dtype=int)
    last_id = np.full((H, W), -1, dtype=int)
    saturated = np.zeros((H, W), dtype=bool)   # 조기 종료가 실제로 걸린 픽셀
    n_isects = 0

    for ty in range(0, H, tile_size):
        for tx in range(0, W, tile_size):
            # 이 타일과 겹치는 Gaussian만 후보로 (isect_tiles의 역할)
            cand = []
            for i, (mx, my) in enumerate(means2d):
                rx, ry = screen_radii(conics_all[i])
                if (mx + rx >= tx and mx - rx <= tx + tile_size
                        and my + ry >= ty and my - ry <= ty + tile_size):
                    cand.append(i)              # 이미 깊이순이므로 순서 보존
            n_isects += len(cand)

            for py in range(ty, min(ty + tile_size, H)):
                for px in range(tx, min(tx + tile_size, W)):
                    T = 1.0
                    acc = np.zeros(3)
                    for i in cand:
                        dx = means2d[i, 0] - (px + 0.5)   # 픽셀 중심 (+0.5)
                        dy = means2d[i, 1] - (py + 0.5)
                        sg = sigma_kernel(conics_all[i], dx, dy)
                        if sg < 0.0:                      # 커널의 valid 검사
                            continue
                        alpha = min(MAX_ALPHA, opacities[i] * np.exp(-sg))
                        if alpha < ALPHA_THRESHOLD:
                            continue
                        next_T = T * (1.0 - alpha)
                        if early_stop and next_T <= TRANSMITTANCE_THRESHOLD:
                            saturated[py, px] = True
                            break                         # 포화: 이 Gaussian 제외하고 종료
                        acc += colors_all[i] * (alpha * T)
                        T = next_T
                        n_processed[py, px] += 1
                        last_id[py, px] = i
                    render[py, px] = acc
                    render_alpha[py, px] = 1.0 - T         # 커널: render_alphas = 1 - T
    return render, render_alpha, n_processed, last_id, saturated, n_isects


render, render_alpha, n_proc, last_id, sat, n_isects = rasterize_to_pixels_toy(
    means2d, conics_all, opacities, colors_all, W, H)
print(f"n_isects (Gaussian-타일 쌍의 수) = {n_isects}   (Gaussian {len(means2d)}개, 타일 16개)")
print(f"render shape={render.shape}  min={render.min():.4f} max={render.max():.4f}")
print(f"alpha  min={render_alpha.min():.4f} max={render_alpha.max():.4f}  (= 1 - T)")
print(f"픽셀당 처리한 Gaussian 수: 평균 {n_proc.mean():.2f} / 최대 {n_proc.max()} (전체 {len(means2d)}개)")
print(f"조기 종료(T<=1e-4)가 실제로 걸린 픽셀 = {sat.mean()*100:.1f}%")
print(f"중심 픽셀(32,32): 색 = {render[32, 32]}, alpha = {render_alpha[32,32]:.6f}, last_id = {last_id[32,32]}")
# 출력: n_isects (Gaussian-타일 쌍의 수) = 160   (Gaussian 12개, 타일 16개)
# 출력: render shape=(64, 64, 3)  min=0.0896 max=0.9628
# 출력: alpha  min=0.6203 max=0.9999  (= 1 - T)
# 출력: 픽셀당 처리한 Gaussian 수: 평균 7.90 / 최대 11 (전체 12개)
# 출력: 조기 종료(T<=1e-4)가 실제로 걸린 픽셀 = 11.4%
# 출력: 중심 픽셀(32,32): 색 = [0.684  0.3484 0.5484], alpha = 0.998388, last_id = 5

# %% [markdown]
# ## 6. 벡터화 torch 구현과 일치 확인
#
# 같은 수학을 `torch.cumprod`로 한 번에 계산한다.
# $T_i=\prod_{j<i}(1-\alpha_j)$는 **exclusive cumulative product**다.
# 이렇게 하면 $\prod$ 기호가 코드에 그대로 드러난다 (조기 종료만 없다).

# %%
def rasterize_vectorized(means2d, conics_all, opacities, colors_all, W, H):
    m = torch.tensor(means2d, dtype=torch.float64)          # [N,2] 깊이순
    cn = torch.tensor(conics_all, dtype=torch.float64)      # [N,3]
    op = torch.tensor(opacities, dtype=torch.float64)       # [N]
    co = torch.tensor(colors_all, dtype=torch.float64)      # [N,3]

    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64), indexing="ij")
    pix = torch.stack([xs + 0.5, ys + 0.5], dim=-1)         # [H,W,2]
    d = m.view(-1, 1, 1, 2) - pix.unsqueeze(0)              # [N,H,W,2]  Delta
    dx, dy = d[..., 0], d[..., 1]

    # sigma = 1/2 Delta^T Sigma'^-1 Delta
    sg = 0.5 * (cn[:, 0].view(-1, 1, 1) * dx**2 + cn[:, 2].view(-1, 1, 1) * dy**2) \
        + cn[:, 1].view(-1, 1, 1) * dx * dy
    alpha = torch.clamp(op.view(-1, 1, 1) * torch.exp(-sg), max=MAX_ALPHA)
    alpha = torch.where((sg < 0) | (alpha < ALPHA_THRESHOLD),
                        torch.zeros_like(alpha), alpha)     # 커널의 valid 검사

    # T_i = prod_{j<i} (1 - alpha_j)  <- exclusive cumprod
    T = torch.cumprod(1.0 - alpha, dim=0)                   # inclusive
    T = torch.cat([torch.ones_like(T[:1]), T[:-1]], dim=0)  # 한 칸 밀어 exclusive로
    w = alpha * T                                           # [N,H,W]  기여도
    img = (w.unsqueeze(-1) * co.view(-1, 1, 1, 3)).sum(0)   # [H,W,3]
    return img.numpy(), w.sum(0).numpy(), w.numpy()


img_v, alpha_v, w_v = rasterize_vectorized(means2d, conics_all, opacities, colors_all, W, H)
print(f"조기 종료 ON : 최대 색 차이 = {np.abs(img_v - render).max():.2e},  "
      f"최대 alpha 차이 = {np.abs(alpha_v - render_alpha).max():.2e}")

render_ns, alpha_ns, n_proc_ns, _, _, _ = rasterize_to_pixels_toy(
    means2d, conics_all, opacities, colors_all, W, H, early_stop=False)
print(f"조기 종료 OFF: 최대 색 차이 = {np.abs(img_v - render_ns).max():.2e},  "
      f"최대 alpha 차이 = {np.abs(alpha_ns - alpha_v).max():.2e}   <- 완전 일치")
# break 시점에 버리는 빛의 양은 '그 직전의 T'이고, T*(1-alpha) <= 1e-4 이므로
#   버려지는 양 <= TRANSMITTANCE_THRESHOLD / (1 - alpha) <= 1e-4 / (1-MAX_ALPHA)
print(f"조기 종료 오차 상한 = 1e-4/(1-MAX_ALPHA) = {TRANSMITTANCE_THRESHOLD/(1-MAX_ALPHA):.1e}"
      f"  (관측된 최대 alpha 오차 {np.abs(alpha_v - render_alpha).max():.2e} 는 그 안)")
print(f"처리한 (픽셀, Gaussian) 쌍: 조기 종료 ON {n_proc.sum():,} vs OFF {n_proc_ns.sum():,} "
      f"({(1 - n_proc.sum()/n_proc_ns.sum())*100:.1f}% 절약)")
# 출력: 조기 종료 ON : 최대 색 차이 = 1.82e-03,  최대 alpha 차이 = 3.05e-03
# 출력: 조기 종료 OFF: 최대 색 차이 = 4.44e-16,  최대 alpha 차이 = 4.44e-16   <- 완전 일치
# 출력: 조기 종료 오차 상한 = 1e-4/(1-MAX_ALPHA) = 1.0e-02  (관측된 최대 alpha 오차 3.05e-03 는 그 안)
# 출력: 처리한 (픽셀, Gaussian) 쌍: 조기 종료 ON 32,370 vs OFF 33,287 (2.8% 절약)

# %% [markdown]
# 조기 종료를 끄면 `cumprod` 벡터화 구현과 부동소수점 오차 수준(~$10^{-16}$)까지 일치한다.
# 켜면 차이가 생기는데, 그 상한은 다음처럼 계산된다. `break` 하는 순간 버리는 빛의 양은
# **그 직전의 투과율 $T$** 전부이고, 조건이 $T(1-\alpha)\le 10^{-4}$ 이므로
#
# $$\text{버리는 양} = T \le \frac{10^{-4}}{1-\alpha} \le \frac{10^{-4}}{1-\texttt{MAX\_ALPHA}} = 10^{-2}$$
#
# 8비트 색 한 단계가 $1/255\approx 4\times10^{-3}$ 이니 최악의 경우에도 몇 LSB 수준이다.
# **눈에 거의 안 보이는 오차를 대가로 뒤쪽 Gaussian 로딩을 건너뛴다**는 거래다.

# %% [markdown]
# ## 7. autograd — 가려진 Gaussian은 gradient가 작다
#
# 식이 전부 곱셈·덧셈·`exp`이므로 미분 가능하다. 가장 깨끗한 항등식은 색에 대한 것이다.
#
# $$\frac{\partial C}{\partial c_i} = w_i = \alpha_i \prod_{j<i}(1-\alpha_j)$$
#
# 즉 **색 gradient가 곧 기여도** $w_i$다. 화면 같은 자리에 겹쳐 있는 동일한
# Gaussian 8개(앞→뒤, $o=0.7$)를 두고, 뒤로 갈수록 gradient가 기하급수적으로
# 줄어드는 것을 확인한다.

# %%
N_stack = 8
o_stack = 0.7
_, cn_stack = make_conic(8.0, 8.0, 0.0)

px, py = 32.5, 32.5   # 픽셀 중심 — Gaussian 중심을 여기에 정확히 두면 Delta = 0

m_s = torch.full((N_stack, 2), 32.5, dtype=torch.float64, requires_grad=True)
op_s = torch.full((N_stack,), o_stack, dtype=torch.float64, requires_grad=True)
co_s = torch.rand(N_stack, 3, dtype=torch.float64, requires_grad=True)
cn_s = torch.tensor(cn_stack, dtype=torch.float64)

dx, dy = m_s[:, 0] - px, m_s[:, 1] - py
sg = 0.5 * (cn_s[0] * dx**2 + cn_s[2] * dy**2) + cn_s[1] * dx * dy
alpha = torch.clamp(op_s * torch.exp(-sg), max=MAX_ALPHA)

T_excl = torch.cat([torch.ones(1, dtype=torch.float64),
                    torch.cumprod(1.0 - alpha, dim=0)[:-1]])   # prod_{j<i}(1-alpha_j)
w = alpha * T_excl
C = (w.unsqueeze(-1) * co_s).sum(0)
C.sum().backward()

print(" i   alpha     T_i(=prod)     w_i        dC/dc_i (autograd)   |dL/dmeans2d_i|")
for i in range(N_stack):
    print(f"{i:2d}  {alpha[i].item():.5f}  {T_excl[i].item():.3e}  {w[i].item():.3e}"
          f"      {co_s.grad[i, 0].item():.3e}      {m_s.grad[i].norm().item():.3e}")
ratio = (T_excl[1:] / T_excl[:-1]).detach().numpy()
print(f"\ndC/dc_i == w_i 인가? {torch.allclose(co_s.grad[:, 0], w.detach())}")
print(f"중심에 정확히 놓았으므로 alpha == o == {alpha[0].item():.6f}")
print(f"T_i가 공비 (1-alpha) = {1-o_stack:.2f} 인 등비수열인가? {np.allclose(ratio, 1-o_stack)}")
print(f"맨 앞 vs 맨 뒤 색 gradient 비 = {(w[0]/w[-1]).item():.1f}배  "
      f"(이론값 (1-a)^-7 = {(1-o_stack)**-(N_stack-1):.1f}배)")
print(f"누적 alpha = sum w = {w.sum().item():.6f},  최종 T = {(1-w.sum()).item():.3e}")
print(f"means2d gradient = {m_s.grad.abs().max().item():.3e}  <- Delta=0 은 alpha의 극점이므로 0")
# 출력:  i   alpha     T_i(=prod)     w_i        dC/dc_i (autograd)   |dL/dmeans2d_i|
# 출력:  0  0.70000  1.000e+00  7.000e-01      7.000e-01      0.000e+00
# 출력:  1  0.70000  3.000e-01  2.100e-01      2.100e-01      0.000e+00
# 출력:  2  0.70000  9.000e-02  6.300e-02      6.300e-02      0.000e+00
# 출력:  3  0.70000  2.700e-02  1.890e-02      1.890e-02      0.000e+00
# 출력:  4  0.70000  8.100e-03  5.670e-03      5.670e-03      0.000e+00
# 출력:  5  0.70000  2.430e-03  1.701e-03      1.701e-03      0.000e+00
# 출력:  6  0.70000  7.290e-04  5.103e-04      5.103e-04      0.000e+00
# 출력:  7  0.70000  2.187e-04  1.531e-04      1.531e-04      0.000e+00
# 출력:
# 출력: dC/dc_i == w_i 인가? True
# 출력: 중심에 정확히 놓았으므로 alpha == o == 0.700000
# 출력: T_i가 공비 (1-alpha) = 0.30 인 등비수열인가? True
# 출력: 맨 앞 vs 맨 뒤 색 gradient 비 = 4572.5배  (이론값 (1-a)^-7 = 4572.5배)
# 출력: 누적 alpha = sum w = 0.999934,  최종 T = 6.561e-05
# 출력: means2d gradient = 0.000e+00  <- Delta=0 은 alpha의 극점이므로 0

# %% [markdown]
# 위 표에서 $T_i=(1-0.7)^i$가 정확히 등비수열이고, $w_i$도 같은 비로 줄어든다.
# 8번째 Gaussian의 색 gradient는 첫 번째의 약 $0.3^7\approx 1/4600$이다 —
# 앞이 막혀 있으면 **역전파가 도달하지 않는다.**
#
# 한편 `dL/dmeans2d`가 워크스루 5단계의 **밀도화 신호**다.
# `DefaultStrategy`가 `info["means2d"].retain_grad()`로 이 값을 받아
# 크기가 임계 이상이면 해당 Gaussian을 split/duplicate 한다.
# (위 실험은 Gaussian이 픽셀 중심에 정확히 있어 $\Delta=0$, 즉 극점이므로
# `means2d` gradient가 0이다. 아래처럼 중심을 살짝 옮기면 살아난다.)

# %%
for off in (0.0, 1.0, 4.0, 8.0, 16.0):
    m1 = torch.tensor([[px + off, py]], dtype=torch.float64, requires_grad=True)
    d0, d1 = m1[0, 0] - px, m1[0, 1] - py
    s1 = 0.5 * (cn_s[0] * d0**2 + cn_s[2] * d1**2) + cn_s[1] * d0 * d1
    al = torch.clamp(torch.tensor(o_stack, dtype=torch.float64) * torch.exp(-s1), max=MAX_ALPHA)
    al.backward()
    print(f"Delta_x={off:5.1f}  sigma={s1.item():7.4f}  alpha={al.item():.6f}  "
          f"|dalpha/dmeans2d| = {m1.grad.norm().item():.3e}")
# 출력: Delta_x=  0.0  sigma= 0.0000  alpha=0.700000  |dalpha/dmeans2d| = 0.000e+00
# 출력: Delta_x=  1.0  sigma= 0.0078  alpha=0.694553  |dalpha/dmeans2d| = 1.085e-02
# 출력: Delta_x=  4.0  sigma= 0.1250  alpha=0.617748  |dalpha/dmeans2d| = 3.861e-02
# 출력: Delta_x=  8.0  sigma= 0.5000  alpha=0.424571  |dalpha/dmeans2d| = 5.307e-02
# 출력: Delta_x= 16.0  sigma= 2.0000  alpha=0.094735  |dalpha/dmeans2d| = 2.368e-02
# 출력: (Delta=0은 극점이라 gradient 0. 1시그마 근처(Delta_x=8)에서 가장 크다 -> 밀도화 신호)

# %% [markdown]
# ## 8. 시각화
#
# - (A) 투과율 $T_i$와 기여도 $w_i=\alpha_i T_i$ — 등비수열로 감쇠, 조기 종료 지점
# - (B) 단일 Gaussian의 알파 지도 $o\exp(-\tfrac12\Delta^\top\Sigma'^{-1}\Delta)$ — 기울어진 타원
#   (흰 등고선은 $\tfrac12\Delta^\top\Sigma'^{-1}\Delta = 0.5,\,4.5$, 즉 $1\sigma$와 $3\sigma$)
# - (C) 미니 래스터라이저 출력 (Gaussian 10개, 16x16 타일 경계 표시)
# - (D) 픽셀별 처리한 Gaussian 수 — 조기 종료가 실제로 일하는 곳

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "(A) T_i = Π(1-α_j) 와 기여도 w_i = α_i T_i  (α=0.5 균일)",
        "(B) 단일 Gaussian의 α = o·exp(-½ ΔᵀΣ'⁻¹Δ)",
        "(C) 미니 래스터라이저 출력 (16×16 타일)",
        "(D) 픽셀별 처리한 Gaussian 수 (조기 종료)",
    ),
    vertical_spacing=0.12, horizontal_spacing=0.10,
)

# (A) 등비수열 감쇠와 조기 종료
a_unif = 0.5
n_layers = 20
idx = np.arange(n_layers)
T_seq = (1 - a_unif) ** idx
w_seq = a_unif * T_seq
k_stop = int(np.ceil(np.log(TRANSMITTANCE_THRESHOLD) / np.log(1 - a_unif)))
fig.add_trace(go.Bar(x=idx, y=w_seq, name="w_i (기여도)",
                     marker_color="rgba(255,127,14,0.7)"), row=1, col=1)
fig.add_trace(go.Scatter(x=idx, y=T_seq, name="T_i (투과율)", mode="lines+markers",
                         line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=idx, y=np.full(n_layers, TRANSMITTANCE_THRESHOLD), mode="lines",
                         name="T = 1e-4 임계", line=dict(color="gray", dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=[k_stop, k_stop], y=[1e-7, 1.5], mode="lines",
                         name=f"조기 종료 지점 (i={k_stop})",
                         line=dict(color="crimson", dash="dash")), row=1, col=1)
fig.update_yaxes(type="log", title_text="값 (log 스케일)", range=[-7, 0.3], row=1, col=1)
fig.update_xaxes(title_text="앞→뒤 층 번호 i", row=1, col=1)

# (B) 단일 Gaussian 알파 지도 + 1/2/3 sigma 등고선
gx, gy = np.arange(W) + 0.5, np.arange(H) + 0.5
GX, GY = np.meshgrid(gx, gy)
_, cn_b = make_conic(10.0, 4.0, 30.0)
sgb = sigma_kernel(cn_b, 32.0 - GX, 32.0 - GY)
alpha_b = np.minimum(MAX_ALPHA, 0.9 * np.exp(-sgb))
fig.add_trace(go.Heatmap(z=alpha_b, x=gx, y=gy, colorscale="Magma", zmin=0, zmax=0.9,
                         colorbar=dict(title="α", len=0.36, y=0.81, x=1.005)), row=1, col=2)
# sigma = 1/2 k^2 이므로 k 시그마 등고선은 sigma = 0.5(k=1), 4.5(k=3)
fig.add_trace(go.Contour(z=sgb, x=gx, y=gy, showscale=False,
                         contours=dict(start=0.5, end=4.5, size=4.0, coloring="none",
                                       showlabels=True, labelfont=dict(color="white", size=9)),
                         line=dict(color="white", width=1),
                         name="½ΔᵀΣ'⁻¹Δ = 0.5(1σ), 4.5(3σ)", showlegend=True), row=1, col=2)
fig.update_yaxes(autorange="reversed", scaleanchor="x2", row=1, col=2)

# (C) 렌더 결과 + 타일 격자 + Gaussian 중심
img8 = (np.clip(render, 0, 1) * 255).astype(np.uint8)
fig.add_trace(go.Image(z=img8, name="render"), row=2, col=1)
for t in range(TILE_SIZE, W, TILE_SIZE):
    fig.add_shape(type="line", x0=t - 0.5, x1=t - 0.5, y0=-0.5, y1=H - 0.5,
                  line=dict(color="rgba(255,255,255,0.4)", width=1), row=2, col=1)
    fig.add_shape(type="line", y0=t - 0.5, y1=t - 0.5, x0=-0.5, x1=W - 0.5,
                  line=dict(color="rgba(255,255,255,0.4)", width=1), row=2, col=1)
fig.add_trace(go.Scatter(x=means2d[:, 0] - 0.5, y=means2d[:, 1] - 0.5, mode="markers+text",
                         text=[str(i) for i in range(len(means2d))],
                         textposition="top center", textfont=dict(color="white", size=9),
                         marker=dict(color="white", size=5, symbol="x"),
                         name="Gaussian 중심 (앞→뒤 번호)"), row=2, col=1)

# (D) 픽셀별 처리 개수
fig.add_trace(go.Heatmap(z=n_proc, colorscale="Viridis",
                         colorbar=dict(title="개수", len=0.36, y=0.19, x=1.005)), row=2, col=2)
fig.update_yaxes(autorange="reversed", scaleanchor="x4", row=2, col=2)

fig.update_layout(
    title_text="rasterize_to_pixels:  C = Σ c_i α_i Π_{j<i}(1-α_j),   α_i = o_i exp(-½ ΔᵀΣ'⁻¹Δ)",
    height=900, width=1180, showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=-0.10, x=0.0),
    template="plotly_white", bargap=0.15,
)

_show(fig)
fig.write_image("expy.png", scale=2)
print("expy.png 저장 완료")
# 출력: expy.png 저장 완료

# %% [markdown]
# ## 정리
#
# | 수식 조각 | 코드에서의 정체 |
# |---|---|
# | $\prod_{j<i}(1-\alpha_j)$ | 픽셀마다 하나씩 들고 있는 스칼라 `T` (루프에서 곱해 나감) |
# | $\alpha_i T_i$ | 커널의 `vis = alpha * T`; 확률처럼 $\sum_i w_i \le 1$ |
# | $\sum_i w_i$ | `render_alphas = 1 - T` |
# | $\frac12\Delta^\top\Sigma'^{-1}\Delta$ | `0.5*(a*dx*dx + c*dy*dy) + b*dx*dy` (conic 3원소) |
# | $j<i$ (순서) | `isect_tiles`의 (tile_id, depth) 64비트 정렬 키 |
# | $\partial C/\partial c_i$ | 정확히 $w_i$ — 가려진 Gaussian은 학습되지 않는다 |
# | (수식에 없음) | `MAX_ALPHA=0.99`, `ALPHA_THRESHOLD=1/255`, `T<=1e-4` 조기 종료 |
