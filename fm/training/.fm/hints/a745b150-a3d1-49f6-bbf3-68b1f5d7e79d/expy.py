# %% [markdown]
# # `step_pre_backward()` — `info["means2d"].retain_grad()`가 하는 일
#
# `DefaultStrategy.step_pre_backward()`의 본문은 사실상 딱 두 줄이다.
#
# ```python
# assert self.key_for_gradient in info          # 보통 "means2d"
# info[self.key_for_gradient].retain_grad()     # 이게 전부
# ```
#
# 왜 이 한 줄이 필요한가? PyTorch는 **leaf 텐서**(직접 만든 `nn.Parameter` 등)에만
# `.grad`를 채워 두고, 그래프 **중간 텐서(non-leaf)** 의 gradient는 backward가
# 지나가는 즉시 버린다(메모리 절약). `means2d`는 leaf인 `means`를 카메라로 투영해서
# 만들어진 중간 결과이므로 기본 상태에서는 `.grad`가 `None`이다.
# 그런데 밀도화(densification)의 핵심 신호가 바로 **화면공간 gradient**
# $\partial \mathcal{L}/\partial \mu_{2D}$ 이므로, `backward()` **전에** 미리
# "이 텐서 gradient는 남겨 둬"라고 표시해 두어야 한다.
#
# 필요 패키지: torch, numpy, plotly, kaleido
# (gsplat 자체는 import하지 않는다 — JIT 커널 빌드에 30분 이상 걸릴 수 있음)

# %%
import numpy as np
import torch

torch.manual_seed(0)


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


print("torch:", torch.__version__)
# 출력: torch: 2.9.1+cu128

# %% [markdown]
# ## 1. 가장 작은 재현: leaf vs non-leaf
#
# $a$는 leaf, $b = 2a$는 non-leaf. $\mathcal{L} = \sum b^2$ 를 backward하면
# $a$.grad는 채워지지만 $b$.grad는 사라진다.

# %%
a = torch.tensor([1.0, 2.0], requires_grad=True)  # leaf (= splats["means"] 역할)
b = a * 2                                          # non-leaf (= info["means2d"] 역할)
loss = (b**2).sum()
loss.backward()

print("a.is_leaf =", a.is_leaf, "| a.grad =", a.grad)
print("b.is_leaf =", b.is_leaf, "| b.grad =", b.grad)
# 출력: a.is_leaf = True | a.grad = tensor([ 8., 16.])
# 출력: b.is_leaf = False | b.grad = None      ← 중간 텐서라 버려졌다
# (PyTorch가 UserWarning으로 "use .retain_grad() on the non-leaf Tensor"라고 직접
#  알려 준다. step_pre_backward()는 그 조언을 그대로 실행하는 훅이다.)

# %%
# retain_grad()를 backward 전에 호출하면 non-leaf도 .grad를 보관한다
a = torch.tensor([1.0, 2.0], requires_grad=True)
b = a * 2
b.retain_grad()  # ← step_pre_backward()가 하는 일
loss = (b**2).sum()
loss.backward()

print("b.grad =", b.grad, " (= dL/db = 2b)")
print("a.grad =", a.grad, " (= dL/da = 2b*2, 체인룰은 그대로)")
# 출력: b.grad = tensor([4., 8.])  (= dL/db = 2b)
# 출력: a.grad = tensor([ 8., 16.])  (= dL/da = 2b*2, 체인룰은 그대로)
# retain_grad는 흘러가는 값을 "복사해 두는" 것뿐 — 그래프나 다른 gradient에 영향 없음

# %% [markdown]
# ## 2. 장난감 스플래팅 파이프라인
#
# gsplat의 `rasterization()`을 아주 작게 흉내낸다. leaf 파라미터는 3D 위치
# `means3d`이고, 핀홀 투영으로 화면 좌표를 만든다.
#
# $$\mu_{2D} = f\frac{(x,y)}{z} + c, \qquad
# \alpha_i(p) = o_i \exp\!\left(-\frac{\lVert p-\mu_{2D,i}\rVert^2}{2 s_i^2}\right)$$
#
# 픽셀 색은 깊이순 앞→뒤 알파 블렌딩:
# $$C(p) = \sum_i c_i\,\alpha_i(p)\prod_{j<i}\bigl(1-\alpha_j(p)\bigr)$$
#
# `rasterization()`이 돌려주는 `info` dict처럼 중간 산출물을 dict에 담아 반환한다.
#
# > 손실은 실제 3DGS의 `0.8·L1 + 0.2·(1-SSIM)` 대신 **MSE**를 쓴다. L1의 gradient는
# > 부호($\pm 1$)만 남아 오차 *크기*에 비례하지 않아서, "많이 어긋난 Gaussian이
# > 더 큰 화면공간 grad를 받는다"는 관계를 장난감 규모에서 보기 어렵기 때문이다.
# > `retain_grad()`의 동작 자체는 손실 종류와 무관하다.

# %%
H = W = 80
FOCAL = 60.0
CX, CY = W / 2, H / 2

yy, xx = torch.meshgrid(
    torch.arange(H, dtype=torch.float32),
    torch.arange(W, dtype=torch.float32),
    indexing="ij",
)
GRID = torch.stack([xx, yy], dim=-1)  # [H,W,2] 픽셀 좌표


def rasterize(means3d, scales, opacities, colors):
    """미분 가능한 장난감 래스터라이저. gsplat rasterization()의 축소판."""
    z = means3d[:, 2].clamp_min(1e-3)
    # ↓ means2d는 leaf가 아니라 계산 그래프의 중간 노드다
    means2d = FOCAL * means3d[:, :2] / z[:, None] + torch.tensor([CX, CY])  # [N,2]
    radii = (3.0 * scales).detach()                                         # [N] 화면 반경

    d = GRID[None] - means2d[:, None, None, :]                # [N,H,W,2]
    alpha = opacities[:, None, None] * torch.exp(
        -0.5 * (d**2).sum(-1) / scales[:, None, None] ** 2
    )                                                          # [N,H,W]

    order = torch.argsort(z.detach())                          # 앞 → 뒤
    alpha, cols = alpha[order], colors[order]
    trans = torch.cumprod(
        torch.cat([torch.ones(1, H, W), 1.0 - alpha[:-1]], 0), dim=0
    )                                                          # [N,H,W] 누적 투과율
    weight = alpha * trans
    render = (weight[..., None] * cols[:, None, None, :]).sum(0)  # [H,W,3]

    info = {  # gsplat info dict와 같은 역할
        "means2d": means2d,
        "radii": radii,
        "width": W,
        "height": H,
        "n_cameras": 1,
        "depths": z,
    }
    return render, info


N = 6
# 서로 겹치지 않게 3x2 격자로 배치 (한 Gaussian의 오차가 옆 Gaussian에 새지 않도록)
means3d = torch.tensor(
    [
        [-0.35, -0.28, 1.00],
        [0.00, -0.28, 1.02],
        [0.35, -0.28, 1.04],
        [-0.35, 0.28, 1.06],
        [0.00, 0.28, 1.08],
        [0.35, 0.28, 1.10],
    ],
    requires_grad=True,
)
scales = torch.tensor([3.0] * N, requires_grad=True)
opacities = torch.tensor([0.9] * N, requires_grad=True)
colors = torch.eye(3).repeat(2, 1).contiguous().requires_grad_(True)  # [6,3]

# GT: 일부 Gaussian만 어긋난 "정답" 씬 → 어긋난 것만 화면공간 grad가 커야 한다
shift = torch.zeros(N, 3)
shift[0, 0] = 0.045   # 화면상 약 2.7px 어긋남
shift[2, 1] = -0.022  # 약 1.3px
shift[4, 0] = 0.005   # 약 0.3px (거의 맞음)
with torch.no_grad():
    gt, _ = rasterize(means3d + shift, scales, opacities, colors)

render, info = rasterize(means3d, scales, opacities, colors)
print("means2d:", tuple(info["means2d"].shape), "| is_leaf =", info["means2d"].is_leaf)
print("화면 좌표:\n", info["means2d"].detach().numpy().round(1))
print("MSE =", round(((render - gt) ** 2).mean().item(), 6))
# 출력: means2d: (6, 2) | is_leaf = False
# 출력: 화면 좌표:
# 출력:  [[19.  23.2]
# 출력:   [40.  23.5]
# 출력:   [60.2 23.8]
# 출력:   [20.2 55.8]
# 출력:   [40.  55.6]
# 출력:   [59.1 55.3]]
# 출력: MSE = 0.000547

# %% [markdown]
# ## 3. `step_pre_backward()`를 빼먹으면?
#
# `step_post_backward()`는 `info["means2d"].grad`를 읽어 통계를 누적한다
# (`gsplat/strategy/default.py`의 `_update_state()`). pre-backward 훅이 없으면
# `.grad`가 `None`이라 그 첫 줄(`.grad.clone()`)에서 바로 터진다.

# %%
render, info = rasterize(means3d, scales, opacities, colors)
loss = ((render - gt) ** 2).mean()
loss.backward()

print("means2d.grad (pre_backward 없음) =", info["means2d"].grad)
try:
    _ = info["means2d"].grad.clone()  # _update_state()가 하는 첫 동작
except AttributeError as e:
    print("→ _update_state()에서 터짐:", type(e).__name__, e)
# 출력: means2d.grad (pre_backward 없음) = None
# 출력: → _update_state()에서 터짐: AttributeError 'NoneType' object has no attribute 'clone'

# %%
# 반면 means3d(leaf)의 grad는 잘 들어온다 — 학습(Adam step) 자체는 문제없이 돌아간다.
# 즉 retain_grad는 "최적화"가 아니라 "밀도화용 계측(instrumentation)"이다.
print("means3d.grad =\n", means3d.grad.numpy().round(5))
# 출력: means3d.grad =
# 출력:  [[-0.01753  0.      -0.00614]
# 출력:   [-0.      -0.      -0.     ]
# 출력:   [ 0.       0.00928  0.0025 ]
# 출력:   [-0.       0.      -0.     ]
# 출력:   [-0.00204 -0.       0.     ]
# 출력:   [-0.       0.       0.     ]]
# 정확히 맞은 Gaussian(1,3,5)은 grad ~0 — 재구성 오차가 없으니 밀 이유가 없다

# %% [markdown]
# ## 4. 올바른 순서
#
# forward → **`step_pre_backward()`** → loss → `backward()` → `step_post_backward()`
#
# (`examples/simple_trainer.py`의 `train()` 루프와 같은 순서)

# %%
def step_pre_backward(info, key_for_gradient="means2d"):
    """DefaultStrategy.step_pre_backward()의 전부."""
    assert key_for_gradient in info, "The 2D means of the Gaussians is required but missing."
    info[key_for_gradient].retain_grad()


means3d.grad = None
render, info = rasterize(means3d, scales, opacities, colors)
step_pre_backward(info)                     # ← 여기
loss = ((render - gt) ** 2).mean()
loss.backward()

g2d = info["means2d"].grad                  # [N,2] 픽셀 좌표계 gradient
print("means2d.grad =\n", g2d.numpy())
# 출력: means2d.grad =
# 출력:  [[-2.9224876e-04  1.3643755e-12]
# 출력:   [-3.9019596e-09 -8.7676838e-11]
# 출력:   [ 2.3845814e-11  1.6085694e-04]
# 출력:   [-1.8419691e-10  2.8148087e-12]
# 출력:   [-3.6736626e-05 -2.8611922e-12]
# 출력:   [-6.2839001e-10  1.0329407e-11]]
# 부호를 보면 GS0은 -x 방향, GS2는 +y 방향으로 밀려야 손실이 준다

# %% [markdown]
# ## 5. `step_post_backward()`가 이 grad로 하는 일
#
# 픽셀 좌표계 gradient를 **NDC $[-1,1]$ 기준으로 되돌려서** 해상도에 무관한 스칼라로
# 만든 뒤(`default.py`의 `_update_state()`), 화면에 보이는 Gaussian에 대해서만 누적한다.
#
# $$g_i \mathrel{+}= \left\lVert\left(\tfrac{W}{2}C\,\partial_x,\ \tfrac{H}{2}C\,\partial_y\right)\right\rVert_2,
# \qquad n_i \mathrel{+}= 1$$
#
# ($C$ = `n_cameras`. $\partial \mathcal{L}/\partial \mu_{\text{px}} \cdot \frac{W}{2}
# = \partial \mathcal{L}/\partial \mu_{\text{ndc}}$ 이므로, 해상도가 바뀌어도 같은
# 임계값을 쓸 수 있다.)
#
# refine 시점(500~15000 스텝, 100스텝마다)에 평균 $g_i/n_i$를 `grow_grad2d`와 비교해
# 크면 성장 대상 — 크기가 작으면 **duplicate**, 크면 **split**.

# %%
grads = g2d.clone()
grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

state = {"grad2d": torch.zeros(N), "count": torch.zeros(N)}
sel = info["radii"] > 0.0                                    # 화면에 보이는 것만
gs_ids = torch.where(sel)[0]
state["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
avg = state["grad2d"] / state["count"].clamp_min(1)

# 실제 기본값은 grow_grad2d=2e-4 (absgrad=True면 8e-4 권장). 이 장난감 씬은
# Gaussian이 6개뿐이라 스케일이 다르므로, 여기서는 시연용 임계값을 쓴다.
GROW_GRAD2D_REAL = 2e-4
GROW_GRAD2D_TOY = 1e-4
for i in range(N):
    tag = "GROW" if avg[i] > GROW_GRAD2D_TOY else "  -  "
    off_px = shift[i, :2].norm().item() * FOCAL / means3d[i, 2].item()
    print(f"GS{i}  |grad|_ndc = {avg[i]:.3e}  {tag}   (GT와 화면상 {off_px:4.1f}px 어긋남)")
# 출력: GS0  |grad|_ndc = 1.169e-02  GROW   (GT와 화면상  2.7px 어긋남)
# 출력: GS1  |grad|_ndc = 1.561e-07    -     (GT와 화면상  0.0px 어긋남)
# 출력: GS2  |grad|_ndc = 6.434e-03  GROW   (GT와 화면상  1.3px 어긋남)
# 출력: GS3  |grad|_ndc = 7.369e-09    -     (GT와 화면상  0.0px 어긋남)
# 출력: GS4  |grad|_ndc = 1.469e-03  GROW   (GT와 화면상  0.3px 어긋남)
# 출력: GS5  |grad|_ndc = 2.514e-08    -     (GT와 화면상  0.0px 어긋남)
# → 어긋남이 큰 Gaussian일수록 화면공간 grad가 크다 = "여기가 아직 안 맞았다,
#   더 쪼개라"는 신호. retain_grad() 한 줄이 이 랭킹 정보를 살려 둔다.

# %% [markdown]
# ## 6. 왜 `absgrad`(AbsGS)라는 변형이 있는가
#
# `means2d.grad`는 **그 Gaussian을 덮는 모든 픽셀의 기여를 합산**한 값이다. 한
# Gaussian이 왼쪽 픽셀에서는 왼쪽으로, 오른쪽 픽셀에서는 오른쪽으로 당겨지면 두 힘이
# **상쇄**되어 합이 0에 가까워진다. 정답이 "한 덩어리가 아니라 두 덩어리"인
# 과소재구성(under-reconstruction) 상황이 딱 이것이라, 쪼개져야 하는데도 신호가 죽는다.
#
# AbsGS는 픽셀별 gradient의 **절대값 합**을 따로 모아(`means2d.absgrad`, CUDA 커널이
# 채워 준다) 이 상쇄를 없앤다. `absgrad=True`일 때 `_update_state()`는
# `.grad` 대신 `.absgrad`를 읽고, 임계값은 `8e-4` 정도를 쓴다.

# %%
# 화면 왼쪽/오른쪽 절반 loss로 나눠 gradient 상쇄를 직접 관찰
mu = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)  # 큰 Gaussian 1개
sc = torch.tensor([8.0], requires_grad=True)
op = torch.tensor([0.9], requires_grad=True)
co = torch.ones(1, 3, requires_grad=True)
with torch.no_grad():  # 정답은 좌우로 벌어진 두 덩어리
    two = torch.tensor([[-0.12, 0.0, 1.0], [0.12, 0.0, 1.0]])
    gt2, _ = rasterize(two, sc.repeat(2), op.repeat(2), co.repeat(2, 1))

r2, i2 = rasterize(mu, sc, op, co)
step_pre_backward(i2)
err = (r2 - gt2) ** 2
mask = torch.zeros(H, W, 1)
mask[:, : W // 2] = 1.0
gl = torch.autograd.grad((err * mask).mean(), i2["means2d"], retain_graph=True)[0][0]
gr = torch.autograd.grad((err * (1 - mask)).mean(), i2["means2d"])[0][0]

print(f"왼쪽 절반이 만드는 grad_x  = {gl[0]:+.3e}")
print(f"오른쪽 절반이 만드는 grad_x = {gr[0]:+.3e}")
print(f"단순 합  |g| (= .grad)      = {(gl + gr).norm():.3e}")
print(f"절대값 합 (= absgrad 취지)  = {(gl.abs() + gr.abs()).norm():.3e}")
print(f"→ 남은 신호 비율 = {(gl + gr).norm() / (gl.abs() + gr.abs()).norm():.2e}  (사실상 0)")
# 출력: 왼쪽 절반이 만드는 grad_x  = +1.320e-03
# 출력: 오른쪽 절반이 만드는 grad_x = -1.320e-03
# 출력: 단순 합  |g| (= .grad)      = 1.181e-10
# 출력: 절대값 합 (= absgrad 취지)  = 2.640e-03
# 출력: → 남은 신호 비율 = 4.47e-08  (사실상 0)
# 완벽히 대칭인 under-reconstruction → .grad는 0으로 상쇄, absgrad만 살아남는다

# %% [markdown]
# ## 7. 실전 주의점
#
# - **호출 시점이 전부다.** `backward()` 뒤에 `retain_grad()`를 부르면 이미 gradient가
#   버려진 뒤라 아무 효과가 없다. 그래서 이름이 `step_`**`pre`**`_backward`.
# - **매 스텝 새로 불러야 한다.** `means2d`는 forward마다 새로 만들어지는 텐서라
#   플래그가 다음 스텝으로 이어지지 않는다.
# - **비용은 거의 없다.** `[C,N,2]`(또는 packed면 `[nnz,2]`) float 하나를 backward가
#   끝날 때까지 붙잡는 정도. 그래프 구조나 다른 gradient 값은 전혀 바뀌지 않는다.
# - `requires_grad=False`인 forward(eval, 또는 Gaussian을 얼려 둔 구간)에서는
#   `retain_grad()`가 예외를 던진다. `simple_trainer.py`가 `_gaussians_frozen`일 때
#   이 훅을 건너뛰는 이유.
# - `MCMCStrategy`는 화면공간 grad를 쓰지 않으므로 `step_pre_backward()`가 비어 있다
#   (`mcmc.py`에 주석으로만 남아 있고, base `Strategy`의 `pass` 구현을 상속).
# - 2DGS에서는 `key_for_gradient="gradient_2dgs"`가 된다 — 훅이 `info`의 어떤 키를
#   붙잡을지만 달라지고 원리는 같다.

# %%
frozen = means3d.detach()  # requires_grad=False
_, info_frozen = rasterize(frozen, scales.detach(), opacities.detach(), colors.detach())
try:
    step_pre_backward(info_frozen)
except RuntimeError as e:
    print("frozen forward에서 retain_grad →", str(e)[:70])
# 출력: frozen forward에서 retain_grad → can't retain_grad on Tensor that has requires_grad=False

# %% [markdown]
# ## 8. 시각화
#
# 왼쪽: GT vs 현재 렌더. 가운데: `means2d.grad`의 **반대 방향** 화살표
# (= 손실을 줄이는 화면상 이동 방향). 오른쪽: NDC로 정규화한 grad 크기와 임계선.
# 정확히 맞은 GS1/GS3/GS5는 grad가 0이라 화살표도 막대도 없다.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

means3d.grad = None
render, info = rasterize(means3d, scales, opacities, colors)
step_pre_backward(info)
((render - gt) ** 2).mean().backward()
g2d = info["means2d"].grad.clone()
gn = g2d.clone()
gn[..., 0] *= W / 2.0
gn[..., 1] *= H / 2.0
mag = gn.norm(dim=-1).detach().numpy()
mu2d = info["means2d"].detach().numpy()
hot = mag > GROW_GRAD2D_TOY

fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=(
        "잔차 |렌더 − GT| (오차가 있는 곳)",
        "렌더 + −means2d.grad 방향",
        "|grad|_ndc — 성장 후보 판정",
    ),
    horizontal_spacing=0.08,
)
resid = (render.detach() - gt).abs().sum(-1).numpy()
resid = (resid / resid.max() * 255).clip(0, 255).astype(np.uint8)
fig.add_trace(
    go.Image(z=np.repeat(resid[..., None], 3, axis=-1), hoverinfo="skip"), row=1, col=1
)
fig.add_trace(
    go.Image(z=(render.detach().numpy() * 255).clip(0, 255).astype(np.uint8)), row=1, col=2
)

# 화살표: 방향은 −grad, 길이는 크기에 따라 6~13px로 압축(로그 스케일 차이가 커서)
dirv = -g2d.detach().numpy()
dirv = dirv / (np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-30)
alen = 6.0 + 7.0 * np.sqrt(mag / mag.max())
d = dirv * alen[:, None]
for i in range(N):
    if not hot[i]:
        continue
    fig.add_annotation(
        x=mu2d[i, 0] + d[i, 0], y=mu2d[i, 1] + d[i, 1], ax=mu2d[i, 0], ay=mu2d[i, 1],
        xref="x2", yref="y2", axref="x2", ayref="y2",
        showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=3,
        arrowcolor="#e45756", row=1, col=2,
    )
fig.add_trace(
    go.Scatter(
        x=mu2d[:, 0], y=mu2d[:, 1], mode="markers+text",
        text=[f"GS{i}" for i in range(N)], textposition="bottom center",
        textfont=dict(color="white", size=11),
        marker=dict(
            size=9, line=dict(width=1, color="white"),
            color=["#e45756" if h else "#54a24b" for h in hot],
        ),
        showlegend=False, hovertext=[f"|grad|_ndc={m:.2e}" for m in mag],
    ),
    row=1, col=2,
)
fig.add_trace(
    go.Bar(
        x=[f"GS{i}" for i in range(N)], y=np.maximum(mag, 1e-9),
        marker_color=["#e45756" if h else "#54a24b" for h in hot],
        showlegend=False,
        text=[f"{m:.1e}" for m in mag],
        textposition="outside",
    ),
    row=1, col=3,
)
fig.add_hline(
    y=GROW_GRAD2D_TOY, line=dict(color="#4c78a8", dash="dash"),
    annotation_text="임계값(시연용)", annotation_position="top left", row=1, col=3,
)
fig.update_yaxes(type="log", title_text="|grad|_ndc (log)", range=[-9.5, -1.4], row=1, col=3)
fig.update_layout(
    title="step_pre_backward()의 retain_grad()가 살려 두는 신호: 화면공간 gradient",
    width=1280, height=450, template="plotly_white",
    margin=dict(t=95, b=40, l=45, r=20),
)
_show(fig)
fig.write_image("expy.png", scale=2)
print("saved expy.png")
# 출력: saved expy.png
