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
# # gsplat 3DGS 학습 과정 워크스루
#
# 이 노트북은 gsplat의 3D Gaussian Splatting 학습 파이프라인을
# [examples/simple_trainer.py](simple_trainer.py)를 따라가며 단계별로 분해해서 실행해 본다.
# 실제 트레이너의 핵심 경로(데이터 → 초기화 → 렌더 → 손실 → 밀도화 → 최적화)만 남기고
# 부가 기능(viewer, pose/appearance 최적화, 분산, 압축 등)은 걷어냈다.
#
# ```
# COLMAP SfM 결과                       매 스텝 반복
# ┌──────────────┐   ┌──────────────────────────────────────────────────┐
# │ poses, K,    │   │  이미지 1장 샘플                                 │
# │ sparse points│   │    → rasterization()  (SH평가→투영→타일링→블렌딩)│
# └──────┬───────┘   │    → loss = 0.8·L1 + 0.2·(1-SSIM)                │
#        │           │    → strategy.step_pre_backward()  (grad 추적)   │
#   Gaussian 초기화  │    → loss.backward()                             │
#   (means/scales/   │    → optimizer.step()  (파라미터별 Adam)         │
#    quats/opacity/  │    → strategy.step_post_backward()               │
#    SH)             │       (duplicate / split / prune / opacity reset)│
#        └──────────►└──────────────────────────────────────────────────┘
# ```
#
# **실행 방법**
# - CUDA GPU + gsplat CUDA extension이 빌드된 환경이 필요하다 (`conda activate gsplat`).
# - VSCode에서 이 파일을 열면 `# %%` 셀 단위로 인터랙티브 실행이 되고,
#   `jupytext --to ipynb training_walkthrough.py`로 .ipynb 변환도 가능하다.
# - 데이터는 Mip-NeRF 360 등 COLMAP 형식 씬이면 아무거나 된다 (아래 `DATA_DIR` 수정).

# %%
import os
import sys
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# examples/ 아래 모듈(datasets.colmap 등)을 import하기 위한 경로 설정
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from gsplat.losses import l1_loss, ssim_loss

# ---------------- 설정 ----------------
DATA_DIR = "/home/sungwoo/projects/data/raw/mipnerf360/2.0.0/garden"  # COLMAP 씬 경로
DATA_FACTOR = 8      # 이미지 다운샘플 배율 (simple_trainer 기본은 4; 빠른 데모용으로 8)
TEST_EVERY = 8       # 8장마다 1장을 검증용으로 분리
MAX_STEPS = 2_000    # 데모용 (논문 재현은 30_000)
SH_DEGREE = 3        # spherical harmonics 최대 차수 → 계수 (3+1)^2 = 16개
DEVICE = "cuda:0"

torch.manual_seed(42)
np.random.seed(42)
print(torch.cuda.get_device_name(0))

# %% [markdown]
# ## 1단계: 데이터 준비 — COLMAP SfM 결과 로드
#
# 3DGS 학습의 입력은 **같은 장면을 여러 각도에서 찍은 사진 + COLMAP SfM 결과**다.
# [datasets/colmap.py](datasets/colmap.py)의 `Parser`가 세 가지를 읽어온다.
#
# | 데이터 | 용도 |
# |---|---|
# | 카메라 포즈 `camtoworlds` [N,4,4] | 각 학습 이미지의 시점 |
# | 내부 파라미터 `Ks` [3,3] (+왜곡계수) | 투영 모델. 왜곡이 있으면 이미지를 미리 undistort |
# | sparse 3D 포인트 + RGB | **Gaussian 초기 위치/색** (`init_type="sfm"`) |
#
# `normalize=True`면 카메라 위치 기준 similarity 변환으로 월드 좌표를 정규화하고
# (`similarity_from_cameras` + `align_principal_axes`), 씬의 대략적 크기가
# `parser.scene_scale`로 계산된다. 이 값은 이후 **learning rate와 밀도화 임계값의
# 기준 단위**가 되므로 중요하다 (simple_trainer.py:458 — `scene_scale * 1.1 * global_scale`).
#
# train/val 분리는 단순히 "매 `test_every`번째 이미지는 val" 규칙이다 (colmap.py:459).

# %%
from datasets.colmap import Parser, Dataset

parser = Parser(
    data_dir=DATA_DIR,
    factor=DATA_FACTOR,
    normalize=True,       # 월드 좌표 정규화
    test_every=TEST_EVERY,
)
trainset = Dataset(parser, split="train")
valset = Dataset(parser, split="val")

scene_scale = parser.scene_scale * 1.1  # simple_trainer.py:458과 동일
print(f"학습 이미지 {len(trainset)}장 / 검증 이미지 {len(valset)}장")
print(f"SfM 포인트 {parser.points.shape[0]:,}개, scene_scale = {scene_scale:.3f}")
print(f"이미지 크기: {list(parser.imsize_dict.values())[0]}")

# %%
# SfM 포인트클라우드와 카메라 위치를 3D로 확인
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
pts, rgb = parser.points, parser.points_rgb / 255.0
sub = np.random.choice(len(pts), min(20_000, len(pts)), replace=False)
ax.scatter(*pts[sub].T, c=rgb[sub], s=0.5)
cam_pos = parser.camtoworlds[:, :3, 3]
ax.scatter(*cam_pos.T, c="red", s=12, marker="^", label="cameras")
ax.set_title("SfM sparse points + camera poses (normalized world)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2단계: Gaussian 파라미터 초기화
#
# 하나의 3D Gaussian은 5종류의 파라미터로 표현된다. 최적화 안정성을 위해
# 제약이 있는 값들은 **비제약 공간에 저장**하고 렌더 직전에 활성화 함수를 통과시킨다.
#
# | 파라미터 | shape | 저장 공간 → 활성화 | 초기값 |
# |---|---|---|---|
# | `means` | [N,3] | 그대로 | SfM 포인트 위치 |
# | `scales` | [N,3] | log → `exp` | log(3-최근접 이웃 평균거리) — 이웃이 멀면 큰 Gaussian |
# | `quats` | [N,4] | 미정규화 → 내부 normalize | 랜덤 |
# | `opacities` | [N] | logit → `sigmoid` | logit(0.1) |
# | `sh0`/`shN` | [N,1,3]/[N,15,3] | SH 계수 | DC = (rgb−0.5)/0.2821, 고차항 = 0 |
#
# 공분산은 quaternion 회전 R과 스케일 대각행렬 S로부터 $\Sigma = R\,S\,S^\top R^\top$로
# 합성되므로 항상 양의 정부호가 보장된다.
#
# **최적화기는 파라미터마다 별도의 Adam**을 쓴다 (`eps=1e-15`). 학습률은 파라미터별로
# 크게 다르며, 특히 `means`의 lr은 `scene_scale`을 곱해 씬 크기에 무관하게 만든다.
# (simple_trainer.py:288 `create_splats_with_optimizers`와 동일한 로직)

# %%
def knn_mean_dist(points: torch.Tensor, k: int = 3, chunk: int = 8192) -> torch.Tensor:
    """각 점에서 k-최근접 이웃까지의 평균 거리 (GPU, 청크 처리)."""
    out = []
    for i in range(0, len(points), chunk):
        d = torch.cdist(points[i : i + chunk], points)  # [chunk, N]
        knn_d = d.topk(k + 1, largest=False).values[:, 1:]  # 자기 자신 제외
        out.append(knn_d.mean(dim=-1))
    return torch.cat(out)


def init_splats_with_optimizers(parser, scene_scale: float, sh_degree: int, device: str):
    points = torch.from_numpy(parser.points).float().to(device)
    rgbs = torch.from_numpy(parser.points_rgb / 255.0).float().to(device)
    N = points.shape[0]

    # 크기: 주변 점 밀도에 맞춰 초기화 (빈틈없이 덮되 과하게 겹치지 않도록)
    dist_avg = knn_mean_dist(points, k=3)
    scales = torch.log(dist_avg)[:, None].repeat(1, 3)          # [N,3] log-space
    quats = torch.rand(N, 4, device=device)                     # [N,4]
    opacities = torch.logit(torch.full((N,), 0.1, device=device))  # [N] logit-space

    # 색: SH 계수. DC(0차)만 SfM 색으로, 고차항은 0으로
    C0 = 0.28209479177387814
    colors = torch.zeros(N, (sh_degree + 1) ** 2, 3, device=device)
    colors[:, 0, :] = (rgbs - 0.5) / C0

    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(points),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        # .contiguous()가 필요: 슬라이스 뷰를 그대로 Parameter로 쓰면 fused Adam이 거부한다
        "sh0": torch.nn.Parameter(colors[:, :1, :].contiguous()),
        "shN": torch.nn.Parameter(colors[:, 1:, :].contiguous()),
    }).to(device)

    # 파라미터별 학습률 (simple_trainer 기본값)
    lrs = {
        "means": 1.6e-4 * scene_scale,  # 위치는 씬 크기에 비례
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,             # 고차 SH는 천천히
    }
    optimizers = {
        name: torch.optim.Adam([{"params": splats[name], "lr": lr, "name": name}],
                               eps=1e-15, fused=True)
        for name, lr in lrs.items()
    }
    return splats, optimizers


splats, optimizers = init_splats_with_optimizers(parser, scene_scale, SH_DEGREE, DEVICE)
print(f"초기 Gaussian 개수: {len(splats['means']):,}")
for k, v in splats.items():
    print(f"  {k:10s} {tuple(v.shape)}")

# %% [markdown]
# ## 3단계: Forward — `rasterization()` 한 번 호출해 보기
#
# [gsplat/rendering.py:234](../gsplat/rendering.py#L234)의 `rasterization()`이 미분 가능한
# 렌더러 전체다. 내부는 4개의 CUDA 커널 단계로 구성된다.
#
# 1. **SH 평가** (`spherical_harmonics`) — 카메라→Gaussian 시선 방향으로 SH 계수를
#    평가해 뷰 의존적 RGB를 얻는다. `sh_degree` 인자로 활성 차수를 제한할 수 있다.
# 2. **투영** (`fully_fused_projection`) — 3D 공분산 $\Sigma = RSS^\top R^\top$를
#    카메라로 투영(EWA splatting)해 2D 공분산(conic), 화면 좌표 `means2d`, 깊이,
#    반경 `radii`를 얻는다. near/far 밖이거나 화면 밖이면 `radii=0`으로 컬링.
# 3. **타일 교차** (`isect_tiles`) — 화면을 16×16 픽셀 타일로 나누고, 각 Gaussian이
#    걸치는 타일마다 (tile_id, depth) 키를 만들어 정렬한다.
# 4. **픽셀 래스터화** (`rasterize_to_pixels`) — 타일별로 깊이순 앞→뒤 알파 블렌딩:
#    $C = \sum_i c_i\,\alpha_i \prod_{j<i}(1-\alpha_j)$,
#    $\alpha_i = o_i \exp(-\tfrac12 \Delta^\top \Sigma'^{-1} \Delta)$.
#    투과율이 임계값 아래로 떨어지면 조기 종료.
#
# 반환되는 `info` dict가 **밀도화 전략의 입력**이 된다. 특히 `info["means2d"]`는
# 화면공간 위치 텐서로, 이것의 gradient가 "이 Gaussian을 더 쪼개야 하는가"의 신호다.

# %%
def rasterize_splats(splats, camtoworlds, Ks, width, height, sh_degree, **kwargs):
    """simple_trainer.py:649 rasterize_splats()의 최소 버전."""
    means = splats["means"]                          # [N,3]
    quats = splats["quats"]                          # [N,4] (내부에서 normalize됨)
    scales = torch.exp(splats["scales"])             # [N,3] log → 실제 크기
    opacities = torch.sigmoid(splats["opacities"])   # [N]   logit → (0,1)
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N,16,3] SH 계수

    return rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),      # world→cam
        Ks=Ks, width=width, height=height,
        sh_degree=sh_degree,
        packed=False,                                # 밀도화 상태 갱신 코드와 맞춤
        **kwargs,
    )


# 첫 학습 이미지를 초기 상태로 렌더해 보기
data0 = trainset[0]
c2w = data0["camtoworld"][None].to(DEVICE)   # [1,4,4]
K = data0["K"][None].to(DEVICE)              # [1,3,3]
gt = data0["image"][None].to(DEVICE) / 255.0 # [1,H,W,3]
H, W = gt.shape[1:3]

with torch.no_grad():
    render, alpha, info = rasterize_splats(splats, c2w, K, W, H, sh_degree=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(gt[0].cpu()); axes[0].set_title("GT")
axes[1].imshow(render[0].clamp(0, 1).cpu()); axes[1].set_title("초기 상태 렌더 (SfM 색만)")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()

print("info keys:", sorted(info.keys()))
print("means2d:", tuple(info["means2d"].shape), "| radii:", tuple(info["radii"].shape))
print(f"화면에 보이는 Gaussian: {(info['radii'] > 0).all(-1).sum().item():,} / {len(splats['means']):,}")

# %% [markdown]
# ## 4단계: 손실 함수
#
# 3DGS의 기본 손실은 놀랄 만큼 단순하다 (simple_trainer.py:961):
#
# $$\mathcal{L} = (1-\lambda)\,\mathcal{L}_{L1} + \lambda\,(1-\mathrm{SSIM}), \qquad \lambda = 0.2$$
#
# - **L1**: 픽셀 단위 색 재구성. 노이즈에 강건.
# - **SSIM**: 11×11 가우시안 윈도우 기반 구조 유사도. 국소 대비/구조를 맞춰
#   L1만 쓸 때 생기는 뭉개짐을 억제한다.
#
# simple_trainer에는 선택적 항이 더 있다 (기본은 모두 꺼짐):
# `depth_loss`(SfM 포인트 disparity L1), `opacity_reg`/`scale_reg`(MCMC 전략용 정규화),
# `random_bkgd`(투명 영역이 배경색으로 도망가는 것 방지).

# %%
with torch.no_grad():
    l1 = l1_loss(render, gt).mean()
    ssim = ssim_loss(render.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))
    total = torch.lerp(l1, ssim, 0.2)  # = 0.8*l1 + 0.2*ssim
print(f"초기 상태 손실: L1={l1:.4f}, SSIM loss={ssim:.4f}, total={total:.4f}")

# %% [markdown]
# ## 5단계: 밀도화(Densification) 전략
#
# SfM 포인트만으로는 씬을 다 덮지 못하므로, 학습 중에 Gaussian을 **늘리고 정리하는**
# 과정이 필수다. [gsplat/strategy/default.py](../gsplat/strategy/default.py)의
# `DefaultStrategy`가 원논문 방식을 구현한다. 매 스텝 두 개의 훅이 불린다.
#
# - `step_pre_backward()` — `info["means2d"].retain_grad()`. 화면공간 gradient를
#   backward 후에도 읽을 수 있게 함.
# - `step_post_backward()` — gradient 통계를 누적하고, 주기마다 refine 실행:
#
# | 동작 | 조건 (기본값) | 효과 |
# |---|---|---|
# | **duplicate** | 화면 grad 평균 > `2e-4` 이고 크기 ≤ 1%·scene_scale | 작은데 오차 큰 곳 → 복제 |
# | **split** | 화면 grad 평균 > `2e-4` 이고 크기 > 1%·scene_scale | 큰데 오차 큰 곳 → 2개로 쪼개고 크기 /1.6 |
# | **prune** | opacity < `0.005`, 또는 크기 > 10%·scene_scale | 기여 없는/비대한 것 제거 |
# | **opacity reset** | 매 `3000`스텝 | 전체 opacity를 0.01로 리셋 → floater 정리 |
#
# refine은 스텝 `500~15000` 구간에서 `100`스텝마다 실행된다. 화면공간 gradient는
# `[-1,1]` NDC 기준으로 정규화해 누적한다 (default.py:248). `absgrad=True`를 주면
# 픽셀별 gradient의 절대값 합(AbsGS)을 쓰는데, 상쇄가 없어 더 민감한 분할 신호가 된다
# (이때 임계값은 0.0008 권장).

# %%
strategy = DefaultStrategy(verbose=True)
strategy.check_sanity(splats, optimizers)           # params/optimizer 키 일치 검사
strategy_state = strategy.initialize_state(scene_scale=scene_scale)
print(strategy)

# %% [markdown]
# ## 6단계: 학습 루프
#
# 이제 전부 조립한다. simple_trainer.py:795 `train()`과 같은 순서다.
#
# 1. 학습 이미지 1장 샘플 (DataLoader, shuffle)
# 2. SH 차수 스케줄: `min(step // 1000, 3)` — 처음엔 DC만 학습해 색부터 안정화
# 3. `rasterization()` forward
# 4. `strategy.step_pre_backward()` → loss 계산 → `loss.backward()`
# 5. 파라미터별 Adam step + zero_grad
# 6. `means` lr 지수 감쇠: 총 스텝에 걸쳐 초기값의 1%까지 (`gamma = 0.01^{1/max\_steps}`)
# 7. `strategy.step_post_backward()` — 밀도화/가지치기
#
# 실제 트레이너는 여기에 checkpoint 저장, eval, viewer 갱신, TensorBoard 로깅이 끼어든다.

# %%
means_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizers["means"], gamma=0.01 ** (1.0 / MAX_STEPS)
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=True, num_workers=4, persistent_workers=True
)
loader_iter = iter(trainloader)

# 학습 경과 기록용
history = {"step": [], "loss": [], "num_gs": []}
snapshots = {}
snap_view = valset[0]  # 고정 시점에서 경과 관찰
snap_c2w = snap_view["camtoworld"][None].to(DEVICE)
snap_K = snap_view["K"][None].to(DEVICE)
snap_gt = snap_view["image"][None].to(DEVICE) / 255.0
snap_steps = {0, 250, 500, 1000, MAX_STEPS - 1}

tic = time.time()
pbar = tqdm(range(MAX_STEPS))
for step in pbar:
    try:
        data = next(loader_iter)
    except StopIteration:
        loader_iter = iter(trainloader)
        data = next(loader_iter)

    camtoworlds = data["camtoworld"].to(DEVICE)   # [1,4,4]
    Ks = data["K"].to(DEVICE)                     # [1,3,3]
    pixels = data["image"].to(DEVICE) / 255.0     # [1,H,W,3]
    height, width = pixels.shape[1:3]

    # (2) SH 차수 스케줄
    sh_degree_to_use = min(step // 1000, SH_DEGREE)

    # (3) forward
    renders, alphas, info = rasterize_splats(
        splats, camtoworlds, Ks, width, height, sh_degree=sh_degree_to_use
    )

    # (4) 밀도화 훅 + 손실 + backward
    strategy.step_pre_backward(splats, optimizers, strategy_state, step, info)

    l1 = l1_loss(renders, pixels).mean()
    ssim = ssim_loss(renders.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
    loss = torch.lerp(l1, ssim, 0.2)
    loss.backward()

    # (5) 파라미터별 Adam
    for opt in optimizers.values():
        opt.step()
        opt.zero_grad(set_to_none=True)
    # (6) means lr 감쇠
    means_lr_scheduler.step()

    # (7) 밀도화: duplicate / split / prune / opacity reset
    strategy.step_post_backward(
        splats, optimizers, strategy_state, step, info, packed=False
    )

    history["step"].append(step)
    history["loss"].append(loss.item())
    history["num_gs"].append(len(splats["means"]))
    if step % 50 == 0:
        pbar.set_description(
            f"loss={loss.item():.3f} | GS={len(splats['means']):,} | sh={sh_degree_to_use}"
        )

    if step in snap_steps:
        with torch.no_grad():
            snap, _, _ = rasterize_splats(
                splats, snap_c2w, snap_K, snap_gt.shape[2], snap_gt.shape[1],
                sh_degree=sh_degree_to_use,
            )
        snapshots[step] = snap[0].clamp(0, 1).cpu()

print(f"학습 완료: {time.time() - tic:.1f}s, 최종 Gaussian {len(splats['means']):,}개")

# %% [markdown]
# ## 결과 확인
#
# 손실 곡선에서 **밀도화의 흔적**을 볼 수 있다. Gaussian 수는 refine 구간(스텝 500~)
# 에서 100스텝마다 계단식으로 늘어난다. (30k 스텝 완주 시에는 3000스텝마다
# opacity reset 직후 loss가 튀었다가 회복하는 패턴, 15000스텝 이후 개수 고정도 보인다.)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
axes[0].plot(history["step"], history["loss"], lw=0.5)
axes[0].set(xlabel="step", ylabel="loss", title="Training loss (0.8·L1 + 0.2·SSIM)")
axes[0].set_yscale("log")
axes[1].plot(history["step"], history["num_gs"])
axes[1].set(xlabel="step", ylabel="#Gaussians", title="밀도화에 따른 Gaussian 개수")
plt.tight_layout(); plt.show()

# %%
# 고정 검증 시점에서 본 학습 경과
steps_sorted = sorted(snapshots)
fig, axes = plt.subplots(1, len(steps_sorted) + 1, figsize=(4 * (len(steps_sorted) + 1), 3.5))
for ax, s in zip(axes, steps_sorted):
    ax.imshow(snapshots[s]); ax.set_title(f"step {s}"); ax.axis("off")
axes[-1].imshow(snap_gt[0].cpu()); axes[-1].set_title("GT (val)"); axes[-1].axis("off")
plt.tight_layout(); plt.show()

# %%
# 검증셋 PSNR (simple_trainer의 eval()에서는 torchmetrics로 PSNR/SSIM/LPIPS를 계산)
@torch.no_grad()
def eval_psnr(n_images: int = 5) -> float:
    psnrs = []
    for i in range(min(n_images, len(valset))):
        d = valset[i]
        gt = d["image"][None].to(DEVICE) / 255.0
        render, _, _ = rasterize_splats(
            splats, d["camtoworld"][None].to(DEVICE), d["K"][None].to(DEVICE),
            gt.shape[2], gt.shape[1], sh_degree=SH_DEGREE,
        )
        mse = F.mse_loss(render.clamp(0, 1), gt)
        psnrs.append(-10.0 * math.log10(mse.item()))
    return float(np.mean(psnrs))

print(f"검증 PSNR (val {min(5, len(valset))}장 평균): {eval_psnr():.2f} dB")
print("참고: garden full 학습(30k step, factor 4) 기준 ~27 dB")

# %% [markdown]
# ## 정리 — simple_trainer.py와의 대응 관계
#
# | 이 노트북 | simple_trainer.py | 비고 |
# |---|---|---|
# | `Parser`/`Dataset` 로드 | `Runner.__init__` (L444) | ncore 백엔드, mask/exposure 로드 생략 |
# | `init_splats_with_optimizers` | `create_splats_with_optimizers` (L288) | 멀티 GPU 분배, sparse/visible Adam 생략 |
# | `rasterize_splats` | `Runner.rasterize_splats` (L649) | appearance/post-processing/왜곡 카메라 생략 |
# | 학습 루프 | `Runner.train` (L795) | pose 최적화, ckpt/ply 저장, viewer 생략 |
# | `eval_psnr` | `Runner.eval` (L1201) | SSIM/LPIPS, 색보정 지표 생략 |
#
# **여기서 더 볼 것들**
# - `packed=True`: 가시 Gaussian만 sparse하게 유지 → 대규모 씬 메모리 절감
#   (`sparse_grad`, `SelectiveAdam`과 조합)
# - `MCMCStrategy` ([gsplat/strategy/mcmc.py](../gsplat/strategy/mcmc.py)):
#   휴리스틱 분할 대신 SGLD 방식 — Gaussian 수 상한을 두고 opacity 기반 확률적
#   재배치 + 노이즈 주입. `opacity_reg`/`scale_reg`와 함께 사용
# - `absgrad=True` + `grow_grad2d=0.0008`: AbsGS 방식의 더 정밀한 밀도화
# - 카메라 모델: pinhole 외 fisheye/ftheta, 왜곡계수, rolling shutter, 3DGUT(`with_ut`)
# - 분산 학습: `rasterization(distributed=True)` — rank별 Gaussian 분할 소유
