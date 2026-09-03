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
# # gsplat `rasterization()` 동작 원리 워크스루
#
# 이 노트북은 [gsplat/rendering.py](../gsplat/rendering.py)의 `rasterization()` 한 줄이
# 내부에서 무엇을 하는지, **각 단계를 손으로 다시 계산해 CUDA 결과와 대조**하면서 따라간다.
# 학습 루프 전체는 [training_walkthrough.py](training_walkthrough.py)에서 다뤘고,
# 여기서는 그 안의 "렌더" 한 스텝만 확대해서 본다.
#
# ```
#   입력: means[N,3] quats[N,4] scales[N,3] opacities[N] colors(SH)[N,K,3]   viewmats[C,4,4] Ks[C,3,3]
#     │
#     ▼  ① 3D 공분산      Σ = R S Sᵀ Rᵀ                       (quat_scale_to_covar_preci)
#     ▼  ② 카메라 좌표     μ_c = R μ + t,  Σ_c = R Σ Rᵀ          ┐
#     ▼  ③ 원근 투영(EWA)  Σ₂ = J Σ_c Jᵀ + eps2d·I              ├ fully_fused_projection  (CUDA 커널 1개)
#          → means2d, conics(=Σ₂⁻¹), radii, depths, 절두체 컬링 ┘
#     ▼  ④ 색 평가         colors = clamp(SH(μ − cam_pos) + 0.5)  (spherical_harmonics)
#     ▼  ⑤ 타일 교차       Gaussian × 덮는 타일 → 64bit 키 [image|tile|depth] radix sort  (isect_tiles)
#     ▼  ⑥ 타일 오프셋     타일별 시작 인덱스                    (isect_offset_encode)
#     ▼  ⑦ 알파 블렌딩     타일 = CUDA 블록, 픽셀 = 스레드, 앞→뒤 순서 누적  (rasterize_to_pixels)
#     │
#   출력: render_colors[C,H,W,D]  render_alphas[C,H,W,1]  meta(중간 텐서들)
# ```
#
# 현재 코드베이스에서 `rasterization()`은 위 단계를 **C++ 오케스트레이터 `rasterization_3dgs`**
# ([Rendering.cpp](../gsplat/cuda/csrc/Rendering.cpp))에 한 번에 넘긴다. 하지만 단계별 CUDA 래퍼
# ([gsplat/cuda/_wrapper.py](../gsplat/cuda/_wrapper.py))와 순수 PyTorch 참조 구현
# ([gsplat/cuda/_torch_impl.py](../gsplat/cuda/_torch_impl.py))이 그대로 남아 있어,
# 이 노트북은 그 둘을 나란히 놓고 검증한다. (tests/test_basic.py가 같은 방식으로 CUDA를 검증한다.)
#
# **실행 방법**
# - CUDA GPU + gsplat CUDA extension 빌드 환경(`conda activate gsplat`). 첫 import 시 JIT 빌드가 수 분 걸릴 수 있다.
# - VSCode에서 `# %%` 셀 단위로 실행하거나, `python examples/rasterization_walkthrough.py`로 통째로 실행.
# - 데이터는 저장소에 포함된 [assets/test_garden.npz](../assets/test_garden.npz)(garden 씬 SfM 포인트)를 쓰므로 별도 준비가 없다.

# %%
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.font_manager as fm

# 그림 제목의 한글 표시용: 설치된 CJK 폰트가 있으면 사용, 없으면 DejaVu로 폴백(한글은 □로 나오지만 실행에는 영향 없음)
_ko_fonts = [f for f in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]
             if f in {x.name for x in fm.fontManager.ttflist}]
plt.rcParams["font.family"] = _ko_fonts + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import (           # 단계별 CUDA 래퍼
    quat_scale_to_covar_preci,
    fully_fused_projection,
    spherical_harmonics,
    isect_tiles,
    isect_offset_encode,
    rasterize_to_pixels,
)
from gsplat.cuda._torch_impl import (        # 순수 PyTorch 참조 구현
    _fully_fused_projection,
    _isect_tiles,
    _isect_offset_encode,
    _spherical_harmonics,
)
from gsplat.cuda._math import _quat_to_rotmat
from gsplat.cuda._constants import ALPHA_THRESHOLD, MAX_ALPHA, TRANSMITTANCE_THRESHOLD
from gsplat._helper import load_test_data

DEVICE = "cuda:0"
torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
print(torch.cuda.get_device_name(0))


def maxdiff(name: str, a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.float() - b.float()).abs().max().item()
    print(f"  {name:28s} max|Δ| = {d:.3e}")
    return d


# %% [markdown]
# ## 0. 한 번에 보기 — `rasterization()` 호출과 `meta`
#
# 먼저 블랙박스로 한 번 호출해 입·출력 모양을 본다. 테스트 씬은 garden의 SfM 포인트(위치, 색)만 있으므로
# 크기·회전·불투명도는 랜덤이다. 색은 SH 계수 형태(`sh_degree=3`, K=16)로 넘겨서 ④단계까지 태운다.
#
# - `render_mode="RGB+ED"`: RGB 3채널 + 기대 깊이(Expected Depth) 1채널.
#   깊이는 색과 똑같이 "채널 하나"로 취급되어 함께 블렌딩되고, ED는 마지막에 alpha로 나눈다.
# - `packed=False`: 중간 텐서를 `[C, N, ...]` 밀집 형태로 받는다(설명하기 쉬움). 기본값 `packed=True`는 9절 참고.

# %%
# gsplat/_helper.py의 테스트용 씬 로더. assets/test_garden.npz(garden 씬 COLMAP SfM 포인트 138,766개 + 카메라 3대)를
# 읽고, [-2,2]³ 박스 안의 점만 남긴 뒤 Gaussian 속성 중 SfM에 없는 것(크기·회전·불투명도)은 랜덤으로 채워 준다.
# 반환값 (모두 DEVICE 위의 텐서, N = 크롭 후 점 개수, C = 카메라 수 = 3):
#   means     [N,3]   3D 중심 (npz의 means3d)          — 실제 SfM 위치
#   quats     [N,4]   회전 쿼터니언 (w,x,y,z), 정규화됨   — randn에서 랜덤 생성
#   scales    [N,3]   축별 크기 (선형, log 아님)         — U[1e-4, 0.02] 랜덤 (아래에서 덮어씀)
#   opacities [N]     불투명도 (0~1, 시그모이드 후 값)    — U[0,1] 랜덤
#   rgbs      [N,3]   색 (0~1 float)                    — 실제 SfM 색 (uint8/255); 아래에서 SH DC 계수로 변환
#   viewmats  [C,4,4] world→camera 변환 (상단 3x3 = R, 4열 = t)
#   Ks        [C,3,3] 카메라 내부 파라미터 (fx, fy, cx, cy)
#   W, H      int     이미지 크기 (px)
means, quats, scales, opacities, rgbs, viewmats, Ks, W, H = load_test_data(device=DEVICE)
N, C = means.shape[0], viewmats.shape[0]
scales = torch.rand_like(scales) * 0.02 + 0.01   # 테스트 기본값(≤0.02)보다 조금 키워 splat이 보이게

SH_DEGREE = 3
K = (SH_DEGREE + 1) ** 2
C0 = 0.28209479177387814                          # SH 0차 기저 상수 Y₀₀
sh_coeffs = torch.zeros(N, K, 3, device=DEVICE)
sh_coeffs[:, 0] = (rgbs - 0.5) / C0               # DC 성분 ← SfM 색
sh_coeffs[:, 1:] = torch.randn(N, K - 1, 3, device=DEVICE) * 0.05   # 고차항: 시점 의존성이 보이도록 약간

print(f"Gaussians N={N:,}  cameras C={C}  image {W}x{H}")

render, alpha, meta = rasterization(
    means, quats, scales, opacities, sh_coeffs, viewmats, Ks, W, H,
    sh_degree=SH_DEGREE, packed=False, render_mode="RGB+ED",
)
print("render_colors", tuple(render.shape), " render_alphas", tuple(alpha.shape))
print("meta:")
for k, v in meta.items():
    print(f"  {k:16s}", tuple(v.shape) if torch.is_tensor(v) else v)

# %%
fig, axes = plt.subplots(1, C + 1, figsize=(4 * (C + 1), 3))
for c in range(C):
    axes[c].imshow(render[c, ..., :3].clamp(0, 1).cpu()); axes[c].set_title(f"camera {c} RGB"); axes[c].axis("off")
d = render[0, ..., 3].cpu(); d[alpha[0, ..., 0].cpu() < 0.5] = float("nan")
axes[-1].imshow(d, cmap="turbo"); axes[-1].set_title("camera 0 expected depth"); axes[-1].axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# `meta`에 담긴 것이 바로 ①~⑥단계의 중간 결과다.
#
# | 키 | 모양 | 뜻 |
# |---|---|---|
# | `radii` | [C,N,2] | 화면에서의 x/y 반경(px, int). **0이면 컬링됨**(절두체 밖, 너무 작음, 너무 투명) |
# | `means2d` | [C,N,2] | 투영된 중심(px). 학습 시 이 텐서의 grad가 밀도화(split/duplicate) 기준 |
# | `depths` | [C,N] | 카메라 좌표 z (정렬 키) |
# | `conics` | [C,N,3] | 2D 공분산의 역행렬 Σ₂⁻¹의 (a,b,c) 상삼각 성분 |
# | `opacities` | [C,N] | 카메라별 불투명도(antialiased 모드에서는 보정계수가 곱해짐) |
# | `tiles_per_gauss` | [C,N] | 각 Gaussian이 덮는 타일 수 |
# | `isect_ids`, `flatten_ids` | [n_isects] | 정렬된 (이미지·타일·깊이) 키와 그에 대응하는 Gaussian 인덱스 |
# | `isect_offsets` | [C,tile_h,tile_w] | 타일별로 `flatten_ids`의 시작 위치 |
#
# 이제 이것들을 하나씩 직접 만들어 본다. 수식이 잘 보이도록 **Gaussian 4개짜리 장난감 씬**을 먼저 쓰고,
# 8절에서 garden 씬 전체로 `rasterization()`을 정확히 재현한다.

# %% [markdown]
# ## 1. 장난감 씬
#
# 카메라는 세계 원점에서 +z를 본다(`viewmat = I`). 64×48 이미지, 타일 16px → 4×3 타일.
# 네 번째 Gaussian은 카메라 뒤(z<0)에 두어 컬링되는 것을 확인한다.

# %%
def quat_z(deg: float) -> torch.Tensor:
    """z축 회전 쿼터니언 (w, x, y, z) — gsplat은 w-first 규약."""
    h = math.radians(deg) / 2
    return torch.tensor([math.cos(h), 0.0, 0.0, math.sin(h)])


toy = {
    "means":     torch.tensor([[0.0, 0.0, 3.0], [0.6, 0.3, 4.0], [-0.5, -0.2, 2.5], [0.0, 0.0, -2.0]]),
    "scales":    torch.tensor([[0.30, 0.12, 0.10], [0.25, 0.25, 0.25], [0.08, 0.30, 0.10], [0.2, 0.2, 0.2]]),
    "quats":     torch.stack([quat_z(30), quat_z(0), quat_z(-20), quat_z(0)]),
    "opacities": torch.tensor([0.8, 0.6, 0.9, 1.0]),
    "colors":    torch.tensor([[0.9, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.3, 0.9], [1.0, 1.0, 1.0]]),
}
toy = {k: v.to(DEVICE) for k, v in toy.items()}
tN = toy["means"].shape[0]
tW, tH = 64, 48
tK = torch.tensor([[[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]]], device=DEVICE)   # [1,3,3]
tview = torch.eye(4, device=DEVICE)[None]                                                     # [1,4,4]
TILE = 16
tile_w, tile_h = math.ceil(tW / TILE), math.ceil(tH / TILE)
print(f"toy: N={tN}, image {tW}x{tH}, tiles {tile_w}x{tile_h}")

# %% [markdown]
# ## 2. ① 3D 공분산: Σ = R S Sᵀ Rᵀ
#
# Gaussian의 모양은 공분산 Σ(3×3 대칭 양정치)로 표현되지만, 학습 파라미터는 `quats`(회전 R)와 `scales`(축 길이 s)다.
# Σ를 직접 최적화하면 양정치성이 깨질 수 있어서 **Σ = (R·diag(s))·(R·diag(s))ᵀ** 로 항상 유효한 Σ를 만든다.
# CUDA 커널 안에서는 이 계산이 투영 커널에 융합(fused)되어 있고, 별도 함수 `quat_scale_to_covar_preci`로도 호출할 수 있다.

# %%
def covar_from_quat_scale(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    R = _quat_to_rotmat(q)                 # [N,3,3]  (내부에서 q를 정규화)
    M = R * s[..., None, :]                # R @ diag(s)
    return M @ M.transpose(-1, -2)         # R S Sᵀ Rᵀ


cov_manual = covar_from_quat_scale(toy["quats"], toy["scales"])
cov_cuda, _ = quat_scale_to_covar_preci(toy["quats"], toy["scales"], compute_preci=False, triu=False)
maxdiff("covar  manual vs CUDA", cov_manual, cov_cuda)

print("\nΣ of Gaussian 0 (z축 30° 회전, s=(0.30,0.12,0.10)):\n", cov_manual[0])
print("sqrt(eigvals(Σ)) =", torch.linalg.eigvalsh(cov_manual[0]).sqrt(), " ← scales를 정렬한 것과 같다")

# %% [markdown]
# ## 3. ②③ 카메라 변환 + 원근 투영 (EWA splatting)
#
# 3D Gaussian을 2D Gaussian으로 근사하는 핵심 아이디어:
#
# 1. 카메라 좌표로: μ_c = R μ + t,  Σ_c = R Σ Rᵀ  (viewmat의 회전/이동)
# 2. 원근 투영 π(x,y,z) = (fx·x/z + cx, fy·y/z + cy)는 비선형이므로, **μ_c에서의 1차 근사(Jacobian J)**로 공분산을 밀어낸다.
#
#    $$J = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix},\qquad
#      \Sigma_{2D} = J\,\Sigma_c\,J^\top + \epsilon\,I$$
#
#    `eps2d=0.3`(px²)은 최소 블러다. 이게 없으면 1px보다 작은 Gaussian이 픽셀 중심 사이로 빠져 사라진다.
# 3. 래스터화에서 실제로 쓰는 것은 Σ₂가 아니라 그 **역행렬(conic)** (a,b,c)와 **축정렬 반경** radii = ceil(3.33·√diag(Σ₂)).
# 4. 컬링: z가 near/far 밖, 또는 반경 사각형이 이미지와 안 겹치면 radii=0.
#
# 이 전체가 CUDA 커널 `projection_ewa_3dgs_fused_fwd_kernel`
# ([ProjectionEWA3DGSFused.cu](../gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu)) 하나에서 **Gaussian×카메라당 스레드 1개**로 처리된다.

# %%
def project_manually(means, covars, viewmat, K, W, H, eps2d=0.3, near=0.01, far=1e10):
    """단일 카메라용. _torch_impl._fully_fused_projection의 요지만 남긴 버전."""
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    means_c = means @ R.T + t                              # ② μ_c = R μ + t
    covars_c = R @ covars @ R.T                            #    Σ_c = R Σ Rᵀ
    x, y, z = means_c.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    means2d = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)          # ③ π(μ_c)
    O = torch.zeros_like(z)
    J = torch.stack([fx / z, O, -fx * x / z**2, O, fy / z, -fy * y / z**2], -1).reshape(-1, 2, 3)
    cov2d = J @ covars_c @ J.transpose(-1, -2)
    cov2d = cov2d + eps2d * torch.eye(2, device=means.device)                # 최소 0.3px² 블러
    inv = torch.linalg.inv(cov2d)
    conics = torch.stack([inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1]], dim=-1)  # (a, b, c)
    radii = torch.ceil(3.33 * cov2d.diagonal(dim1=-2, dim2=-1).sqrt())      # 3.33σ 사각 반경

    valid = (z > near) & (z < far)
    inside = ((means2d[:, 0] + radii[:, 0] > 0) & (means2d[:, 0] - radii[:, 0] < W)
              & (means2d[:, 1] + radii[:, 1] > 0) & (means2d[:, 1] - radii[:, 1] < H))
    radii = torch.where((valid & inside)[:, None], radii, torch.zeros_like(radii)).int()
    return radii, means2d, z, conics, cov2d


radii_m, means2d_m, depths_m, conics_m, cov2d_m = project_manually(toy["means"], cov_manual, tview[0], tK[0], tW, tH)

# 순수 PyTorch 참조 (배치 차원 [1,N,...] 로 반환)
radii_t, means2d_t, depths_t, conics_t, _ = _fully_fused_projection(toy["means"], cov_manual, tview, tK, tW, tH)
# CUDA (quats/scales를 직접 넘기면 ①까지 커널 안에서 처리)
radii_c, means2d_c, depths_c, conics_c, _ = fully_fused_projection(
    toy["means"], None, toy["quats"], toy["scales"], tview, tK, tW, tH)

valid = (radii_c[0] > 0).all(-1)
print("radii   manual/torch/CUDA:\n", torch.stack([radii_m, radii_t[0], radii_c[0]], 1).cpu().numpy().tolist())
maxdiff("means2d manual vs CUDA", means2d_m[valid], means2d_c[0][valid])
maxdiff("conics  manual vs CUDA", conics_m[valid], conics_c[0][valid])
maxdiff("depths  manual vs CUDA", depths_m[valid], depths_c[0][valid])
maxdiff("means2d torch  vs CUDA", means2d_t[0][valid], means2d_c[0][valid])
print("\n컬링된 Gaussian(radii=0):", (~valid).nonzero().flatten().tolist(), " ← z=-2 (카메라 뒤)")

# %% [markdown]
# 참고: 참조 구현 `_persp_proj`는 J를 만들기 전에 x/z, y/z를 시야각의 1.3배 안으로 clamp한다(화면 밖 멀리 있는
# Gaussian의 Jacobian이 폭주하는 것을 막는 Inria 원본의 트릭). 장난감 씬은 모두 시야 안이라 결과가 같다.
#
# **불투명도 인지 반경**: `rasterization()` 내부에서는 `opacities`도 투영 커널에 넘긴다. 그러면 반경이
# 3.33σ 대신 `min(3.33, sqrt(2·ln(α / (1/255))))·σ` 로 줄어든다 — 알파가 1/255 아래로 떨어지는 지점 밖은
# 어차피 그리지 않으므로, 투명한 Gaussian은 더 작은 사각형만 타일링하면 된다.

# %%
for name, opac in [("장난감 씬 그대로", toy["opacities"]),
                   ("g0의 opacity를 0.05로", torch.tensor([0.05, 0.6, 0.9, 1.0], device=DEVICE))]:
    radii_op, *_ = fully_fused_projection(
        toy["means"], None, toy["quats"], toy["scales"], tview, tK, tW, tH, opacities=opac)
    ext = torch.sqrt(2 * torch.log(opac / ALPHA_THRESHOLD)).clamp(max=3.33)
    print(f"[{name}] opacity = {opac.tolist()}")
    print(f"   extend(σ 배수) = {[round(e, 2) for e in ext.tolist()]}")
    print(f"   radii 3.33σ    = {radii_c[0].tolist()}")
    print(f"   opacity-aware  = {radii_op[0].tolist()}")

# %%
def draw_splats(ax, means2d, cov2d, radii, colors, W, H, tile, title):
    """투영된 2D Gaussian(1σ, 3σ 타원)과 radii 사각형, 타일 격자를 그린다."""
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.set_title(title)
    for x in range(0, W + 1, tile): ax.axvline(x, color="gray", lw=0.5)
    for y in range(0, H + 1, tile): ax.axhline(y, color="gray", lw=0.5)
    for i in range(means2d.shape[0]):
        if radii[i].min() <= 0:
            continue
        evals, evecs = torch.linalg.eigh(cov2d[i])
        ang = math.degrees(math.atan2(evecs[1, 0].item(), evecs[0, 0].item()))
        c = colors[i].tolist()
        for k, a in [(1, 0.9), (3, 0.4)]:
            ax.add_patch(Ellipse(means2d[i].tolist(), 2 * k * evals[0].sqrt().item(), 2 * k * evals[1].sqrt().item(),
                                 angle=ang, fill=False, color=c, lw=1.5, alpha=a))
        rx, ry = radii[i].tolist()
        ax.add_patch(Rectangle((means2d[i, 0].item() - rx, means2d[i, 1].item() - ry), 2 * rx, 2 * ry,
                               fill=False, color=c, ls="--", lw=0.8))
        ax.text(means2d[i, 0].item(), means2d[i, 1].item(), str(i), color=c, ha="center", va="center", fontsize=9)


fig, ax = plt.subplots(figsize=(6, 4.5))
draw_splats(ax, means2d_m.cpu(), cov2d_m.cpu(), radii_c[0].cpu(), toy["colors"].cpu(), tW, tH, TILE,
            "투영된 Gaussian (실선: 1σ/3σ, 점선: radii 사각형, 격자: 16px 타일)")
plt.show()

# %% [markdown]
# ## 4. ④ 색: Spherical Harmonics 평가
#
# 색이 SH 계수([N,K,3])로 주어지면, Gaussian마다 **카메라에서 본 방향** d = normalize(μ − cam_pos)에 대해
# c = Σₖ Yₖ(d)·coeffₖ 를 계산한다. 카메라 위치는 viewmat에서 cam_pos = −Rᵀt 로 복원한다.
# 그 뒤 `+0.5`(DC 오프셋 — 초기화 때 `(rgb−0.5)/C0`로 뺀 것의 역) 후 0 미만을 clamp한다.
# 장난감 씬은 색을 직접(RGB) 주었으니 garden 씬으로 확인한다.

# %%
R_cam, t_cam = viewmats[:, :3, :3], viewmats[:, :3, 3]
campos = -torch.einsum("cij,ci->cj", R_cam, t_cam)               # −Rᵀ t   [C,3]
dirs = means[None] - campos[:, None]                              # [C,N,3]

sh_torch = _spherical_harmonics(SH_DEGREE, dirs, sh_coeffs)       # [C,N,3]
sh_cuda = spherical_harmonics(SH_DEGREE, means, viewmats, sh_coeffs)
maxdiff("SH torch vs CUDA", sh_torch, sh_cuda)

colors_view = torch.clamp_min(sh_cuda + 0.5, 0.0)                 # rasterize_to_pixels로 들어가는 최종 색
print("카메라별 색 차이(시점 의존성) 평균:", (colors_view[0] - colors_view[1]).abs().mean().item())

# %% [markdown]
# ## 5. ⑤⑥ 타일 교차와 정렬
#
# 픽셀마다 N개 Gaussian을 모두 검사하면 너무 느리다. 대신 화면을 16×16 타일로 나누고,
# **각 Gaussian이 어느 타일들을 덮는지**를 (Gaussian, 타일) 쌍의 리스트로 만든다.
#
# 각 쌍에 64비트 키를 붙인다:
# ```
#  [ image_id | tile_id | float32(depth) 비트 ]      ← 상위 → 하위
# ```
# 이 키를 radix sort하면 **같은 타일의 Gaussian이 연속으로 모이고, 그 안에서는 가까운 것부터** 정렬된다.
# (양수 float의 비트 패턴은 정수로 비교해도 순서가 보존된다.) 그 결과가 `isect_ids`(키)와 `flatten_ids`(Gaussian 인덱스).
#
# CUDA([IntersectTile.cu](../gsplat/cuda/csrc/IntersectTile.cu))는 `intersect_tile_kernel`을 두 번 돈다 —
# 1차: Gaussian별 타일 수 세기 → prefix sum → 2차: 키 채우기 → CUB radix sort.

# %%
# AABB 모드(conics/opacities 없음): radii 사각형이 겹치는 타일 전부 — 순수 PyTorch 참조와 동일한 규칙
tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
    means2d_c, radii_c, depths_c, TILE, tile_w, tile_h, n_images=1)
_tpg, _ids, _fids = _isect_tiles(means2d_c, radii_c, depths_c, TILE, tile_w, tile_h)
print("tiles_per_gauss CUDA :", tiles_per_gauss[0].tolist(), " torch:", _tpg[0].tolist())
print("isect_ids  일치:", torch.equal(isect_ids, _ids), " flatten_ids 일치:", torch.equal(flatten_ids, _fids))
print("n_isects =", isect_ids.numel())

# %%
# 키 해독: 상위 비트 image_id | 중간 tile_id | 하위 32비트 depth
tile_n_bits = (tile_w * tile_h - 1).bit_length()
depth_key = (isect_ids & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
tile_id = (isect_ids >> 32) & ((1 << tile_n_bits) - 1)
image_id = isect_ids >> (32 + tile_n_bits)
gauss_id = flatten_ids.long() % tN

print(f"tile_n_bits={tile_n_bits}   (idx) image  tile(y,x)  depth   gaussian")
for i in range(isect_ids.numel()):
    ty, tx = divmod(tile_id[i].item(), tile_w)
    print(f"  {i:3d}   {image_id[i].item():3d}   ({ty},{tx})     {depth_key[i].item():5.2f}    g{gauss_id[i].item()}")

# %% [markdown]
# `isect_offset_encode`는 정렬된 키에서 타일이 바뀌는 지점을 찾아 **타일별 시작 오프셋**을 만든다.
# 타일 t의 Gaussian 목록은 `flatten_ids[offsets[t] : offsets[t+1]]` (마지막 타일은 n_isects까지).
# 빈 타일은 이전 오프셋을 그대로 물려받아 길이 0이 된다.

# %%
isect_offsets = isect_offset_encode(isect_ids, 1, tile_w, tile_h)          # [1, tile_h, tile_w]
_offsets = _isect_offset_encode(isect_ids, 1, tile_w, tile_h)
print("offsets 일치:", torch.equal(isect_offsets, _offsets))
print("isect_offsets [tile_h, tile_w]:\n", isect_offsets[0].cpu().numpy())
flat = torch.cat([isect_offsets.flatten(), isect_offsets.new_tensor([isect_ids.numel()])])
print("타일별 Gaussian 수:\n", (flat[1:] - flat[:-1]).reshape(tile_h, tile_w).cpu().numpy())

# %% [markdown]
# **AccuTile**: `rasterization()`은 `conics`/`opacities`도 넘겨서, 사각형 대신 **알파가 1/255 이상인 타원**과
# 실제로 겹치는 타일만 고른다. 결과 이미지는 같고(잘린 타일의 기여는 어차피 임계값 아래) 정렬·블렌딩 비용만 줄어든다.

# %%
tpg_accu, isect_ids_accu, flatten_ids_accu = isect_tiles(
    means2d_c, radii_c, depths_c, TILE, tile_w, tile_h, n_images=1,
    conics=conics_c, opacities=toy["opacities"][None])
print("tiles_per_gauss  AABB   :", tiles_per_gauss[0].tolist())
print("tiles_per_gauss  AccuTile:", tpg_accu[0].tolist())

# %% [markdown]
# ## 6. ⑦ 알파 블렌딩 (rasterize_to_pixels)
#
# 이제 픽셀 p의 색은 그 타일의 Gaussian 목록을 **앞→뒤 순으로** 훑으며 다음처럼 누적된다.
#
# $$\sigma_i = \tfrac12(a\,dx^2 + c\,dy^2) + b\,dx\,dy,\quad
#   \alpha_i = \min(0.99,\ o_i e^{-\sigma_i}),\quad
#   C_p = \sum_i c_i\,\alpha_i\,T_i,\quad T_{i+1} = T_i(1-\alpha_i)$$
#
# - dx, dy는 픽셀 **중심**(px+0.5)과 means2d의 차. σ는 마할라노비스 거리의 절반.
# - α < 1/255 (`ALPHA_THRESHOLD`)인 Gaussian은 건너뛴다. α는 0.99 (`MAX_ALPHA`)로 상한.
# - 투과율 T가 1e-4 (`TRANSMITTANCE_THRESHOLD`) 이하가 되면 그 픽셀은 끝(그 Gaussian은 **제외**하고 종료).
# - render_alpha = 1 − T.
# - 임계값들은 Python 쪽 [gsplat/cuda/_constants.py](../gsplat/cuda/_constants.py)와 CUDA 쪽
#   [gsplat/cuda/include/Common.h](../gsplat/cuda/include/Common.h)(`GAUSSIAN_EXTEND=3.33f` 포함)에 같은 값으로 정의되어 있다.
#
# CUDA 커널 `rasterize_to_pixels_3dgs_fwd_kernel<CDIM, TILE=16, CTA=256>`
# ([RasterizeToPixels3DGSSerialBatchFwd.cu](../gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu))의 구조:
#
# ```
# 블록(blockIdx)  = 타일 하나          →  range = [isect_offsets[tile], isect_offsets[tile+1])
# 스레드(tid)     = 타일 안 픽셀 하나   →  T=1, pix_out=0
# for batch in range(0, len(range), 256):
#     각 스레드가 Gaussian 1개씩 shared memory에 적재 (id, xy, opacity, conic)   __syncthreads()
#     for g in batch(앞→뒤):  σ, α 계산 → 건너뛰기/누적/종료 판정
#     __syncthreads_count(done) == 256 이면 타일 전체 조기 종료
# pix_out += background * T ;  render_alpha = 1 - T ;  last_ids = 마지막 기여 Gaussian (backward용)
# ```
# 아래 `rasterize_naive`는 이 커널을 순수 PyTorch로 옮긴 것이다. 타일 루프 = 블록, 타일 안 픽셀 텐서 = 스레드들,
# Gaussian 루프 = 직렬 순회. (실제 `_torch_impl._rasterize_to_pixels`는 nerfacc로 벡터화한 버전이다.)

# %%
def rasterize_naive(means2d, conics, colors, opacities, W, H, tile_size, isect_offsets, flatten_ids):
    """단일 이미지. means2d[N,2] conics[N,3] colors[N,D] opacities[N], isect_offsets[tile_h,tile_w]."""
    th, tw = isect_offsets.shape
    n_isects = flatten_ids.numel()
    offsets = torch.cat([isect_offsets.flatten(), isect_offsets.new_tensor([n_isects])]).tolist()
    D = colors.shape[-1]
    img = torch.zeros(H, W, D, device=means2d.device)
    T_map = torch.ones(H, W, device=means2d.device)
    n_contrib = torch.zeros(H, W, dtype=torch.int32, device=means2d.device)

    for tile in range(th * tw):                                   # ← CUDA: 블록 하나
        ty, tx = divmod(tile, tw)
        y0, y1 = ty * tile_size, min((ty + 1) * tile_size, H)
        x0, x1 = tx * tile_size, min((tx + 1) * tile_size, W)
        py, px = torch.meshgrid(torch.arange(y0, y1, device=img.device) + 0.5,
                                torch.arange(x0, x1, device=img.device) + 0.5, indexing="ij")
        T = torch.ones_like(px)                                    # ← CUDA: 스레드(픽셀)별 레지스터
        out = torch.zeros(px.shape + (D,), device=img.device)
        done = torch.zeros_like(px, dtype=torch.bool)
        cnt = torch.zeros_like(px, dtype=torch.int32)

        for k in range(offsets[tile], offsets[tile + 1]):         # ← 이 타일의 Gaussian, 앞→뒤
            g = flatten_ids[k].item()
            dx, dy = means2d[g, 0] - px, means2d[g, 1] - py
            a, b, c = conics[g]
            sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
            alpha = torch.clamp_max(opacities[g] * torch.exp(-sigma), MAX_ALPHA)
            valid = (sigma >= 0) & (alpha >= ALPHA_THRESHOLD) & ~done
            next_T = T * (1 - alpha)
            saturated = valid & (next_T <= TRANSMITTANCE_THRESHOLD)  # 이 Gaussian은 제외하고 종료
            blend = valid & ~saturated
            done |= saturated
            out += (blend * alpha * T)[..., None] * colors[g]       # c_i · α_i · T_i
            T = torch.where(blend, next_T, T)
            cnt += blend.int()
            if bool(done.all()):
                break                                              # ← __syncthreads_count 조기 종료

        img[y0:y1, x0:x1] = out; T_map[y0:y1, x0:x1] = T; n_contrib[y0:y1, x0:x1] = cnt
    return img, 1.0 - T_map, n_contrib


img_naive, alpha_naive, n_contrib = rasterize_naive(
    means2d_c[0], conics_c[0], toy["colors"], toy["opacities"], tW, tH, TILE, isect_offsets[0], flatten_ids)

img_cuda, alpha_cuda = rasterize_to_pixels(
    means2d_c, conics_c, toy["colors"][None], toy["opacities"][None], tW, tH, TILE, isect_offsets, flatten_ids)
maxdiff("render naive vs CUDA", img_naive, img_cuda[0])
maxdiff("alpha  naive vs CUDA", alpha_naive, alpha_cuda[0, ..., 0])

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
axes[0].imshow(img_naive.cpu()); axes[0].set_title("naive (PyTorch 루프)")
axes[1].imshow(img_cuda[0].cpu()); axes[1].set_title("CUDA rasterize_to_pixels")
im = axes[2].imshow(alpha_cuda[0, ..., 0].cpu(), cmap="gray", vmin=0, vmax=1); axes[2].set_title("render_alpha = 1 − T")
im = axes[3].imshow(n_contrib.cpu(), cmap="viridis"); axes[3].set_title("픽셀별 블렌딩된 Gaussian 수"); plt.colorbar(im, ax=axes[3])
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 7. 전체 파이프라인 재조립 — garden 씬에서 `rasterization()`과 비트 단위 비교
#
# 위 단계를 순서대로 이어 붙인 `rasterize_stepwise`가 C++ 오케스트레이터와 같은 결과를 내는지 확인한다.
# `rasterization()` 내부 순서(Rendering.cpp `rasterization_3dgs`)와 맞추기 위해
# 투영에 `opacities`를 넘기고(불투명도 인지 반경), 타일 교차에 `conics`/`opacities`를 넘긴다(AccuTile).

# %%
def rasterize_stepwise(means, quats, scales, opacities, sh_coeffs, viewmats, Ks, W, H,
                       sh_degree, tile_size=16, eps2d=0.3, near_plane=0.01, far_plane=1e10):
    C, N = viewmats.shape[0], means.shape[0]
    # ①②③ 공분산 + 카메라 변환 + 투영 (fused CUDA 커널)
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, quats, scales, viewmats, Ks, W, H,
        eps2d=eps2d, near_plane=near_plane, far_plane=far_plane, opacities=opacities)
    opac = opacities[None].expand(C, N)                                        # 카메라별 불투명도 [C,N]
    # ⑤⑥ 타일 교차 + 정렬 + 오프셋
    tw, th = math.ceil(W / tile_size), math.ceil(H / tile_size)
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tw, th, n_images=C, conics=conics, opacities=opac)
    isect_offsets = isect_offset_encode(isect_ids, C, tw, th)                  # [C, th, tw]
    # ④ SH → 색  (컬링된 Gaussian은 masks로 건너뜀)
    masks = (radii > 0).all(dim=-1)
    colors = torch.clamp_min(spherical_harmonics(sh_degree, means, viewmats, sh_coeffs, masks=masks) + 0.5, 0.0)
    # ⑦ 알파 블렌딩
    render, alpha = rasterize_to_pixels(means2d, conics, colors, opac, W, H, tile_size, isect_offsets, flatten_ids)
    meta = dict(radii=radii, means2d=means2d, depths=depths, conics=conics, tiles_per_gauss=tiles_per_gauss,
                isect_ids=isect_ids, flatten_ids=flatten_ids, isect_offsets=isect_offsets)
    return render, alpha, meta


with torch.no_grad():
    r_fused, a_fused, m_fused = rasterization(
        means, quats, scales, opacities, sh_coeffs, viewmats, Ks, W, H, sh_degree=SH_DEGREE, packed=False)
    r_step, a_step, m_step = rasterize_stepwise(
        means, quats, scales, opacities, sh_coeffs, viewmats, Ks, W, H, sh_degree=SH_DEGREE)

print("forward:")
maxdiff("render_colors", r_fused, r_step)
maxdiff("render_alphas", a_fused, a_step)
valid = (m_fused["radii"] > 0).all(dim=-1)   # 컬링된(radii=0) 항목은 커널이 early-return해 값을 쓰지 않으므로(torch.empty 쓰레기값) 제외
for k in ["means2d", "depths", "conics"]:
    maxdiff(f"meta[{k}] (radii>0만)", m_fused[k][valid], m_step[k][valid])
for k in ["radii", "tiles_per_gauss", "isect_ids", "flatten_ids", "isect_offsets"]:
    print(f"  meta[{k}] 일치: {torch.equal(m_fused[k], m_step[k])}")
print(f"  n_isects = {m_fused['isect_ids'].numel():,}   가시 Gaussian = {(m_fused['radii'] > 0).all(-1).sum().item():,} / {C * N:,}")

# %% [markdown]
# ### backward — 그래디언트도 같은가
#
# 두 경로 모두 각 단계 CUDA op의 autograd(Function)를 타므로 grad도 같아야 한다. 학습에서 실제로 쓰는 것이
# 이 grad다: `means2d.grad`의 크기(화면 이동량)가 DefaultStrategy의 split/duplicate 기준이 된다.

# %%
params = {"means": means, "quats": quats, "scales": scales, "opacities": opacities, "sh_coeffs": sh_coeffs}
for p in params.values():
    p.requires_grad_(True)
weights = torch.randn_like(r_fused)


def grads_of(fn):
    for p in params.values():
        p.grad = None
    out = fn()
    (out * weights).sum().backward()
    return {k: p.grad.clone() for k, p in params.items()}


g_fused = grads_of(lambda: rasterization(
    means, quats, scales, opacities, sh_coeffs, viewmats, Ks, W, H, sh_degree=SH_DEGREE, packed=False)[0])
g_step = grads_of(lambda: rasterize_stepwise(
    means, quats, scales, opacities, sh_coeffs, viewmats, Ks, W, H, sh_degree=SH_DEGREE)[0])
print("backward:")
for k in params:
    d = maxdiff(f"grad[{k}]", g_fused[k], g_step[k])
    print(f"  {'':28s} (|grad| max = {g_fused[k].abs().max().item():.3e})")
for p in params.values():
    p.requires_grad_(False); p.grad = None

# %%
def bench(fn, n=10):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3


with torch.no_grad():
    t_fused = bench(lambda: rasterization(means, quats, scales, opacities, sh_coeffs, viewmats, Ks, W, H,
                                          sh_degree=SH_DEGREE, packed=False))
    t_step = bench(lambda: rasterize_stepwise(means, quats, scales, opacities, sh_coeffs, viewmats, Ks, W, H,
                                              sh_degree=SH_DEGREE))
print(f"forward 시간 (C={C}, N={N:,}, {W}x{H}):  rasterization() {t_fused:.2f} ms   stepwise {t_step:.2f} ms")
print("→ 계산 커널은 같고, C++ 오케스트레이터는 파이썬 오버헤드·중간 텐서 cat 몇 개를 절약할 뿐이다.")

# %% [markdown]
# ## 8. 타일 부하 시각화
#
# 타일마다 Gaussian 수가 크게 다르다. CUDA 블록 하나가 타일 하나를 맡으므로 이 분포가 곧 **블록별 작업량 불균형**이고,
# 밀도가 높은 씬에서 래스터화 시간이 어디서 나오는지 보여준다.

# %%
th, tw = m_fused["isect_offsets"].shape[-2:]
flat = torch.cat([m_fused["isect_offsets"].flatten(), m_fused["isect_offsets"].new_tensor([m_fused["isect_ids"].numel()])])
per_tile = (flat[1:] - flat[:-1]).reshape(C, th, tw)

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
axes[0].imshow(r_fused[0].clamp(0, 1).cpu()); axes[0].set_title("camera 0"); axes[0].axis("off")
im = axes[1].imshow(per_tile[0].cpu(), cmap="magma"); axes[1].set_title("타일별 Gaussian 수 (isect_offsets 차분)"); plt.colorbar(im, ax=axes[1])
axes[2].hist(m_fused["tiles_per_gauss"][0][m_fused["radii"][0].min(-1).values > 0].cpu().numpy(), bins=30, log=True)
axes[2].set(xlabel="tiles per Gaussian", ylabel="count (log)", title="Gaussian별 덮는 타일 수")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 9. `packed=True` — 가시 Gaussian만 남기는 희소 표현
#
# 기본값 `packed=True`에서는 투영 커널이 `[C,N,...]` 대신 **radii>0인 (카메라, Gaussian) 쌍만** `[nnz,...]`로 반환하고,
# 어느 카메라·어느 Gaussian인지는 `camera_ids`/`gaussian_ids`(COO 인덱스)로 알려 준다. 카메라 수가 많거나
# 씬이 크면 메모리가 크게 줄고, 이후 단계(isect_tiles, rasterize_to_pixels)도 같은 packed 입력을 그대로 받는다.
# `sparse_grad=True`와 조합하면 grad도 희소로 돌아온다.

# %%
with torch.no_grad():
    _, _, meta_p = rasterization(means, quats, scales, opacities, sh_coeffs, viewmats, Ks, W, H,
                                 sh_degree=SH_DEGREE, packed=True)
nnz = meta_p["means2d"].shape[0]
print(f"dense  means2d {tuple(m_fused['means2d'].shape)}  →  packed means2d {tuple(meta_p['means2d'].shape)}  (nnz/CN = {nnz / (C * N):.1%})")
print("camera_ids  [:8] =", meta_p["camera_ids"][:8].tolist())
print("gaussian_ids[:8] =", meta_p["gaussian_ids"][:8].tolist())
print("render 동일:", torch.allclose(
    rasterization(means, quats, scales, opacities, sh_coeffs, viewmats, Ks, W, H, sh_degree=SH_DEGREE, packed=True)[0],
    r_fused, atol=1e-5))

# %% [markdown]
# ## 정리 — 단계별 코드 대응표
#
# | 단계 | Python 래퍼 (`gsplat/cuda/_wrapper.py`) | torch.ops.gsplat | CUDA 소스 / 커널 | 순수 PyTorch 참조 |
# |---|---|---|---|---|
# | ① 공분산 | `quat_scale_to_covar_preci` | `quat_scale_to_covar_preci` | QuatScaleToCovarCUDA.cu | `_math._quat_scale_to_covar_preci` |
# | ②③ 투영 | `fully_fused_projection` | `projection_ewa_3dgs_fused` / `_packed` | ProjectionEWA3DGSFused.cu `projection_ewa_3dgs_fused_fwd_kernel` | `_torch_impl._fully_fused_projection` |
# | ④ SH | `spherical_harmonics` | `spherical_harmonics_fwd` | SphericalHarmonicsCUDA.cu (+ `L1Plus`, `ViewDirection` 변형) | `_torch_impl._spherical_harmonics` |
# | ⑤ 타일 교차 | `isect_tiles` | `intersect_tile` | IntersectTile.cu `intersect_tile_kernel` (2패스) + CUB radix sort | `_torch_impl._isect_tiles` |
# | ⑥ 오프셋 | `isect_offset_encode` | `intersect_offset` | IntersectTile.cu `intersect_offset_kernel` | `_torch_impl._isect_offset_encode` |
# | ⑦ 블렌딩 | `rasterize_to_pixels` | `rasterize_to_pixels_3dgs` | RasterizeToPixels3DGSSerialBatch{Fwd,Bwd}.cu `rasterize_to_pixels_3dgs_fwd_kernel<CDIM,16,256>` | `_torch_impl._rasterize_to_pixels` (nerfacc), 이 노트북의 `rasterize_naive` |
# | 전체 | `rendering.rasterization()` | `rasterization_3dgs` | Rendering.cpp `rasterization_3dgs()` | `rendering._rasterization()` |
#
# **backward 커널의 요지** (RasterizeToPixels3DGSSerialBatchBwd.cu): forward가 저장한 최종 T와 `last_ids`(픽셀별 마지막 기여
# Gaussian)에서 출발해 같은 타일 목록을 **뒤→앞으로** 순회하며 T를 `T /= (1−α)`로 복원하고, ∂L/∂color, ∂L/∂α → ∂L/∂conic,
# ∂L/∂means2d, ∂L/∂opacity를 계산한다. Gaussian 하나에 대한 기여가 여러 픽셀(스레드)에 흩어져 있으므로 warp 단위로
# 합친 뒤 `atomicAdd`로 모은다. `absgrad=True`면 ∂L/∂means2d의 절댓값 합을 따로 누적한다(AbsGS 밀도화 기준).
#
# **여기서 더 볼 것들**
# - `rasterize_mode="antialiased"`: eps2d 블러 전후 행렬식 비 √(det₀/det)를 불투명도에 곱해 화면 크기가 작아질 때 밝기 보존(Mip-Splatting)
# - `with_ut=True`: Jacobian 대신 Unscented Transform으로 투영 — 어안/F-theta/롤링셔터 카메라 지원(3DGUT)
# - `with_eval3d=True`: 2D 근사 없이 픽셀 광선과 3D Gaussian의 응답을 직접 평가(RasterizeToPixelsFromWorld3DGS*.cu)
# - `render_mode="D"/"ED"`, `extra_signals`: 깊이·임의 특징을 색과 같은 채널로 블렌딩. 채널 수는 Config.h의 `GSPLAT_NUM_CHANNELS`에 컴파일된 값만 가능
# - `rasterization_2dgs`: 표면 지향 2D Gaussian(디스크) 버전 — 광선-평면 교차로 σ를 구한다
