# gsplat의 분산 학습은 어떻게 동작하는가

> **답**: `rasterization(distributed=True)`를 사용해 **rank별로 Gaussian을 분할 소유**한다.
> 각 GPU는 자기 몫의 Gaussian만 파라미터/옵티마이저/densification까지 통째로 관리하고,
> 렌더링 시점에만 NCCL 통신 두 번으로 서로의 Gaussian을 빌려 쓴다.

---

## 1. 핵심 아이디어: 데이터 병렬이 아니라 "모델(Gaussian) 병렬"

일반적인 DDP는 **모델 전체를 모든 GPU가 복제**하고 데이터만 나눈다. 3DGS에서 이 방식은
곧바로 막힌다. 대규모 씬은 Gaussian이 수천만 개까지 늘어나고, Gaussian 자체가 학습
파라미터(means/quats/scales/opacities/SH)이므로 복제 비용이 곧 메모리 한계가 된다.

gsplat은 반대로 간다.

| | 카메라(이미지) | Gaussian(파라미터) |
|---|---|---|
| DDP(일반) | rank별 분할 | 전 rank 복제 |
| gsplat `distributed=True` | rank별 분할 | **rank별 분할 소유(sharding)** |

즉 rank $r$은 전체 Gaussian 집합 $G$의 서로 겹치지 않는 부분집합 $G_r$만 들고 있고,
$G = \bigcup_r G_r$ 이다. 어떤 rank도 씬 전체를 메모리에 갖지 않는다. 이 설계는
[On Scaling Up 3D Gaussian Splatting Training (Grendel-GS, arXiv:2406.18533)](https://arxiv.org/abs/2406.18533)
논문을 따른 것이고, `gsplat/rendering.py`의 `rasterization()` docstring이 이 논문을
직접 인용한다.

## 2. 소유권 분할이 실제로 일어나는 지점

초기화 때 단 세 줄이다. `/home/sungwoo/projects/swcho/gsplat/examples/simple_trainer.py`
의 `create_splats_with_optimizers()`:

```python
# Distribute the GSs to different ranks (also works for single rank)
points = points[world_rank::world_size]
rgbs   = rgbs[world_rank::world_size]
scales = scales[world_rank::world_size]
```

SfM 포인트클라우드를 **stride 슬라이싱**으로 잘라 rank마다 다른 부분을 준다.
(공간 분할이 아니라 인덱스 인터리빙이므로, 각 rank의 Gaussian은 씬 전체에 골고루
흩어진다 → 로드 밸런싱에 유리.) 이후 이 rank-로컬 텐서가 그대로
`torch.nn.ParameterDict`가 되고, **옵티마이저도 로컬 파라미터에만 붙는다**.
Gaussian 파라미터에 대한 `all_reduce`는 어디에도 없다 — 복제본이 없으니 동기화할
대상 자체가 없다.

학습률만 전역 배치 크기에 맞춰 보정한다(같은 파일):

```python
BS = batch_size * world_size          # 전역 유효 배치
... lr=lr * math.sqrt(BS) ...         # sqrt scaling rule
```

## 3. 렌더링 한 스텝의 통신: seam 두 개

문제는 명확하다. rank 0이 담당하는 카메라의 이미지를 그리려면 **다른 rank가 소유한
Gaussian**도 그 픽셀에 기여해야 한다. gsplat은 이걸 파이프라인 전체를 뜯어고치지 않고,
정확히 **두 개의 이음새(seam)** 에만 통신을 주입해서 해결한다.
(`/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/csrc/DistributedCollectives.h` 주석)

```
rank r 의 한 스텝
──────────────────────────────────────────────────────────────────
  [로컬 Gaussian G_r]   [로컬 카메라 C_r]
                              │
      Seam A: all-gather ─────┤  모든 rank의 카메라를 모음 → C_global
                              ▼
   project(G_r  ×  C_global)      ← 내 Gaussian을 "모든" 카메라에 투영
                              │      (여기서 나온 메타데이터가 densification용)
      Seam B: all-to-all ─────┤  투영 결과를 "그 카메라를 소유한 rank"로 발송
                              ▼
   tile 정렬 + rasterize(C_r)      ← 내 카메라의 이미지만 합성
                              ▼
        loss(내 이미지 vs 내 GT) → backward
        (역방향은 같은 collective의 reverse: all-to-all→all-gather)
──────────────────────────────────────────────────────────────────
```

### Seam A — 카메라 all-gather (투영 전)

`gather_cameras_for_distributed()`가 rank별 Gaussian 수 `N_world`, 카메라 수
`C_world`와 함께 `viewmats`/`Ks`를 모은다. 그 결과 각 rank는 **자기 Gaussian을 전역
카메라 전부에 투영**한다. `Rendering.cpp`에서 `C`가 잠시 전역 카메라 수로 바뀌는 것이
이 구간이다. 카메라 텐서는 Gaussian에 비해 훨씬 작으므로 all-gather 비용이 싸다.

### Seam B — 투영 결과 all-to-all (타일링 전)

`scatter_projection_for_distributed()`가 투영된 2D 정보
(`radii, means2d, depths, conics, opacities, features`)를 **그 카메라를 소유한 rank로
보낸다**. 여기서 인덱스 재매핑이 같이 일어난다:

- `camera_ids`: 전역 → 수신 rank 기준 로컬
- `gaussian_ids`: 송신 rank 기준 로컬 → 전역

이후 각 rank는 **자기 카메라의 이미지만** 타일 정렬·알파 합성한다. 즉 "누가 무엇을
소유하는가"가 두 축으로 나뉘어 있고, seam B가 Gaussian-주도 레이아웃에서
카메라-주도 레이아웃으로 갈아타는 지점이다.

### 그래디언트

`gsplat/distributed.py`의 `all_gather_tensor_list` / `all_to_all_tensor_list`는
입력이 `requires_grad`면 `torch.distributed.nn.functional`(미분 가능 버전)을 쓴다.
따라서 **다른 rank의 이미지 손실에서 내 Gaussian으로 그래디언트가 자동으로 역류**한다.
rasterization docstring의 표현 그대로 "allows for gradients to flow back to the
Gaussians living in other ranks". 사용자가 손으로 통신을 짤 일이 없다.

## 4. Densification(밀도화)은 완전히 rank-로컬

이게 가장 헷갈리는 부분이다. `DefaultStrategy` / `MCMCStrategy`
(`gsplat/strategy/*.py`)에는 `world_size`, `rank`, `distributed` 문자열이
**단 한 번도 등장하지 않는다**. 분산을 전혀 모른다.

가능한 이유는 `Rendering.cpp`가 반환 메타데이터를 **seam B 이전에** 캡처하기 때문이다:

```cpp
// Record the returned metadata now: the pre-scatter, rank-local projection
// (Gaussian axis = local N; the densification strategy indexes it by gaussian_id).
```

즉 `meta["gaussian_ids"]`, `means2d.grad`(성장 신호)는 **로컬 N 축 기준**으로 돌아온다.
그래서 각 rank는 자기 Gaussian만 보고 독립적으로 split/duplicate/prune 한다.
결과적으로 rank마다 Gaussian 개수가 서로 달라지며(그래서 문서가 "balanced computation을
위해 rank별 개수를 비슷하게 두길 권장하지만 강제하진 않는다"고 말한다),
`N_world`가 매 스텝 다시 all-gather되는 이유도 이것이다.

## 5. 체크포인트와 평가

- 체크포인트: `ckpt_{step}_rank{world_rank}.pt` — **rank 수만큼 파일이 나온다.**
  전체 씬은 이 파일들의 합집합이다. 단일 파일 재구성은 사용자 몫.
- DDP를 쓰는 건 Gaussian이 아니라 **보조 모듈들**이다: `pose_adjust`, `app_module`
  같은 작은 MLP/임베딩은 복제되므로 `DDP(...)`로 감싸고, 저장할 땐 `.module.state_dict()`.
- 지표 집계와 이미지 저장은 `world_rank == 0`에서만. 단, 렌더 호출 자체는 collective를
  포함하므로 **모든 rank가 같은 횟수로 eval 루프를 돌아야 한다**(안 그러면 데드락).
- 뷰어는 분산 학습에서 비활성화된다("Viewer is disabled in distributed training.").

## 6. 실행 방법

`gsplat/distributed.py`의 `cli(fn, cfg)`가 런처다.

```bash
# 단일 노드, 보이는 GPU 수만큼 자동 spawn
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/simple_trainer.py default --data_dir ...
```

- 단일 노드: `torch.cuda.device_count()`만큼 `torch.multiprocessing.spawn`
- 다중 노드: `OMPI_COMM_WORLD_*` 환경변수(MPI/mpirun)를 읽어 rank 배치
- 각 워커는 `nccl` 백엔드로 `init_process_group` → `fn(local_rank, world_rank, world_size, cfg)` → `barrier` → `destroy_process_group`
- 사용자 함수 시그니처는 `def main(local_rank, world_rank, world_size, cfg)` 고정

트레이너 쪽에서 플래그를 켜는 곳은 딱 한 줄이다(`Runner.rasterize_splats`):

```python
distributed=self.world_size > 1,
```

`world_size == 1`이면 자동으로 꺼진다. 반대로 GPU 1장에서 `distributed=True`를 켜도
gather/scatter가 **항등 연산**이 되어 단일 GPU 경로와 수치적으로 동일하다 —
디버깅용으로 이 경로를 그대로 켜볼 수 있게 만든 설계다.

## 7. `distributed=True`의 제약 (Rendering.cpp 검증부)

분산 경로는 클래식 3DGS 파이프라인에만 붙어 있어서 다음을 **명시적으로 거부**한다:

- 배치 차원(`means.dim() != 2`) 불가 — 카메라 배치는 되지만 Gaussian 배치는 안 됨
- `sparse_grad`, `absgrad`, `with_ut`(3DGUT), `with_eval3d`, `return_normals` 불가
- `rays`, rolling shutter, 렌즈 왜곡 계수, LiDAR 계수 불가
- `camera_model="pinhole"`만, `global_z_order=True`만
- 색상은 per-Gaussian `(N, D)` 레이아웃만 (SH는 허용, per-camera 색상은 불가)
- NCCL 기본 프로세스 그룹만 지원 (`_get_default_nccl_process_group_name()`이 검증)
- rank별 **카메라 개수는 반드시 동일**해야 함(Gaussian 개수는 달라도 됨)

또 하나 함정: `simple_trainer.py`는 `DefaultStrategy`일 때 `absgrad`를 넘기므로,
`absgrad=True` 설정과 멀티 GPU를 동시에 쓰면 위 검증에 걸린다.

---

## 한 줄 요약

**Gaussian은 쪼개서 각자 소유하고(파라미터·옵티마이저·densification까지 로컬),
이미지를 그릴 때만 "카메라 all-gather → 투영 → 결과 all-to-all"이라는 두 번의
통신으로 서로의 Gaussian을 빌려 쓴다. 그래디언트는 그 통신을 거꾸로 타고 주인에게
돌아간다.**

### 참고 파일

- `/home/sungwoo/projects/swcho/gsplat/gsplat/rendering.py` — `rasterization(distributed=...)`, `_get_default_nccl_process_group_name()`
- `/home/sungwoo/projects/swcho/gsplat/gsplat/distributed.py` — `all_gather_int32`, `all_to_all_tensor_list`, `cli`
- `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/csrc/DistributedCollectives.h` / `.cpp` — seam A/B 구현
- `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/csrc/Rendering.cpp` — 오케스트레이터와 제약 검증
- `/home/sungwoo/projects/swcho/gsplat/examples/simple_trainer.py` — Gaussian 분할, lr 스케일, rank별 ckpt
- `/home/sungwoo/projects/swcho/gsplat/tests/_test_distributed.py` — collective 유닛 테스트
