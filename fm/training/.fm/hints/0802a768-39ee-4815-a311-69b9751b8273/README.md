# `packed=False`로 두는 이유

**Q.** 이 노트북에서 `packed=False`로 두는 이유는?

**A.** 밀도화 상태 갱신 코드와 맞추기 위해서다. `packed=True`는 가시 Gaussian만 sparse하게 유지하는 다른 표현을 쓴다.

---

## 1. 노트북에서 `packed`가 등장하는 두 곳

`training_walkthrough.py`에는 `packed`가 **한 쌍**으로 나온다. 이게 핵심이다.

```python
# 3단계: rasterize_splats() 안 — 렌더러 쪽
return rasterization(
    means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
    viewmats=torch.linalg.inv(camtoworlds),
    Ks=Ks, width=width, height=height,
    sh_degree=sh_degree,
    packed=False,                                # 밀도화 상태 갱신 코드와 맞춤
    **kwargs,
)

# 학습 루프 (7) — 밀도화 전략 쪽
strategy.step_post_backward(
    splats, optimizers, strategy_state, step, info, packed=False
)
```

`rasterization()`이 `info`(meta) dict를 만들고, `DefaultStrategy.step_post_backward()`가 그 dict를 **읽어서** 해석한다. 두 호출의 `packed` 값이 **같아야** 해석이 맞는다. 즉 `packed=False`는 "성능상 더 좋아서"가 아니라 **두 코드의 텐서 레이아웃 계약을 일치시키는 스위치**다.

## 2. `packed`가 실제로 바꾸는 것: dense [C,N,…] vs sparse [nnz,…]

`gsplat/rendering.py`의 `rasterization()` docstring이 명시한 트레이드오프:

> **Memory-Speed Trade-off**: If `packed` is True, the intermediate results are packed into sparse tensors, which is more memory efficient but might be slightly slower. … If `packed` is False, the intermediate results are with shape `[..., C, N, ...]`, which is faster but might consume more memory.

| | `packed=False` (dense) | `packed=True` (sparse/packed) |
|---|---|---|
| `info["means2d"]` | `[C, N, 2]` — 카메라 × 전체 Gaussian | `[nnz, 2]` — 실제로 보이는 (카메라, Gaussian) 쌍만 |
| `info["radii"]` | `[C, N, 2]`, 안 보이면 `0` | `[nnz, 2]`, 보이는 것만 |
| `info["gaussian_ids"]` | **`None`** | `[nnz]` — 각 행이 몇 번 Gaussian인지 |
| `info["camera_ids"]` / `batch_ids` | **`None`** | `[nnz]` |
| 메모리 | C×N 전체를 잡음 | 가시 항목만 |
| 컬링된 Gaussian 표현 | 자리는 남고 `radii=0` | 애초에 행이 없음 |

`rendering.py`에서 dense일 때 id 텐서를 명시적으로 지워버린다:

```python
if not packed:
    batch_ids = None
    camera_ids = None
    gaussian_ids = None
```

카메라 하나가 씬의 일부만 보는 대규모 씬(예: 도시 스케일, 수백만 Gaussian)에서는 `[C,N,...]`이 대부분 0인 낭비이므로 `packed=True`가 메모리를 크게 아낀다. 대신 인덱싱/scatter가 늘어 조금 느려질 수 있다.

## 3. 밀도화 상태 갱신 코드가 왜 이 값을 알아야 하나

`gsplat/strategy/default.py`의 `DefaultStrategy._update_state()`가 바로 그 "밀도화 상태 갱신 코드"다. Gaussian별 누적 통계 `state["grad2d"]`, `state["count"]`, `state["radii"]`를 갱신하는데, **`packed`에 따라 완전히 다른 분기**를 탄다.

```python
# 업데이트 대상 인덱스를 뽑는 부분
if packed:
    # grads is [nnz, 2]
    gs_ids = info["gaussian_ids"]                     # [nnz]  ← 렌더러가 준 id를 그대로 사용
    radii = info["radii"].max(dim=-1).values          # [nnz]
else:
    # grads is [C, N, 2]
    sel = (info["radii"] > 0.0).all(dim=-1)           # [C, N]  ← 가시 마스크를 직접 만든다
    gs_ids = torch.where(sel)[1]                      # [nnz]
    grads = grads[sel]                                # [nnz, 2]
    radii = info["radii"][sel].max(dim=-1).values     # [nnz]

state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
```

- **packed 경로**: 렌더러가 이미 "보이는 것만" 골라 놓았고 `gaussian_ids`로 원래 인덱스를 알려준다. 전략은 그걸 신뢰해서 `index_add_`한다.
- **dense 경로**: 모든 (카메라, Gaussian) 칸이 존재하므로 전략이 스스로 `radii > 0`로 가시성을 판정하고, `torch.where(sel)[1]`로 **N축 인덱스**를 뽑아 Gaussian id를 복원한다.

두 경로 모두 결과적으로 `[nnz]`로 압축해 같은 통계를 만들지만, **입력 텐서의 shape 가정이 정반대**다. 이 누적된 `grad2d/count`가 `_grow_gs()`에서 `grads = state["grad2d"] / count.clamp_min(1)` → `grads > grow_grad2d` 판정으로 duplicate/split 대상을 고르고, `_prune_gs()`가 opacity·scale로 prune하니, 여기서 잘못 채워지면 밀도화 전체가 망가진다.

## 4. 값이 어긋나면 무슨 일이 생기나

노트북처럼 두 곳을 다 `False`로 맞춰 두면 안전하다. 어긋나면:

- **렌더러 `packed=True` + 전략 `packed=False`**: 전략이 `[C,N,2]`를 기대하고 `[nnz,2]`에 `.all(dim=-1)` 마스크를 씌운 뒤 `torch.where(sel)[1]`을 하므로 IndexError/차원 오류, 혹은 (운 나쁘게 통과하면) **의미 없는 인덱스**로 통계를 누적한다.
- **렌더러 `packed=False` + 전략 `packed=True`**: `info["gaussian_ids"]`가 `None`이므로 `index_add_`에서 즉시 터진다. (`_update_state` 앞의 assert는 키 **존재**만 확인하고 `None` 여부는 보지 않는다 — 조용히 지나간 뒤 뒤에서 죽는다.)

노트북에서 `packed`에 의존하는 곳이 하나 더 있다. 3단계의 가시 Gaussian 카운트 출력이다.

```python
print(f"화면에 보이는 Gaussian: {(info['radii'] > 0).all(-1).sum().item():,} / {len(splats['means']):,}")
```

이건 `radii`가 `[C,N,2]`라는 dense 가정을 깔고 있다. `packed=True`면 `radii`가 이미 가시분만 담은 `[nnz,2]`라서 이 식은 "전체 대비 가시 비율"이 아니라 그냥 `nnz`를 세는 꼴이 되어 출력의 의미가 사라진다. 즉 노트북의 **설명용 진단 코드까지** dense 레이아웃에 묶여 있다.

## 5. `rasterization()`의 기본값은 `True`라서 명시가 필요하다

놓치기 쉬운 점: `gsplat/rendering.py`의 시그니처 기본값은 dense가 아니다.

```python
def rasterization(
    ...
    packed: bool = True,      # rendering.py:251 — 렌더러 기본은 packed!
```

반면 전략 쪽 기본값은 `False`다.

```python
def step_post_backward(self, params, optimizers, state, step, info,
                       packed: bool = False, scene=None):   # strategy/default.py:179
```

그래서 `rasterization()` 호출에서 `packed`를 **생략하면** 렌더러는 packed, 전략은 dense로 갈려 4절의 오류가 난다. 노트북의 `packed=False,  # 밀도화 상태 갱신 코드와 맞춤` 주석은 "기본값을 일부러 뒤집어 전략의 기본값에 맞췄다"는 뜻이다. `examples/simple_trainer.py`도 같은 선택을 config 기본값으로 못박아 둔다.

```python
# Use packed mode for rasterization, this leads to less memory usage but slightly slower.
packed: bool = False        # simple_trainer.py:183
```

그리고 실제 호출에서 두 곳 모두 **같은 `cfg.packed`를 전달**해 짝을 유지한다 (`rasterize_splats`의 `packed=self.cfg.packed`, 밀도화 훅의 `packed=cfg.packed`). 즉 원본 트레이너는 "한 스위치, 두 소비자" 구조로 짝을 강제하고, 노트북은 학습용으로 단순화하면서 양쪽을 리터럴 `False`로 고정한 것이다.

## 6. `packed=True`로 가면 함께 열리는 것들 / 지켜야 할 것들

노트북 마지막 "여기서 더 볼 것들"이 이 방향을 가리킨다.

> `packed=True`: 가시 Gaussian만 sparse하게 유지 → 대규모 씬 메모리 절감 (`sparse_grad`, `SelectiveAdam`과 조합)

`gaussian_ids`가 생기면 그걸 활용하는 최적화가 붙는다.

```python
# sparse_grad: packed 전제 — grad를 COO sparse로 바꿔 SparseAdam에 넘긴다
if cfg.sparse_grad:
    assert cfg.packed, "Sparse gradients only work with packed mode."
    gaussian_ids = info["gaussian_ids"]
    ...
    self.splats[k].grad = torch.sparse_coo_tensor(
        indices=gaussian_ids[None], values=grad[gaussian_ids], size=self.splats[k].size(), ...)

# visible_adam(SelectiveAdam): 가시 마스크를 만드는 방식도 packed에 따라 갈린다
if cfg.visible_adam:
    if cfg.packed:
        visibility_mask = torch.zeros_like(self.splats["opacities"], dtype=bool)
        visibility_mask.scatter_(0, info["gaussian_ids"], 1)
    else:
        visibility_mask = (info["radii"] > 0).all(-1).any(0)
```

`sparse_grad`는 docstring에도 "This argument is only effective when `packed` is True"로 명시돼 있고, 트레이너는 `assert`로 강제한다. 정리하면 `packed`는 단독 성능 플래그가 아니라 **`info`의 표현 규약**이고, 그 규약을 읽는 소비자(밀도화 전략, sparse 옵티마이저, 진단 출력)가 전부 같은 값을 봐야 한다. 학습용 워크스루에서는 dense 쪽이 코드가 직관적이고(`[C,N,...]` shape로 바로 이해됨) 전략 기본값과도 맞아서 `packed=False`가 자연스러운 선택이다.

---

### 한 줄 요약
`packed`는 `rasterization()`이 돌려주는 `info`의 레이아웃(`[C,N,…]` dense + `gaussian_ids=None` ↔ `[nnz,…]` sparse + `gaussian_ids` 제공)을 결정하고, `DefaultStrategy._update_state()`가 그 레이아웃에 따라 분기하므로 렌더러와 밀도화 훅의 `packed`는 반드시 짝이 맞아야 한다. `rasterization()`의 기본값이 `True`인 탓에 dense 경로를 쓰려면 노트북처럼 `packed=False`를 **명시**해야 한다.

### 관련 소스
- `gsplat/rendering.py:234` `rasterization()` — 기본값 `packed=True`, `if not packed: gaussian_ids = None`
- `gsplat/strategy/default.py:179` `step_post_backward(..., packed=False)` / `:263` `_update_state()`의 packed 분기
- `examples/simple_trainer.py:183` `cfg.packed`, `:732`/`:1162` 렌더러·전략에 같은 값 전달, `:1109` `sparse_grad` assert
