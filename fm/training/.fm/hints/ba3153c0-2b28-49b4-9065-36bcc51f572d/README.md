# `packed=True` — 가시 Gaussian만 sparse하게

**Q.** `packed=True` 옵션의 이점과 함께 쓰는 기능은?
**A.** 가시 Gaussian만 sparse하게 유지해 대규모 씬의 메모리를 절감한다. `sparse_grad`, `SelectiveAdam`과 조합해 쓴다.

---

## 1. 무엇이 "packed" 되는가

`rasterization()`(`gsplat/rendering.py:234`)의 내부 4단계 중 **투영(projection)** 단계는
카메라 C개 × Gaussian N개에 대한 중간 결과(`means2d`, `depths`, `conics`, `radii`, SH로 평가한 색 …)를 만든다.

| 모드 | 중간 텐서 shape | 의미 |
|---|---|---|
| `packed=False` (dense) | `[..., C, N, ...]` | **모든** 카메라-Gaussian 쌍을 다 담는다. 화면 밖 Gaussian은 `radii=0`으로 마킹만 될 뿐 자리는 그대로 차지 |
| `packed=True` (sparse) | `[..., nnz, ...]` | 컬링을 통과한 **가시 쌍만** 1차원으로 압축(pack). `nnz` = 실제로 보이는 (camera, gaussian) 쌍의 개수 |

즉 packed 모드는 "C×N 짜리 조밀한 표를 만들고 대부분을 0으로 두는" 대신
"살아남은 항목만 나열한 리스트"를 만든다. 희소 행렬의 dense vs COO 표현과 정확히 같은 트레이드오프다.

packed 모드에서는 `info`(meta) dict에 **`gaussian_ids`, `camera_ids`, `batch_ids`** (각각 `[nnz]`)가 채워지고,
`packed=False`에서는 이 세 키가 `None`이 된다 (`gsplat/rendering.py:655`).
이 인덱스 배열이 "압축된 행 i가 원래 어느 Gaussian/카메라였는지"를 되짚는 열쇠이고,
아래 `sparse_grad`/`SelectiveAdam` 조합이 전부 이 배열 위에 세워진다.

## 2. 이점: 메모리 ↔ 속도 트레이드오프

gsplat 공식 문서(docstring, `rendering.py:348`)의 표현 그대로:

> **Memory-Speed Trade-off**: If `packed` is True, the intermediate results are packed into sparse
> tensors, which is more memory efficient but **might be slightly slower**. This is especially helpful
> when the scene is large and **each camera sees only a small portion of the scene**.

- **이점**: 메모리 절감. 절감 폭은 가시율 `nnz / (C·N)`에 비례한다.
  수백만 개 Gaussian을 가진 대규모 씬에서 한 뷰가 보는 비율이 5~10%라면 중간 텐서는 10~20배 작아진다.
- **비용**: 약간의 속도 저하. 압축/인덱싱(스캔·컴팩션) 오버헤드와 비정렬(uncoalesced) 메모리 접근 때문이다.
  씬이 작고 카메라가 거의 모든 Gaussian을 보는 상황이라면 packed는 이득 없이 손해만 본다.
- 워크스루가 `packed=False`를 쓴 이유도 이것이다: 작은 씬 + 밀도화 상태 갱신 코드를 dense `[C, N]` 형태에 맞춰 놓았기 때문
  (`training_walkthrough.py`의 `rasterize_splats()`와 `strategy.step_post_backward(..., packed=False)`).

## 3. 함께 쓰는 기능 ①: `sparse_grad=True`

문서(`rendering.py:355`) 요약: `sparse_grad=True`면 **`means`, `quats`, `scales`의 gradient가 COO sparse layout**으로 저장된다.
그리고 결정적으로 —

> This argument is **only effective when `packed` is True**.

`packed=True`가 선행 조건인 이유는 간단하다. "어느 Gaussian이 이번 스텝에 gradient를 받았는가"의 인덱스,
즉 `info["gaussian_ids"]`가 packed 모드에서만 존재하기 때문이다.
`examples/simple_trainer.py:1108`은 이를 assert로 못박는다.

```python
if cfg.sparse_grad:
    assert cfg.packed, "Sparse gradients only work with packed mode."
    gaussian_ids = info["gaussian_ids"]
    for k in self.splats.keys():
        grad = self.splats[k].grad
        if grad is None or grad.is_sparse:
            continue
        self.splats[k].grad = torch.sparse_coo_tensor(
            indices=gaussian_ids[None],   # [1, nnz]
            values=grad[gaussian_ids],    # [nnz, ...]
            size=self.splats[k].size(),   # [N, ...]
            is_coalesced=len(Ks) == 1,
        )
```

dense gradient `[N, ...]`를 보이는 항목만 뽑아 `[nnz, ...]` COO 텐서로 바꿔 끼운다.
sparse gradient는 일반 `Adam`이 먹지 못하므로 옵티마이저도 갈아끼워야 한다(`simple_trainer.py:362`).

```python
if sparse_grad:
    optimizer_class = torch.optim.SparseAdam   # sparse gradient 전용
elif visible_adam:
    optimizer_class = SelectiveAdam            # Taming-3DGS의 visible Adam
else:
    optimizer_class = torch.optim.Adam
```

CLI로는 `--packed --sparse_grad`. gsplat은 이 조합을 **experimental**로 표시한다
(`# Use sparse gradients for optimization. (experimental)`, `simple_trainer.py:184`).

## 4. 함께 쓰는 기능 ②: `SelectiveAdam` (`--visible_adam`)

`SelectiveAdam`(`gsplat/optimizers/selective_adam.py:21`)은 `torch.optim.Adam`을 상속하되
**visibility 마스크를 인자로 받는 `step(visibility)`** 를 갖는다.
보이지 않은 Gaussian은 모멘텀 `exp_avg`/`exp_avg_sq` 갱신과 파라미터 업데이트를 통째로 건너뛰고,
연산 전체가 CUDA 커널 하나(`gsplat.cuda._wrapper.adam`)로 fuse되어 있다.
Taming 3DGS 논문에서 제안된 두 옵티마이저 중 하나다.

마스크를 만드는 방법이 바로 packed 여부에 따라 갈린다(`simple_trainer.py:1122`).

```python
if cfg.visible_adam:
    if cfg.packed:
        visibility_mask = torch.zeros_like(self.splats["opacities"], dtype=bool)
        visibility_mask.scatter_(0, info["gaussian_ids"], 1)   # nnz 인덱스를 그대로 마스크로
    else:
        visibility_mask = (info["radii"] > 0).all(-1).any(0)   # [C,N] radii에서 유도
...
for optimizer in self.optimizers.values():
    optimizer.step(visibility_mask) if cfg.visible_adam else optimizer.step()
```

`SelectiveAdam`은 packed가 **필수는 아니다**(dense에서는 `radii>0`으로 마스크를 만들 수 있다).
다만 packed일 때 `gaussian_ids` → `scatter_` 한 줄이면 끝나고, dense `[C,N]` 마스크를 만들 필요도 없어 자연스럽게 궁합이 맞는다.
`sparse_grad`와 `visible_adam`은 **둘 중 하나만** 고른다(위 if/elif — 옵티마이저 클래스가 배타적이다).

## 5. 밀도화 전략도 packed를 알아야 한다

`DefaultStrategy.step_post_backward(..., packed=...)`는 `means2d.grad`를 Gaussian별로 누적하는데,
그 텐서의 모양이 모드에 따라 다르므로 플래그를 그대로 전달해야 한다(`gsplat/strategy/default.py:263`).

```python
if packed:
    gs_ids = info["gaussian_ids"]                    # [nnz] — 이미 압축돼 있음
    radii  = info["radii"].max(dim=-1).values        # [nnz]
else:
    sel    = (info["radii"] > 0.0).all(dim=-1)       # [C, N]
    gs_ids = torch.where(sel)[1]                     # [nnz]
    grads  = grads[sel]                              # [C,N,2] → [nnz,2]
    radii  = info["radii"][sel].max(dim=-1).values
state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
```

dense 경로는 결국 `sel` 마스크로 직접 압축한다 — **packed는 이 압축을 CUDA 커널 안으로 앞당겨,
`[C,N]` 중간 텐서를 애초에 만들지 않는 것**이라고 보면 정확하다.
워크스루가 `packed=False`와 `step_post_backward(..., packed=False)`를 짝지어 둔 이유가 여기 있다. 둘이 어긋나면 shape이 깨진다.

## 6. 언제 켜는가 — 실전 요약

```bash
# 대규모 씬 + VRAM 부족: 메모리 우선
python simple_trainer.py default --packed ...
# 더 공격적으로 (실험적)
python simple_trainer.py default --packed --sparse_grad ...
# 속도까지 노릴 때 (Taming 3DGS 계열, sparse_grad와 배타)
python simple_trainer.py default --packed --visible_adam ...
```

- **켠다**: Gaussian 수가 수백만, 카메라 하나가 씬의 일부만 보는 대규모/야외 씬, VRAM이 빠듯할 때.
- **끄는 게 낫다**: 소규모 오브젝트 씬, 대부분의 Gaussian이 모든 뷰에 보이는 경우 — 압축 오버헤드만 남는다.
- **같이 보면 좋은 이웃 옵션**: `radius_clip`(너무 작은 Gaussian 조기 스킵으로 속도↑), `distributed=True`(랭크별 Gaussian 분할 소유).
  셋 다 "전부를 다루지 말고 필요한 것만 다루자"는 같은 아이디어의 다른 층위다.

## 한 줄 정리

`packed=True` = 중간 결과를 `[C, N]` dense가 아니라 **가시 쌍 `[nnz]` 만 담은 sparse 표현**으로 →
메모리 절감(약간 느려짐), 그리고 그 부산물인 `gaussian_ids` 덕분에
**`sparse_grad`(+`SparseAdam`)** 와 **`SelectiveAdam`(`--visible_adam`)** 이 붙을 수 있다.
