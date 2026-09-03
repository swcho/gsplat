# `zero_grad(set_to_none=True)`를 쓰는 이유

> **Q.** `zero_grad(set_to_none=True)`를 쓰는 이유는?
> **A.** gradient 텐서를 0으로 채우는 대신 `None`으로 만들어 메모리와 연산을 아낀다. PyTorch의 권장 기본 동작이다.

---

## 1. 두 가지 초기화 방식

PyTorch에서 `loss.backward()`는 gradient를 **덮어쓰지 않고 누적**한다(`p.grad += ...`). 그래서 매 스텝 시작 전에 이전 스텝의 gradient를 치워야 하는데, 방법이 둘이다.

| | `set_to_none=False` (옛 기본값) | `set_to_none=True` (현재 기본값) |
|---|---|---|
| 하는 일 | `p.grad.zero_()` — 기존 버퍼를 0으로 **채운다** | `p.grad = None` — 버퍼 참조를 **끊는다** |
| 커널 | 파라미터마다 `memset` 커널 1회 launch | 커널 없음 (파이썬 attribute 대입) |
| 메모리 | gradient 버퍼가 학습 내내 상주 | 버퍼가 해제되어 allocator 풀로 반환 |
| 다음 backward | 상주 버퍼에 `+=` 누적 | 새 텐서를 할당해 **대입** (누적 아님) |

`None`이 된 상태에서 다음 backward가 오면 autograd가 grad 텐서를 새로 만들어 그냥 붙인다. 결과는 수학적으로 동일하고, 대신 "0으로 채우는 작업" 자체가 통째로 사라진다.

---

## 2. 무엇을 아끼는가

### (a) 연산 — 쓸모없는 memset 커널 제거

0으로 채우기는 gradient 텐서 전체를 훑는 GPU 커널이다. 이 값들은 다음 backward에서 **어차피 전부 새 값으로 덮인다**(첫 누적이 `0 + g` 이므로). 즉 0으로 채우는 순간의 연산은 결과에 아무 기여를 하지 않는다.

gsplat 학습 루프는 파라미터 그룹마다 별도 optimizer를 두므로(`means / scales / quats / opacities / sh0 / shN` 6개) 스텝당 6번의 memset이 나간다. 기본 30,000 스텝이면 **18만 번의 불필요한 커널 launch**다. 커널 하나하나는 짧지만, launch 오버헤드 + 메모리 대역폭 소모가 누적된다.

### (b) 메모리 — 피크 사용량 감소

`None`이 되면 gradient 텐서의 참조 카운트가 0이 되어 PyTorch caching allocator가 그 블록을 회수한다. 다음 backward 전까지 그 메모리를 **다른 용도로 재사용**할 수 있다는 뜻이다.

3DGS에서 이 크기는 무시할 수 없다. Gaussian 100만 개(fp32) 기준 gradient 총량:

| 파라미터 | shape | 크기 |
|---|---|---|
| `means` | `[N,3]` | 12 MB |
| `scales` | `[N,3]` | 12 MB |
| `quats` | `[N,4]` | 16 MB |
| `opacities` | `[N]` | 4 MB |
| `sh0` | `[N,1,3]` | 12 MB |
| `shN` | `[N,15,3]` | **180 MB** |
| **합계** | | **≈ 236 MB** |

고차 SH 계수(`shN`)가 압도적이다. 그리고 gradient가 살아 있어도 되는 구간은 `backward()` ~ `optimizer.step()` 사이뿐인데, 그 직후에 오는 것이 하필 **densification**이다.

```python
# training_walkthrough.py:378-390
loss.backward()

# (5) 파라미터별 Adam
for opt in optimizers.values():
    opt.step()
    opt.zero_grad(set_to_none=True)   # ← 여기서 236 MB가 풀려난다
# (6) means lr 감쇠
means_lr_scheduler.step()

# (7) 밀도화: duplicate / split / prune / opacity reset
strategy.step_post_backward(...)      # ← 여기서 새 파라미터 텐서를 크게 할당한다
```

`duplicate`/`split`은 파라미터를 `torch.cat`으로 늘리고 Adam의 `exp_avg`/`exp_avg_sq` 상태까지 새로 만든다(`gsplat/strategy/ops.py`의 `_update_param_with_optimizer`). 즉 **가장 메모리가 빠듯한 순간 직전에** gradient 버퍼를 비워 두는 것이라, 단순한 "조금 아낀다"가 아니라 OOM 여부를 가르는 차이가 될 수 있다.

### (c) 의미 — optimizer가 해당 파라미터를 건너뛴다

`p.grad is None`인 파라미터는 optimizer가 아예 **스킵**한다. gradient가 0으로 채워져 있으면 스킵되지 않고 update 식이 그대로 실행되는데, 이때 옵티마이저에 따라 결과가 달라진다.

- **SGD + momentum**: 0인 gradient로도 momentum 버퍼가 계속 감쇠하며 파라미터를 밀어낸다.
- **weight decay**: gradient가 0이어도 파라미터를 계속 줄인다.
- **Adam**: 0 gradient가 `exp_avg`/`exp_avg_sq`에 섞여 들어가 통계를 희석한다.

즉 `set_to_none=True`는 "이번 스텝에 gradient를 받지 않은 파라미터는 건드리지 않는다"는 더 자연스러운 의미론을 준다. 3DGS에서는 이 스텝에 화면에 보이지 않은 Gaussian이 늘 존재하므로(`radii == 0`으로 컬링) 이 성질이 유의미하다. 같은 문제의식을 더 밀어붙인 것이 gsplat의 `SelectiveAdam`(`--visible-adam`)으로, 아예 가시성 마스크를 넘겨 보이는 Gaussian만 업데이트한다.

---

## 3. 실제 코드 위치

**워크스루** (`.fm/assets/training_walkthrough.py:381`)

```python
for opt in optimizers.values():
    opt.step()
    opt.zero_grad(set_to_none=True)
```

**실제 트레이너** (`examples/simple_trainer.py:1132-1147`) — 파라미터 optimizer뿐 아니라 camera pose / appearance / post-processing optimizer까지 **전부** 같은 패턴이다.

```python
for optimizer in self.optimizers.values():
    if cfg.visible_adam:
        optimizer.step(visibility_mask)
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
for optimizer in self.pose_optimizers:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
# ... app_optimizers, post_processing_optimizers 동일
```

---

## 4. 자주 걸리는 함정

### (a) `.grad`가 `None`일 수 있다 — 접근 전에 가드

`zero_grad()` 직후 `p.grad.norm()` 같은 코드를 쓰면 `AttributeError: 'NoneType' object has no attribute ...`가 난다. gsplat도 gradient를 직접 만지는 곳에서는 명시적으로 방어한다 (`examples/simple_trainer.py:1108-1114`):

```python
if cfg.sparse_grad:
    for k in self.splats.keys():
        grad = self.splats[k].grad
        if grad is None or grad.is_sparse:   # ← None 가드
            continue
        ...
```

### (b) densification은 `means2d.grad`를 쓰는데, 순서가 괜찮은가?

괜찮다. 그리고 그 이유가 이 카드의 핵심과 맞닿아 있다.

`DefaultStrategy`는 화면공간 gradient로 duplicate/split을 결정한다 (`gsplat/strategy/default.py:243-247`).

```python
grads = info["means2d"].grad.clone()   # 또는 .absgrad
```

호출 순서는 `backward()` → `opt.step()` → `zero_grad(set_to_none=True)` → `step_post_backward()`이므로 겉보기엔 "grad를 지우고 나서 grad를 읽는" 것처럼 보인다. 하지만 **`info["means2d"]`는 어떤 optimizer의 `param_groups`에도 들어 있지 않은 비-leaf 중간 텐서**다. `optimizer.zero_grad()`는 자기 param_group에 등록된 텐서만 건드리므로 `means2d.grad`는 손상되지 않는다. 이 텐서에 grad가 남아 있는 것 자체는 `step_pre_backward()`가 미리 불러 둔 `retain_grad()` 덕분이다 (`default.py:170`).

정리하면 두 종류의 gradient가 각자 다른 생명주기를 갖는다.

| | leaf 파라미터 grad (`splats[k].grad`) | `info["means2d"].grad` |
|---|---|---|
| 살아남게 하는 것 | leaf라서 기본 저장 | `retain_grad()` 명시 호출 |
| 소비자 | Adam `step()` | `step_post_backward()`의 밀도화 통계 |
| 언제 사라지나 | `zero_grad(set_to_none=True)` | `info` dict가 다음 스텝에 교체될 때 GC |

### (c) gradient accumulation을 쓴다면 호출 위치가 다르다

여러 미니배치를 모아 한 번에 update할 때는 `zero_grad`를 **누적 구간의 시작에서만** 불러야 한다. 매 backward마다 부르면 누적이 사라진다. gsplat 기본 루프는 스텝당 이미지 1장이라 이 문제와는 무관하다.

### (d) `None`인 상태로 저장/로드

체크포인트에 `state_dict()`를 저장할 때 gradient는 애초에 포함되지 않으므로 영향 없다.

---

## 5. PyTorch 버전 히스토리 (왜 "권장 기본 동작"인가)

- **~1.6**: `zero_grad()`는 `p.grad.zero_()`뿐. 성능을 신경 쓰는 코드는 수동으로 `p.grad = None`을 대입했다.
- **1.7**: `set_to_none` 인자 추가. 기본값은 하위 호환을 위해 `False`.
- **2.0**: 기본값이 `True`로 **변경**. 공식 튜닝 가이드(*Performance Tuning Guide* — "Use `zero_grad(set_to_none=True)` instead of `zero_grad()`")가 권장하던 것이 표준이 됐다.

따라서 PyTorch 2.x에서 `zero_grad(set_to_none=True)`는 기본값과 동일하다. 그럼에도 gsplat이 인자를 명시하는 이유는 (1) 1.x에서 돌려도 같은 동작을 보장하고, (2) "여기서 gradient 버퍼를 놓아 준다"는 의도를 코드에 남기기 위해서다.

---

## 6. 한 문단 요약

`zero_grad()`가 gradient를 0으로 채우는 것은, 어차피 다음 backward가 전부 덮어쓸 값을 위해 GPU 커널을 돌리고 큰 버퍼를 계속 붙들고 있는 낭비다. `set_to_none=True`는 그 버퍼를 그냥 놓아 버려 (a) memset 커널을 없애고, (b) 메모리를 allocator에 반환하고, (c) gradient를 받지 못한 파라미터가 optimizer에 의해 갱신되지 않게 한다. gsplat처럼 Gaussian 수백만 개의 `shN` gradient만으로 수백 MB가 나가고, 그 직후 densification이 파라미터와 Adam 상태를 통째로 재할당하는 학습 루프에서는 (b)가 특히 크게 작용한다. PyTorch 2.0부터는 이것이 기본값이다.
