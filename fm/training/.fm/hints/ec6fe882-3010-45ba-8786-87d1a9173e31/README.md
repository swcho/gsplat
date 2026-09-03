# duplicate 연산의 조건과 효과

## 한 줄 요약

**화면공간 gradient 평균이 `grow_grad2d`(기본 `2e-4`)를 넘고, 3D 크기가 `grow_scale3d`(기본 `0.01`) × `scene_scale` 이하인 Gaussian**을 그대로 한 개 복제한다. "재현이 안 되는데(오차 큼) 몸집이 작은(더 쪼갤 게 아니라 더 많이 필요한)" 자리에 Gaussian 개수를 늘려 표현력을 올리는 densification 연산이다.

---

## 1. 어디서 일어나는가

`DefaultStrategy`(원 3DGS 논문 방식)의 `step_post_backward()` 안에서 주기적으로 호출된다.
[`gsplat/strategy/default.py`](../../../../../gsplat/strategy/default.py), [`gsplat/strategy/ops.py`](../../../../../gsplat/strategy/ops.py)

```
매 스텝:  step_pre_backward()  → means2d.retain_grad()
          loss.backward()
          step_post_backward() → _update_state()  (grad2d, count 누적)
                                 └ refine 주기이면:
                                     _grow_gs()  → duplicate / split
                                     _prune_gs() → remove
                                     통계 리셋 (grad2d.zero_(), count.zero_())
```

refine이 실제로 돌아가는 게이트는 세 조건의 AND다.

```python
if (step > self.refine_start_iter          # 기본 500
    and step % self.refine_every == 0      # 기본 100
    and step % self.reset_every >= self.pause_refine_after_reset):
```
게다가 함수 진입 직후 `step >= refine_stop_iter`(기본 15,000)이면 그냥 return한다.
→ **스텝 500~15,000 구간에서 100스텝마다** duplicate/split/prune이 일어난다. 즉 duplicate은 매 스텝이 아니라 refine 타이밍에만 발생하는 "배치 연산"이다.

## 2. 조건 ①: 화면 grad 평균 > `2e-4`

`_grow_gs()` 첫 줄이 조건의 좌변을 만든다.

```python
count = state["count"]
grads = state["grad2d"] / count.clamp_min(1)     # ← "평균"의 의미
is_grad_high = grads > self.grow_grad2d          # grow_grad2d = 0.0002 = 2e-4
```

여기서 중요한 디테일 세 가지.

- **무엇의 gradient인가**: 3D 위치 `means`가 아니라 **화면에 투영된 2D 좌표 `means2d`의 gradient**다. 그래서 `step_pre_backward()`에서 `info["means2d"].retain_grad()`를 걸어 non-leaf 텐서의 grad를 살려 둔다 (2DGS는 `key_for_gradient="gradient_2dgs"`를 쓴다). 이 값이 크다는 것은 "이 Gaussian을 화면에서 조금 움직이면 loss가 많이 줄어든다" = **그 픽셀 영역의 재현이 아직 나쁘다**는 신호다.
- **정규화**: raw grad는 NDC(`[-1,1]`) 기준이므로 픽셀 스케일로 환산한다.
  ```python
  grads[..., 0] *= info["width"]  / 2.0 * info["n_cameras"]
  grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
  ```
  해상도·배치 카메라 수에 임계값이 흔들리지 않게 맞춰 주는 것이라, `2e-4`라는 절대 수치는 이 정규화를 전제로 한 값이다.
- **평균의 분모 `count`**: `radii > 0`(즉 실제로 화면에 보인 카메라)에서만 grad norm과 count를 `index_add_`로 누적한다. 그래서 "보인 횟수로 나눈 평균"이며, 한두 뷰에서만 보인 Gaussian이 누적합만으로 불리해지지 않는다.

```python
sel = (info["radii"] > 0.0).all(dim=-1)
state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))   # x,y 벡터의 L2 norm
state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
```

> `absgrad=True`(AbsGS)를 쓰면 `.grad` 대신 `.absgrad`(픽셀별 gradient의 절대값 합)를 누적한다. 부호 상쇄가 없어 신호가 훨씬 커지므로 임계값을 `0.0008` 정도로 올려야 한다. 이때 `rasterization(..., absgrad=True)`도 같이 켜야 값이 채워진다.

## 3. 조건 ②: 크기 ≤ 1% · scene_scale

```python
is_small = (torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"])   # 0.01 * scene_scale
is_dupli = is_grad_high & is_small
```

- `params["scales"]`는 **로그 공간**에 저장되므로 `torch.exp`로 실제 크기로 되돌린다.
- `.max(dim=-1)`: 3축 중 **가장 긴 축**을 대표 크기로 본다.
- `scene_scale`은 카메라 중심들의 퍼짐에서 계산된 씬의 물리적 크기다(`examples/datasets/colmap.py`의 `self.scene_scale = np.max(dists)`, 트레이너에서 `* 1.1`). 임계값을 이 값에 곱하기 때문에 **방 한 칸이든 야외 씬이든 같은 하이퍼파라미터가 통한다** — `means`의 learning rate를 `1.6e-4 * scene_scale`로 잡는 것과 같은 정신이다.

## 4. duplicate vs split — 같은 grad, 갈리는 크기

`is_small`의 여집합이 곧 split 대상이다. 두 연산은 **grad 조건을 공유하고 크기 조건만 반대**다.

| | 조건 | 개수 변화 | 새 Gaussian의 성질 |
|---|---|---|---|
| **duplicate** | grad 평균 > `2e-4` **and** 크기 ≤ `0.01`·scene_scale | N → N+1 (원본 유지) | 원본과 **모든 파라미터가 동일한 복사본** |
| **split** | grad 평균 > `2e-4` **and** 크기 > `0.01`·scene_scale | N → N+2 (원본 제거) | 공분산에서 뽑은 위치로 흩고 scale을 `/1.6` |

직관: 오차가 큰 자리에서 Gaussian이 **이미 크면** 하나가 넓은 영역을 뭉개고 있는 것이므로 잘게 **쪼개야**(split) 하고, **작으면** 디테일을 담을 개체 수 자체가 부족한 것이므로 **개수를 늘려야**(duplicate) 한다. 전자는 "해상도 부족", 후자는 "커버리지 부족" 처방이다.

`_grow_gs()`의 실행 순서에도 이 구분이 반영된다.

```python
if n_dupli > 0:
    duplicate(..., mask=is_dupli, ...)      # 먼저 복제 → 텐서 뒤에 append
# 복제로 새로 생긴 것은 split 대상에서 제외
is_split = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)])
if n_split > 0:
    split(..., mask=is_split, ...)
```
duplicate이 파라미터 텐서 **끝에 붙이는** 방식이라 기존 인덱스가 보존되고, 그래서 `is_split` 마스크를 `False`로 패딩만 해 주면 그대로 재사용된다. 한 refine에서 같은 Gaussian이 복제되고 또 쪼개지는 이중 처리를 막는 장치다.

## 5. duplicate이 실제로 하는 일

`gsplat/strategy/ops.py`의 `duplicate()`는 놀랄 만큼 단순하다.

```python
sel = torch.where(mask)[0]

def param_fn(name, p):                      # 모든 파라미터를 그대로 concat
    return torch.nn.Parameter(torch.cat([p, p[sel]]), requires_grad=p.requires_grad)

def optimizer_fn(key, v):                   # Adam 모멘텀은 0으로 시작
    return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

_update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
for k, v in state.items():                  # grad2d / count / radii 러닝 통계도 함께 복사
    if isinstance(v, torch.Tensor):
        state[k] = torch.cat((v, v[sel]))
```

포인트:

- **means / scales / quats / opacities / sh 계수 전부 그대로 복사.** 위치 지터도, opacity 보정도, 크기 축소도 없다(split과의 결정적 차이).
- **Adam의 `exp_avg` / `exp_avg_sq`는 0으로 초기화.** 원본의 관성을 물려받지 않고 새 개체로 학습을 시작한다.
- **러닝 통계(`grad2d`, `count`, `radii`)도 복사**해 두지만, refine 직후 `state["grad2d"].zero_(); state["count"].zero_()`로 어차피 비워진다.
- 처음에는 두 Gaussian이 **완전히 겹쳐** 있어 렌더 결과가 순간적으로 바뀌지 않는다(알파 합성상 opacity가 약간 두꺼워지는 효과는 있다). 이후 gradient가 둘을 서로 다른 방향으로 밀어내면서 갈라지고, 그 결과 같은 영역을 두 개가 분업해 더 세밀한 디테일을 표현하게 된다. 즉 duplicate의 효과는 **즉시**가 아니라 **몇 백 스텝에 걸쳐** 나타난다.

## 6. 학습 곡선에서 보이는 흔적

- 스텝 500 이후 100스텝마다 Gaussian 수가 계단식으로 증가한다(`verbose=True`면 `Step 600: N GSs duplicated, M GSs split.` 로그가 찍힌다).
- 매 3,000스텝의 opacity reset(`prune_opa * 2.0 = 0.01`)과 짝을 이룬다: duplicate/split이 늘리고, prune(opacity < `0.005`, 크기 > `0.1`·scene_scale)이 정리한다. `pause_refine_after_reset`을 학습 이미지 수 정도로 주면 reset 직후 통계가 오염된 구간에서 성장을 잠시 멈출 수 있다.
- 15,000스텝(=`refine_stop_iter`) 이후에는 개수가 고정되고 순수 파라미터 최적화만 남는다.

## 7. 자주 걸리는 함정

- **`2e-4`는 씬 스케일과 무관한 값**이다. 크기 조건만 `scene_scale`로 정규화된다. grad 쪽은 해상도/카메라 수로 정규화되므로 임계값을 만질 이유는 보통 `absgrad` 전환뿐이다.
- `scales`를 exp 없이 비교하면 조건이 완전히 뒤집힌다(로그 공간의 음수 값).
- duplicate 여부는 **개별 뷰의 gradient가 아니라 refine 주기 100스텝 동안 누적된 뷰 평균**으로 결정된다. 한 프레임에서 크게 튄 gradient만으로는 트리거되지 않는다.
- `revised_opacity`는 split에만 적용되는 옵션이다. duplicate은 opacity를 손대지 않는다.
- MCMC 전략(`gsplat/strategy/mcmc.py`)에는 duplicate/split이 없다. 총 개수 상한(`cap_max`)을 두고 `relocate` / `sample_add`로 개체를 옮기고 채우는 다른 패러다임이다.
