# split 연산의 조건과 효과

> **Q.** split 연산의 조건과 효과는?
> **A.** 화면 grad 평균 > `2e-4`이고 크기 > 1%·scene_scale일 때 발생한다. 큰데 오차가 큰 Gaussian을 2개로 쪼개고 크기를 1.6으로 나눈다.

---

## 1. split이 왜 필요한가

SfM 초기 포인트만으로는 씬을 다 덮지 못한다. 그래서 `DefaultStrategy`(원논문 3DGS 방식)는 학습 중에 Gaussian 수를 **늘리고 정리**한다. 이때 "오차가 큰 곳"을 찾는 신호는 **화면공간(screen-space) gradient**, 즉 `info["means2d"]`의 gradient다. 이 값이 크다는 건 "이 Gaussian을 화면에서 조금 움직이면 loss가 많이 줄어든다" = **아직 이 영역이 제대로 표현되지 않았다**는 뜻이다.

오차가 큰 Gaussian을 만났을 때 선택지는 두 가지고, **크기(3D scale)** 로 갈린다.

| 동작 | 조건 (기본값) | 해석 |
|---|---|---|
| **duplicate** | grad 평균 > `2e-4` **AND** 크기 ≤ 1%·scene_scale | 작은데 오차 큼 → "여기 Gaussian이 부족하다" → 그대로 복제해서 개수를 늘림 |
| **split** | grad 평균 > `2e-4` **AND** 크기 > 1%·scene_scale | 큰데 오차 큼 → "하나가 너무 넓은 영역을 뭉개고 있다" → 잘게 쪼갬 |

즉 split은 **under-reconstruction을 해상도로 해결하는** 연산이고, duplicate는 **coverage로 해결하는** 연산이다.

---

## 2. 조건을 코드로 확인

`gsplat/strategy/default.py` → `DefaultStrategy._grow_gs()`:

```python
count = state["count"]
grads = state["grad2d"] / count.clamp_min(1)      # 관측 횟수로 나눈 "평균"

is_grad_high = grads > self.grow_grad2d            # grow_grad2d = 0.0002 = 2e-4
is_small = (
    torch.exp(params["scales"]).max(dim=-1).values # 3축 중 가장 긴 축
    <= self.grow_scale3d * state["scene_scale"]    # grow_scale3d = 0.01 → 1%
)
is_dupli = is_grad_high & is_small
is_large = ~is_small
is_split = is_grad_high & is_large                 # ← split 조건

if step < self.refine_scale2d_stop_iter:           # 기본 0 → 보통 비활성
    is_split |= state["radii"] > self.grow_scale2d # 화면에서 너무 큰 것도 split
```

읽을 포인트가 네 개 있다.

1. **`grad2d`는 누적값, 실제 판단은 평균.** `_update_state()`가 매 스텝 `state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))`로 **보인 카메라에서만** gradient 크기를 더하고, `state["count"]`에 관측 횟수를 더한다. refine 시점에 `grad2d / count`로 나누므로, 많은 뷰에서 보인 Gaussian이 단순히 합이 커서 유리해지는 일이 없다.
2. **gradient는 NDC → 픽셀 스케일로 정규화된다.**
   ```python
   grads[..., 0] *= info["width"]  / 2.0 * info["n_cameras"]
   grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
   ```
   `means2d`의 gradient는 `[-1,1]` NDC 기준이라 해상도에 따라 크기가 달라진다. 이걸 이미지 크기로 곱해 픽셀 단위로 맞추기 때문에 `2e-4`라는 절대 임계값이 여러 데이터셋에서 통한다.
3. **크기 조건은 `scene_scale`에 상대적이다.** 임계값은 고정 미터값이 아니라 `0.01 * scene_scale`. 워크스루에서 `scene_scale = parser.scene_scale * 1.1` (simple_trainer.py:458과 동일)로 계산하며, 이 값이 학습률과 밀도화 임계값 모두의 기준 단위가 된다. 덕분에 방 한 칸 씬과 야외 씬에 같은 하이퍼파라미터를 쓸 수 있다.
4. **비교 대상은 `exp(scales)`의 최댓값**(가장 긴 축). `params["scales"]`는 log 공간이라 `exp`로 되돌린 뒤, 3축 중 하나라도 크면 "큰 Gaussian"으로 본다.

### `is_small`과 `is_large`는 정확히 여집합

`is_large = ~is_small`이므로 grad가 높은 Gaussian은 **반드시 duplicate 또는 split 하나로** 분류된다. 둘 다 되거나 둘 다 안 되는 경우는 없다. 카드의 답에서 두 조건이 `≤ 1%` / `> 1%`로 딱 갈리는 이유다.

### 언제 실행되는가 (조건의 나머지 절반)

`step_post_backward()`의 게이트를 통과해야 위 판정이 아예 돌아간다.

```python
if step >= self.refine_stop_iter:  return          # 15_000 이후 밀도화 종료
if (step > self.refine_start_iter                  # 500 이후 시작
    and step % self.refine_every == 0              # 100 스텝마다
    and step % self.reset_every >= self.pause_refine_after_reset):
```

즉 **스텝 500~15000 구간에서 100스텝마다** duplicate/split/prune이 일어난다. `pause_refine_after_reset`을 주면 opacity reset 직후 일정 스텝 동안은 통계가 오염됐다고 보고 밀도화를 쉰다.

---

## 3. 효과: `split()`이 실제로 하는 일

`gsplat/strategy/ops.py` → `split()`:

```python
sel  = torch.where(mask)[0]    # 쪼갤 것
rest = torch.where(~mask)[0]   # 남길 것

scales  = torch.exp(params["scales"][sel])
quats   = F.normalize(params["quats"][sel], dim=-1)
rotmats = normalized_quat_to_rotmat(quats)              # [N,3,3]
samples = torch.einsum("nij,nj,bnj->bni",
                       rotmats, scales,
                       torch.randn(2, len(scales), 3))  # [2,N,3]

def param_fn(name, p):
    repeats = [2] + [1] * (p.dim() - 1)
    if name == "means":
        p_split = (p[sel] + samples).reshape(-1, 3)      # 자식 위치
    elif name == "scales":
        p_split = torch.log(scales / 1.6).repeat(2, 1)   # ← 1.6으로 나눔
    elif name == "opacities" and revised_opacity:
        new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
        p_split = torch.logit(new_opacities).repeat(repeats)
    else:
        p_split = p[sel].repeat(repeats)                 # quats/sh는 그대로 복사
    return torch.nn.Parameter(torch.cat([p[rest], p_split]))
```

### (a) 부모는 사라진다 — 1개 → 2개

새 파라미터는 `cat([p[rest], p_split])`이다. `rest`는 **선택되지 않은 것들**이므로 부모는 결과 텐서에 포함되지 않는다. 선택된 N개가 사라지고 2N개가 붙으므로 **총 개수는 순증 +N**. "2개로 쪼갠다"는 표현 그대로, 복제(원본 유지 + 사본 추가)와 다르다.

| | duplicate | split |
|---|---|---|
| 원본 | **유지** (`cat([p, p[sel]])`) | **삭제** (`cat([p[rest], p_split])`) |
| 개수 변화 | +N | +N (N 제거, 2N 추가) |
| 크기 | 그대로 | **/1.6** |
| 위치 | 완전히 동일 | 부모 분포에서 샘플링 |

### (b) 자식의 위치는 부모 타원체 내부에서 샘플링

`samples = R · diag(scale) · randn(3)`. 이건 **부모 Gaussian 자신의 분포에서 뽑은 난수**다 (표준정규를 부모의 축/크기로 변환). 그래서 자식 두 개는 부모가 덮고 있던 타원체 안쪽 어딘가에, 부모의 형태 방향으로 흩어져 놓인다. `randn`이므로 결정적이 아니고, 1σ 밖으로 나가는 자식도 생긴다.

부모의 회전(`quats`)과 색(`sh0`/`shN`)은 그대로 복사되므로, 자식들은 **부모와 같은 방향·같은 색인 더 작은 두 개**로 출발한 뒤 이후 스텝에서 각자 갈라진다.

### (c) 크기 `/1.6`의 의미

`1.6`은 원논문 3DGS의 heuristic 상수(φ=1.6)다. 자식의 표준편차를 부모의 `1/1.6 ≈ 0.625`배로 줄인다 — log 공간에서는 `log(scale) - log(1.6) ≈ log(scale) - 0.47`을 뺀 것과 같다. 부피로는 `(1/1.6)³ ≈ 0.244`배, 두 개를 합쳐도 `≈ 0.49`배라 **덮는 부피는 절반쯤으로 줄어든다**. 이게 의도다: 넓게 뭉개던 영역을 더 촘촘하고 날카로운 단위로 다시 채워 세부 디테일이 살아날 여지를 만든다. 너무 크게 나누면(예: /4) 구멍이 생겨 다시 채우는 데 스텝을 낭비하고, 너무 작게 나누면(예: /1.05) 쪼갠 효과가 없다.

### (d) opacity는 기본적으로 손대지 않는다

`revised_opacity=False`(기본)면 두 자식이 부모의 opacity를 그대로 물려받는다. 원논문과 동일한 동작이지만, 겹친 두 개가 같은 α를 가지므로 합성 알파는 순간적으로 부모보다 진해진다. `revised_opacity=True`(arXiv:2404.06109, 실험적)로 켜면
`α_child = 1 - sqrt(1 - α_parent)`
를 쓴다. 자식 둘이 겹쳐 합성될 때 `1-(1-α_c)² = α_p`가 되어 **분할 전후의 불투명도가 보존**된다.

### (e) optimizer 모멘텀은 0으로 초기화

```python
def optimizer_fn(key, v):
    v_split = torch.zeros((2 * len(sel), *v.shape[1:]))
    return torch.cat([v[rest], v_split])
```
Adam의 `exp_avg`/`exp_avg_sq`를 자식에 대해 0으로 시작한다. 부모의 관성을 물려주면 갓 태어난 Gaussian이 엉뚱한 방향으로 튀기 때문이다. `_update_param_with_optimizer()`가 파라미터 교체와 optimizer state 재구성을 함께 처리한다.

### (f) 러닝 스탯도 함께 재배열

`state`의 텐서(`grad2d`, `count`, `radii`)도 같은 순서로 `cat([v[rest], v[sel].repeat(2)])` 되어 인덱스 정합성이 유지된다. 다만 refine 블록 끝에서 `state["grad2d"].zero_()`, `state["count"].zero_()`로 전부 리셋되므로, 다음 100스텝은 새 통계로 다시 판단한다.

---

## 4. duplicate와 split의 실행 순서 (미묘한 부분)

`_grow_gs()`는 **duplicate를 먼저** 실행하고, 그 다음 split 마스크 뒤에 0을 이어 붙인다.

```python
if n_dupli > 0:
    duplicate(..., mask=is_dupli)
# 복제로 새로 생긴 GS는 split 대상이 되지 않도록 False 패딩
is_split = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool)])
if n_split > 0:
    split(..., mask=is_split, revised_opacity=self.revised_opacity)
```

duplicate가 텐서 **뒤에** N_dupli개를 붙이므로, 기존 `is_split` 마스크는 길이가 안 맞게 된다. 여기서 `False`로 패딩해 **같은 refine 스텝에서 복제된 Gaussian이 곧바로 split되는 일을 막는다**. (애초에 duplicate 대상은 `is_small`이므로 split 조건과 배타적이지만, 마스크 길이를 맞추기 위해 필수다.)

---

## 5. 숫자로 감 잡기

`scene_scale = 5.0`인 씬(예: MipNeRF360 계열)이라면:

- split/duplicate 경계: `0.01 × 5.0 = 0.05` → 가장 긴 축이 **5cm 이상**이면 큰 것으로 분류
- prune 경계: `0.1 × 5.0 = 0.5` → **50cm 넘게 비대해진** Gaussian은 제거
- 가장 긴 축이 `0.08`이고 평균 화면 grad가 `3e-4`인 Gaussian → `is_grad_high=True`, `is_small=False` → **split**
  - 자식 크기: `0.08 / 1.6 = 0.05` (경계선까지 내려온다 → 다음 refine에서는 duplicate 쪽으로 분류될 수 있음)
  - 자식 위치: 부모 중심에서 `R·diag(0.08,·,·)·randn` 만큼 떨어진 두 점

반복되는 split은 크기를 `1/1.6`씩 기하적으로 줄이므로, 4번 split되면 `1/1.6⁴ ≈ 0.15`배가 된다. 이 하강과 prune(너무 작아 opacity가 죽는 것 제거)이 맞물려 Gaussian 크기 분포가 적정 수준에 수렴한다.

---

## 6. 관련 하이퍼파라미터 튜닝

| 파라미터 | 기본값 | split에 미치는 영향 |
|---|---|---|
| `grow_grad2d` | `0.0002` | 낮추면 split/duplicate가 폭발적으로 늘어 VRAM 초과. `absgrad=True`면 **`0.0008` 권장** |
| `grow_scale3d` | `0.01` | 높이면 split보다 duplicate로 몰림 (큰 것 기준이 올라감) |
| `grow_scale2d` | `0.05` | `refine_scale2d_stop_iter > 0`일 때만 동작. **화면에서 큰** Gaussian도 강제 split (grad 조건 무시하고 `|=`) |
| `refine_scale2d_stop_iter` | `0` (꺼짐) | 양수로 켜면 위 2D 기준 활성화 — 거대한 blob/floater를 초기에 깨는 데 유효 |
| `revised_opacity` | `False` | `True`면 자식 opacity를 `1-sqrt(1-α)`로 보정 |
| `refine_every` / `refine_start_iter` / `refine_stop_iter` | `100` / `500` / `15000` | split이 일어나는 시간 창 |

`absgrad=True`는 픽셀별 gradient의 **절대값 합**(AbsGS, arXiv:2404.10484)을 누적한다. 부호가 상쇄되지 않아 "여러 방향으로 동시에 당겨지는" 큰 Gaussian이 훨씬 뚜렷하게 잡히고, 그래서 임계값을 4배(`8e-4`) 올려 균형을 맞춘다. 이때 `rasterization(..., absgrad=True)`도 같이 켜야 `info["means2d"].absgrad`가 채워진다.

---

## 7. 한 줄 요약

> **조건** — refine 스텝(500~15000, 100마다)에서 `mean(grad2d)/count > 2e-4` **AND** `max(exp(scales)) > 0.01·scene_scale`
> **효과** — 부모를 지우고, 부모 분포에서 샘플링한 두 위치에 크기 `/1.6`인 자식 2개를 생성(회전·색은 복사, optimizer 모멘텀은 0). 개수는 +N, 덮는 부피는 절반쯤으로 줄어 디테일 여지를 만든다.

### 참고 파일
- `gsplat/strategy/default.py` — `DefaultStrategy._grow_gs()`, `_update_state()`, `step_post_backward()`
- `gsplat/strategy/ops.py` — `split()`, `duplicate()`
- `examples/simple_trainer.py:458` — `scene_scale` 계산
- 원논문: [3D Gaussian Splatting (arXiv:2308.04079)](https://arxiv.org/abs/2308.04079) §5 Adaptive Density Control
