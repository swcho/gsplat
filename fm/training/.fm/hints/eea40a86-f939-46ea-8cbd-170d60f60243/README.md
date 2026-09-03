# `step_post_backward()`가 하는 일은?

> **답**: gradient 통계를 누적하고, 정해진 주기마다 refine(duplicate/split/prune/opacity reset)을 실행한다.

---

## 1. 어디에 끼어드는 훅인가

`DefaultStrategy`는 학습 루프에 **두 개의 콜백**으로만 개입한다. 전략 객체는 파라미터를 소유하지 않고, 매 스텝 넘겨받은 `params` / `optimizers` / `state`를 **제자리(in-place)로 갈아끼운다**.

```
(3) rasterization()  →  info dict  (means2d, radii, width, height, n_cameras, ...)
(4) strategy.step_pre_backward(...)     ← info["means2d"].retain_grad()  단 한 줄
    loss = 0.8*L1 + 0.2*SSIM ;  loss.backward()
(5) for opt in optimizers.values(): opt.step(); opt.zero_grad(set_to_none=True)
(6) means_lr_scheduler.step()
(7) strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
    └─ 여기서 Gaussian 개수 N이 바뀐다
```

`step_pre_backward()`는 **읽을 준비**만 한다. `info["means2d"]`는 leaf가 아닌 중간 텐서라서 backward가 끝나면 `.grad`가 버려지는데, `retain_grad()`로 붙잡아 둔다. 실제 일은 전부 `step_post_backward()`가 한다.

> 참고로 (5)의 `opt.zero_grad(set_to_none=True)`가 (7)보다 먼저 와도 괜찮다. optimizer는 `means/scales/quats/opacities/sh0/shN` 같은 **leaf 파라미터**의 grad만 지우고, `means2d`는 optimizer에 등록된 적이 없으므로 retain된 grad가 그대로 살아 있다.

소스: `gsplat/strategy/default.py` — `step_pre_backward` (L158), `step_post_backward` (L172), `_update_state` (L226), `_grow_gs` (L286), `_prune_gs` (L343).

---

## 2. 본문 구조 — 세 개의 블록

```python
def step_post_backward(self, params, optimizers, state, step, info, packed=False, scene=None):
    if step >= self.refine_stop_iter:          # ← ① 조기 종료 (기본 15_000)
        return

    self._update_state(params, state, info, packed=packed)   # ← ② 통계 누적 (매 스텝)

    if (step > self.refine_start_iter                        # ← ③ refine (주기적)
        and step % self.refine_every == 0
        and step % self.reset_every >= self.pause_refine_after_reset):
        n_dupli, n_split = self._grow_gs(...)   # duplicate → split
        n_prune = self._prune_gs(...)           # prune
        state["grad2d"].zero_(); state["count"].zero_()   # 통계 리셋
        torch.cuda.empty_cache()

    if step % self.reset_every == 0 and step > 0:            # ← ④ opacity reset (독립 조건)
        reset_opa(params, optimizers, state, value=self.prune_opa * 2.0)
```

핵심 포인트 네 가지:

- **②는 매 스텝, ③은 주기적.** "gradient 통계를 누적하고, 정해진 주기마다 refine"이라는 답안이 정확히 이 구조를 가리킨다.
- `refine_stop_iter`(15000) 이후에는 **아무 일도 안 한다**. 통계 누적조차 멈추고 Gaussian 개수가 고정된다. 워크스루의 "15000스텝 이후 개수 고정"이 이것.
- **opacity reset은 refine 블록 밖에 있다.** `reset_every`(3000)의 배수마다 refine 여부와 무관하게 실행된다.
- refine 조건의 세 번째 절 `step % reset_every >= pause_refine_after_reset`은 opacity reset **직후** 잠깐 refine을 쉬게 하는 옵션이다. 기본 0이라 항상 통과하고, 보통 학습 이미지 개수만큼 주는 것을 권한다(리셋된 opacity가 한 epoch 돌며 회복할 시간을 주려고).

---

## 3. ② 통계 누적 (`_update_state`) — 무엇을 어떻게 쌓는가

`state`는 `initialize_state()`가 만든 세 칸짜리 dict다. 디바이스를 알 수 없어서 텐서는 첫 스텝에 지연 생성된다.

| 키 | 모양 | 의미 |
|---|---|---|
| `grad2d` | `[N]` | 화면공간 gradient **norm의 누적합** |
| `count` | `[N]` | 그 Gaussian이 **보인 횟수** 누적 |
| `scene_scale` | scalar | 크기 임계값의 기준 (COLMAP 카메라 분포로 추정) |
| `radii` | `[N]` | (옵션) 화면 반경 최대값 — `refine_scale2d_stop_iter > 0`일 때만 |

### (a) gradient를 화면 `[-1,1]` 기준으로 정규화

```python
grads = info["means2d"].grad.clone()      # absgrad=True면 .absgrad
grads[..., 0] *= info["width"]  / 2.0 * info["n_cameras"]
grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
```

`means2d`는 **픽셀 좌표**다. `width/2`를 곱하면 "픽셀당" 미분이 "NDC 1단위당" 미분으로 바뀌어 **해상도에 무관한 값**이 된다. 이래서 640×480으로 학습해도, 1600×1200으로 학습해도 같은 `grow_grad2d=2e-4`를 쓸 수 있다. `n_cameras`를 곱하는 건 loss가 배치 내 카메라 수로 평균되어 gradient가 `1/C`로 희석된 것을 되돌리는 보정.

### (b) 보이는 것만 골라 `index_add_`

```python
sel = (info["radii"] > 0.0).all(dim=-1)   # [C, N] — 화면 안 & near/far 안
gs_ids = torch.where(sel)[1]              # [nnz]
grads  = grads[sel]                       # [nnz, 2]
state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
```

컬링된(`radii == 0`) Gaussian은 **분모에도 분자에도 안 들어간다**. 그래서 나중에 `grad2d / count`가 "보인 스텝들에 대한 평균"이 되고, 카메라에 거의 안 잡히는 Gaussian이 낮은 평균 때문에 부당하게 densify 대상에서 빠지는 일이 없다. `packed=True`면 rasterization이 이미 `[nnz]`로 압축해 주므로 `info["gaussian_ids"]`를 그대로 쓴다 — 워크스루가 `packed=False`를 forward와 훅 양쪽에 **일치시킨** 이유가 이 분기다.

### (c) `absgrad`

`absgrad=True`면 `.grad` 대신 `.absgrad`(픽셀별 gradient의 **절대값 합**)를 쓴다. 한 Gaussian이 이미지의 왼쪽은 +로, 오른쪽은 −로 틀렸을 때 보통 gradient는 상쇄되어 0에 가깝지만 absgrad는 그렇지 않다. "위치는 맞는데 하나로 표현하기엔 과한" 경우를 잡아내므로 분할 신호가 더 민감하다(AbsGS, arXiv:2404.10484). 대신 값의 스케일이 커지므로 임계값을 `grow_grad2d=0.0008`로 올려야 하고, `rasterization(..., absgrad=True)`도 같이 켜야 `.absgrad` 속성이 채워진다.

---

## 4. ③ refine — 네 가지 동작

조건은 `step > 500` **및** `step % 100 == 0`이므로, refine이 실제로 도는 스텝은 `600, 700, 800, ..., 14900`이다 (`step >= 15000`이면 맨 위에서 `return`하므로 15000은 포함되지 않는다). 즉 **500~15000 구간에서 100스텝마다**.

### duplicate와 split (`_grow_gs`)

```python
grads = state["grad2d"] / count.clamp_min(1)          # 보인 스텝 평균
is_grad_high = grads > self.grow_grad2d               # 2e-4
is_small = exp(params["scales"]).max(-1).values <= self.grow_scale3d * scene_scale   # 1%
is_dupli = is_grad_high &  is_small
is_split = is_grad_high & ~is_small
```

| 동작 | 조건 | 하는 일 |
|---|---|---|
| **duplicate** | grad 평균 > `2e-4` **및** 크기 ≤ 1%·scene_scale | 똑같은 Gaussian을 그대로 뒤에 `torch.cat` — "작은데 오차가 큰 곳"에 밀도를 더한다 |
| **split** | grad 평균 > `2e-4` **및** 크기 > 1%·scene_scale | 원본을 지우고 2개로 — "큰데 오차가 큰 곳"을 잘게 쪼갠다 |

`count.clamp_min(1)`은 한 번도 안 보인 Gaussian의 0 나눗셈을 막는다(분자도 0이므로 결과는 0).

`split`의 실제 구현(`gsplat/strategy/ops.py:175`)이 재미있다.

```python
samples = einsum("nij,nj,bnj->bni", rotmats, scales, torch.randn(2, N, 3))  # [2,N,3]
means:  p[sel] + samples          # 자기 공분산에서 뽑은 오프셋만큼 흔들어 배치
scales: log(scales / 1.6)         # 크기를 1.6로 나눔
그 외:  p[sel].repeat(2, ...)     # quats/opacity/SH는 그대로 복사
```

새 두 개의 위치는 **원본 Gaussian 자신의 타원체 분포에서 샘플링**한 오프셋으로 정한다. 즉 원본이 덮던 영역 안에 확률적으로 흩뿌리는 것. 크기를 `1.6`으로 나누는 것은 원논문의 경험값(φ=1.6)이다. `revised_opacity=True`면 opacity도 `1 - sqrt(1 - o)`로 조정해 두 개를 겹쳤을 때의 합성 알파를 원본과 맞춘다(arXiv:2404.06109, 실험적).

**순서에 숨은 디테일**: duplicate를 먼저 실행하고, 그 다음 `is_split` 마스크 뒤에 `n_dupli`개의 `False`를 이어 붙인다.

```python
is_split = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)])
```

방금 복제되어 텐서 뒤에 붙은 새 Gaussian이 같은 스텝에서 **또 split되지 않게** 막는 장치다. duplicate는 뒤에 append(인덱스 보존), split은 `[p[rest], p_split]`로 **재배열**하므로 이 순서가 아니면 마스크 인덱스가 깨진다.

### prune (`_prune_gs`)

```python
is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa   # 0.005
if step > self.reset_every:                                                # 3000 이후
    is_too_big = exp(params["scales"]).max(-1).values > self.prune_scale3d * scene_scale  # 10%
    is_prune |= is_too_big
```

- **투명도 기준은 항상** 적용: 화면에 기여하지 않는 Gaussian을 버린다.
- **크기 기준은 `step > reset_every`(3000) 이후에만.** 초기에 SfM 포인트에서 시작한 큰 Gaussian들이 아직 줄어들 기회를 못 얻었는데 학살당하는 것을 피한다.
- 화면 크기 기반 prune(`prune_scale2d`)은 원 구현의 버그(graphdeco-inria/gaussian-splatting#123)로 실제로는 동작하지 않았고, gsplat은 완결성을 위해 구현해 두되 `refine_scale2d_stop_iter=0`으로 **기본 비활성**이다.

### opacity reset (`reset_opa`)

`reset_every`(3000)마다, `value = prune_opa * 2 = 0.01`로.

```python
opacities = torch.clamp(p, max=torch.logit(torch.tensor(value)).item())
```

이름은 "reset"이지만 실제로는 **clamp(상한 걸기)**다. 이미 0.01보다 투명한 것은 그대로 두고, 불투명한 것만 0.01로 끌어내린다. 카메라 근처에 떠서 오차를 가려주기만 하던 **floater**를 강제로 투명하게 만들어, 정말 필요한 것만 다시 불투명해지게 하고 나머지는 다음 prune에서 `< 0.005`로 걸려 사라지게 한다. 그래서 학습 곡선에서 3000스텝마다 loss가 튀었다가 회복하는 톱니가 보인다.

---

## 5. 개수가 바뀌면 optimizer도 같이 수술한다

`N`이 변하면 파라미터 텐서만 갈아치우면 안 된다. Adam의 `exp_avg` / `exp_avg_sq` 모멘트도 모양이 맞아야 한다. `ops.py`의 네 함수는 모두 `_update_param_with_optimizer(param_fn, optimizer_fn, ...)`를 통해 **파라미터 · optimizer state · strategy state를 한꺼번에** 재구성한다.

| 대상 | duplicate | split | remove | reset_opa |
|---|---|---|---|---|
| 파라미터 | `cat([p, p[sel]])` | `cat([p[rest], p_split])` | `p[keep]` | opacity만 clamp |
| Adam 모멘트 | 새 항목 **0** | 새 항목 **0** | 같이 슬라이싱 | **전체 0으로** |
| `state["grad2d"]/["count"]` | `cat((v, v[sel]))` | 같이 재배열 | 같이 슬라이싱 | 그대로 |

새로 생긴 Gaussian의 모멘트를 0으로 두는 건 남의 관성을 물려받지 않게 하기 위한 것. `reset_opa`가 모멘트를 **전부** 0으로 미는 건, 방금 강제로 눌러 놓은 opacity를 Adam의 옛 관성이 즉시 되돌려 놓는 것을 막기 위한 것이다.

이 수술이 "optimizer마다 param_group이 정확히 하나, 그리고 `params`와 `optimizers`의 키가 일치"를 요구한다. 워크스루가 루프 전에 `strategy.check_sanity(splats, optimizers)`를 부르는 이유이고, 파라미터를 개별 optimizer로 쪼개 만드는 이유다.

refine 블록 마지막에 `state["grad2d"].zero_(); state["count"].zero_()`로 통계를 비우는 것도 필수다. 텐서 배치가 방금 바뀌었으므로 이전 누적값은 더 이상 같은 Gaussian을 가리키지 않는다.

---

## 6. 기본값 요약 (`DefaultStrategy` dataclass 필드)

| 필드 | 기본값 | 역할 |
|---|---|---|
| `grow_grad2d` | `0.0002` | duplicate/split 발동 grad 임계값 (absgrad면 `0.0008`) |
| `grow_scale3d` | `0.01` | 이 크기 이하 → duplicate, 초과 → split |
| `prune_opa` | `0.005` | 이 아래 opacity는 제거 (reset 값 = ×2 = `0.01`) |
| `prune_scale3d` | `0.1` | scene_scale의 10% 넘으면 비대하다고 판단해 제거 |
| `refine_start_iter` / `refine_stop_iter` | `500` / `15_000` | refine 구간 |
| `refine_every` / `reset_every` | `100` / `3000` | refine 주기 / opacity reset 주기 |
| `pause_refine_after_reset` | `0` | reset 후 refine 유예 (학습 이미지 수 권장) |
| `grow_scale2d` / `prune_scale2d` / `refine_scale2d_stop_iter` | `0.05` / `0.15` / `0.0` | 화면 크기 기반 분할·제거 (기본 꺼짐) |
| `key_for_gradient` | `"means2d"` | 2DGS는 `"gradient_2dgs"` |

`verbose=True`면 매 refine마다 `Step 600: 1234 GSs duplicated, 567 GSs split. Now having ...` 로그가 찍힌다.

---

## 7. 한 줄 정리와 흔한 함정

**한 줄**: `step_post_backward()`는 매 스텝 `means2d.grad`를 해상도 무관 스케일로 정규화해 보이는 Gaussian에만 누적하고(`grad2d`, `count`), 100스텝마다 그 평균으로 duplicate/split/prune을 실행해 Gaussian 개수를 바꾸며(optimizer 모멘트까지 함께 재구성), 3000스텝마다 opacity를 0.01로 눌러 floater를 정리한다.

- ❌ "refine이 매 스텝 돈다" → 통계 누적만 매 스텝. refine은 `refine_every`(100)마다.
- ❌ "opacity reset도 refine 블록 안" → 별개의 `if`. refine 구간 밖 조건에서도 `refine_stop_iter` 전까지 3000의 배수마다 실행된다.
- ❌ "duplicate와 split 조건이 겹칠 수 있다" → 크기 조건이 `is_small` / `~is_small`로 **상호 배타**다. 한 Gaussian은 둘 중 하나만.
- ❌ "step_pre_backward가 통계를 모은다" → 그건 `retain_grad()` 한 줄뿐.
- ⚠️ forward의 `packed`와 `step_post_backward(..., packed=)`가 어긋나면 `_update_state`가 텐서 모양을 잘못 해석한다. 반드시 일치시킬 것.
- ⚠️ `absgrad=True`는 `rasterization()`과 전략 **양쪽에** 줘야 하고, 임계값도 함께 올려야 한다.

**대안 전략**: `MCMCStrategy`(`gsplat/strategy/mcmc.py`)는 gradient 기반 heuristic 대신 Gaussian 개수를 `cap_max`로 고정하고, 죽은 Gaussian을 살아 있는 것 위로 **재배치(relocate)** 하며 noise를 주입한다 — SGLD 관점의 샘플링이라 grad 임계값 튜닝이 필요 없다.
