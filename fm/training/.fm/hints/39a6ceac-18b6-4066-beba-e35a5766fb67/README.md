# 매 학습 스텝의 7단계 순서

**Q.** 매 학습 스텝에서 반복되는 동작을 순서대로 나열하면?

**A.** 이미지 1장 샘플 → `rasterization()` → loss = 0.8·L1 + 0.2·(1−SSIM) → `step_pre_backward()` → `loss.backward()` → `optimizer.step()` → `step_post_backward()`(duplicate/split/prune/opacity reset)

---

## 1. 전체 그림

3DGS 학습은 "COLMAP으로 한 번 만든 초기 상태"를 놓고, 그 뒤로는 **똑같은 7단계를 3만 번 반복**하는 구조다. 워크스루 노트북 서두의 다이어그램이 이 구조를 그대로 보여준다.

```
COLMAP SfM 결과                       매 스텝 반복
┌──────────────┐   ┌──────────────────────────────────────────────────┐
│ poses, K,    │   │  이미지 1장 샘플                                 │
│ sparse points│   │    → rasterization()  (SH평가→투영→타일링→블렌딩)│
└──────┬───────┘   │    → loss = 0.8·L1 + 0.2·(1-SSIM)                │
       │           │    → strategy.step_pre_backward()  (grad 추적)   │
  Gaussian 초기화  │    → loss.backward()                             │
  (means/scales/   │    → optimizer.step()  (파라미터별 Adam)         │
   quats/opacity/  │    → strategy.step_post_backward()               │
   SH)             │       (duplicate / split / prune / opacity reset)│
       └──────────►└──────────────────────────────────────────────────┘
```

왼쪽(데이터 로드 + Gaussian 초기화)은 **한 번만**, 오른쪽 박스는 **매 스텝**이라는 게 핵심 대비다. NeRF류와 달리 네트워크 가중치가 없고, 최적화 대상이 곧 씬을 이루는 Gaussian 파라미터 자체(`means`/`scales`/`quats`/`opacities`/`sh0`/`shN`)이기 때문에, "파라미터 개수 자체가 스텝마다 변한다"는 점이 루프의 마지막 단계(밀도화)를 특별하게 만든다.

## 2. 단계별로 무슨 일이 일어나는가

| # | 동작 | 실제 코드 | 하는 일 |
|---|---|---|---|
| 1 | 이미지 1장 샘플 | `DataLoader(batch_size=1, shuffle=True)`의 `next(loader_iter)` | `camtoworld`[1,4,4], `K`[1,3,3], `image`[1,H,W,3] 한 세트. `StopIteration`이면 iterator 재생성(= 무한 순환) |
| 2 | `rasterization()` | `gsplat/rendering.py` | SH 평가 → `fully_fused_projection`(EWA 투영) → `isect_tiles`(16×16 타일 정렬) → `rasterize_to_pixels`(앞→뒤 알파 블렌딩). 반환은 `renders`, `alphas`, **`info`** |
| 3 | loss | `l1_loss` + `ssim_loss` | `torch.lerp(l1, ssim, 0.2)` = `0.8·L1 + 0.2·(1−SSIM)` |
| 4 | `step_pre_backward()` | `gsplat/strategy/default.py:158` | 딱 한 줄: `info["means2d"].retain_grad()` |
| 5 | `loss.backward()` | — | 파라미터 grad + 위에서 붙잡아 둔 `means2d.grad`가 채워진다 |
| 6 | `optimizer.step()` | 파라미터별 Adam 6개 | `opt.step()` → `opt.zero_grad(set_to_none=True)`, 그리고 `means` lr 지수 감쇠 스케줄러 |
| 7 | `step_post_backward()` | `default.py:172` | grad 통계 누적 → 주기적으로 duplicate / split / prune, 그리고 3000스텝마다 opacity reset |

### (1) 이미지 1장 샘플 — batch_size=1

배치가 1인 건 우연이 아니다. 손실에 SSIM(11×11 윈도우)이 들어가고 Gaussian 개수가 스텝마다 바뀌므로, 여러 뷰를 묶어 얻는 이득이 크지 않고 대신 뷰마다 화면공간 gradient 통계를 깔끔하게 쌓을 수 있다. `shuffle=True`로 뷰 순서를 섞어 특정 시점에 과적합되는 것을 막는다.

### (2) `rasterization()` — 미분 가능 렌더러 + 밀도화 신호 생산기

이 호출의 산출물은 픽셀만이 아니다. **`info` dict가 7단계의 입력**이 된다.

- `info["means2d"]` [C,N,2] — 화면공간 위치. **이것의 gradient가 "이 Gaussian을 쪼개야 하나?"의 신호**
- `info["radii"]` — 화면 반경. 0이면 컬링됨(= 이 뷰에서 안 보임)
- `info["width"]`, `info["height"]`, `info["n_cameras"]`, `info["gaussian_ids"]` — grad를 NDC로 정규화하고 가시 Gaussian만 골라내는 데 쓰임

SH 차수는 스텝에 따라 `min(step // 1000, 3)`으로 올라간다. 처음엔 DC(0차)만 학습해 기본 색을 안정화하고, 뷰 의존 성분은 나중에 푼다.

### (3) loss = 0.8·L1 + 0.2·(1−SSIM)

$$\mathcal{L} = (1-\lambda)\,\mathcal{L}_{L1} + \lambda\,(1-\mathrm{SSIM}),\qquad \lambda = 0.2$$

코드는 이걸 `torch.lerp(l1loss, ssimloss, 0.2)` 한 줄로 쓴다(`lerp(a,b,w) = a + w(b−a)`이므로 정확히 `0.8·L1 + 0.2·ssimloss`). `ssim_loss`가 이미 `1−SSIM` 형태를 반환하므로 그대로 더하면 된다. L1은 색 재구성·노이즈 강건성, SSIM은 국소 구조/대비를 담당해 L1 단독의 뭉개짐을 막는다. depth loss, opacity/scale 정규화, random background는 모두 **선택 항이고 기본은 꺼짐**이다.

### (4) `step_pre_backward()` — 순서가 중요한 이유 ①

내용은 `info["means2d"].retain_grad()` 뿐이다. `means2d`는 파라미터가 아니라 **중간 텐서**라서, PyTorch는 backward가 끝나면 그 `.grad`를 버린다. `retain_grad()`를 미리 걸어 두지 않으면 5단계 이후 `info["means2d"].grad`가 `None`이 되고, 7단계의 밀도화 판단 근거가 사라진다.

그래서 이 훅의 계약은 **"forward 뒤 ~ backward 앞"** 이다. 카드의 순서(loss 계산 → `step_pre_backward()` → `backward()`)와 `simple_trainer.py`의 실제 순서(forward → `step_pre_backward()` → loss → `backward()`)는 미묘하게 다른데, **둘 다 맞다.** loss 계산은 grad 저장 정책에 영향을 주지 않으므로 loss 앞이든 뒤든 상관없고, `backward()` 앞이기만 하면 된다. `DefaultStrategy`의 docstring 예시도 카드와 같은 배치를 쓴다.

```python
render_image, render_alpha, info = rasterization(...)
strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
loss = ...
loss.backward()
strategy.step_post_backward(params, optimizers, strategy_state, step, info)
```

### (5)(6) `backward()` → 파라미터별 Adam

옵티마이저가 하나가 아니라 **파라미터마다 별도의 Adam**(`eps=1e-15`)이라는 점이 3DGS 특유의 구조다. 학습률 스케일이 파라미터마다 자릿수 단위로 다르기 때문이다.

| 파라미터 | lr | 메모 |
|---|---|---|
| `means` | `1.6e-4 × scene_scale` | 위치는 씬 크기에 비례 + 총 스텝에 걸쳐 초기값의 1%까지 지수 감쇠(`gamma = 0.01^(1/max_steps)`) |
| `scales` | `5e-3` | log 공간 |
| `quats` | `1e-3` | 내부에서 normalize |
| `opacities` | `5e-2` | logit 공간 |
| `sh0` | `2.5e-3` | DC |
| `shN` | `2.5e-3 / 20` | 고차 SH는 20배 천천히 |

각 `opt.step()` 직후 `opt.zero_grad(set_to_none=True)`로 grad를 비운다. 이때 `means2d.grad`는 아직 살아 있다(파라미터가 아니므로 옵티마이저가 건드리지 않는다) — 7단계가 그걸 읽는다.

### (7) `step_post_backward()` — 순서가 중요한 이유 ②

이 훅이 **`optimizer.step()` 뒤**에 오는 이유는, 여기서 Gaussian을 복제/분할/제거하면 **파라미터 텐서의 행 개수 N이 바뀌고 그에 맞춰 Adam의 exp_avg/exp_avg_sq 상태까지 같이 재배치**되기 때문이다. 옵티마이저 스텝 전에 텐서 크기를 바꾸면 grad와 상태가 어긋난다. (`simple_trainer.py`도 주석으로 "Run post-backward steps after backward and optimizer"라고 못 박아 둔다.)

훅 내부는 두 층이다.

**(a) 매 스텝: `_update_state()` — grad 통계 누적**

```python
grads = info["means2d"].grad.clone()          # absgrad=True면 .absgrad
grads[..., 0] *= info["width"]  / 2.0 * info["n_cameras"]   # → [-1,1] NDC 기준
grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
sel = (info["radii"] > 0.0).all(dim=-1)       # 이 뷰에서 보이는 것만
state["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
state["count"].index_add_(0, gs_ids, 1.0)     # 보인 횟수
```

즉 **"보였을 때의 화면공간 grad 크기 합"과 "보인 횟수"** 를 각각 누적한다. 나중에 `grad2d / count`로 평균을 낸다 — 자주 보인 Gaussian이 유리해지지 않게 하는 정규화다.

**(b) 주기적: refine**

조건은 `step > 500` and `step % 100 == 0` and `step < 15000`(`refine_stop_iter`에서 조기 return).

| 동작 | 조건(기본값) | 효과 |
|---|---|---|
| **duplicate** | grad 평균 > `2e-4` **and** 최대 스케일 ≤ `0.01 × scene_scale` | 작은데 오차 큰 곳 → 그대로 복제(빈틈 채우기) |
| **split** | grad 평균 > `2e-4` **and** 최대 스케일 > `0.01 × scene_scale` | 큰데 오차 큰 곳 → 2개로 쪼개고 크기 /1.6 (디테일 세분화) |
| **prune** | `sigmoid(opacity) < 0.005`, 또는 (`step > 3000`일 때) 최대 스케일 > `0.1 × scene_scale` | 기여 없는 것·비대해진 것 제거 |
| **opacity reset** | `step % 3000 == 0 and step > 0` | 전체 opacity를 `prune_opa × 2 = 0.01`로 리셋 → 다음 prune 때 floater가 쓸려 나감 |

세부 순서도 의도적이다: **duplicate → split → prune** 순이고, duplicate로 새로 생긴 것들은 같은 스텝에서 split 대상에서 제외된다(`is_split`에 `zeros(n_dupli)`를 concat). refine이 끝나면 `grad2d`/`count`를 0으로 초기화하고 `torch.cuda.empty_cache()`를 부른다.

임계값들이 전부 `scene_scale`에 곱해져 있다는 점도 눈여겨볼 것. `scene_scale`은 1단계의 COLMAP 정규화에서 나온 값(`parser.scene_scale * 1.1`)이라, "크다/작다" 판정과 `means` 학습률이 씬 크기에 무관해진다.

## 3. 자주 헷갈리는 지점

- **`step_pre_backward`가 loss 앞인가 뒤인가?** 어느 쪽이든 무관, `backward()` 앞이면 된다. 이 훅의 존재 이유는 loss가 아니라 `retain_grad()`다.
- **`step_post_backward`를 `backward()` 직후에 두면?** 안 된다. `optimizer.step()` **뒤**여야 한다(파라미터/Adam 상태 크기 동시 변경).
- **λ = 0.2가 어디에 붙는가?** SSIM 항에 0.2, L1에 0.8. `lerp(l1, ssim, 0.2)`의 세 번째 인자가 SSIM 가중치다.
- **`packed=False`인 이유** — 밀도화 상태 갱신 코드가 `[C,N,2]` 조밀 레이아웃을 가정한 경로와 맞물린다. `packed=True`면 `gaussian_ids`/`[nnz,2]` 경로로 갈라진다(`_update_state`가 두 경로를 모두 지원).
- **`MCMCStrategy`는 7단계만 다르다.** 1~6단계는 동일하고, 마지막이 휴리스틱 duplicate/split/prune 대신 상한 개수 아래에서의 opacity 기반 확률적 재배치 + 노이즈 주입(SGLD)으로 바뀐다. 이때는 `opacity_reg`/`scale_reg`를 손실에 켜 준다.

## 4. 암기용 압축

> **"뽑고(1) 그리고(2) 재고(3) 붙잡고(4) 흘리고(5) 밟고(6) 늘린다(7)"**
>
> 샘플 → 렌더 → loss → retain_grad → backward → Adam → densify

앞 3개는 "이번 뷰가 얼마나 틀렸나", 뒤 4개는 "그 틀림을 어떻게 파라미터와 개수에 반영하나"로 갈린다.
