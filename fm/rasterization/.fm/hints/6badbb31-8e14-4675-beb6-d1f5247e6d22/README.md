# `absgrad=True` — AbsGS 밀도화 기준

**Q.** `absgrad=True` 옵션은 무엇을 하는가?

**A.** ∂L/∂means2d의 **절댓값 합**을 따로 누적한다. AbsGS 방식의 밀도화 기준으로 사용된다.

워크스루의 해당 문장(§ backward 커널의 요지):

> Gaussian 하나에 대한 기여가 여러 픽셀(스레드)에 흩어져 있으므로 warp 단위로 합친 뒤 `atomicAdd`로 모은다.
> `absgrad=True`면 ∂L/∂means2d의 절댓값 합을 따로 누적한다(AbsGS 밀도화 기준).

---

## 1. 배경 — 원 3DGS의 밀도화 기준

3DGS 학습은 고정된 Gaussian 집합으로 시작하지 않는다. 일정 주기(`refine_every=100`)마다
"이 Gaussian은 자기가 맡은 영역을 제대로 못 그리고 있다"고 판단되는 것들을 **복제(duplicate)** 하거나
**분할(split)** 한다. 그 판단 기준이 **뷰 공간(화면 좌표) 위치 gradient**, 즉 `means2d.grad`다.

gsplat `DefaultStrategy._update_state`가 하는 일이 정확히 그것이다
(`/home/sungwoo/projects/swcho/gsplat/gsplat/strategy/default.py`):

```python
grads = info["means2d"].grad.clone()          # [C, N, 2] 또는 [nnz, 2]
grads[..., 0] *= info["width"]  / 2.0 * info["n_cameras"]   # NDC → 픽셀 스케일 정규화
grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
...
state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))   # 뷰별 노름을 누적
state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
```

그리고 `_grow_gs`에서:

```python
grads = state["grad2d"] / count.clamp_min(1)   # 뷰 수로 나눈 평균
is_grad_high = grads > self.grow_grad2d        # 기본 0.0002
is_small  = scales.max(-1) <= grow_scale3d * scene_scale
is_dupli  = is_grad_high & is_small            # 작은 것 → 복제
is_split  = is_grad_high & ~is_small           # 큰 것  → 분할
```

즉 **"뷰마다 계산한 위치 grad 벡터의 노름을, 그 Gaussian이 보인 뷰 수로 평균"** 한 스칼라 하나가
그 Gaussian의 "재구성 부족 정도"를 대표한다.

## 2. 문제 — gradient collision (AbsGS 논문의 핵심 관찰)

AbsGS(Ye et al., 2024, arXiv:2404.10484)의 관찰은 이렇다. 위 기준은
**over-reconstruction 영역의 큰 Gaussian을 잡아내지 못한다.**

이유는 grad가 **어떻게 합쳐지는지**에 있다. 한 Gaussian은 화면에서 수십~수천 개 픽셀을 덮는다.
backward에서 각 픽셀 p가 그 Gaussian에 대해 자기 몫의 ∂L/∂means2d 기여를 만들고,
그것들이 전부 **부호를 가진 채로 더해져서** 최종 `means2d.grad`가 된다.

```
means2d.grad = Σ_p  g_p        (g_p ∈ R², 픽셀 p가 만든 위치 grad 기여)
```

그런데 하나의 큰 흐릿한 Gaussian이 디테일이 많은 영역을 덮고 있으면,
왼쪽 픽셀들은 "왼쪽으로 가라", 오른쪽 픽셀들은 "오른쪽으로 가라"고 말한다.
각 g_p의 크기는 큰데 방향이 서로 반대라서 **합이 상쇄되어 0에 가까워진다.**
이게 AbsGS가 말하는 **gradient collision**이다.

결과적으로:

- 실제로는 "이 Gaussian은 여러 픽셀에 서로 모순되는 요구를 받고 있다 = 하나로는 못 표현한다"인데,
- 기준은 `‖Σ_p g_p‖ ≈ 0`을 보고 **"grad가 작다 = 잘 맞고 있다"** 고 잘못 판정한다.
- 그래서 큰 블러 Gaussian이 split되지 않고 그대로 남고, 그 영역은 영원히 흐릿하게 수렴한다
  (= over-reconstruction, 논문이 지목한 블러의 원인).

반대로 작은 Gaussian은 덮는 픽셀이 적어 상쇄가 덜 일어나므로 상대적으로 잘 잡힌다.
즉 이 편향은 **큰 Gaussian에 불리하게** 작동한다.

## 3. 해법 — homodirectional gradient

AbsGS의 처방은 단순하다. **합치기 전에 절댓값을 취한다.**

```
means2d.absgrad = Σ_p  |g_p|        (성분별 절댓값, 그 다음 합)
```

논문 표현으로는 **homodirectional(동일 방향) view-space positional gradient**다.
모든 픽셀 기여를 같은 방향으로 정렬해 놓고 더하는 셈이라, 상쇄가 원천적으로 일어나지 않는다.
따라서 이 값은 "이 Gaussian이 픽셀들로부터 받는 요구의 **총량**"을 재고,
요구가 서로 모순될수록(= split이 필요할수록) 오히려 커진다.

⚠️ 두 값의 관계는 삼각부등식이므로 항상

```
‖Σ_p g_p‖  ≤  ‖Σ_p |g_p|‖
```

즉 **absgrad는 항상 일반 grad 이상**이다. 이 부등식이 뒤에 나올 임계값 조정의 이유다.

## 4. 왜 backward 커널 안에서 해야 하는가 (이 카드의 핵심)

여기가 이 옵션이 단순한 후처리가 아니라 **CUDA 커널 플래그**인 이유다.

autograd가 파이썬 쪽에 돌려주는 `means2d.grad`는 **이미 모든 픽셀에 대해 합산이 끝난 값**이다.
`Σ_p g_p`만 손에 들어오고, 개별 g_p는 커널이 끝나는 순간 사라진다.
`means2d.grad.abs()`를 해 봐야 그건 `|Σ_p g_p|`이지 `Σ_p |g_p|`가 아니다 — 이미 상쇄가 끝난 뒤다.

**픽셀 단위 절댓값은 픽셀 단위 기여가 아직 살아 있는 곳, 즉 backward 커널 내부에서만 가능하다.**
그래서 gsplat은 backward 커널에 별도의 출력 버퍼 `v_means2d_abs`를 하나 더 붙이고,
`v_xy_local`을 계산하는 그 자리에서 절댓값 사본을 같이 만든다.

`/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu`:

```cpp
// 커널 시그니처에 절댓값 전용 출력 버퍼가 따로 있다 (nullptr이면 비활성)
vec2 *__restrict__ v_means2d_abs,  // [..., N, 2] or [nnz, 2]
```

```cpp
if (opac * vis <= MAX_ALPHA) {
    const float v_sigma = -opac * vis * v_alpha;
    v_conic_local = { 0.5f * v_sigma * delta.x * delta.x,
                            v_sigma * delta.x * delta.y,
                      0.5f * v_sigma * delta.y * delta.y };
    v_xy_local    = { v_sigma * (conic.x * delta.x + conic.y * delta.y),
                      v_sigma * (conic.y * delta.x + conic.z * delta.y) };
    if (v_means2d_abs != nullptr) {
        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};   // ← 여기가 전부다
    }
    v_opacity_local = vis * v_alpha;
}
```

이 `v_xy_abs_local`이 일반 grad와 **똑같은 집계 경로**(warp reduction → atomicAdd)를 한 번 더 탄다.
워크스루가 말한 "warp 단위로 합친 뒤 `atomicAdd`로 모은다"가 두 벌 돈다는 뜻이다:

```cpp
warpSum(v_xy_local, warp);
if (v_means2d_abs != nullptr) {
    warpSum(v_xy_abs_local, warp);          // ← 추가 warp reduction
}
...
if (warp.thread_rank() == 0) {
    float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
    atomicAdd_system(v_xy_ptr,     v_xy_local.x);
    atomicAdd_system(v_xy_ptr + 1, v_xy_local.y);

    if (v_means2d_abs != nullptr) {
        float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
        atomicAdd_system(v_xy_abs_ptr,     v_xy_abs_local.x);   // ← 추가 atomicAdd
        atomicAdd_system(v_xy_abs_ptr + 1, v_xy_abs_local.y);   // ← (Gaussian당 2개)
    }
}
```

핵심 대칭 구조: **`v_means2d`는 부호 있는 채로, `v_means2d_abs`는 절댓값을 취한 뒤 누적.**
같은 커널, 같은 순회, 같은 reduction — 다른 건 `abs()` 하나뿐이다.

> `v_means2d_abs == nullptr`이면 절댓값 경로 전체가 컴파일 타임이 아닌 런타임 분기로 통째로 건너뛴다.
> 그래서 `absgrad=False`(기본값)일 때 추가 비용이 사실상 없다.

## 5. gsplat API 흐름

절댓값 grad는 **어떤 텐서의 진짜 gradient도 아니다.** autograd가 반환하는 값이 아니라
"학습 통계"에 가깝다. 그래서 gsplat은 이것을 `means2d` 텐서에 **속성으로 얹어서** 밖으로 내보낸다.

```python
# 1) 렌더링 시 켠다 — gsplat/rendering.py
render_colors, render_alphas, meta = rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, W, H,
    absgrad=True,          # ← backward 커널에 v_means2d_abs 버퍼를 붙여 준다
)
```

```python
# 2) rendering.py 내부: 커널이 돌려준 버퍼를 means2d 텐서의 속성으로 붙인다
if absgrad and not with_eval3d:
    means2d.absgrad = means2d_absgrad
```

```python
# 3) loss.backward() 이후, 그 속성을 밀도화 전략이 읽는다 — strategy/default.py
if self.absgrad:
    grads = info[self.key_for_gradient].absgrad.clone()   # ← .grad 대신 .absgrad
else:
    grads = info[self.key_for_gradient].grad.clone()
```

주의할 순서: forward 시점에 `meta["means2d"].absgrad`는 아직 **0으로 채워진 빈 버퍼**다.
`_wrapper.py`의 `_RasterizeToPixels.backward`가 커널 결과를
저장해 둔 그 버퍼에 **in-place로 복사**해 넣는다:

```python
# gsplat/cuda/_wrapper.py — backward 안
# The abs gradient is not a returned input grad; surface it by filling the
# saved means2d.absgrad holder in place.
if ctx.absgrad and v_means2d_abs is not None:
    means2d_absgrad.copy_(v_means2d_abs)
```

`forward`에서 `ctx.mark_non_differentiable(last_ids, means2d_absgrad)`로 이 홀더를
**미분 대상이 아니라고 못 박아** 둔 것도 같은 맥락이다 — autograd 그래프에 끼어들면 안 되는 사이드 채널이다.
따라서 `.absgrad`를 읽는 시점은 반드시 **`loss.backward()` 이후**(= `step_post_backward`)여야 한다.

```python
# 4) 전략 쪽도 같은 플래그를 켜 준다
strategy = DefaultStrategy(absgrad=True, grow_grad2d=0.0008)
```

두 곳을 **동시에** 켜야 한다. `rasterization(absgrad=True)`만 켜면 계산만 하고 안 쓰이고,
`DefaultStrategy(absgrad=True)`만 켜면 `.absgrad` 속성이 없어 터진다.
`examples/simple_trainer.py`가 이 커플링을 이렇게 처리한다:

```python
absgrad=(
    self.cfg.strategy.absgrad
    if isinstance(self.cfg.strategy, DefaultStrategy)
    else False
),
```

## 6. 임계값이 다른 이유 — 0.0002 → 0.0008

`DefaultStrategy` 독스트링이 명시한다:

> Which typically leads to better results but requires to set the `grow_grad2d` to a
> higher value, e.g., 0.0008.

§3의 부등식 `‖Σ g_p‖ ≤ ‖Σ |g_p|‖` 때문이다. absgrad는 **정의상 항상 더 크다.**
같은 임계값 0.0002를 그대로 쓰면 거의 모든 Gaussian이 `is_grad_high`를 통과해
무차별 복제·분할이 일어나고 Gaussian 수가 폭발한다.

권장 배수가 대략 4배(0.0002 → 0.0008)인 것은 경험값이다. 스케일이 통째로 바뀐 새 통계량이므로,
**absgrad를 켰다면 `grow_grad2d`는 반드시 다시 튜닝해야 하는 하이퍼파라미터**라고 보는 편이 맞다.
정규화(`* width/2 * n_cameras`)와 뷰 수 평균(`/ count`)은 두 경우 모두 동일하게 적용되므로,
차이는 순수하게 "부호 상쇄 여부"에서만 온다.

## 7. 효과와 트레이드오프

**얻는 것**
- gradient collision에 가려져 있던 **큰 블러 Gaussian이 제대로 `is_split`로 분류**된다.
  `is_split = is_grad_high & is_large`에서 그동안 `is_grad_high`가 False로 죽던 케이스가 살아난다.
- 고주파 디테일(잔가지, 텍스처, 얇은 구조) 복원이 눈에 띄게 좋아진다 — 논문 제목의 "Fine Details"가 이것.
- 구현이 가볍고 다른 3DGS 변형(Mip-Splatting, MCMC 등)에 그대로 얹힌다.

**치르는 것**
- split이 더 활발해지므로 **Gaussian 수가 늘어나는 경향**이 있다 → VRAM·렌더 시간 증가.
  (논문은 임계값을 적절히 올리면 "reduced or similar memory consumption"으로 더 나은 품질을 얻는다고 보고한다.
  즉 임계값 튜닝이 이 트레이드오프를 결정한다.)
- `grow_grad2d` 재튜닝이 필수. 안 하면 그냥 Gaussian 폭발이다.

**계산 비용**
- backward 커널에서 Gaussian당 **`atomicAdd` 2개(x, y) + warp reduction 1회**가 추가된다.
- forward는 완전히 무영향. backward에서도 이미 존재하는 순회·reduction 구조에 편승하므로
  새 커널 런치나 추가 메모리 순회가 없다. 실측 오버헤드는 보통 수 % 수준.
- 추가 메모리는 `[..., N, 2]` (packed면 `[nnz, 2]`) float 버퍼 하나.

**제약**
- `with_eval3d=True`와는 함께 쓸 수 없다. `rendering.py`의 조건이
  `if absgrad and not with_eval3d:` 이므로 이 경우 `.absgrad` 속성이 아예 붙지 않는다.
  (3D 평가 경로는 별도의 backward 커널을 타서 `v_means2d_abs` 출력이 없다.)

---

## 한 줄 요약

`absgrad=True`는 backward 커널이 픽셀별 ∂L/∂means2d를 **부호 있는 합(`v_means2d`)과
절댓값 합(`v_means2d_abs`) 두 벌로** 누적하게 만든다. 후자는 부호 상쇄(gradient collision)로
큰 Gaussian이 "grad가 작다"고 오판되는 것을 막아 주는 AbsGS의 homodirectional gradient이며,
`meta["means2d"].absgrad` 속성으로 나와 `DefaultStrategy(absgrad=True)`의 split/duplicate 기준이 된다.
항상 일반 grad보다 크므로 `grow_grad2d`는 0.0002 → 0.0008로 올려 준다.

## 참고

- AbsGS: Recovering Fine Details for 3D Gaussian Splatting (Ye et al., 2024) — https://arxiv.org/abs/2404.10484
- `gsplat/rendering.py` — `absgrad` 파라미터, `means2d.absgrad` 부착
- `gsplat/cuda/_wrapper.py` — `_RasterizeToPixels.forward/backward`의 `v_means2d_abs` 홀더 처리
- `gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu` — `abs(v_xy)` 누적 실체
- `gsplat/strategy/default.py` — `absgrad` 플래그, `grow_grad2d` 임계값
