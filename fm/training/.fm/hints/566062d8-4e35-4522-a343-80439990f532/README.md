# `absgrad=True` (AbsGS) — 무엇이 다른가

> **Q.** `absgrad=True` 옵션(AbsGS)은 무엇이 다른가?
> **A.** 픽셀별 gradient의 절대값 합을 쓴다. 부호 상쇄가 없어 더 민감한 분할 신호가 되며, 이때 임계값은 0.0008이 권장된다.

논문: [AbsGS: Recovering Fine Details for 3D Gaussian Splatting (arXiv:2404.10484)](https://arxiv.org/abs/2404.10484)

---

## 1. 배경 — 원래 3DGS의 "분할 신호"는 무엇인가

`DefaultStrategy`는 매 스텝 **화면공간 gradient**를 누적해서, 그 값이 임계값 `grow_grad2d`를 넘는 Gaussian을 duplicate/split 대상으로 고른다 (`gsplat/strategy/default.py:297`).

```python
is_grad_high = grads > self.grow_grad2d      # grow_grad2d 기본값 2e-4
is_small     = exp(scales).max(-1) <= grow_scale3d * scene_scale
is_dupli = is_grad_high & is_small           # 작은데 오차 큼 → 복제
is_split = is_grad_high & ~is_small          # 큰데 오차 큼 → 2개로 쪼갬
```

여기서 `grads`가 정확히 무엇인지가 이 카드의 핵심이다. 래스터라이저의 backward는 **픽셀 하나하나에 대해** 그 Gaussian의 2D 중심 `means2d`가 어느 방향으로 움직여야 손실이 줄어드는지를 계산한다. `gsplat/cuda/csrc/RasterizeToPixels3DGSDevice.cuh:166-169`:

```cpp
v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
              v_sigma * (conic.y * delta.x + conic.z * delta.y)};
```

`delta = (mean.x - px, mean.y - py)`, 즉 **Gaussian 중심에서 해당 픽셀까지의 상대 위치**가 들어간다. 따라서 이 픽셀별 기여의 *부호는 픽셀이 Gaussian의 왼쪽/오른쪽 중 어디에 있느냐에 따라 뒤집힌다.*

그리고 원래 방식은 이 기여들을 그냥 더한다 (`RasterizeToPixels3DGSSerialBatchBwd.cu:284, 306-307`):

```cpp
warpSum(v_xy_local, warp);                       // warp 내부 합
atomicAdd_system(v_xy_ptr,     v_xy_local.x);    // 전역 버퍼에 부호 그대로 누적
atomicAdd_system(v_xy_ptr + 1, v_xy_local.y);
```

수식으로 쓰면 한 뷰에서 Gaussian $i$의 신호는

$$g_i = \Big\| \sum_{p \in \Omega_i} g_{i,p} \Big\|_2$$

**먼저 벡터합(부호 포함), 그 다음 크기**다.

---

## 2. 문제 — gradient collision (부호 상쇄)

큰 Gaussian 하나가 고주파 디테일(잔디, 나뭇잎, 격자무늬, 글자)을 덮고 있는 상황을 생각해 보자.

- 왼쪽 절반 픽셀들은 "중심을 **왼쪽으로** 옮겨라"라고 말한다.
- 오른쪽 절반 픽셀들은 "중심을 **오른쪽으로** 옮겨라"라고 말한다.

두 요구는 방향이 반대이므로 벡터합에서 서로 **상쇄**된다. 결과적으로 $\sum_p g_{i,p} \approx 0$ 이 되어, **재구성 오차가 실제로는 매우 큰데도 분할 신호는 거의 0으로 측정된다.** 이 Gaussian은 임계값을 넘지 못해 영원히 쪼개지지 않고, 그 영역은 큰 Gaussian 하나로 뭉개진 채 남는다. AbsGS 논문이 지적한 **over-reconstruction**(과대 재구성 = blur) 현상이다.

즉 원래 기준은 "이 Gaussian이 **한 방향으로 일관되게** 밀리고 있는가"를 재는 것이지, "이 Gaussian이 **얼마나 틀렸는가**"를 재는 것이 아니다.

---

## 3. AbsGS의 처방 — 합 전에 절대값

고치는 방법은 놀랍도록 간단하다. **합을 취하기 전에** 픽셀별로 절대값을 씌운다. `RasterizeToPixels3DGSDevice.cuh:162-165`:

```cpp
if(compute_abs)                                  // == absgrad
{
    v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
}
```

그리고 이 값을 **원래 gradient와 별개의 버퍼**에 따로 누적한다 (`RasterizeToPixels3DGSSerialBatchBwd.cu:287, 311-313`):

```cpp
warpSum(v_xy_abs_local, warp);
...
float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
atomicAdd_system(v_xy_abs_ptr,     v_xy_abs_local.x);
atomicAdd_system(v_xy_abs_ptr + 1, v_xy_abs_local.y);
```

수식으로는

$$g_i^{\text{abs}} = \left\| \Big( \sum_{p} |g_{i,p,x}|,\ \sum_{p} |g_{i,p,y}| \Big) \right\|_2$$

**먼저 성분별 절대값 합, 그 다음 크기**다. 논문은 이를 *homodirectional*(동일방향) view-space positional gradient라고 부른다 — 모든 픽셀의 요구를 같은 방향으로 정렬해서 더한 것이기 때문이다. 부호 상쇄가 원리적으로 불가능하므로, 오차가 큰 Gaussian은 반드시 큰 값을 받는다.

> 중요: 이것은 **분할 판단에만 쓰이는 별도 통계량**이다. 파라미터를 실제로 업데이트하는 `means2d.grad`(정상 gradient)는 그대로 유지된다. Adam이 절대값 gradient로 학습하는 게 아니다.

---

## 4. 파이썬 쪽 연결 — 두 곳 모두 켜야 한다

**(a) 래스터라이저**가 abs 버퍼를 계산해서 텐서에 붙여 준다 (`gsplat/rendering.py:652-653`):

```python
if absgrad and not with_eval3d:
    means2d.absgrad = means2d_absgrad
```

**(b) 전략**이 `.grad` 대신 `.absgrad`를 읽는다 (`gsplat/strategy/default.py:244-247`):

```python
if self.absgrad:
    grads = info[self.key_for_gradient].absgrad.clone()
else:
    grads = info[self.key_for_gradient].grad.clone()
grads[..., 0] *= info["width"]  / 2.0 * info["n_cameras"]    # [-1,1] NDC 정규화
grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
```

이후 누적과 평균은 두 모드에서 완전히 동일하다 (`default.py:275, 294`):

```python
state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))   # 뷰별 norm을 누적
...
grads = state["grad2d"] / count.clamp_min(1)                # 관측 횟수로 평균
```

`grads.norm(dim=-1)`을 뷰마다 취한 뒤 더하므로, **뷰 사이의 상쇄는 원래도 없었다.** AbsGS가 해결하는 것은 *한 뷰 안에서 픽셀들 사이의* 상쇄뿐이다.

`simple_trainer.py`는 전략 설정을 그대로 래스터라이저에 넘긴다 (`examples/simple_trainer.py:733-737`):

```python
absgrad=(self.cfg.strategy.absgrad
         if isinstance(self.cfg.strategy, DefaultStrategy) else False),
```

따라서 CLI에서는 `--strategy.absgrad --strategy.grow-grad2d 0.0008` 한 세트로 켜면 (a),(b)가 함께 맞춰진다. **직접 코드를 쓸 때 `DefaultStrategy(absgrad=True)`만 주고 `rasterization(...)`에 `absgrad=True`를 빼먹으면 `means2d`에 `.absgrad` 속성이 없어 `AttributeError`가 난다.** docstring이 명시적으로 경고하는 함정이다 (`default.py:49-51`).

---

## 5. 왜 임계값이 `0.0002` → `0.0008`인가

절대값 부등식에서 성분별로

$$\sum_p |g_{i,p,x}| \ \ge\ \Big|\sum_p g_{i,p,x}\Big|, \qquad \sum_p |g_{i,p,y}| \ \ge\ \Big|\sum_p g_{i,p,y}\Big|$$

이고, 노름은 각 성분의 절대값에 대해 단조증가하므로 **언제나**

$$g_i^{\text{abs}} \ \ge\ g_i$$

가 성립한다. 등호는 모든 픽셀 기여가 같은 방향(같은 사분면)일 때만 성립하고, 실제로는 넓게 퍼진 Gaussian일수록 부등호가 크게 벌어진다.

즉 AbsGS 신호는 **원래 신호보다 체계적으로 큰 스케일**을 가진다. 여기에 기존 임계값 `2e-4`를 그대로 쓰면 거의 모든 Gaussian이 임계값을 통과해서 매 100스텝마다 개수가 폭발하고 VRAM이 터진다. 그래서 스케일 변화를 보상하는 4배 높은 값 `0.0008`이 권장된다 (`default.py:47-49`).

| | 원래 3DGS | AbsGS (`absgrad=True`) |
|---|---|---|
| 픽셀 기여 결합 | 벡터합 → 크기 | 성분별 절대값 합 → 크기 |
| 상쇄 | 있음 (부호 반대끼리) | 없음 |
| 재는 것 | 이동 방향의 **일관성** | 오차의 **총량** |
| 스케일 | 작음 | 항상 ≥ 원래 값 |
| 권장 `grow_grad2d` | `0.0002` | `0.0008` |
| 파라미터 업데이트 | `means2d.grad` | `means2d.grad` (동일, 변화 없음) |
| 효과 | 고주파 영역 뭉개짐 | 얇은/고주파 구조 디테일 복원 |

---

## 6. 제약과 주의사항

- **`distributed=True`와 함께 못 쓴다.** `gsplat/cuda/csrc/Rendering.cpp:194`에 하드 체크가 있다: `TORCH_CHECK(!absgrad, "distributed=True does not support absgrad=True")`. abs 통계를 rank 간에 주고받는 경로가 없기 때문이다.
- **`with_eval3d=True`(3DGUT 계열)에서는 조용히 붙지 않는다.** `rendering.py:652`의 `if absgrad and not with_eval3d` 때문에 `.absgrad`가 설정되지 않으므로, 이 조합에서 `DefaultStrategy(absgrad=True)`를 쓰면 실패한다.
- **`MCMCStrategy`에는 없는 옵션이다.** MCMC는 gradient 임계값 휴리스틱 자체를 쓰지 않고 opacity 기반 확률적 재배치 + SGLD 노이즈로 densify하므로 `absgrad` 파라미터가 존재하지 않는다. 그래서 `simple_trainer.py`도 `isinstance(..., DefaultStrategy)`로 가드한다.
- **2DGS에서도 동작한다.** `key_for_gradient="gradient_2dgs"`로 바꿔 쓰면 같은 로직이 2DGS의 대응 gradient에 적용된다.
- 메모리 비용은 `means2d`와 같은 크기의 버퍼 하나(`at::zeros_like(means2d)`, `Rasterization.cpp:886-890`)로 무시할 만하다. abs 버퍼는 `absgrad=False`일 때 아예 할당되지 않는다(`at::empty({0})`).

---

## 7. 한 문장 요약

`absgrad=True`는 **densification 판단 기준을 "픽셀 기여의 벡터합 크기"에서 "픽셀 기여 크기의 합"으로 바꾸는 스위치**다. 부호 상쇄가 사라져 고주파 영역을 덮은 큰 Gaussian이 드디어 분할 신호를 받게 되고, 신호 스케일이 커진 만큼 임계값을 `0.0002`에서 `0.0008`로 올려 균형을 맞춘다.

### 참고 코드 위치

| 파일 | 줄 | 내용 |
|---|---|---|
| `gsplat/cuda/csrc/RasterizeToPixels3DGSDevice.cuh` | 162-165 | 픽셀 기여에 `abs()` 적용 |
| `gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu` | 287, 311-313 | 별도 버퍼로 warpSum + atomicAdd |
| `gsplat/cuda/csrc/Rasterization.cpp` | 886-890 | abs 버퍼 조건부 할당 |
| `gsplat/cuda/csrc/Rendering.cpp` | 194 | `distributed` 비호환 체크 |
| `gsplat/rendering.py` | 382-387, 652-653 | 문서 note, `.absgrad` 속성 부착 |
| `gsplat/strategy/default.py` | 44-51, 111, 244-247 | docstring 경고, 필드, 분기 |
| `examples/simple_trainer.py` | 733-737 | 전략 설정 → 래스터라이저 전달 |
