# 화면공간 gradient의 정규화 기준: `[-1,1]` NDC

**Q.** 화면공간 gradient는 어떤 기준으로 정규화해서 누적하는가?
**A.** `[-1,1]` NDC 기준으로 정규화해 누적한다(`default.py:248`). 해상도에 무관한 임계값을 쓸 수 있게 된다.

---

## 1. 문제 설정: 밀도화의 신호는 "화면에서의 위치 gradient"

3DGS의 밀도화(densification)는 "이 Gaussian이 담당하는 영역에서 오차가 크다"를 어떻게 감지할까?
원논문의 답은 **화면공간 위치의 gradient 크기**다. 어떤 Gaussian을 화면에서 조금 옮겼을 때
손실이 크게 줄어든다면, 그 자리에 있는 Gaussian 하나로는 그 영역의 디테일을 표현하지 못한다는 뜻이므로
복제(duplicate)하거나 분할(split)해야 한다.

워크스루의 표현을 그대로 옮기면 (`training_walkthrough.py`, 5단계):

> 반환되는 `info` dict가 **밀도화 전략의 입력**이 된다. 특히 `info["means2d"]`는
> 화면공간 위치 텐서로, 이것의 gradient가 "이 Gaussian을 더 쪼개야 하는가"의 신호다.

그래서 `DefaultStrategy.step_pre_backward()`가 `info["means2d"].retain_grad()`를 호출해
backward 이후에도 이 중간 텐서의 gradient를 읽을 수 있게 남겨 둔다
(`gsplat/strategy/default.py:170`).

## 2. `means2d`의 단위는 "픽셀"이다

`fully_fused_projection`이 내놓는 `means2d`는 **픽셀 좌표**다 (shape `[C, N, 2]`,
`packed=True`면 `[nnz, 2]`). 즉 `retain_grad()`로 얻는 `means2d.grad`의 단위는

$$\frac{\partial \mathcal{L}}{\partial x_{\text{pix}}} \quad [\text{손실}/\text{픽셀}]$$

이다. 이 값을 그대로 임계값과 비교하면 **해상도가 바뀔 때마다 임계값을 다시 튜닝해야 한다**.
1600px 폭으로 렌더할 때의 "1픽셀"과 400px 폭으로 렌더할 때의 "1픽셀"은 씬에서 차지하는
비중이 4배 다르기 때문이다. gsplat 예제는 `-d 4`, `-d 2` 같은 다운스케일 옵션을
일상적으로 쓰므로 이건 실질적인 문제다.

## 3. 해법: 픽셀 gradient를 NDC gradient로 환산 (`default.py:243-249`)

```python
# normalize grads to [-1, 1] screen space
if self.absgrad:
    grads = info[self.key_for_gradient].absgrad.clone()
else:
    grads = info[self.key_for_gradient].grad.clone()
grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]   # ← line 248
grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]  # ← line 249
```

핵심은 **연쇄법칙에 의한 단위 변환**이다. NDC(Normalized Device Coordinates)에서
화면 가로는 $x_n \in [-1, 1]$, 세로는 $y_n \in [-1, 1]$로 표현되고 픽셀 좌표와는

$$x_{\text{pix}} = \frac{x_n + 1}{2}\,W, \qquad y_{\text{pix}} = \frac{y_n + 1}{2}\,H$$

로 대응된다. 따라서

$$\frac{\partial x_{\text{pix}}}{\partial x_n} = \frac{W}{2}, \qquad
\frac{\partial \mathcal{L}}{\partial x_n}
= \frac{\partial \mathcal{L}}{\partial x_{\text{pix}}} \cdot \frac{W}{2}$$

즉 line 248의 `* width / 2.0`은 새로 렌더를 하거나 근사를 하는 게 아니라, 이미 계산된
픽셀 단위 gradient에 야코비안 $W/2$를 곱해 **NDC 단위 gradient로 정확히 환산**하는 것이다.
`height / 2.0`도 y축에 대해 같다. 결과의 의미는 "Gaussian을 화면 절반 폭만큼 옮길 때의
손실 변화율"이므로 해상도에 의존하지 않는다.

### 곁가지: `* n_cameras`는 무엇인가

같은 줄에 `* info["n_cameras"]`가 함께 곱해져 있다. 학습 손실은 배치 안의 카메라들에 대해
**평균**으로 계산된다(`simple_trainer.py`의 `l1_loss(colors, pixels).mean()` 등). 그래서
카메라 $C$대를 한 배치로 묶으면 각 카메라가 gradient에 기여하는 크기가 $1/C$로 줄어든다.
`* n_cameras`는 이 배치 평균을 되돌려서, 누적되는 통계가 **배치 크기와 무관한 "카메라 1대 기준"**
값이 되게 한다. 정리하면 line 248/249는 두 가지 무관성을 동시에 만든다.

| 곱하는 항 | 없애는 의존성 |
|---|---|
| `width / 2.0`, `height / 2.0` | 렌더 **해상도**에 대한 의존성 (픽셀 → NDC) |
| `n_cameras` | **배치 크기**에 대한 의존성 (배치 평균 되돌리기) |

## 4. 누적: 보이는 뷰에 대해서만, 노름의 합 / 횟수

정규화된 gradient는 곧바로 러닝 통계에 더해진다(`default.py`, `_update_state`):

```python
sel = (info["radii"] > 0.0).all(dim=-1)   # [C, N]  화면에 실제로 보이는 것만
gs_ids = torch.where(sel)[1]              # [nnz]
grads = grads[sel]                        # [nnz, 2]
...
state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
```

- `radii > 0`인, 즉 near/far·화면 밖 컬링을 통과한 Gaussian만 통계에 들어간다.
  뒤에 있어서 보이지도 않는 Gaussian이 "gradient 0"으로 평균을 깎는 일을 막는다.
- 누적되는 값은 2차원 벡터가 아니라 그 **L2 노름** $\sqrt{(\partial\mathcal{L}/\partial x_n)^2 + (\partial\mathcal{L}/\partial y_n)^2}$ 이다.
  방향은 버리고 크기만 쓴다.
- `count`는 그 Gaussian이 보인 (뷰, 스텝) 횟수를 센다.

refine 시점(`_grow_gs`)에서 평균을 낸다:

```python
count = state["count"]
grads = state["grad2d"] / count.clamp_min(1)
is_grad_high = grads > self.grow_grad2d      # grow_grad2d = 2e-4
```

그래서 최종 판정 기준은 **"보인 뷰들에 대한 NDC gradient 노름의 평균"** 이고,
이것을 상수 임계값 `grow_grad2d = 2e-4`와 비교한다. 이 상수가 400px 렌더와 1600px 렌더에서
똑같이 통하는 이유가 바로 line 248의 정규화다. refine 직후 `state["grad2d"].zero_()`,
`state["count"].zero_()`로 통계를 리셋하고 다음 100스텝 구간을 새로 모은다.

이후 분기는 크기 기준으로 갈린다(워크스루 5단계 표와 동일).

| 동작 | 조건 (기본값) |
|---|---|
| duplicate | 평균 NDC grad > `2e-4` **and** 3D 크기 ≤ `0.01`·scene_scale |
| split | 평균 NDC grad > `2e-4` **and** 3D 크기 > `0.01`·scene_scale |

## 5. 같은 철학의 이웃 코드: `radii`도 정규화된다

바로 아래에서 화면상 반경도 같은 방식으로 해상도 무관하게 만든다.

```python
state["radii"][gs_ids] = torch.maximum(
    state["radii"][gs_ids],
    # normalize radii to [0, 1] screen space
    radii / float(max(info["width"], info["height"])),
)
```

여기서는 `[0,1]` 기준이고(반경이므로 음수가 없다), 그래서 임계값 `grow_scale2d = 0.05`가
"화면 긴 변의 5%보다 크게 보이면 쪼갠다"라는 해상도 무관한 규칙으로 읽힌다.
즉 **정규화 기준을 코드에 박아 하이퍼파라미터를 해상도 독립으로 만드는 것**이
`DefaultStrategy` 전반의 설계 원칙이다.

## 6. `absgrad`를 켜도 정규화 기준은 그대로다

`absgrad=True`면 `.grad` 대신 `.absgrad`를 가져온다. 이는 픽셀별 gradient를 부호까지
합산해서 상쇄시키지 않고 **절댓값을 합산**한 AbsGS 방식의 값이다
([AbsGS, arXiv:2404.10484](https://arxiv.org/abs/2404.10484)).

주목할 점은 line 248/249의 `* W/2`, `* H/2`, `* n_cameras`가 `absgrad` 분기 밖에 있다는 것이다.
즉 **어느 쪽을 쓰든 정규화 기준은 동일하게 `[-1,1]` NDC**다. 달라지는 건 값의 스케일뿐이고,
상쇄가 없어 값이 전반적으로 커지므로 임계값을 `2e-4` → `0.0008` 정도로 올려 주면 된다
(워크스루 마지막 절의 권장값과 동일). 또한 이 값을 얻으려면 `rasterization(..., absgrad=True)`로
렌더해서 CUDA 커널이 `means2d.absgrad`를 채워 주게 해야 한다
(`gsplat/cuda/_wrapper.py:1559-1560`).

## 7. 한 줄 요약과 흔한 오해

- **정답 요약**: 픽셀 단위 `means2d.grad`에 야코비안 $W/2$, $H/2$를 곱해 `[-1,1]` NDC 단위로
  환산하고, 여기에 배치 평균 보정 `n_cameras`를 곱한 뒤 L2 노름을 뷰마다 누적해 평균을 낸다.
  덕분에 `grow_grad2d = 2e-4`가 해상도·배치 크기와 무관한 상수가 된다.
- **오해 1**: "gradient를 `[-1,1]` 범위로 클리핑한다"가 아니다. 정규화되는 것은 gradient의
  **좌표계(단위)**이고, gradient 값 자체의 범위는 제한하지 않는다.
- **오해 2**: 정사각형이 아닌 이미지에서는 x축과 y축에 서로 다른 상수($W/2$ vs $H/2$)가 곱해진다.
  이건 버그가 아니라 NDC의 정의가 두 축을 각각 `[-1,1]`로 만드는 것이기 때문이다
  (즉 종횡비는 NDC 좌표계 안으로 흡수된다).
- **오해 3**: 나누는 것은 refine 주기(100)가 아니라 `count`, 즉 **실제로 보인 횟수**다.
