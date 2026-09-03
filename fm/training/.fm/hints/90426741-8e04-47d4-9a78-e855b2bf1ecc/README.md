# 밀도화(densification)가 3DGS 학습에 필수인 이유

> **Q.** 밀도화(densification)가 3DGS 학습에 필수인 이유는?
> **A.** SfM 포인트만으로는 씬을 다 덮지 못하기 때문이다. 학습 중에 Gaussian을 늘리고 정리하는 과정이 필요하다.

---

## 1. 핵심 한 줄

3DGS의 학습 대상은 **"Gaussian 하나하나의 파라미터"** 뿐인데, 정작 화질을 좌우하는 것은
**"Gaussian이 몇 개 있고 어디에 있느냐"** 다. 후자는 gradient descent로 못 바꾸므로,
학습 루프 안에서 **개수를 이산적으로 바꾸는 별도 절차**가 필요하다. 그것이 밀도화다.

## 2. 출발점: SfM 포인트는 "씬"이 아니라 "매칭에 성공한 코너 목록"이다

워크스루 1단계(`training_walkthrough.py`, "1단계: 데이터 준비 — COLMAP SfM 결과 로드")에서
Gaussian의 초기 위치·색은 COLMAP의 sparse 포인트에서 온다. `init_type="sfm"`
(`examples/simple_trainer.py:154`, `:311`)이 바로 그 경로다.

문제는 SfM 포인트가 만들어지는 방식이다. COLMAP은 **여러 사진에서 같은 특징점(SIFT류)을
반복 매칭해 삼각측량이 성공한 점만** 남긴다. 그래서 구조적으로 다음 영역에는 포인트가 거의 없다.

| SfM이 포인트를 못 만드는 곳 | 이유 |
|---|---|
| 무텍스처 벽·바닥·하늘 | 특징점이 안 잡히고, 잡혀도 매칭이 애매함 |
| 반사·투명 표면 (창문, 금속, 물) | 시점마다 외관이 달라져 매칭 실패 |
| 얇은 구조 (나뭇가지, 철망, 케이블) | 특징점이 불안정하고 outlier로 걸러짐 |
| 시점 커버리지가 얕은 배경/원경 | 시차(parallax)가 부족해 삼각측량 불안정 |
| 반복 패턴 (타일, 벽돌) | 잘못된 매칭이 RANSAC에서 제거됨 |

또한 **밀도의 절대량 자체가 부족하다.** Mip-NeRF 360급 씬의 SfM 포인트는 보통
수만~수십만 개인데, 같은 씬을 3DGS로 제대로 재구성하면 최종 Gaussian은 **100만~600만 개**
규모가 된다. 즉 초기 포인트는 최종 필요량의 몇 % 수준이다. 밀도화 없이 학습하면
"처음 준 점 개수"가 그대로 표현력 상한이 된다.

## 3. 초기화 로직이 이 문제를 더 노골적으로 드러낸다

워크스루 2단계의 초기 스케일 설정을 보자.

```python
dist_avg = knn_mean_dist(points, k=3)
scales = torch.log(dist_avg)[:, None].repeat(1, 3)   # log-space
```

각 Gaussian의 크기를 **3-최근접 이웃까지의 평균 거리**로 잡는다. 빈틈 없이 덮으려는
의도인데, 부작용이 그대로 따라온다.

- 포인트가 촘촘한 곳 → 작고 선명한 Gaussian (좋음)
- 포인트가 희박한 곳 → **이웃이 멀어서 거대한 blob 하나**가 넓은 면적을 담당

거대한 blob은 저주파 성분밖에 표현할 수 없다. 워크스루 4단계 직전의
"초기 상태 렌더 (SfM 색만)" 이미지가 흐릿한 이유가 이것이다. 이 blob은
파라미터를 어떻게 최적화해도 고주파 디테일을 만들 수 없고, **쪼개져야만** 해결된다.

## 4. 왜 최적화만으로는 해결되지 않는가 (개수는 미분 가능하지 않다)

학습 루프에서 gradient가 흐르는 대상은 `means / scales / quats / opacities / sh0 / shN`
뿐이다. Gaussian 개수 N은 텐서의 **shape**이고, 손실을 N으로 미분한다는 개념 자체가 없다.

따라서 gradient descent가 할 수 있는 것과 할 수 없는 것이 갈린다.

| gradient가 할 수 있는 일 | 할 수 없는 일 |
|---|---|
| 기존 Gaussian을 옮기고, 늘리고, 회전하고, 색·불투명도 조정 | 새 Gaussian을 만들기 |
| 필요 없는 Gaussian의 opacity를 0에 가깝게 낮추기 | 실제로 메모리에서 없애기 |
| 하나의 blob을 최적 위치의 하나의 blob으로 만들기 | 하나의 blob을 두 개로 쪼개기 |

여기서 전형적인 실패 모드가 나온다. 넓은 영역을 하나의 Gaussian이 담당하고 있으면,
그 영역 안의 서로 다른 픽셀들이 **서로 반대 방향의 gradient**를 보낸다(왼쪽은 더
어둡게, 오른쪽은 더 밝게). 평균을 내면 파라미터는 거의 안 움직이고 loss는 정체된다.
대신 **화면공간(2D) gradient의 크기**만 계속 커진다. 이것이 3DGS가 사용하는 신호다:

> **"화면공간 gradient가 크다 = 이 Gaussian은 자기 담당 구역을 혼자 감당 못 하고 있다 =
> 여기에 Gaussian이 더 필요하다"**

그래서 `step_pre_backward()`가 `info["means2d"].retain_grad()`를 호출한다
(`gsplat/strategy/default.py:170`). backward가 끝난 뒤에도 중간 텐서의 gradient를
읽어야 하기 때문이다. 이 값은 `[-1,1]` NDC 기준이 되도록 이미지 폭/높이로 스케일링한 뒤
누적된다(`default.py:226` `_update_state`).

```python
grads[..., 0] *= info["width"]  / 2.0 * info["n_cameras"]
grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
...
state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
```

`grad2d / count`, 즉 **보인 횟수로 나눈 평균**을 쓰는 것도 중요하다. 그렇지 않으면
많은 카메라에 등장하는 Gaussian이 부당하게 유리해진다.

## 5. gsplat `DefaultStrategy`가 실제로 하는 네 가지

`gsplat/strategy/default.py`의 `DefaultStrategy`가 원논문 방식을 구현한다.
매 스텝 `step_pre_backward()` / `step_post_backward()` 두 훅이 불리고,
후자가 통계 누적 + 주기적 refine을 담당한다.

| 동작 | 조건 (기본값) | 효과 | 코드 |
|---|---|---|---|
| **duplicate** | 평균 화면 grad > `2e-4` **and** 크기 ≤ `1%`·scene_scale | 작은데 오차 큰 곳 → 그대로 복제 (같은 위치·같은 크기) | `default.py:303` / `ops.py:141` |
| **split** | 평균 화면 grad > `2e-4` **and** 크기 > `1%`·scene_scale | 큰데 오차 큰 곳 → 2개로 쪼개고 크기 `/1.6` | `default.py:307` / `ops.py:175` |
| **prune** | `opacity < 0.005`, 또는 크기 > `10%`·scene_scale | 기여 없는/비대한 것 제거 | `default.py:343` |
| **opacity reset** | 매 `3000` 스텝 | 전체 opacity를 `prune_opa*2 = 0.01`로 리셋 | `default.py:222` |

두 성장 동작의 분기 기준을 코드로 보면 이렇다.

```python
is_grad_high = grads > self.grow_grad2d                       # 2e-4
is_small = torch.exp(params["scales"]).max(dim=-1).values \
           <= self.grow_scale3d * state["scene_scale"]        # 0.01 * scene_scale
is_dupli = is_grad_high & is_small     # under-reconstruction  → 복제
is_split = is_grad_high & ~is_small    # over-reconstruction   → 분할
```

**왜 두 갈래로 나누는가?** 원논문의 표현을 쓰면

- *under-reconstruction*: 그 자리에 필요한 형상 대비 **Gaussian이 모자란** 상태.
  크기는 적절하지만 커버가 안 됨 → **복제**해서 개수를 늘린다. 복제된 쌍은 같은 위치에서
  시작하지만 각자 다른 gradient를 받아 자연스럽게 갈라진다.
- *over-reconstruction*: 하나가 **너무 넓은 영역을 뭉개고 있는** 상태
  (§3의 blob) → **분할**해서 해상도를 올린다.

분할은 단순 복제가 아니라 원래 Gaussian의 공분산을 확률분포로 보고 그 안에서 두 점을
샘플링한다(`ops.py:199`).

```python
samples = torch.einsum("nij,nj,bnj->bni", rotmats, scales,
                       torch.randn(2, len(scales), 3, device=device))
...
p_split = (p[sel] + samples).reshape(-1, 3)      # means: 공분산 내부에서 샘플링
p_split = torch.log(scales / 1.6).repeat(2, 1)   # scales: 1/1.6 로 축소
```

`1.6`은 원논문의 실험적 상수 φ다. 2개가 원래 영역을 대략 같은 밀도로 덮으면서도
서로 과하게 겹치지 않는 절충값이다.

## 6. "늘리기"만이 아니라 "정리"가 같이 필요한 이유

카드 답의 후반부("늘리고 **정리하는**")가 실무적으로 더 중요할 때가 많다.

1. **개수 폭발 방지.** duplicate/split은 100스텝마다 조건을 만족하는 모든 Gaussian에
   적용된다. 제거가 없으면 개수가 지수적으로 늘어 VRAM이 먼저 터진다.
2. **floater 제거.** 특정 시점에서만 그럴듯하게 보이는 반투명 덩어리가 공중에 생긴다.
   opacity가 낮게 수렴하지만 gradient가 거의 안 흘러서 스스로 사라지지 않는다.
   → `prune_opa = 0.005` 미만 제거.
3. **opacity reset이라는 강제 재심사.** 3000스텝마다 **모든** Gaussian의 opacity를
   0.01로 되돌린다. 정말 필요한 Gaussian은 다음 refine 구간에서 opacity를 다시 끌어올리고,
   필요 없던 것은 못 올라와서 다음 prune에 걸린다. 즉 "전원 재신임 투표"에 가깝다.
4. **비대한 Gaussian 제거.** `prune_scale3d = 0.1` — 씬 크기의 10%를 넘는 Gaussian은
   배경을 통째로 덮는 유령이 되므로 제거한다. 단 이 조건은 `step > reset_every`
   이후에만 켜진다(초기의 큰 초기화 blob을 성급히 죽이지 않기 위해).

모든 임계값이 **`scene_scale` 상대값**이라는 점도 눈여겨볼 만하다
(`training_walkthrough.py` 1단계: `scene_scale = parser.scene_scale * 1.1`).
그래서 실내 책상 씬이든 야외 정원 씬이든 같은 하이퍼파라미터가 통한다.

## 7. 언제 하는가 — refine 스케줄

```python
if step >= self.refine_stop_iter:            # 15_000
    return
self._update_state(...)                      # 매 스텝 통계 누적
if (step > self.refine_start_iter            # 500
    and step % self.refine_every == 0        # 100
    and step % self.reset_every >= self.pause_refine_after_reset):
    self._grow_gs(...)                       # duplicate → split
    self._prune_gs(...)                      # prune
    state["grad2d"].zero_(); state["count"].zero_()
```

- `500` 이전에는 아직 색/위치가 엉망이라 gradient 신호가 신뢰할 수 없다 → 대기.
- `100`스텝마다: 매 스텝 구조를 바꾸면 Adam의 모멘텀 상태가 계속 리셋되어(새 Gaussian의
  optimizer state는 0으로 채워진다 — `ops.py`의 `optimizer_fn`) 최적화가 불안정해진다.
- refine 직후 `grad2d`/`count`를 0으로 비운다 → 통계는 항상 "최근 100스텝" 기준.
- `15_000`(총 30k의 절반) 이후 중단: 마지막 절반은 **구조를 고정한 채 파라미터만 수렴**시켜
  마무리한다. 끝까지 densify하면 수렴하지 못한 새 Gaussian이 남아 화질이 오히려 나빠진다.

그래서 워크스루의 결과 확인 섹션에서 언급된 학습 곡선 패턴이 나온다.

- Gaussian 개수: 스텝 500 이후 **100스텝마다 계단식 증가**
- 3000스텝 배수 직후: opacity reset 때문에 **loss가 튀었다가 회복**
- 15000스텝 이후: **개수 고정**, loss만 완만히 감소

## 8. 대안 관점 — MCMC 전략과 비교하면 필요성이 더 분명해진다

`gsplat/strategy/mcmc.py`의 `MCMCStrategy`는 같은 문제를 다른 방식으로 푼다.
휴리스틱 grad 임계값 대신 SGLD 관점을 쓴다.

| | `DefaultStrategy` | `MCMCStrategy` |
|---|---|---|
| 성장 신호 | 화면공간 gradient 임계값 | opacity 기반 확률적 재배치 |
| 개수 제어 | prune으로 간접 조절 (결과 예측 어려움) | `cap_max`(기본 `1_000_000`) 상한 고정 |
| 죽은 Gaussian | prune으로 삭제 | `min_opacity` 이하를 살아있는 곳으로 **relocate** |
| 추가 항 | — | `noise_lr`(5e5) 노이즈 주입 + `opacity_reg`/`scale_reg` |

접근은 정반대지만 **둘 다 "학습 중에 Gaussian 집합을 재편한다"는 전제는 공유한다.**
즉 밀도화는 3DGS 구현의 선택적 최적화가 아니라 **필수 구성요소**다.
`--absgrad` + `grow_grad2d=0.0008` (AbsGS) 같은 변종도 전부
"성장 신호를 어떻게 더 잘 측정할까"라는 같은 문제의 개선안이다.

## 9. 정리 — 카드 답을 왜 그렇게 쓰는가

| 답의 구절 | 근거 |
|---|---|
| "SfM 포인트만으로는 씬을 다 덮지 못한다" | 무텍스처/반사/얇은 구조/원경에 포인트 없음 + 절대량이 최종 필요량의 몇 % (§2) |
| "학습 중에" | 개수는 미분 불가 → optimizer 밖의 이산 연산이어야 함 (§4) |
| "늘리고" | duplicate(under-reconstruction) + split(over-reconstruction) (§5) |
| "정리하는" | prune + opacity reset — 없으면 개수 폭발·floater 잔존 (§6) |

**흔한 오해 정리**

- ❌ "SfM 포인트를 더 뽑아 넣으면 밀도화가 필요 없다."
  → 무텍스처 영역에는 애초에 뽑을 특징점이 없다. 게다가 어디에 얼마나 필요한지는
  **렌더 오차를 봐야만** 알 수 있는데, 그 정보는 학습 중에만 생긴다.
- ❌ "`init_type="random"`으로 시작하면 초기화 문제가 없다."
  → 랜덤 초기화도 동작하지만(`simple_trainer.py:314`) SfM 초기화보다 수렴이 느리고
  품질이 낮다. 어느 쪽이든 밀도화가 없으면 개수 상한에 묶인다.
- ❌ "opacity가 낮은 Gaussian은 어차피 렌더에 안 보이니 놔둬도 된다."
  → 보이지 않아도 정렬·타일 처리·메모리 비용은 그대로 낸다. 게다가 α-블렌딩 순서에
  끼어들어 뒤쪽 Gaussian의 gradient를 왜곡할 수 있다.
