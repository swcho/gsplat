# 손실에서 L1 항의 역할

**Q.** 손실에서 L1 항이 담당하는 역할은?

**A.** 픽셀 단위 색 재구성을 담당하며 노이즈에 강건하다. 다만 L1만 쓰면 결과가 뭉개지는 경향이 있다.

---

## 1. 손실 안에서 L1이 놓인 자리

워크스루(`training_walkthrough.py` 4단계, L266–285)가 정리한 3DGS의 기본 손실은 단 두 항이다.

$$\mathcal{L} = (1-\lambda)\,\mathcal{L}_{L1} + \lambda\,(1-\mathrm{SSIM}), \qquad \lambda = 0.2$$

`examples/simple_trainer.py`의 학습 루프(L946–961)가 그대로 이 식이다.

```python
if masks is not None:
    # Exclude masked pixels (e.g. ego vehicle) from L1.
    l1loss = l1_loss(colors[masks], pixels[masks]).mean()
    colors_ssim = colors * masks[..., None]
    pixels_ssim = pixels * masks[..., None]
else:
    l1loss = l1_loss(colors, pixels).mean()
    colors_ssim = colors
    pixels_ssim = pixels
ssimloss = ssim_loss(
    colors_ssim.permute(0, 3, 1, 2), pixels_ssim.permute(0, 3, 1, 2)
)
loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)   # ssim_lambda = 0.2 (L171)
```

`torch.lerp(a, b, w) = a + w·(b − a) = (1−w)·a + w·b`이므로 **L1이 가중치 0.8, SSIM이 0.2**다. 즉 L1은 "보조 항"이 아니라 손실의 몸통이고, SSIM은 그 위에 얹는 구조 보정항이다. 그래서 카드의 답 첫 문장 — "픽셀 단위 색 재구성을 담당" — 이 문자 그대로 맞다. **어떤 픽셀이 어떤 색이어야 하는가**를 알려주는 신호는 전부 L1에서 나온다.

구현 자체는 한 줄이다 (`gsplat/losses.py` L53–63).

```python
def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Element-wise L1 (absolute error) loss."""
    return F.l1_loss(pred, target, reduction="none")
```

`reduction="none"`이 중요하다. 모듈 docstring이 밝히듯 gsplat의 손실들은 **감축하지 않은(per-element)** 값을 돌려주고, 감축은 호출자가 한다. 그래서 위 학습 루프가 `.mean()`을 직접 붙이고, 마스크가 있으면 `colors[masks]`로 **픽셀을 골라낸 뒤** 평균할 수 있다.

## 2. "픽셀 단위"라는 말이 실제로 뜻하는 것

L1은 픽셀 $i$마다 $|c_i - \hat c_i|$를 따로 계산하고, 픽셀 사이를 섞지 않는다. 이 점이 SSIM과의 결정적 차이이고, 파이프라인에서 세 가지 실질적 결과를 낳는다.

**(a) 마스킹이 정확히 된다.** 위 코드에서 L1은 `colors[masks]`로 유효 픽셀만 뽑아 평균한다. SSIM은 11×11 가우시안 윈도우를 쓰는 패치 기반이라 그럴 수 없어서, 양쪽을 0으로 곱해 "가려진 패치가 색을 임의 값으로 끌지 않게" 하는 우회를 쓴다(코드 주석 그대로). 즉 자율주행 데이터의 ego vehicle처럼 **버려야 하는 픽셀**을 깔끔히 배제하는 일은 L1만 할 수 있다.

**(b) 이미지 경계도 감독된다.** `ssim_loss`는 `fused_ssim`이 있으면 `padding="valid"`로 호출된다(`gsplat/losses.py` L181–186). valid 패딩은 윈도우가 완전히 들어가는 영역만 보므로 **테두리 5픽셀에는 SSIM gradient가 없다**. 그 영역의 유일한 supervision이 L1이다.

**(c) 밀도화의 신호원이 된다.** L1(과 SSIM) gradient가 `info["means2d"]`까지 흘러가고, `DefaultStrategy`가 그것을 누적한 `state["grad2d"]`를 `grow_grad2d = 2e-4`와 비교해 duplicate/split을 결정한다(`gsplat/strategy/default.py` L100, L273, L295–298). 픽셀 오차가 큰 곳 → 화면공간 gradient가 큼 → Gaussian이 그 위치에 더 생긴다. **어디에 Gaussian이 부족한가**를 판단하는 근거도 결국 픽셀 단위 색 오차다.

## 3. 왜 "노이즈에 강건"한가 — L1 vs L2

같은 파일에 `mse_loss`(L2)도 있는데(`losses.py` L66–76) 3DGS는 L1을 쓴다. 이유는 gradient의 모양에 있다.

$$\frac{\partial}{\partial \hat c}|\hat c - c| = \mathrm{sign}(\hat c - c) \in \{-1, +1\}, \qquad \frac{\partial}{\partial \hat c}(\hat c - c)^2 = 2(\hat c - c)$$

L1의 gradient는 **크기가 항상 1**이고 방향만 알려준다. L2는 오차에 비례해 커진다. 픽셀 오차가 `[0.01, 0.01, 0.9]`일 때:

| | 픽셀별 gradient 크기 |
|---|---|
| L1 | `1.0, 1.0, 1.0` |
| L2 | `0.02, 0.02, 1.8` |

L2에서는 크게 튀는 한 픽셀이 나머지 전부를 합친 것보다 90배 큰 힘으로 파라미터를 끌어당긴다. 3DGS 학습 데이터에는 이런 "튀는 픽셀"이 구조적으로 존재한다 — 뷰마다 달라지는 스페큘러 하이라이트, 노출·화이트밸런스 불일치, 지나가는 사람이나 차, JPEG/센서 노이즈, COLMAP 포즈의 미세 오차. 이 픽셀들은 **어떤 Gaussian 배치로도 동시에 만족시킬 수 없는** 모순된 요구인데, L2는 거기에 최대 가중치를 준다.

통계적으로 말하면 L2의 최적해는 **평균**, L1의 최적해는 **중앙값**이다. 한 뷰만 크게 다른 관측 `[0.30, 0.32, 0.31, 0.95]`에 대해

```
mean (L2 최적)   = 0.4700
median (L1 최적) = 0.3150
```

L2는 이상치 하나에 끌려 0.47이라는, 어느 뷰에도 맞지 않는 색으로 수렴한다. L1은 0.315 — 다수 뷰가 동의하는 색을 지킨다. 멀티뷰 재구성에서 이 성질은 그대로 "고스팅/떠다니는 floater가 덜 생긴다"로 나타난다.

같은 논리로 gsplat의 다른 항들도 L1을 쓴다. `depth_l1_loss`(L209–224)는 SfM 포인트의 **희소하고 노이즈 많은** 깊이를 disparity 공간에서 L1으로 비교하고, `pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)`(simple_trainer L1005)도 마찬가지다. "신뢰도가 균일하지 않은 타깃에는 L1" 이라는 일관된 선택이다.

## 4. 왜 "L1만 쓰면 뭉개지는가"

강건함의 대가가 이것이다. 원인은 두 겹이다.

**(1) 픽셀 독립성 — 구조를 볼 눈이 없다.** L1은 이웃 픽셀과의 대비·분산·상관을 전혀 보지 않는다. 국소 대비가 절반으로 죽어도, 개별 픽셀 오차의 합이 작으면 L1은 만족한다. SSIM은 반대로 $\mu, \sigma^2, \sigma_{12}$를 윈도우로 재기 때문에(`torch_ssim_loss` L129–148) "평균은 맞는데 대비가 죽었다"를 직접 벌한다. 워크스루가 "국소 대비/구조를 맞춰 L1만 쓸 때 생기는 뭉개짐을 억제한다"고 쓴 게 이 뜻이다.

**(2) 불확실할 때 뭉개는 쪽이 이득이다.** 고주파 텍스처를 1픽셀 어긋나게 맞추는 것과, 그냥 평평하게 뭉개는 것 중 어느 쪽이 손실이 낮은가. 교대 스트라이프 `[0,1,0,1,...]`를 타깃으로 두고 비교하면

| 예측 | L1 | L2 |
|---|---|---|
| 1픽셀 밀린 스트라이프 (디테일 살아있음) | **1.0000** | 1.0000 |
| 전부 0.5 (완전히 뭉갬) | **0.5000** | 0.2500 |

정렬이 조금 어긋난 디테일은 **디테일이 아예 없는 것보다 2배 더 벌받는다**. Gaussian 위치·포즈·해상도가 완벽할 수 없는 실제 학습에서 최적화기는 이 계산을 매 스텝 하고, 안전한 쪽(평균값으로 뭉개기)을 고른다. 이것이 L1 단독 학습이 저주파로 수렴하는 메커니즘이다. 그리고 이 "안전 선택"을 깨는 것이 SSIM의 역할이다 — 대비가 죽으면 $\sigma_{12}$ 항이 벌을 주므로, 뭉개기가 더 이상 공짜가 아니게 된다.

두 항의 관계를 한 줄로: **L1은 색을 맞추고(어디로 갈지), SSIM은 선명함을 지킨다(뭉개지 못하게).** 0.8/0.2라는 비율은 "기본은 L1의 강건한 색 신호로 끌고 가되, 뭉개짐만 SSIM으로 제동한다"는 뜻이다.

## 5. 실제 값으로 확인하기

워크스루는 학습 전 초기 상태(SfM 색만 넣은 Gaussian)에서 두 항을 따로 찍어 본다(L281–285).

```python
with torch.no_grad():
    l1 = l1_loss(render, gt).mean()
    ssim = ssim_loss(render.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))
    total = torch.lerp(l1, ssim, 0.2)   # = 0.8*l1 + 0.2*ssim
print(f"초기 상태 손실: L1={l1:.4f}, SSIM loss={ssim:.4f}, total={total:.4f}")
```

그리고 학습 루프(L375–378)에서 매 스텝 같은 조합을 만들고 `loss.backward()`를 부른다. 로그 곡선의 제목도 `"Training loss (0.8·L1 + 0.2·SSIM)"`(L420)이다.

주의할 대비 한 가지: **학습은 L1로 하지만 평가는 L2 계열로 한다.** 검증 PSNR은 `mse = F.mse_loss(render.clamp(0,1), gt)`에서 나온다(워크스루 L447, simple_trainer의 `eval()`도 torchmetrics PSNR). 그래서 "L1 손실이 조금 올라갔는데 PSNR은 좋아졌다" 같은 어긋남이 원리적으로 가능하다. 손실 값 자체와 리포트 지표를 같은 축으로 읽지 말 것.

## 6. 한 줄 정리 & 자주 틀리는 지점

> L1은 손실의 **가중치 0.8을 차지하는 몸통**으로, 픽셀마다 독립적으로 색을 맞춘다. gradient 크기가 ±1로 고정되어 이상치 픽셀에 끌려가지 않는 대신(최적해가 평균이 아니라 중앙값), 이웃 픽셀 간 구조를 보지 못해 단독으로는 저주파/뭉개진 해로 수렴한다. 그 제동이 나머지 0.2의 SSIM이다.

- ✗ "L1이 보조 항이고 SSIM이 주 항" → 반대다. 0.8 : 0.2.
- ✗ "L1이 노이즈에 강건하니 L2보다 항상 낫다" → 강건함의 대가가 뭉개짐이다. 그래서 단독으로 쓰지 않는다.
- ✗ "뭉개지는 이유는 L1이 작은 오차를 무시해서" → 반대다. L1은 작은 오차에도 크기 1의 gradient를 준다. 뭉개짐의 원인은 **픽셀 독립성**(구조를 못 봄)과 **미정렬 디테일이 무(無)디테일보다 더 큰 벌을 받는 구조**다.
- ✗ "SSIM이 있으니 L1은 없어도 된다" → SSIM은 valid 패딩으로 테두리를 감독하지 못하고, 패치 기반이라 마스킹도 정확히 못 한다. 절대 색 수준(밝기 오프셋)도 SSIM 단독으로는 약하게 잡힌다.
- ✓ `l1_loss`는 `reduction="none"`을 돌려준다. `.mean()`을 잊으면 스칼라가 아니다.
- ✓ 학습 손실은 L1 기반, 평가 PSNR은 MSE 기반이다.

## 참고 위치

- `/home/sungwoo/projects/swcho/gsplat/fm/training/.fm/assets/training_walkthrough.py` (L266–285 손실 설명·초기값, L375–378 학습 루프, L420 곡선, L447 PSNR)
- `/home/sungwoo/projects/swcho/gsplat/gsplat/losses.py` (L53–63 `l1_loss`, L66–76 `mse_loss`, L110–148 `torch_ssim_loss`, L181–186 fused_ssim/valid 패딩, L209–224 `depth_l1_loss`)
- `/home/sungwoo/projects/swcho/gsplat/examples/simple_trainer.py` (L171 `ssim_lambda=0.2`, L946–961 손실 조합과 마스크 처리, L1005 pose L1, L1021 `train/l1loss` 로깅)
- `/home/sungwoo/projects/swcho/gsplat/gsplat/strategy/default.py` (L100 `grow_grad2d`, L273 `grad2d` 누적, L295–298 grow 판정)
