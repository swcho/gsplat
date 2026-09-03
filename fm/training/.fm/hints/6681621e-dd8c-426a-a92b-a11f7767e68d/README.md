# `random_bkgd` — 투명도 overfit 방지 장치

**Q.** `random_bkgd` 옵션의 목적은?
**A.** 투명 영역이 특정 배경색으로 도망가는(overfit) 것을 방지한다. 배경을 매번 랜덤하게 바꿔 학습한다.

## 1. 코드에서의 위치

워크스루 4단계(손실 함수)에서 "기본은 꺼져 있는 선택적 항" 중 하나로 언급된다
(`fm/training/.fm/assets/training_walkthrough.py:278` — `depth_loss`, `opacity_reg`/`scale_reg`와 나란히).

실제 정의와 사용부는 `examples/simple_trainer.py`이며, 코드는 딱 세 줄이다.

```python
# examples/simple_trainer.py:191-192
# Use random background for training to discourage transparency
random_bkgd: bool = False
```

```python
# examples/simple_trainer.py:929-932  (render 직후, loss 직전)
if cfg.random_bkgd:
    bkgd = torch.rand(1, 3, device=device)
    colors = colors + bkgd * (1.0 - alphas)
```

포인트 세 가지:

- `torch.rand(1, 3)`이므로 **스텝마다 새 랜덤 RGB**가 뽑힌다. 배치 안의 모든 뷰/픽셀이 같은 색을 공유하는 "한 장의 단색 배경"이다.
- `colors + bkgd * (1 - alphas)` 는 정확히 알파 합성(over 연산)이다. `rasterization()`의 출력은 **premultiplied**(이미 알파가 곱해진 누적색)이라서 `colors * alphas` 같은 추가 곱셈이 필요 없다.
- 학습 루프에만 있다. eval/렌더링 경로는 고정 배경(`backgrounds=None` → 검정, 또는 viewer의 `render_tab_state.backgrounds`)을 쓰므로 평가 지표는 오염되지 않는다.

## 2. 왜 투명도가 "도망가는"가

`gsplat`의 기본 래스터화는 배경을 주지 않으면 검정(0) 위에 합성한 것과 같다. 그래서
어떤 픽셀의 최종 렌더값은

```
render = C_accum            (배경 = 검정, 즉 0)
render = C_accum + b·(1-α)  (배경 = 색 b)
```

배경이 **항상 검정으로 고정**되어 있으면, 반투명 Gaussian(α<1)도 누적색 `C_accum`만 GT에 맞추면 손실이 0이 된다.
즉 "알파가 부족한 만큼 색을 더 진하게" 라는 편법으로 GT를 재현할 수 있고, 옵티마이저는 굳이 α를 1로 밀 이유가 없다.
결과적으로

- 물체 표면이 여러 겹의 흐릿한 반투명 Gaussian으로 표현되고(floaters, foggy 표면),
- 그 겉보기 색은 **학습 때 쓴 배경색에만 맞춰져 있다**. 배경을 흰색으로 바꿔 렌더하면 물체가 밝게 번지고 실루엣이 무너진다.
- 배경이 없는(마스크된/알파 있는) 데이터에서는 특히 심하다. "빈 공간"과 "검정 배경"이 손실상 구별되지 않기 때문.

이것이 답의 "특정 배경색으로 도망간다(overfit)"는 표현이다. 모델이 장면 형상을 학습하는 대신 **한 개의 배경 상수에 커플링된 색 보정**을 학습해 버리는 shortcut이다.

## 3. 랜덤 배경이 이 shortcut을 없애는 원리

배경 b가 매 스텝 랜덤이면, 반투명 픽셀의 렌더값 `C + b(1-α)`는 b에 따라 **흔들린다**. 이 흔들림은 어떤 고정 `C`로도 상쇄할 수 없다.
b ~ U[0,1]에 대한 기대 L1을 최적 `C`에서 계산하면

```
min_C E_b |C + b(1-α) - g| = (1-α) · E|b - 0.5| = 0.25 · (1-α)
```

즉 **잔여 투명도에 정비례하는 손실 하한**이 생긴다. 검정 고정 배경에서는 이 값이 전부 0이었다.

| α (픽셀 불투명도) | 검정 고정 배경의 달성 가능 L1 | 랜덤 배경의 달성 가능 L1 |
|---|---|---|
| 1.0 | 0.0000 | 0.0000 |
| 0.8 | 0.0000 | 0.0500 |
| 0.5 | 0.0000 | 0.1250 |
| 0.2 | 0.0000 | 0.2000 |

(GT 회색 0.6, b를 20만 회 샘플링해 수치 확인한 값. 정확히 `0.25·(1-α)`와 일치한다.)

따라서 gradient는 "불투명하게 만들어라"는 방향으로 흐른다. 물체가 있는 픽셀은 α→1로, 진짜 빈 공간은 어차피 GT가 배경색을 따라가지 않으므로 α→0으로 정리된다.
데이터 증강(data augmentation)으로 보면: **배경색을 nuisance 변수로 랜덤화해 모델이 그 변수에 의존하지 못하게 하는** 전형적인 기법이다. NeRF/Instant-NGP 계열의 "random background color", nerfstudio splatfacto의 `background_color="random"`과 같은 아이디어다.

## 4. 언제 켜나 / 무엇과 헷갈리지 말 것

- **켜면 좋은 경우**: 알파/마스크가 있는 오브젝트 중심 데이터(NeRF-synthetic 류), 배경이 단색으로 잘려 나간 데이터, 반투명 floaters가 남아 뷰어에서 배경을 바꿨을 때 색이 튀는 경우.
- **기본이 `False`인 이유**: 배경 전체가 실제로 촬영된 일반 장면(MipNeRF360 등)에서는 배경도 학습 대상이라 랜덤화가 오히려 노이즈가 되고, PSNR 지표가 살짝 내려갈 수 있다.
- `opacity_reg`(MCMC 전략의 불투명도 L1 정규화)와는 목적이 반대다. `opacity_reg`는 α를 **낮춰** Gaussian을 솎아내는 항이고, `random_bkgd`는 남은 Gaussian을 **불투명하게** 만드는 압력이다.
- `rasterization(backgrounds=...)` 인자와도 다르다. 그쪽은 래스터라이저 내부에서 배경을 합성해 주는 경로이고, `random_bkgd`는 학습 루프에서 렌더 결과에 직접 더하는 후처리다(그래서 `alphas`가 필요하고, gradient도 α로 그대로 흐른다).
