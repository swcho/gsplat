# opacity reset — 언제, 어떻게, 왜

**Q.** opacity reset은 언제, 어떻게 일어나며 목적은?
**A.** 매 3000스텝마다 전체 opacity를 0.01로 리셋한다. 카메라 앞에 떠 있는 floater를 정리하는 효과가 있다.

---

## 1. 언제 — `step % reset_every == 0 and step > 0`

`DefaultStrategy.step_post_backward()`의 **맨 마지막 블록**이다
(`gsplat/strategy/default.py:218`).

```python
if step % self.reset_every == 0 and step > 0:
    reset_opa(
        params=params,
        optimizers=optimizers,
        state=state,
        value=self.prune_opa * 2.0,   # 0.005 * 2 = 0.01
    )
```

기본값 (`gsplat/strategy/default.py:100~112`):

| 필드 | 기본값 | 의미 |
|---|---|---|
| `reset_every` | `3000` | 리셋 주기 |
| `prune_opa` | `0.005` | prune 임계값 → 리셋 목표값은 이것의 **2배** |
| `refine_start_iter` / `refine_stop_iter` | `500` / `15_000` | refine(그리고 reset) 유효 구간 |
| `refine_every` | `100` | duplicate/split/prune 주기 |
| `pause_refine_after_reset` | `0` | 리셋 후 refine을 잠시 멈추는 스텝 수 |

주의할 점 두 가지:

- **리셋도 `refine_stop_iter`에서 멈춘다.** 함수 첫 줄이
  `if step >= self.refine_stop_iter: return` 이므로, 기본 30k 학습에서 실제로 리셋이
  일어나는 스텝은 **3000 / 6000 / 9000 / 12000 — 총 4번**뿐이다. 15000은 `>=`에
  걸려 이미 return되고, 그 이후 15k~30k 구간은 밀도화가 완전히 멈춘 "수렴 전용"
  구간이라 opacity를 건드리지 않는다.
- **refine 블록과 같은 스텝에서 연달아 실행된다.** 3000은 `refine_every=100`의
  배수이므로 같은 호출 안에서 `_grow_gs` → `_prune_gs` → (통계 zero) → `reset_opa`
  순으로 처리된다. 즉 **prune이 먼저, reset이 나중**이다. 리셋으로 낮아진 opacity는
  그 스텝에 잘리지 않고 다음 refine(=100스텝 뒤)부터 심판대에 오른다.
- 짧은 학습으로 줄일 때는 `Config.adjust_steps(factor)`가
  `reset_every`도 같이 스케일한다 (`examples/simple_trainer.py:274`).

## 2. 어떻게 — "설정"이 아니라 `clamp`

`reset_opa()` (`gsplat/strategy/ops.py:271`)의 실제 동작:

```python
def param_fn(name, p):
    if name == "opacities":
        opacities = torch.clamp(p, max=torch.logit(torch.tensor(value)).item())
        return torch.nn.Parameter(opacities, requires_grad=p.requires_grad)

def optimizer_fn(key, v):
    return torch.zeros_like(v)     # Adam exp_avg / exp_avg_sq 를 0으로
```

여기서 읽어야 할 디테일:

1. **`clamp(max=...)`이지 대입이 아니다.** opacity가 이미 0.01보다 낮은 Gaussian은
   그대로 남는다(그래서 그 다음 prune에서 0.005 미만인 것들이 살아남아 잘린다).
   0.01보다 높은 **모든** Gaussian은 정확히 0.01로 눌린다. 카드의 "전체 opacity를
   0.01로 리셋"은 이 상한 클램프를 말한다.
2. **파라미터는 raw(pre-sigmoid) 값이다.** 저장된 `params["opacities"]`는 logit이고
   렌더링 시 `torch.sigmoid`가 적용된다(`examples/simple_trainer.py:670`).
   그래서 임계값도 `logit(0.01) = -4.5951`로 변환해서 클램프한다.
   초기값은 `init_opacity=0.1` → `logit = -2.1972`
   (`examples/simple_trainer.py:332`)이므로, 리셋은 초기값보다도 **한참 아래**로
   내려보내는 강한 개입이다.
3. **Adam 상태도 함께 리셋된다.** `optimizer_fn`이 `opacities` optimizer의
   `exp_avg`/`exp_avg_sq`를 0으로 만든다(`step`은 유지).
   `_update_param_with_optimizer(..., names=["opacities"])`로 호출되므로
   means/scales/quats/SH의 optimizer 상태는 건드리지 않는다. 즉 opacity만
   "모멘텀 초기화 후 재출발"한다.
4. Gaussian 개수는 변하지 않는다. duplicate/split/prune과 달리 `state`의
   `grad2d`/`count` 텐서를 재정렬할 필요가 없다.

## 3. 왜 — floater를 "무죄추정"에서 끌어내리기

floater(부유물)는 **카메라 근처 공중에 뜬 반투명 blob**이다. 학습 초반, 특정 뷰의
잔차를 카메라 바로 앞에 얇은 반투명 막을 세워 덮어버리면 그 뷰의 loss는 즉시
내려간다. 근접한 Gaussian은 화면에서 크게 투영돼 픽셀을 많이 덮으므로 이 지름길이
gradient 관점에서 매우 매력적이다. 문제는 이게 **기하가 아니라 뷰 종속적 눈속임**이고,
새 시점에서는 흐릿한 안개·구름으로 나타난다는 것.

이런 floater를 그냥 두면 스스로 사라지지 않는다. opacity가 어중간하게 높으면
`prune_opa=0.005` 임계값에 절대 걸리지 않아 prune이 손을 댈 수 없기 때문이다.

opacity reset은 여기에 **주기적 신뢰 초기화**를 넣는다.

- 리셋 직후 모든 Gaussian은 사실상 투명(0.01)해진다 — 렌더링 기여도가 거의 0.
- 이후 100스텝 남짓 동안, **여러 뷰에서 일관되게 loss를 낮추는** Gaussian만
  gradient를 받아 opacity가 다시 올라온다. `opacities_lr=5e-2`는 means의 `1.6e-4`에
  비해 매우 큰 학습률이고, Adam이 gradient 크기를 정규화하므로 raw 공간에서
  스텝당 약 0.05씩 움직인다 → `logit(0.01) = -4.60`에서 `logit(0.5) = 0`까지
  **약 92스텝**. 즉 다음 refine(100스텝 뒤)까지 "진짜 표면"은 대체로 회복한다.
- 반대로 소수 뷰에만 기여하던 floater는 회복 신호가 약해 0.005 밑에 머물고,
  다음 `_prune_gs`에서 `sigmoid(opacity) < 0.005` 조건에 걸려 **삭제된다**
  (`gsplat/strategy/default.py:349`).

부수 효과로 **densification 과잉 억제**도 있다: 리셋은 전체 Gaussian 수를 주기적으로
깎아내려 duplicate/split로 폭증한 개수를 다시 눌러준다. 학습 곡선에서
"3000스텝마다 loss가 튀었다가 회복 + Gaussian 개수 급감"이 보이는 것이 그 흔적이다
(asset `training_walkthrough.py`의 결과 확인 절 참고).

### `pause_refine_after_reset`

리셋 직후 몇 스텝은 gradient 통계(`grad2d`, `count`)가 "거의 투명한 상태"에서
쌓인 값이라 split/duplicate 판단에 쓰기에 부적절하다. 그래서
`step % reset_every >= pause_refine_after_reset` 조건으로 refine을 잠시 미룰 수 있다
(`gsplat/strategy/default.py:191`). 기본값 0은 "멈추지 않음"이고, docstring은
**학습 이미지 수**로 설정하는 것을 권한다 — 모든 뷰가 최소 한 번은 opacity를
회복시킬 기회를 갖게 하려는 것.

## 4. 비교 — MCMCStrategy에는 opacity reset이 없다

`MCMCStrategy`(`gsplat/strategy/mcmc.py`)는 Gaussian 수 상한을 고정하고,
opacity가 낮은 "죽은" Gaussian을 `relocate()`로 살아 있는 것 근처에 **재배치**하며
매 스텝 위치에 노이즈를 주입한다(SGLD). 죽은 것을 지우고 다시 뽑는 구조 자체가
floater 정리 역할을 하므로 별도의 주기적 opacity 클램프가 필요 없다.
그래서 opacity reset은 `DefaultStrategy`(원논문 3DGS 계열) 고유의 장치다.

## 한 줄 요약

`refine_stop_iter` 이전 구간에서 **매 3000스텝**, `reset_opa()`가 raw opacity를
`logit(prune_opa*2) = logit(0.01)`로 **클램프**하고 Adam 모멘텀을 0으로 만든다 →
모두를 투명하게 만든 뒤 **재획득 경쟁**을 시켜, 회복하지 못한 floater를 다음 prune에서
제거하는 것이 목적이다.
