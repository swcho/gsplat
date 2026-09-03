# `eval_psnr()`에 `@torch.no_grad()`를 붙인 이유

**Q.** `eval_psnr()` 함수에 `@torch.no_grad()`를 붙인 이유는?
**A.** 평가에는 gradient가 필요 없으므로 계산 그래프를 만들지 않아 메모리와 시간을 아낀다.

---

## 1. 문제의 코드

`training_walkthrough.py` (L437~451):

```python
# 검증셋 PSNR (simple_trainer의 eval()에서는 torchmetrics로 PSNR/SSIM/LPIPS를 계산)
@torch.no_grad()
def eval_psnr(n_images: int = 5) -> float:
    psnrs = []
    for i in range(min(n_images, len(valset))):
        d = valset[i]
        gt = d["image"][None].to(DEVICE) / 255.0
        render, _, _ = rasterize_splats(
            splats, d["camtoworld"][None].to(DEVICE), d["K"][None].to(DEVICE),
            gt.shape[2], gt.shape[1], sh_degree=SH_DEGREE,
        )
        mse = F.mse_loss(render.clamp(0, 1), gt)
        psnrs.append(-10.0 * math.log10(mse.item()))
    return float(np.mean(psnrs))
```

이 함수가 하는 일은 **렌더 → GT와 비교 → 숫자 하나 뽑기**가 전부다.
`mse.item()`으로 파이썬 float를 꺼내는 순간 텐서와의 연결은 끊어지고,
`loss.backward()`도 `optimizer.step()`도 호출되지 않는다.
즉 **gradient가 쓰일 곳이 한 군데도 없다.**

그런데 `splats["means"]`, `splats["scales"]` 등은 학습용 파라미터라
`requires_grad=True` 상태다. 아무 보호 없이 이 함수를 호출하면 PyTorch는
"나중에 backward가 올지도 모른다"고 가정하고 **렌더링 전 과정의 계산 그래프를
빠짐없이 만들어 둔다.** 그 비용을 통째로 없애는 스위치가 `torch.no_grad()`다.

## 2. `torch.no_grad()`가 실제로 끄는 것

`torch.no_grad()`는 스레드-로컬한 **grad mode 플래그**를 끄는 컨텍스트 매니저다.
이 블록 안에서 벌어지는 일:

| 항목 | grad mode ON (기본) | `no_grad` 안 |
|---|---|---|
| 출력 텐서의 `grad_fn` | `MeanBackward0` 등이 붙음 | `None` |
| 출력 텐서의 `requires_grad` | 입력이 하나라도 True면 True | 항상 False |
| backward용 중간 activation | 그래프가 참조를 잡고 있어 **해제 불가** | 다음 연산 뒤 즉시 해제 |
| `ctx.save_for_backward` (custom Function) | 텐서를 저장 | 그래프가 안 만들어져 저장분이 살아남지 않음 |

핵심은 세 번째 줄이다. **연산 결과 텐서 자체의 메모리가 아니라,
"backward에서 다시 쓰려고 붙잡아 둔 중간 결과들"이 메모리를 먹는다.**
grad mode가 켜져 있으면 이들이 스코프를 벗어나도 그래프가 참조를 쥐고 있어
해제되지 않고, forward가 끝날 때까지 계속 쌓인다.

> 주의: `@torch.no_grad()`는 **autograd만** 끈다. `model.eval()`(Dropout/BatchNorm
> 모드 전환)과는 완전히 별개이고, `requires_grad` 속성을 바꾸는 것도 아니다.
> 블록을 벗어나면 파라미터는 그대로 학습 가능한 상태로 돌아온다.

## 3. 숫자로 확인 (이 저장소 환경, CUDA)

12층 MLP를 200k×64 입력에 대해 forward만 돌린 측정값:

```
grad ON  peak MB: 691.9   requires_grad=True   grad_fn=MeanBackward0
no_grad  peak MB: 203.6   requires_grad=False  grad_fn=None
grad ON  fwd 3.21 ms   /   no_grad fwd 3.22 ms
```

- **메모리: 692 MB → 204 MB (약 3.4배 절감).** 차이 488 MB가 전부
  "backward를 위해 붙잡아 둔 중간 텐서"다.
- **시간: forward 자체는 거의 동일.** 정직하게 말하면 `no_grad`의 이득은
  *연산 시간*보다 *메모리*에서 압도적으로 크다. 시간 이득이 나오는 경로는
  (a) 그래프 노드 생성·`save_for_backward` 부기 오버헤드, (b) 할당량이 줄어
  캐싱 allocator가 덜 쫓기고 `cudaMalloc`/동기화가 덜 발생하는 효과,
  (c) OOM으로 인한 재시도·스왑이 안 일어나는 것 — 연산량 자체가 큰 워크로드에서는
  (a)가 상대적으로 묻힌다.

## 4. 3DGS/gsplat에서 특히 중요한 이유

일반 CNN보다 이 프로젝트에서 `no_grad`의 무게가 훨씬 크다.

1. **평가는 학습보다 해상도가 높다.** 학습 루프는 `patch`나 축소 이미지로 돌 수
   있지만 검증은 보통 원본 해상도 풀프레임이다. 픽셀 수가 곱으로 늘고,
   그래프가 잡는 버퍼도 같이 곱으로 늘어난다.
2. **Gaussian 개수가 학습 중에 계속 늘어난다.** `DefaultStrategy`의
   duplicate/split 때문에 후반부엔 수십만~수백만 개가 된다. `rasterization()`의
   backward는 `means2d`, `conics`, `colors`, tile–Gaussian intersection 인덱스 등
   Gaussian 수에 비례하는 버퍼를 전부 붙잡아야 한다.
3. **`rasterization()`은 커스텀 `autograd.Function`이다.** grad mode가 켜져 있으면
   CUDA 커널이 backward용 텐서를 `save_for_backward`로 물고 있게 되고,
   `info` 딕셔너리의 텐서들도 `grad_fn`을 달고 나온다. `no_grad` 안에서는 이
   저장 자체가 일어나지 않아 렌더 결과 이미지만 남는다.
4. **밀도화 훅과의 충돌 방지.** 학습 루프는 `strategy.step_pre_backward()`에서
   `info["means2d"].retain_grad()`를 부르고, `step_post_backward()`에서 그
   화면공간 gradient 통계를 누적한다. 평가에서 만들어진 그래프/gradient가
   섞여 들어가면 밀도화 판단이 오염될 수 있다. `no_grad`는 그 경로를 원천 차단한다.
5. **학습 중간에 주기적으로 불린다.** 한 번 OOM 안 났다고 끝이 아니라,
   학습 파라미터의 gradient 버퍼 + Adam 상태(exp_avg, exp_avg_sq)가 이미 GPU에
   올라가 있는 상태에서 평가가 끼어든다. 남은 여유 메모리가 적은 시점이라
   평가 쪽 그래프가 그대로 OOM의 방아쇠가 된다.

## 5. 같은 아이디어, 다른 문법 — 워크스루 안의 다른 사용처

이 노트북은 gradient가 필요 없는 지점마다 일관되게 같은 처리를 한다:

| 위치 | 형태 | 목적 |
|---|---|---|
| L252 | `with torch.no_grad():` | 초기 상태 렌더 (시각화용) |
| L281 | `with torch.no_grad():` | 초기 손실값 출력 (backward 없음) |
| L401 | `with torch.no_grad():` | 학습 루프 중 스냅샷 렌더 |
| L437 | `@torch.no_grad()` | `eval_psnr()` 전체 |

**데코레이터 vs `with` 블록**은 기능이 같고 범위만 다르다.
함수 **전체**가 gradient 불필요일 때는 데코레이터가 깔끔하고
(들여쓰기 한 단계를 줄이고, early return이 여러 개여도 빠짐없이 덮인다),
학습 루프 안처럼 함수의 **일부**만 감쌀 때는 `with` 블록을 쓴다.
`eval_psnr()`은 첫 줄부터 마지막 줄까지 전부 평가 코드라 데코레이터가 맞는 선택이다.

## 6. 대응 관계 — `simple_trainer.py`

워크스루의 `eval_psnr`은 `Runner.eval`(L1201)의 축소판이고, 원본도 똑같이
데코레이터를 쓴다:

```python
    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
```

`simple_trainer.py`에서 `@torch.no_grad()`가 붙은 메서드는 L1200(`eval`),
L1302, L1380, L1411, L1428 — 평가/렌더링/저장 계열 메서드 전부다.
"결과만 뽑아 쓰는 함수에는 예외 없이 붙인다"가 이 저장소의 관례라고 봐도 된다.

## 7. 알아 둘 함정과 이웃 개념

- **학습 코드에 실수로 붙이면 조용히 망가진다.** 에러가 안 나고 `loss.backward()`가
  "element 0 of tensors does not require grad"로 터지거나, 더 나쁘게는
  파라미터 업데이트만 안 되고 루프가 멀쩡히 도는 것처럼 보인다.
- **제너레이터 함수에는 데코레이터를 쓰지 말 것.** `yield`가 있는 함수에 붙이면
  값을 내보내고 재진입하는 사이에 모드가 의도대로 유지되지 않는다. `with` 블록을 쓴다.
- **스레드-로컬이다.** DataLoader worker나 별도 스레드로 넘어간 코드에는
  전파되지 않는다. 그 안에서 필요하면 다시 걸어야 한다.
- **`torch.inference_mode()`는 더 강한 버전.** 버전 카운터/뷰 추적까지 끄기 때문에
  더 빠르고 메모리도 덜 쓰지만, 그 안에서 만든 텐서는 나중에 autograd 계산에
  아예 참여할 수 없다. 순수 추론 서빙에는 이쪽이 낫고, 평가 결과를 다시 학습
  경로에 넣을 여지가 있으면 `no_grad`가 안전하다.
- **`.detach()`와의 차이.** `detach()`는 이미 만들어진 그래프에서 텐서 하나를
  떼어내는 사후 처리라 **그래프 생성 비용은 이미 지불한 뒤**다. `no_grad`는
  애초에 만들지 않는다 — 메모리를 아끼려면 `no_grad`가 맞다.

## 8. 한 줄 요약

평가 함수는 **숫자 하나만 뽑고 backward를 안 한다.** 그런데 파라미터가
`requires_grad=True`인 이상 PyTorch는 기본적으로 backward용 그래프와 중간 버퍼를
전부 만들어 둔다 — 고해상도 × 수십만 Gaussian이라 그 비용이 학습 forward보다도 크다.
`@torch.no_grad()`는 "여기서 backward는 절대 안 온다"고 선언해 그 낭비를 통째로
없애는, 비용 0의 안전장치다.
