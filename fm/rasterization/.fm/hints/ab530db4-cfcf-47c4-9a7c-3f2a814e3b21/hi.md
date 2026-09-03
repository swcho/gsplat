# 알파 블렌딩의 누적 공식 — 반투명 유리판을 겹쳐 보기

**Q.** 알파 블렌딩의 누적 공식(앞→뒤 순회)은?

**A.**
$$\alpha_i = \min(0.99,\ o_i e^{-\sigma_i}),\qquad
C_p = \sum_i c_i\,\alpha_i\,T_i,\qquad
T_{i+1} = T_i(1-\alpha_i)$$
최종적으로 `render_alpha = 1 - T`.

---

## 1. 비유부터: 색유리 여러 장을 겹쳐 놓고 보기

눈앞에 색이 있는 **반투명 유리판**을 여러 장 세워 놓았다고 하자. 판을 눈에서 가까운 순으로 1번, 2번, 3번, … 이라고 번호를 붙인다.

각 판 $i$는 두 가지 성질을 갖는다.

| 기호 | 뜻 |
|---|---|
| $c_i$ | 그 판의 **색** (예: 빨강 $(1,0,0)$) |
| $\alpha_i$ | 그 판이 **막는 비율**. $\alpha_i = 0.3$이면 뒤에서 오는 빛의 30%를 붙잡아 자기 색으로 바꾸고, 70%는 통과시킨다 |

우리가 보고 싶은 것은 이 판들을 다 겹쳤을 때 눈에 들어오는 **최종 색 하나**다. 그게 픽셀 색 $C_p$이다.

3D Gaussian Splatting에서 "유리판"은 화면에 투영된 타원(2D Gaussian)이고, 번호는 카메라로부터의 **깊이 순서**다. 그래서 "앞→뒤 순회"란 곧 **눈에서 가까운 판부터 차례로 처리한다**는 뜻이다.

## 2. $\alpha_i$가 어디서 오는가 — $\min(0.99,\ o_i e^{-\sigma_i})$

한 Gaussian은 판 전체가 균일하게 불투명한 게 아니라, **중심에서 진하고 가장자리로 갈수록 옅어진다.** 고교에서 배운 정규분포의 종 모양 곡선을 떠올리면 된다.

픽셀 중심 $(p_x, p_y)$와 Gaussian 중심 $(\mu_x, \mu_y)$의 차를 $dx, dy$라 하면

$$\sigma_i = \tfrac12\left(a\,dx^2 + c\,dy^2\right) + b\,dx\,dy$$

여기서 $(a, b, c)$는 conic이라 부르는 세 숫자로, 타원의 모양(폭·기울기)을 정한다. $\sigma$는 **"중심에서 얼마나 멀리 떨어졌나"를 타원 모양에 맞게 잰 거리의 제곱의 절반**이다. 원형 Gaussian이라면 $a = c = 1/s^2$, $b = 0$이 되어 $\sigma = \frac{dx^2+dy^2}{2s^2}$ — 정확히 정규분포 지수부의 그 형태다.

그러면

$$e^{-\sigma_i} \in (0, 1]$$

은 중심에서 $1$, 멀어질수록 $0$으로 떨어지는 종 모양 값이다. 여기에 그 Gaussian 고유의 **불투명도** $o_i \in [0,1]$를 곱하면

$$\alpha_i = o_i\,e^{-\sigma_i}$$

즉 "이 판이 이 픽셀 위치에서 실제로 막는 비율". 마지막으로 $\min(0.99, \cdot)$ 상한을 씌운다 — 이유는 §7에서 다룬다.

## 3. 투과율 $T_i$: 곱셈으로 쌓이는 이유

**정의.** $T_i$ = 뒤에서 오는 빛이 아니라, 반대로 생각하는 게 편하다. **$i$번째 판까지 도달할 수 있는 시선의 비율.**

시작할 때 $T_1 = 1$ (아무것도 안 가렸으니 100%가 1번 판에 닿는다).

1번 판을 지나면, 1번이 $\alpha_1$만큼 붙잡았으므로 통과하는 것은 $1 - \alpha_1$. 따라서

$$T_2 = T_1(1-\alpha_1) = 1-\alpha_1$$

2번 판을 지나면 **그 중에서 다시** $1-\alpha_2$만 통과한다.

$$T_3 = T_2(1-\alpha_2) = (1-\alpha_1)(1-\alpha_2)$$

일반적으로

$$\boxed{\;T_{i+1} = T_i(1-\alpha_i)\quad\Longleftrightarrow\quad T_i = \prod_{j<i}(1-\alpha_j)\;}$$

**왜 곱셈인가?** 확률과 통계에서 배운 독립 사건의 곱 법칙과 정확히 같은 구조다. "1번을 통과한다"와 "2번을 통과한다"가 순차적으로 일어나는 사건이라면, 둘 다 통과할 확률은 각각의 확률의 곱이다. 덧셈이 아닌 이유는 자명하다: 판 3장이 각각 50%를 막는다면 남는 것은 $1 - 1.5 = -0.5$가 아니라 $0.5^3 = 0.125$다. 물리에서 배운 **Beer–Lambert 법칙**(매질을 지날수록 빛의 세기가 지수적으로 감쇠)의 이산 버전이라고 봐도 좋다.

## 4. 픽셀 색: "그 판이 막은 비율 × 그 판까지 도달한 비율"로 가중합

$i$번째 판이 실제로 눈에 보내는 빛은 얼마인가?

- 그 판까지 도달한 빛의 비율: $T_i$
- 그 중에서 그 판이 붙잡아 자기 색으로 내보내는 비율: $\alpha_i$

두 사건이 연달아 일어나므로 이 판의 **기여 가중치**는 곱이다.

$$w_i = \alpha_i\,T_i$$

(주의: 이 판이 내보낸 빛은 그 앞의 판들을 통과해 눈에 오지만, 그 통과분은 이미 $T_i$가 아니라… 아니, 정확히는 $T_i$가 **앞쪽 판들을 통과한 비율**이므로 $\alpha_i T_i$ 하나로 왕복이 아니라 편도만 세는 근사다. 3DGS는 이 emission-absorption 모델을 쓴다 — 빛은 뒤에서 앞으로 한 방향으로만 흐른다.)

그러면 픽셀 색은 각 판 색의 가중합:

$$\boxed{\;C_p = \sum_i c_i\,\alpha_i\,T_i\;}$$

앞→뒤 루프로 쓰면 이렇게 된다.

```
T = 1;  C = 0
for i in 앞 → 뒤:
    C += c_i * alpha_i * T
    T *= (1 - alpha_i)
```

이 두 줄이 gsplat의 CUDA 커널(`RasterizeToPixels3DGSSerialBatchFwd.cu`) 안쪽에 그대로 들어 있다:

```cuda
const float next_T = T[p] * (1.0f - alpha);
const float vis    = alpha * T[p];        // = α_i T_i
pix_out[p][k]     += c_ptr[k] * vis;
T[p]               = next_T;
```

## 5. 가중치의 합 = $1 - T_{end}$ — 망원급수 증명

여기가 이 카드의 핵심 정체다. **가중치 $\alpha_i T_i$를 다 더하면 무엇이 되는가?**

주장:
$$\sum_{i=1}^{n} \alpha_i T_i = 1 - \prod_{i=1}^{n}(1-\alpha_i) = 1 - T_{n+1}$$

**증명 (망원급수).** $T_{i+1} = T_i(1-\alpha_i)$를 이항하면

$$\alpha_i T_i = T_i - T_i(1-\alpha_i) = T_i - T_{i+1}$$

즉 **각 항이 이웃한 두 $T$의 차**다. 이걸 $i=1$부터 $n$까지 더하면 가운데가 전부 소거된다.

$$\sum_{i=1}^n \alpha_i T_i = (T_1 - T_2) + (T_2 - T_3) + \cdots + (T_n - T_{n+1}) = T_1 - T_{n+1} = 1 - T_{n+1}$$

($T_1 = 1$이므로.) $\blacksquare$

**수학적 귀납법으로도.** $n=1$일 때 $\alpha_1 T_1 = \alpha_1 = 1 - (1-\alpha_1)$ ✓. $n=k$에서 성립한다고 가정하면
$$\sum_{i=1}^{k+1}\alpha_i T_i = (1 - T_{k+1}) + \alpha_{k+1}T_{k+1} = 1 - T_{k+1}(1-\alpha_{k+1}) = 1 - T_{k+2}$$ ✓

### 그래서 `render_alpha = 1 - T`

이 등식이 말하는 바:

$$\text{render\_alpha} = 1 - T_{end} = \sum_i \alpha_i T_i = \text{"이 픽셀에서 무언가에 가려진 총량"}$$

- 아무 Gaussian도 없으면 $T_{end} = 1$ → alpha $= 0$ → **완전히 비어 있는 픽셀** (배경이 그대로 보임)
- 불투명한 것이 꽉 차 있으면 $T_{end} \approx 0$ → alpha $\approx 1$ → **완전히 채워진 픽셀**

그래서 커널은 마지막에 딱 한 줄을 쓴다.

```cuda
render_alphas[pix_id[p]] = 1.0f - T[p];
```

$T$를 따로 저장할 필요가 없다. **$T$와 alpha는 같은 정보의 두 표현**이고, backward 커널은 실제로 `float T_final = 1.0f - render_alphas[pix_id];`로 저장된 alpha에서 $T$를 되살려 쓴다.

부가 효과: 배경색 $c_{bg}$를 합성할 때 남은 몫이 정확히 $T_{end}$이므로
$$C_p^{\text{final}} = \sum_i c_i \alpha_i T_i + T_{end}\,c_{bg}$$
이고, 가중치 총합이 $\sum_i \alpha_i T_i + T_{end} = 1$이 되어 **볼록결합(convex combination)** 이 완성된다. 색이 범위를 벗어나지 않음이 보장된다.

## 6. 왜 앞→뒤 **순서**여야 하는가

알파 블렌딩은 **교환법칙이 성립하지 않는다.** 앞에 있는 판이 뒤에 있는 판을 가리기 때문이다. 숫자로 확인해 보자. 판 2장, 색은 1차원(밝기)이라 하자.

| | 색 $c$ | $\alpha$ |
|---|---|---|
| 빨강 판 | $1.0$ | $0.8$ |
| 파랑 판 | $0.0$ | $0.8$ |

**빨강이 앞:** $T_1=1$ → 기여 $1.0 \times 0.8 \times 1 = 0.8$, $T_2 = 0.2$ → 기여 $0.0 \times 0.8 \times 0.2 = 0$. 합 $= \mathbf{0.80}$

**파랑이 앞:** $T_1=1$ → 기여 $0.0 \times 0.8 \times 1 = 0$, $T_2 = 0.2$ → 기여 $1.0 \times 0.8 \times 0.2 = 0.16$. 합 $= \mathbf{0.16}$

같은 두 판인데 **결과가 5배 차이 난다.** 앞의 판이 $\alpha=0.8$로 대부분을 막아버려 뒤 판이 거의 보이지 않기 때문이다.

흥미롭게도 **alpha 자체는 순서와 무관**하다: $1 - \prod(1-\alpha_i)$는 곱의 순서를 바꿔도 같다. 두 경우 모두 $1 - 0.2\times0.2 = 0.96$. 순서가 바꾸는 것은 **색**뿐이다.

이것이 래스터화 파이프라인이 굳이 (Gaussian, 타일) 쌍을 **깊이 키로 정렬**하는 이유다. 정렬이 없으면 색이 틀린다. 반대로 말하면, 정렬을 근사(타일 단위 정렬)해도 3DGS가 그럭저럭 잘 보이는 이유는, 잘못된 순서가 alpha는 건드리지 않고 색만 조금 흔들기 때문이다.

## 7. 왜 $\alpha \le 0.99$인가 — backward의 $1/(1-\alpha)$

$\min(0.99, \cdot)$이라는 상한(`MAX_ALPHA`)은 화질 때문이 아니라 **학습(backward) 때문**이다.

forward는 $T$를 앞→뒤로 굴리며 $C$를 쌓는다. 그런데 backward는 **뒤→앞으로** 순회하면서 각 $i$의 $T_i$를 알아야 한다. $T$를 모든 $i$에 대해 저장하면 메모리가 터지므로, 커널은 최종값 $T_{end}$ 하나만 저장하고 **역으로 되돌린다**:

$$T_{i} = \frac{T_{i+1}}{1-\alpha_i}$$

```cuda
float T_final = 1.0f - render_alphas[pix_id];   // 저장된 alpha에서 T 복원
float T = T_final;
...
float ra = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
T *= ra;                                        // 뒤→앞으로 T를 되감기
```

여기서 $\alpha = 1$이면 $1/(1-\alpha) = 1/0$ — **0으로 나누기**다. $\alpha$가 1에 아주 가깝기만 해도 이 나눗셈이 폭발해 그래디언트가 NaN/Inf가 된다.

$$\alpha \le 0.99 \;\Longrightarrow\; \frac{1}{1-\alpha} \le 100$$

로 나눗셈이 확실히 유한해진다. (안전벨트로 `MIN_ONE_MINUS_ALPHA = 1e-6` 바닥도 함께 걸려 있다.) forward만 놓고 보면 $\alpha=1$을 허용해도 아무 문제 없다 — 이 상한은 순전히 **미분 가능성을 지키기 위한 값**이다.

## 8. 하한 $1/255$과 조기 종료 $T < 10^{-4}$

두 임계값 모두 **정확도를 거의 잃지 않으면서 계산을 줄이는** 장치다.

### $\alpha < 1/255$ (`ALPHA_THRESHOLD`)이면 건너뛴다

최종 이미지는 채널당 8bit, 즉 표현 가능한 최소 색 차이가 $1/255$다. $\alpha_i T_i < 1/255$인 기여는 반올림하면 **어차피 0**이다. 이걸 계산하는 것은 낭비이므로 아예 건너뛴다.

Gaussian은 꼬리가 무한히 길기 때문에(정규분포!) 이 하한이 없으면 **모든 Gaussian이 모든 픽셀에 아주 조금씩 기여**해 계산이 끝나지 않는다. $1/255$ 하한은 사실상 Gaussian을 **유한한 반경으로 잘라내는** 역할을 하고, 이 반경이 곧 타일 교차에서 쓰는 `radii`가 된다.

### $T \le 10^{-4}$ (`TRANSMITTANCE_THRESHOLD`)이면 그 픽셀은 끝

$T$가 $10^{-4}$까지 떨어졌다는 것은 **뒤에서 오는 빛의 99.99%가 이미 가려졌다**는 뜻이다. 남은 Gaussian이 아무리 밝아도 기여는 $\alpha_i T_i \le 10^{-4}$로 $1/255 \approx 0.004$의 1/40도 안 된다. 그래서 그 픽셀은 루프를 **중단**한다. 이게 3DGS를 실시간으로 만드는 큰 요인 중 하나다 — 앞쪽에 불투명한 것이 있으면 뒤의 수천 개를 건너뛴다.

주의할 디테일: 커널은 $T_{i+1} \le 10^{-4}$가 되는 그 Gaussian을 **더하지 않고 배제한 채** 종료한다(exclusive).

```cuda
const float next_T = T[p] * (1.0f - alpha);
if (next_T <= TRANSMITTANCE_THRESHOLD) { done_mask |= (1u << p); continue; }  // 제외하고 종료
```

### 두 상수의 관계

`_constants.py`의 주석이 설계 의도를 밝힌다:

$$\text{TRANSMITTANCE\_THRESHOLD} = (1 - \text{MAX\_ALPHA})^2 = (0.01)^2 = 10^{-4}$$

즉 **"최대 불투명도 Gaussian이 두 장 겹쳐야 겨우 포화된다"**는 기준으로 잡았다. 한 장만으로 끝나면 너무 공격적이고, 더 낮추면 $T$가 float32에서 위험할 만큼 작아져 backward의 $T/(1-\alpha)$ 되감기가 정밀도를 잃는다.

## 9. NeRF 볼륨 렌더링과 같은 식이다

이 카드의 공식은 3DGS만의 것이 아니다. NeRF의 볼륨 렌더링 적분

$$C = \int_{t_n}^{t_f} T(t)\,\sigma(t)\,c(t)\,dt,\qquad
T(t) = \exp\!\left(-\int_{t_n}^{t}\sigma(s)\,ds\right)$$

를 구간 $[t_i, t_{i+1}]$로 이산화하면 정확히 같은 형태가 나온다. 구간 길이를 $\delta_i$라 할 때

$$\alpha_i = 1 - e^{-\sigma_i \delta_i},\qquad
T_i = \prod_{j<i}(1-\alpha_j),\qquad
C \approx \sum_i T_i\,\alpha_i\,c_i$$

**대응 관계:**

| NeRF (연속) | 3DGS (이산) |
|---|---|
| 광선 위의 샘플 점 $t_i$ | 깊이순으로 정렬된 Gaussian $i$ |
| 밀도 $\sigma(t)$와 구간 $\delta_i$ | 불투명도 $o_i$와 종 모양 $e^{-\sigma_i}$ |
| $\alpha_i = 1 - e^{-\sigma_i\delta_i}$ | $\alpha_i = \min(0.99,\ o_i e^{-\sigma_i})$ |
| $T(t) = \exp(-\int\sigma)$ | $T_i = \prod_{j<i}(1-\alpha_j)$ |
| $\int T\sigma c\,dt$ | $\sum_i c_i\alpha_i T_i$ |

($\exp$의 지수 법칙 $\exp(-\sum) = \prod\exp(-\cdot)$ 덕에 연속 적분의 지수가 이산의 곱과 정확히 대응한다.)

차이는 **어디서 $\alpha$가 오는가**뿐이다. NeRF는 MLP를 광선 위 수백 곳에서 질의해 얻고, 3DGS는 이미 화면에 투영된 타원식으로 **닫힌 형태로 즉시** 계산한다. 그래서 3DGS가 수백 배 빠르다. 누적 방식은 완전히 동일하다.

## 10. 한 줄 요약

> 반투명 판을 앞에서부터 통과시키면 도달률 $T$는 $(1-\alpha)$씩 **곱해져** 줄고, 각 판의 기여는 "막은 비율 × 도달한 비율" $= \alpha_i T_i = T_i - T_{i+1}$이라 **망원급수로 접혀** 총합이 $1 - T_{end}$가 된다. 그래서 색은 $\sum c_i\alpha_i T_i$, alpha는 그냥 $1-T$다.

---

## 관련 코드

| 역할 | 위치 |
|---|---|
| 워크스루 ⑦ 절 + `rasterize_naive` | `.fm/assets/rasterization_walkthrough.py` (6. ⑦ 알파 블렌딩) |
| CUDA forward | `gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu` |
| $\sigma, \alpha$ 계산 | `gsplat/cuda/csrc/RasterizeToPixels3DGSDevice.cuh` `eval_gaussian_weight` |
| CUDA backward ($T$ 되감기) | `gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu` (`ra = 1/(1-alpha)`) |
| 임계값 상수 | `gsplat/cuda/_constants.py`, `gsplat/cuda/include/Common.h` |
| 순수 PyTorch 참조 | `gsplat/cuda/_torch_impl.py` `_rasterize_to_pixels` |
