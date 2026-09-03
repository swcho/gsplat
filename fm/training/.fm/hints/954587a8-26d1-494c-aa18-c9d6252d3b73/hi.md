# 반투명 색유리 겹쳐 보기 — 알파 블렌딩 공식의 정체

> 전제: 등비수열의 곱, 지수함수, 확률의 곱셈정리(독립사건), 정규분포, 벡터와 행렬의 곱, 이차곡선(타원). 대학 수준 개념(공분산 행렬, 마할라노비스 거리)은 아래에서 고교 개념에서부터 쌓아 올린다.

## 0. 우리가 풀려는 문제

앞 단계까지 끝내면 화면 위에 **뿌옇게 번진 타원 얼룩**(2D로 눌러 찍힌 Gaussian) 수십만 개가 놓여 있다. 각 얼룩은 세 가지 정보를 갖는다.

| 기호 | 이름 | 뜻 |
|---|---|---|
| $c_i$ | color | 이 얼룩의 색 (RGB 3성분) |
| $o_i$ | opacity | 이 얼룩이 얼마나 진한가, $0<o_i<1$ |
| $\Sigma'_i$ | 2D 공분산 | 화면에서 이 얼룩이 어떤 타원으로 퍼지는가 |

이제 **픽셀 하나**의 색을 정해야 한다. 그 픽셀 위에는 얼룩 수십~수백 개가 겹쳐 있다. 어떻게 하나의 색을 뽑을까?

카드가 묻는 것이 바로 그 답이다.

$$C = \sum_i c_i\,\alpha_i \prod_{j<i}(1-\alpha_j),\qquad \alpha_i = o_i \exp\!\left(-\tfrac12 \Delta^\top \Sigma'^{-1} \Delta\right)$$

이 식은 두 부분으로 쪼개서 읽어야 한다.

- **바깥쪽** $\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j)$ — "겹친 반투명 층들을 어떻게 합치나" (§1~§4)
- **안쪽** $\alpha_i = o_i\exp(\cdots)$ — "이 픽셀에서 이 얼룩의 알파는 얼마인가" (§5~§6)

---

## 1. 반투명 색유리 한 장

색깔이 든 반투명 유리판 한 장을 흰 벽 앞에 세웠다고 하자. 유리판은 자기 색 $c_1$을 내고, 뒤에서 오는 빛은 일부만 통과시킨다.

- 유리판이 가리는 비율 = $\alpha_1$ (알파, 불투명도)
- 뒤를 통과시키는 비율 = $1-\alpha_1$ (투과율, transmittance)

그럼 눈에 보이는 색은

$$C = c_1\,\alpha_1 + C_{\text{배경}}\,(1-\alpha_1)$$

$\alpha_1=1$이면 완전 불투명해서 유리 색만 보이고, $\alpha_1=0$이면 유리가 없는 것과 같다. **가중평균**이다. 여기까지는 그냥 상식이다.

## 2. 두 장, 세 장

이제 유리판을 **앞에서부터** 1번, 2번 순서로 두 장 세웠다. 눈은 앞에 있다.

- 1번 유리는 자기 색을 $\alpha_1$만큼 보탠다 → $c_1\alpha_1$
- 2번 유리에서 나온 빛은 **1번 유리를 뚫고** 나와야 한다. 그러니 $(1-\alpha_1)$이 곱해진다 → $c_2\alpha_2\,(1-\alpha_1)$

$$C = c_1\alpha_1 + c_2\alpha_2(1-\alpha_1) + \cdots$$

세 장이면 3번 유리의 빛은 2번과 1번을 **둘 다** 뚫어야 한다.

$$C = c_1\alpha_1 + c_2\alpha_2(1-\alpha_1) + c_3\alpha_3(1-\alpha_1)(1-\alpha_2)$$

패턴이 보인다. $i$번째 유리 앞에는 $1,2,\dots,i-1$번 유리가 있으므로, 곱해지는 투과율은

$$T_i = (1-\alpha_1)(1-\alpha_2)\cdots(1-\alpha_{i-1}) = \prod_{j<i}(1-\alpha_j)$$

$$\boxed{\;C = \sum_i c_i\,\alpha_i\,T_i = \sum_i c_i\,\alpha_i\prod_{j<i}(1-\alpha_j)\;}$$

**카드의 첫 번째 식이 그냥 "반투명 유리판 여러 장 겹치기"다.** $\prod_{j<i}$는 "나보다 앞에 있는 것들 전부"라는 뜻이고, 하는 일은 "앞에 가려진 만큼 내 기여를 깎는다"다.

### 왜 곱셈인가 — 확률로 읽기

$1-\alpha_j$를 "빛 알갱이 하나가 $j$번 유리를 무사히 통과할 확률"이라고 보자. 각 유리를 통과하는 사건이 서로 독립이라고 하면, 확률의 곱셈정리로

$$P(\text{1번부터 } i-1\text{번까지 다 통과}) = \prod_{j<i}(1-\alpha_j)$$

$\alpha_i$는 "$i$번 유리에서 흡수·산란될 확률"이고, $\alpha_i T_i$는 "**정확히 $i$번 유리에서 눈에 들어온 빛의 비율**"이다. 즉 $w_i = \alpha_i T_i$는 확률분포처럼 생겼고 $\sum_i w_i \le 1$이다. 그래서 이 $w_i$를 **가중치(weight)** 또는 **기여도**라 부른다. 식은 결국

$$C = \sum_i w_i\,c_i \quad(\text{가중평균}), \qquad \sum_i w_i = 1 - \prod_i (1-\alpha_i)$$

이고, 이 $\sum_i w_i$ 값이 gsplat이 `render_alphas`로 돌려주는 **누적 알파**다.

## 3. 재귀식 — 컴퓨터가 실제로 계산하는 방법

식에 $\prod$이 있으니 매번 곱을 다 다시 계산할까? 아니다. 투과율 하나만 들고 앞→뒤로 한 번 훑으면 된다.

$$T \leftarrow 1,\quad C\leftarrow 0$$
$$\text{각 } i \text{에 대해:}\quad C \mathrel{+}= c_i\,\alpha_i\,T,\qquad T \mathrel{\leftarrow} T\,(1-\alpha_i)$$

이게 CUDA 커널에 그대로 있다 (`RasterizeToPixels3DGSSerialBatchFwd.cu`).

```cpp
const float next_T = T * (1.0f - alpha);
const float vis    = alpha * T;          // = w_i
for (k = 0; k < CDIM; ++k) pix_out[k] += c_ptr[k] * vis;
T = next_T;
```

`T`는 픽셀마다 하나씩 들고 있는 스칼라다. $\prod$ 기호는 "루프 안에서 곱해 나가는 변수"의 수식 표기일 뿐이다.

## 4. 순서가 결과를 바꾼다 — 그래서 깊이순 정렬

$\prod_{j<i}$의 $j<i$는 "$i$보다 **앞에 있는** 것"이다. 그러니 순서가 정해져 있어야 한다. 순서를 바꾸면 답이 달라지는지 확인해 보자. 빨강 $c_1=1$, 파랑 $c_2=0$, 둘 다 $\alpha=0.5$:

| 순서 | 계산 | 결과 |
|---|---|---|
| 빨강 앞 | $1(0.5) + 0(0.5)(0.5)$ | $0.50$ |
| 파랑 앞 | $0(0.5) + 1(0.5)(0.5)$ | $0.25$ |

같은 얼룩 두 개인데 값이 두 배 차이 난다. **알파 블렌딩은 교환법칙이 성립하지 않는다.** (가중평균 자체는 교환 가능하지만 가중치 $T_i$가 순서에 의존한다.)

그래서 앞 단계인 `isect_tiles`가 깊이로 정렬을 해 준다. 정렬 키는 64비트 정수 하나로 만든다.

$$\text{key} = \underbrace{\text{image\_id}}_{\text{상위}}\;\Vert\;\text{tile\_id}\;\Vert\;\underbrace{\text{float32(depth)의 비트 패턴}}_{\text{하위 32비트}}$$

양수 float32는 비트 패턴을 그대로 정수로 읽어도 크기 순서가 보존된다. 이 성질을 이용해 실수 정렬을 정수 radix sort 한 번으로 처리한다. 정렬 결과 같은 타일의 얼룩들이 메모리에서 연속으로, 그 안에서 **앞→뒤(near→far)** 로 놓인다.

### 왜 "타일별로"인가

화면을 $16\times16$ 픽셀 타일로 자르는 이유는 세 가지다.

1. **정렬을 국소화한다.** 전체 화면을 통째로 정렬하는 게 아니라, 타일 단위로 나눠 각 타일이 자기와 겹치는 얼룩만 갖는다. 겹치지 않는 얼룩은 아예 후보에서 빠진다.
2. **한 타일의 256개 픽셀이 같은 얼룩 목록을 공유한다.** GPU 블록 하나가 타일 하나를 맡고, 얼룩 데이터(`means2d`, `conics`, `opacity`)를 **공유 메모리**에 배치로 한 번 올려 256픽셀이 재사용한다. 커널의 `xy_opacity_batch`, `conic_batch`가 그것이다.
3. **조기 종료를 블록 단위로 할 수 있다.** §7에서 설명한다.

정렬을 타일별로 해도 되는 근거는 "픽셀은 자기 타일 안에만 있고, 타일 안에서의 깊이 순서는 그 타일 모든 픽셀에 대해 동일하다"는 것이다. 엄밀히는 얼룩 하나가 여러 타일에 걸치므로 얼룩–타일 쌍마다 키를 하나씩 만든다(이 쌍의 개수가 `n_isects`).

---

## 5. $\alpha_i$의 정체 — 2차원 정규분포 값

이제 안쪽 식이다.

$$\alpha_i = o_i \exp\!\left(-\tfrac12 \Delta^\top \Sigma'^{-1} \Delta\right)$$

$\alpha_i$가 얼룩마다 하나씩인 상수가 아니라 **픽셀마다 다른 값**이라는 점이 핵심이다. 얼룩 중심 근처 픽셀에서는 진하고, 멀어지면 옅어진다. 유리판 비유로 하면 "가운데가 진하고 가장자리로 갈수록 투명해지는 얼룩진 유리판"이다.

### 1차원에서 출발

고교에서 배운 정규분포는

$$f(x) \propto \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) = \exp\!\left(-\tfrac12\cdot \Delta\cdot\frac{1}{\sigma^2}\cdot\Delta\right),\qquad \Delta = x-\mu$$

이렇게 다시 쓴 형태를 기억하자. **"편차를 두 번 곱하고 가운데에 분산의 역수를 끼운다."**

### 2차원으로

이제 화면이라 좌표가 두 개다. $\Delta$는 스칼라가 아니라 벡터다.

$$\Delta = \begin{pmatrix}\Delta x\\ \Delta y\end{pmatrix} = (\text{픽셀 중심}) - (\text{얼룩 중심 } \texttt{means2d})$$

$\frac{1}{\sigma^2}$ 자리에 들어갈 것은 수 하나가 아니라 $2\times2$ 행렬 $\Sigma'^{-1}$이다. $\Sigma'$가 2D **공분산 행렬**(퍼짐을 나타내는 행렬)이고, 1차원에서 $\frac{1}{\sigma^2}$가 분산의 역수였던 것처럼 역행렬을 쓴다. $\Sigma'^{-1} = \begin{pmatrix}a&b\\b&c\end{pmatrix}$로 두면

$$\Delta^\top \Sigma'^{-1}\Delta = a\,\Delta x^2 + 2b\,\Delta x\,\Delta y + c\,\Delta y^2$$

**$\Delta x, \Delta y$의 이차식**이다. 그리고 이 값이 상수인 자리, 즉

$$a\,\Delta x^2 + 2b\,\Delta x\Delta y + c\,\Delta y^2 = k$$

는 기하에서 배운 **회전된 타원의 방정식**이다. $b\ne0$이면 축이 기울고, $a,c$가 각 방향의 뾰족함을 정한다. 그래서 코드는 이 행렬을 **`conics`**(원뿔곡선)라고 부르고, 대칭이라 3개 값 $(a,b,c)$만 저장한다.

> 이 이차식 $\Delta^\top\Sigma'^{-1}\Delta$의 제곱근을 **마할라노비스 거리**라 한다. "유클리드 거리를 재는데, 퍼진 방향으로는 관대하게 재는" 거리다. 원형 얼룩($\Sigma'=\sigma^2 I$)에서는 그냥 $\|\Delta\|^2/\sigma^2$로 되돌아간다.

### 코드와 대조

CUDA 커널(`RasterizeToPixels3DGSDevice.cuh`)의 딱 세 줄이 위 수식 전부다.

```cpp
const float sigma = 0.5f * (conic.x*dx*dx + conic.z*dy*dy) + conic.y*dx*dy;
const float vis   = __expf(-sigma);
const float alpha = min(MAX_ALPHA, opac * vis);
```

`conic = (a, b, c)`이므로 `sigma` $= \frac12(a\Delta x^2 + c\Delta y^2) + b\,\Delta x\Delta y = \frac12\Delta^\top\Sigma'^{-1}\Delta$. 수식의 $\frac12$가 코드의 `0.5f`이고, $2b\Delta x\Delta y$의 2와 $\frac12$가 약분되어 `conic.y*dx*dy`가 계수 없이 나온다. 곱셈·덧셈·지수 하나뿐이다 — 나눗셈도 역행렬 계산도 픽셀 단계에는 없다. 그것이 앞 단계에서 $\Sigma'$가 아니라 **$\Sigma'^{-1}$을 미리 넘겨주는 이유**다.

부호도 확인해 두자. 커널은 `dx = mean - pixel`, PyTorch 참조 구현(`_torch_impl.py`)은 `pixel - mean`을 쓴다. 이차식이라 $\Delta$의 부호가 뒤집혀도 값이 같으므로 둘은 동일하다.

## 6. $o_i$와 지수항의 역할 분담

$$\alpha_i = \underbrace{o_i}_{\text{얼룩 고유의 진하기}} \times \underbrace{\exp(-\tfrac12\Delta^\top\Sigma'^{-1}\Delta)}_{\text{픽셀 위치에 따른 감쇠, 최대 1}}$$

- $o_i$는 학습되는 파라미터다. 다만 그대로 저장하지 않고 logit으로 저장해 `sigmoid`를 씌운다 → $0<o_i<1$이 자동 보장된다 (워크스루의 `torch.sigmoid(splats["opacities"])`).
- 지수항은 얼룩 중심에서 정확히 1이고, 멀어지면 단조 감소한다. 따라서 $\alpha_i \le o_i$이고, 중심 픽셀에서만 $o_i$에 도달한다.

$\Delta=0$일 때 값이 1이라는 것도 중요하다. 확률밀도함수라면 앞에 $\frac{1}{2\pi\sqrt{\det\Sigma'}}$ 같은 정규화 상수가 붙어야 하는데, **여기엔 없다.** 우리는 확률을 재는 게 아니라 "가리는 비율"을 재는 것이므로 최댓값이 1이어야 하고, 진하기 조절은 $o_i$가 따로 맡는다. 정규화 상수를 붙이면 작은 얼룩이 무한히 진해져 버린다.

---

## 7. 실제 구현의 안전장치들

수식은 위 두 줄이지만 실전 커널에는 상수 세 개가 더 있다 (`gsplat/cuda/include/Common.h`, `gsplat/cuda/_constants.py`).

| 상수 | 값 | 무엇을 막는가 |
|---|---|---|
| `MAX_ALPHA` | $0.99$ | $\alpha=1$이면 역전파에서 $\frac{1}{1-\alpha}$가 발산한다. 상한을 씌워 막는다 |
| `ALPHA_THRESHOLD` | $1/255$ | 8비트 색으로 1단계도 못 바꾸는 기여는 건너뛴다 |
| `TRANSMITTANCE_THRESHOLD` | $10^{-4}$ | 남은 투과율이 이 아래면 **조기 종료** |

**(a) 조기 종료.** $T$는 앞→뒤로 가면서 $(1-\alpha)$를 계속 곱하니 단조 감소한다. $T$가 $10^{-4}$까지 떨어지면 뒤에 남은 얼룩들이 기여할 수 있는 총량은 그 시점의 $T$ 전부($\sum_{k\ge i} w_k \le T$)뿐이다. 그래서 픽셀을 "완료"로 표시하고 루프를 끊는다.

버리는 양의 상한을 정확히 세어 보면, 조건이 $T(1-\alpha)\le10^{-4}$이므로

$$\text{버리는 빛} \;=\; T \;\le\; \frac{10^{-4}}{1-\alpha} \;\le\; \frac{10^{-4}}{1-\texttt{MAX\_ALPHA}} = 10^{-2}$$

이다. 8비트 색 한 단계가 $1/255\approx4\times10^{-3}$이니 최악의 경우에도 몇 LSB 수준이고, 보통은 훨씬 작다.

```cpp
if (next_T <= TRANSMITTANCE_THRESHOLD) { done_mask |= (1u << p); continue; }
...
if (__syncthreads_count(done_mask == ALL_DONE) >= BATCH_SIZE) break;
```

두 번째 줄이 §4에서 미룬 이유다. **타일의 256개 픽셀이 모두 포화되면 블록 전체가 나머지 얼룩 로딩을 그만둔다.** 실내 장면처럼 겹침이 심한 곳에서 이게 큰 절약이다. 이 조기 종료가 성립하는 것도 깊이순 정렬 덕이다 — 앞에서부터 보니까 "이 뒤는 다 안 보인다"고 말할 수 있다.

또 `TRANSMITTANCE_THRESHOLD` $=10^{-4} = (1-0.99)^2 = (1-\texttt{MAX\_ALPHA})^2$ 인 것도 우연이 아니다. 주석에 적혀 있다: "최대 불투명도 Gaussian이 **두 개** 겹쳐야 임계에 도달하도록" 고른 값이다.

**(b) 출력 두 개.** 루프가 끝나면 커널은 이렇게 쓴다.

```cpp
render_alphas[pix] = 1.0f - T[p];
render_colors[pix] = pix_out[p][k] + T[p] * backgrounds[k];   // 배경 있으면
```

- 최종 $T$는 "끝까지 뚫고 나온 비율"이니 $1-T = \sum_i w_i$ 가 누적 알파(§2)다. 이게 `rasterize_splats`가 돌려주는 `alpha`이고, 워크스루에서 배경색 합성이나 마스크 손실에 쓰인다.
- 배경은 남은 투과율 $T$만큼 섞인다. §1의 $C_\text{배경}(1-\alpha_1)$을 $n$장으로 일반화한 항이다.

**(c) `last_ids`.** 마지막으로 기여한 얼룩의 인덱스를 저장한다. 역전파에서 뒤→앞으로 되짚어 갈 때 어디서 시작할지 알려주는 값이다.

---

## 8. 왜 z-buffer가 아니고 이 식인가

일반 3D 게임은 "가장 앞의 것만 그린다"(z-buffer)를 쓴다. 3DGS가 그렇게 하지 않는 이유는 **미분** 때문이다.

- "가장 앞"을 고르는 연산은 계단 함수라 미분이 0 아니면 정의되지 않는다. 얼룩을 조금 움직여도 손실이 안 변하거나 갑자기 튄다.
- 반면 $C=\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j)$는 $c_i, o_i, \Delta, \Sigma'^{-1}$ 전부에 대해 매끄럽게 미분 가능하다. 지수함수와 곱셈·덧셈만으로 되어 있으니 당연하다.

특히 $\Delta$를 통해 $\texttt{means2d}$에 대한 gradient가 생기는데, 워크스루 5단계가 쓰는 밀도화 신호가 정확히 이것이다.

$$\frac{\partial \mathcal{L}}{\partial\,\texttt{means2d}_i}\ \text{가 크다} \;\Rightarrow\; \text{"이 얼룩은 화면에서 옮겨지고 싶어 한다"} \;\Rightarrow\; \text{split/duplicate}$$

또 $w_i = \alpha_i T_i$가 gradient의 크기를 자동으로 조절한다. 앞이 꽉 막혀 $T_i\approx0$인 얼룩은 색에 기여하지 않으니 gradient도 거의 0이다 — **가려진 것은 학습되지 않는다.** 이는 버그가 아니라 물리적으로 옳은 동작이고, 동시에 "가려진 얼룩이 영원히 방치될 수 있다"는 3DGS의 알려진 약점(그래서 opacity reset, MCMC의 relocate 같은 장치가 필요한 이유)이기도 하다.

---

## 9. 한 줄 정리

`rasterize_to_pixels`는 픽셀 하나마다 **투과율 스칼라 $T$ 하나**를 들고 그 픽셀에 걸친 얼룩들을 깊이순 앞→뒤로 훑으며, 매번 2D 정규분포 값 $\exp(-\frac12\Delta^\top\Sigma'^{-1}\Delta)$에 불투명도 $o_i$를 곱해 $\alpha_i$를 얻고, $c_i\alpha_i T$를 색에 더한 뒤 $T$에 $(1-\alpha_i)$를 곱한다. 타일 단위로 묶어 정렬·공유메모리·조기 종료를 얻고, 전체가 미분 가능하므로 이 한 식이 3DGS 학습의 통로가 된다.

## 참고 위치

- `gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu:222` — 앞→뒤 블렌딩 루프
- `gsplat/cuda/csrc/RasterizeToPixels3DGSDevice.cuh:45` — `eval_gaussian_weight()` (sigma → alpha)
- `gsplat/cuda/include/Common.h:97` — `ALPHA_THRESHOLD`, `MAX_ALPHA`, `TRANSMITTANCE_THRESHOLD`
- `gsplat/cuda/_torch_impl.py:786` — PyTorch 참조 구현의 sigma/alpha 계산
- `gsplat/cuda/_torch_impl.py:405` — `isect_tiles`의 (tile_id, depth) 64비트 정렬 키
