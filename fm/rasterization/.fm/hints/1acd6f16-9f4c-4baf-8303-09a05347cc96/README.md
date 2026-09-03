# 래스터화 커널이 backward를 위해 저장하는 값

> **Q.** 래스터화 커널이 backward를 위해 저장하는 값은?
>
> **A.** 최종 투과율 T와 `last_ids`(픽셀별 마지막으로 기여한 Gaussian 인덱스)다. backward는 이 둘에서 출발해 역순으로 순회한다.

---

## 1. 왜 문제인가 — 일반 autograd라면 저장해야 할 것

알파 블렌딩 forward는 픽셀 $p$의 타일 목록을 앞→뒤로 훑으며 이렇게 누적한다.

$$\sigma_i = \tfrac12(a\,dx^2 + c\,dy^2) + b\,dx\,dy,\qquad
\alpha_i = \min\!\big(0.99,\ o_i e^{-\sigma_i}\big)$$

$$T_1 = 1,\qquad T_{i+1} = T_i\,(1-\alpha_i),\qquad
C_p = \sum_i c_i\,\alpha_i\,T_i \;+\; T_{\text{final}}\,\mathrm{bg}$$

PyTorch의 일반적인 autograd 규칙대로라면 곱셈 노드마다 피연산자를 붙잡아 둬야 하므로, **모든 $(i,p)$ 쌍에 대해 $T_i$와 $\alpha_i$를 전부 텐서로 남겨야** 한다. 그런데 그 개수는 이미지 크기가 아니라

$$\sum_{p}\ (\text{픽셀 } p \text{ 에 블렌딩된 Gaussian 수})$$

이고, 이건 씬 밀도에 따라 얼마든 커지는 데이터 의존적인 양이다. 워크스루의 `n_contrib` 맵(“픽셀별 블렌딩된 Gaussian 수”, `rasterize_naive`가 반환하고 4번째 패널로 그리는 그 이미지)이 바로 이 합의 분포다. 1920×1080에서 픽셀당 평균 200개만 잡아도

- 저장해야 할 원소: $2.07\text{M} \times 200 \times 2 \approx 8.3\times10^{8}$
- fp32로 약 **3.3 GB / 프레임**

학습은 프레임을 계속 돌리므로 이 방식은 시작부터 불가능하다. 반면 뒤에 나올 방식이 실제로 저장하는 건 픽셀당 스칼라 2개, 즉 $2.07\text{M} \times (4+4)\,\mathrm{B} \approx$ **16.6 MB**다. $O(\sum_p n_p)$ 대 $O(HW)$의 차이다.

## 2. 대신 쓰는 전략 — 재계산(recompute)

핵심 관찰은 **$T$의 점화식이 가역**이라는 것이다.

$$T_{i+1} = T_i(1-\alpha_i)\quad\Longleftrightarrow\quad \boxed{\,T_i = \frac{T_{i+1}}{1-\alpha_i}\,}$$

그리고 $\alpha_i$는 저장해 둘 필요가 없다. $\alpha_i$를 만드는 재료 — `means2d`, `conics`, `opacities` — 는 어차피 backward가 grad를 돌려줘야 하는 **입력**이라 이미 손에 있고, 픽셀 좌표만 있으면 $\sigma_i$, $\alpha_i$를 그 자리에서 다시 계산할 수 있다(`eval_gaussian_weight`). 따라서:

> **마지막 $T$ 하나**만 있으면, 뒤→앞으로 가면서 $\alpha_i$를 재계산하고 $T \mathrel{/}= (1-\alpha_i)$로 나눠 가며 모든 중간 $T_i$를 순서대로 되살릴 수 있다.

중간 활성값을 저장하는 대신 **다시 계산하는(recompute) 것으로 메모리와 연산을 맞바꾼** 전형적인 gradient-checkpointing 계열 설계이며, 원 3DGS 논문 Sec. 6이 명시적으로 취한 방식이다(픽셀별 블렌딩 목록을 남기는 대신 누적 불투명도만 저장하고, back-to-front 순회에서 각 점의 $\alpha$로 나눠 중간값을 복원한다).

## 3. 그런데 왜 `last_ids`도 필요한가

$T$ 하나만으로는 **어디서부터 되감기를 시작할지**를 모른다. 이유는 forward의 조기 종료다.

- 투과율이 `TRANSMITTANCE_THRESHOLD = 1e-4` 이하가 되면 그 픽셀은 거기서 끝난다(그 Gaussian은 **제외**하고 종료).
- 종료 지점은 **픽셀마다 다르다.** 앞쪽에 불투명한 Gaussian이 놓인 픽셀은 10개만에 끝나고, 빈 하늘을 보는 픽셀은 타일 목록 끝까지 간다.
- 종료 지점 뒤의 Gaussian들은 그 픽셀에 기여가 **정확히 0**이므로 grad도 0이다.

그래서 forward는 “이 픽셀이 마지막으로 실제 블렌딩한 Gaussian의 인덱스”를 픽셀마다 `last_ids`에 적어 둔다. backward는

1. `last_ids`가 가리키는 지점에서 시작해 (그 뒤는 건드리지 않고 = 낭비 없이),
2. 거기서 앞쪽으로 되감으며 $T$를 복원한다.

정리하면 **`T_final`은 “되감기의 초기값”, `last_ids`는 “되감기의 시작 위치”**다. 둘 다 픽셀당 스칼라 하나라서 저장 비용이 $O(HW)$에 머무른다.

## 4. 실제 텐서 이름 — `render_alphas = 1 − T`

여기서 실무적인 포인트 하나. forward 커널이 저장하는 텐서 이름은 `T`가 아니라 `render_alphas`다.

`gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu` 말미:

```cpp
// Here T is the transmittance AFTER the last gaussian in this pixel.
render_alphas[pix_id[p]] = 1.0f - T[p];
...
last_ids[pix_id[p]] = static_cast<int32_t>(cur_idx[p]);
```

`render_alpha = 1 − T`는 **어차피 커널이 사용자에게 돌려주는 출력**(불투명도 맵, 워크스루의 세 번째 패널 `render_alpha = 1 − T`)이다. 즉 backward를 위해 텐서를 하나 더 만드는 게 아니라, **이미 있는 출력을 재활용**해서 T를 담는다. backward는 첫 줄에서 그대로 되돌린다.

`gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu`:

```cpp
// this is the T AFTER the last gaussian in this pixel
float T_final           = 1.0f - render_alphas[pix_id];
float T                 = T_final;
// the contribution from gaussians behind the current one
float buffer[CDIM]      = {0.f};
// index of last gaussian to contribute to this pixel
const int32_t bin_final = inside ? last_ids[pix_id] : 0;
```

`last_ids`의 값은 forward의 `cur_idx[p] = batch_start + t`, 즉 **`flatten_ids` 배열에 대한 전역 인덱스**다(타일 내 상대 오프셋도 아니고 Gaussian id도 아니다). 초깃값은 0이라, 아무것도 블렌딩하지 못한 픽셀은 `bin_final = 0`이 되어 backward에서 사실상 전부 건너뛴다.

### `ctx.save_for_backward` 목록

`gsplat/cuda/_wrapper.py`의 `RegisterRasterizeToPixels3DGS.setup_context`:

```python
_render_colors, render_alphas, means2d_absgrad, last_ids = output
ctx.mark_non_differentiable(last_ids, means2d_absgrad)
...
ctx.save_for_backward(
    means2d, conics, colors, opacities, backgrounds, masks,
    isect_offsets, flatten_ids,
    render_alphas, last_ids, means2d_absgrad,
)
```

| 저장물 | 크기 | 역할 |
|---|---|---|
| `means2d`, `conics`, `colors`, `opacities` | $O(N)$ | forward **입력**. α와 색을 backward에서 **재계산**하는 재료 |
| `backgrounds`, `masks` | 작음 | 배경 항·타일 마스크 |
| `isect_offsets`, `flatten_ids` | $O(\text{tiles})$, $O(n_{\text{isects}})$ | forward와 **똑같은 타일별 Gaussian 목록·순서**를 다시 걷기 위함 |
| **`render_alphas`** | $O(HW)$ | $T_{\text{final}} = 1 - \text{render\_alphas}$ ← **정답의 “최종 투과율 T”** |
| **`last_ids`** | $O(HW)$ | 픽셀별 마지막 기여 Gaussian 인덱스 ← **정답의 “되감기 시작점”** |
| `means2d_absgrad` | $O(N)$ | AbsGS용 홀더(backward가 in-place로 채움) |

“중간 활성값”은 하나도 없다. 저장 목록은 **forward의 입력 + 픽셀 크기 스칼라 맵 2개**가 전부다. `last_ids`는 `mark_non_differentiable`로 표시되어 autograd 그래프에 얽히지 않는다(정수 인덱스이므로).

## 5. backward 커널의 역순 배치 순회 — forward와 대칭

forward는 타일 목록을 256개씩 배치로 끊어 shared memory에 적재하고 앞→뒤로 소비한다. backward는 **그 구조를 그대로 뒤집는다.**

| | forward (`...Fwd.cu`) | backward (`...Bwd.cu`) |
|---|---|---|
| 블록 | 타일 하나 | 타일 하나 (동일) |
| 배치 적재 | `batch_start = range_start + BATCH_SIZE*b`, `idx = batch_start + tid` | `batch_end = range_end - 1 - block_size*b`, `idx = batch_end - tr` |
| 소비 순서 | `for t = 0 … batch_size` (앞→뒤) | `for t = max(0, batch_end - warp_bin_final) … batch_size` (뒤→앞) |
| 상태 갱신 | `T *= (1 - alpha)` | `T /= max(1e-6, 1 - alpha)` |
| 종료 | `T ≤ 1e-4` → `done_mask`, `__syncthreads_count`로 타일 조기 종료 | 시작을 `last_ids`로 잘라 뒤쪽을 아예 안 봄 |

“역순으로 순회한다”는 게 실제 코드에서 어떻게 표현되는지가 요점이다. 워크스루 개념 코드로 쓰면 `for (idx = bin_final - 1; idx >= range_start; --idx)` 꼴이고, 실제 커널은 이를 shared-memory 배치 단위로 구현한 것이다. 두 군데서 “뒤쪽 건너뛰기”가 일어난다.

```cpp
const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
...
for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
    bool valid = inside;
    if (batch_end - t > bin_final) { valid = 0; }   // 픽셀 단위 컷
    ...
}
```

- **warp 단위 컷**: warp 32개 레인의 `bin_final` 최댓값(`warp_bin_final`)보다 뒤인 구간은 루프 자체를 건너뛴다.
- **픽셀 단위 컷**: 그 안에서도 자기 `bin_final`보다 뒤면 `valid = 0`.

즉 `last_ids`는 정확성(어디서 $T$ 되감기를 시작할지)뿐 아니라 **성능**(기여 0인 구간을 아예 안 도는 것)에도 직결된다.

한 가지 비대칭: forward는 `PIXELS_PER_THREAD`(스레드 1개가 픽셀 여러 개)에 `done_mask` 비트마스크로 조기 종료를 관리하는 반면, backward는 스레드 1개 = 픽셀 1개다. 이는 Gaussian 하나의 grad를 warp 32개 레인에 걸쳐 `warpSum`으로 합친 뒤 lane 0만 `atomicAdd`하기 위한 배치다.

## 6. 색 grad를 위해 함께 되감는 것 — 누적 버퍼 $S$

$T$만 되감아서는 부족하다. $\partial C/\partial \alpha_i$가 **$i$보다 뒤에 있는 모든 Gaussian의 색 기여**에 의존하기 때문이다.

$$C_p = \sum_j c_j \alpha_j T_j,\qquad T_j = T_i(1-\alpha_i)\!\!\prod_{i<k<j}\!\!(1-\alpha_k)\ \ (j>i)$$

$\alpha_i$를 흔들면 (a) 자기 항 $c_i\alpha_i T_i$가 변하고, (b) 뒤의 모든 항이 공통 인자 $(1-\alpha_i)$를 통해 변한다.

$$\frac{\partial C}{\partial \alpha_i}
= c_i T_i \;+\; \sum_{j>i} c_j\alpha_j \frac{\partial T_j}{\partial \alpha_i}
= c_i T_i \;-\; \frac{1}{1-\alpha_i}\sum_{j>i} c_j\alpha_j T_j$$

뒤쪽 누적치를 $\;\mathrm{buf}_{i+1} \equiv \sum_{j>i} c_j\alpha_j T_j\;$로 두고, 이를 $T_{i+1}$로 정규화한 “뒤에서 온 색” $S_{i+1} = \mathrm{buf}_{i+1}/T_{i+1}$을 쓰면 $\mathrm{buf}_{i+1} = T_i(1-\alpha_i)S_{i+1}$ 이므로

$$\boxed{\ \frac{\partial C}{\partial \alpha_i} = T_i\big(c_i - S_{i+1}\big)\ }$$

**직관**: $\alpha_i$를 키우면 자기 색 $c_i$를 더 넣는 대신, 뒤쪽이 합성해 만들던 색 $S_{i+1}$을 그만큼 가린다. 이득은 그 차이 $c_i - S_{i+1}$이고, 앞쪽이 이미 가린 만큼 $T_i$로 감쇠된다. 앞쪽 Gaussian일수록 $T_i \approx 1$이라 grad가 크다.

핵심은 **$S$(=`buffer`)도 $T$처럼 뒤→앞 순회로 자연스럽게 굴러간다**는 것이다. 뒤에서부터 오므로 현재 Gaussian을 처리하는 시점의 `buffer`는 정확히 “뒤쪽 누적치”이고, 처리 후에 자기 기여를 더하면 다음(앞쪽) 단계용 버퍼가 된다. 커널에서:

```cpp
float ra        = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
T              *= ra;                 // T_{i+1} -> T_i
const float fac = alpha * T;          // α_i T_i
for (k) v_rgb_local[k] = fac * v_render_c[k];               // ∂L/∂c_i = α_i T_i · v_C

float v_alpha = 0.f;
for (k) v_alpha += (rgbs_batch[t*CDIM+k] * T - buffer[k] * ra) * v_render_c[k];
//                  c_i·T_i           -   buf_{i+1}/(1-α_i)     ← 위 유도와 같은 식
v_alpha += T_final * ra * v_render_a;                        // ∂A/∂α_i = T_final/(1-α_i)
if (backgrounds) v_alpha += -T_final * ra * accum;           // 배경 항

for (k) buffer[k] += rgbs_batch[t*CDIM+k] * fac;             // buf_{i+1} -> buf_i
```

`v_alpha`가 나오면 나머지는 연쇄법칙이다. $\alpha_i = o_i\,\mathrm{vis}$, $\mathrm{vis} = e^{-\sigma_i}$이므로 $\partial\alpha/\partial\sigma = -o_i\,\mathrm{vis}$이고, $\sigma$가 `conic`·`delta`의 이차형식이므로 `v_conic`, `v_xy`(= $\partial L/\partial$`means2d`), `v_opacity = vis * v_alpha`가 곧바로 따라 나온다. 참고로 $\alpha$가 상한 0.99에 걸린 Gaussian은 `if (opac * vis <= MAX_ALPHA)` 가드로 conic/means2d/opacity grad가 0이 된다(클램프 구간에서 미분 0).

이 grad들은 Gaussian 하나에 대해 여러 픽셀(스레드)로 흩어져 있으므로 `warpSum`으로 warp 내 합산 후 lane 0이 `atomicAdd_system`으로 전역에 모은다. `absgrad=True`면 $|\partial L/\partial \text{means2d}|$를 따로 누적하는데(AbsGS 밀도화 기준), 이것도 같은 루프에서 처리된다. 워크스루가 말하듯 `means2d.grad`의 크기는 DefaultStrategy의 split/duplicate 판단 기준이 되므로 이 값 자체가 학습의 밀도 제어에 직접 쓰인다.

## 7. 나눗셈 안정성 — 0.99 상한이 하는 일

되감기가 **나눗셈** $T/(1-\alpha)$이라는 점이 이 설계의 유일한 수치적 약점이다. $\alpha \to 1$이면 발산한다. gsplat은 세 겹으로 막는다.

`gsplat/cuda/include/Common.h`:

```c
#define ALPHA_THRESHOLD         (1.f / 255.f)
// MAX_ALPHA and TRANSMITTANCE_THRESHOLD are chosen so that the equivalent of
// a maximal opacity Gaussian has to be rasterized twice to reach the threshold,
// without getting the transmittance too small for numerical stability of
// the backward pass.
// i.e. TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2
#define MAX_ALPHA               0.99f
#define TRANSMITTANCE_THRESHOLD 1e-4f
// Floor for (1 - alpha) when computing 1/(1-alpha) in backward rasterization.
#define MIN_ONE_MINUS_ALPHA     1e-6f
```

1. **`MAX_ALPHA = 0.99`** — forward에서 $\alpha_i = \min(0.99,\ o_i e^{-\sigma_i})$로 잘라 두면 $1-\alpha_i \ge 0.01$이 보장된다. 따라서 backward의 `ra = 1/(1-α) ≤ 100`으로 **상한이 걸린다.** 저장하는 게 $T$ 하나뿐이라 backward가 나눗셈을 $n_p$번 연달아 하는데, 그 각각이 유계라는 뜻이다. 즉 0.99 상한은 “거의 불투명한 Gaussian 표현”을 위한 게 아니라 상당 부분 **backward의 수치 안정성을 위한 장치**다.
2. **`TRANSMITTANCE_THRESHOLD = 1e-4 = (1 − 0.99)²`** — 주석이 밝히듯 이 값은 우연이 아니라 `MAX_ALPHA`에서 유도됐다. 최대 불투명 Gaussian이 두 번은 깔려야 도달하는 값이고, 동시에 $T_{\text{final}}$이 fp32에서 상대오차가 커질 만큼 작아지지 않게 한다. forward 코드 주석도 “$T$는 backward에서 쓰이고 매우 작아질 수 있어 double로 저장하는 게 이상적이지만 backward가 1.5배 느려져서 float를 유지한다”고 남겨 두었다.
3. **`MIN_ONE_MINUS_ALPHA = 1e-6`** — `fmaxf`로 마지막 바닥을 깐다. 위 1번 때문에 정상 경로에서는 실제로 걸릴 일이 없는 보험이다.

되감기 오차 관점에서도 정리하면, $T_i$를 나눗셈으로 복원하므로 오차가 앞으로 갈수록 누적된다. 그런데 grad 크기 $\propto T_i$이고 앞쪽일수록 $T_i$가 1에 가까워 나눗셈 횟수가 적으므로, **큰 grad를 갖는 항일수록 오차가 적게 쌓이는** 방향이라 실용상 문제가 되지 않는다. 워크스루 7장이 `rasterization()` 경로와 `rasterize_stepwise` 경로의 grad를 `maxdiff`로 비교해 일치를 확인하는 것도 이 경로가 실제로 통과하는 지점이다.

## 8. 트레이드오프 요약

| | 전부 저장(일반 autograd) | 재계산(gsplat/3DGS) |
|---|---|---|
| 메모리 | $O\!\left(\sum_p n_p\right)$ — 데이터 의존, 수 GB | $O(HW)$ — `render_alphas` + `last_ids`, 수십 MB |
| backward 연산 | $\alpha$ 재계산 불필요 | $\sigma,\alpha$를 **다시** 계산 → forward와 비슷한 양의 flops 추가 (전체 ≈ 2×) |
| 메모리 대역폭 | 큰 활성 텐서 write/read | `means2d`/`conics`/`colors`를 다시 read (shared memory로 타일 내 재사용) |
| 수치 | $T_i$ 정확 | $T/(1-\alpha)$ 되감기 → 상한/하한 상수로 방어 |

GPU에서 flops는 남고 메모리 대역폭과 용량이 병목이므로, “연산 2배를 내고 메모리를 픽셀 수준으로 낮춘다”는 교환은 압도적으로 유리하다. 애초에 저장하는 쪽은 GB 단위라 선택지가 아니었다.

---

## 한 줄 정리

forward는 **`render_alphas`($=1-T_{\text{final}}$)와 `last_ids` 딱 두 개의 픽셀 크기 맵**만 남긴다. backward는 `last_ids`가 가리키는 지점부터 타일 목록을 뒤→앞으로 돌며, $\alpha_i$를 `means2d`/`conics`/`opacities`에서 재계산하고 $T_i = T_{i+1}/(1-\alpha_i)$로 투과율을, `buffer`로 뒤쪽 색 누적치 $S$를 함께 되감아 모든 grad를 만든다.

## 관련 소스

| 대상 | 경로 |
|---|---|
| 워크스루 6장(블렌딩)·정리표(backward 커널 요지) | `/home/sungwoo/projects/swcho/gsplat/fm/rasterization/.fm/assets/rasterization_walkthrough.py` |
| forward 커널 (`render_alphas`, `last_ids` 기록) | `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu` |
| backward 커널 (`T_final`, `bin_final`, 역순 배치, `buffer`) | `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu` |
| `ctx.save_for_backward` 목록 | `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/_wrapper.py` (`RegisterRasterizeToPixels3DGS`) |
| 상수 정의 | `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/include/Common.h`, `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/_constants.py` |
