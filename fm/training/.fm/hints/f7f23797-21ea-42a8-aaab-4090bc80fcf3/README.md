# 픽셀 래스터화의 조기 종료 (early termination)

**Q.** 픽셀 래스터화에서 조기 종료는 언제 발생하는가?

**A.** 누적 투과율(transmittance) `T`가 임계값 아래로 떨어지면 남은 Gaussian을 처리하지 않고 종료한다. 뒤쪽 Gaussian이 화면에 기여하지 못하므로 계산을 아낀다.

---

## 1. 어디서 벌어지는 일인가

워크스루(`training_walkthrough.py`)가 정리한 `rasterization()`의 4단계 중 **마지막 단계**다.

```
SH 평가 → 투영(EWA) → 타일 교차(16×16, depth 정렬) → 픽셀 래스터화 ← 여기
```

`rasterize_to_pixels`는 타일마다 **깊이순으로 앞→뒤** 알파 블렌딩을 한다.

$$C = \sum_i c_i\,\alpha_i \prod_{j<i}(1-\alpha_j), \qquad \alpha_i = o_i \exp\!\left(-\tfrac12 \Delta^\top \Sigma'^{-1} \Delta\right)$$

여기서 뒤쪽 곱 $\prod_{j<i}(1-\alpha_j)$가 바로 **누적 투과율** `T`다. "지금까지 만난 Gaussian들을 통과해서 이 픽셀까지 아직 남아 있는 빛의 비율"이라는 뜻이고, 시작값은 `T = 1.0`이다. Gaussian 하나를 블렌딩할 때마다

```
T ← T * (1 - alpha)
```

로 단조 감소한다. `T`가 작다는 것은 **앞쪽 Gaussian들이 이미 픽셀을 다 덮었다**는 뜻이므로, 그 뒤에 있는 Gaussian은 화면 색에 사실상 아무 영향을 못 준다. 그래서 그 지점에서 루프를 끊는 것이 조기 종료다.

## 2. 실제 코드의 조건

핵심 로직은 dense/sparse 커널이 공유하는 device 함수에 한 군데로 모여 있다 — `gsplat/cuda/csrc/RasterizeToPixels3DGSDevice.cuh`의 `rasterize_to_pixels_3dgs_blend_fwd()`:

```cpp
const float alpha  = gw.alpha;
const float next_T = T * (1.0f - alpha);
if (next_T <= TRANSMITTANCE_THRESHOLD)
{
    return true; // pixel saturated; exclusive of this gaussian
}
const float vis = alpha * T;
for (uint32_t k = 0; k < CDIM; ++k) pix_out[k] += color_ptr[k] * vis;
cur_idx = gaussian_idx;
T       = next_T;
return false;
```

읽을 때 주의할 점 세 가지.

1. **판정은 `next_T`로 한다.** 즉 "이 Gaussian을 반영한 *뒤*의 T"를 미리 계산해서 임계값과 비교한다.
2. **경계 조건이 exclusive다.** 조건을 만족하면 `pix_out`에 더하기 **전에** 빠져나온다. 즉 임계값을 넘기게 만든 그 Gaussian 자체는 색에 기여하지 않고, `T`도 갱신되지 않는다. 코드 주석의 `exclusive of this gaussian`이 그 뜻이다.
3. **비교가 `<=`다.** `T`가 정확히 임계값이 되어도 종료다.

임계값 자체는 `gsplat/cuda/include/Common.h`에 있다.

```cpp
#define ALPHA_THRESHOLD         (1.f / 255.f)
#define GAUSSIAN_EXTEND         3.33f
// MAX_ALPHA and TRANSMITTANCE_THRESHOLD are chosen so that the equivalent of
// a maximal opacity Gaussian has to be rasterized twice to reach the threshold,
// i.e. TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2
#define MAX_ALPHA               0.99f
#define TRANSMITTANCE_THRESHOLD 1e-4f
```

`T ≤ 1e-4`면 남은 Gaussian의 최대 기여가 8비트 색 한 단계($1/255 ≈ 3.9\times10^{-3}$)보다 훨씬 작으므로, 시각적으로 손실이 없다.

### 임계값이 왜 `1e-4`인가

주석이 설계 근거를 그대로 적어놨다. 알파는 `MAX_ALPHA = 0.99`로 clamp되므로 한 Gaussian이 `T`를 줄일 수 있는 최대 배율은 `1 - 0.99 = 0.01`이다. 따라서

$$\texttt{TRANSMITTANCE\_THRESHOLD} = (1 - \texttt{MAX\_ALPHA})^2 = 0.01^2 = 10^{-4}$$

로 잡으면 **불투명도가 최대인 Gaussian이라도 최소 2개는 블렌딩되어야** 종료 조건에 도달한다. 한 개로 픽셀이 끝나버리는 일을 막으면서, 동시에 `T`가 backward에서 `1/(1-alpha)`로 역추적될 때 float32로 다루기 힘들 만큼 작아지지도 않게 하는 타협점이다.

## 3. 조기 종료 ≠ Gaussian 하나 건너뛰기

같은 루프 안에 비슷하게 생긴, 그러나 **성질이 다른** 두 가지 skip이 있다. 카드에서 묻는 것은 뒤쪽 것이다.

| | 조건 | 효과 | 상태 |
|---|---|---|---|
| **개별 skip** (`ALPHA_THRESHOLD`) | `sigma < 0` 또는 `alpha < 1/255` | 이 Gaussian만 무시하고 **다음 Gaussian으로 계속** | 픽셀은 살아 있음 |
| **조기 종료** (`TRANSMITTANCE_THRESHOLD`) | `next_T <= 1e-4` | 이 픽셀의 **루프를 끝냄** | 픽셀 `done` |

`eval_gaussian_weight()`의 `valid` 플래그가 앞쪽, `next_T` 비교가 뒤쪽이다. "기여가 작아서 이 Gaussian을 버린다"와 "이미 다 가려졌으니 이 픽셀은 끝이다"를 헷갈리지 말 것.

## 4. 픽셀 단위 → 타일 단위로 번지는 절약

CUDA 커널은 픽셀 하나씩 도는 게 아니라 **한 CTA가 타일 하나**를 맡고, 스레드 하나가 여러 픽셀(`PIXELS_PER_THREAD`)을 담당한다. 그래서 종료 상태를 비트마스크로 관리한다 (`RasterizeToPixels3DGSSerialBatchFwd.cu`).

```cpp
constexpr uint32_t ALL_DONE = (1u << PIXELS_PER_THREAD) - 1u;
...
for (uint32_t t = 0; (t < batch_size) && (done_mask != ALL_DONE); ++t)
{
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        if (done_mask & (1u << p)) continue;           // 이 픽셀은 이미 끝
        ...
        const float next_T = T[p] * (1.0f - alpha);
        if (next_T <= TRANSMITTANCE_THRESHOLD)
        {                                              // this pixel is done: exclusive
            done_mask |= (1u << p);
            continue;
        }
        ...
    }
}
// end early if entire tile is done
if (__syncthreads_count(done_mask == ALL_DONE) >= BATCH_SIZE) break;
```

절약이 세 층으로 일어난다.

1. **픽셀 층** — 끝난 픽셀은 `done_mask` 비트로 걸러 알파 블렌딩 산술을 건너뛴다.
2. **스레드 층** — 스레드가 맡은 픽셀이 전부 끝나면(`done_mask == ALL_DONE`) 배치 내부 루프 자체를 안 돈다.
3. **타일 층** — CTA의 모든 스레드가 끝났으면 `__syncthreads_count`로 그것을 합의하고 **배치 루프를 break**한다. 이 시점부터는 뒤쪽 Gaussian을 shared memory로 **로드조차 하지 않는다**. 진짜 큰 절약은 여기서 나온다 — 산술뿐 아니라 global memory 트래픽이 사라진다.

주의: 3층의 break는 타일 전체 합의가 필요하다. 한 픽셀이라도 아직 투명하면(예: 하늘이 보이는 픽셀) 그 타일은 계속 Gaussian을 읽어야 한다. 즉 조기 종료의 이득은 **장면이 얼마나 불투명한가**에 크게 좌우된다. 실내처럼 앞면이 꽉 막힌 장면에서 이득이 크고, 배경이 뻥 뚫린 장면에서는 작다.

## 5. 종료 시점이 남기는 두 가지 산출물

조기 종료는 그냥 "빠지는" 게 아니라 forward 출력을 정의한다.

```cpp
// Here T is the transmittance AFTER the last gaussian in this pixel.
render_alphas[pix_id[p]] = 1.0f - T[p];
render_colors[...]       = backgrounds == nullptr ? pix_out[p][k]
                                                 : (pix_out[p][k] + T[p] * backgrounds[k]);
last_ids[pix_id[p]]      = static_cast<int32_t>(cur_idx[p]);
```

- **`alpha = 1 - T`** — `rasterization()`이 두 번째로 돌려주는 그 alpha 맵이다. 조기 종료로 나온 픽셀은 `T ≤ 1e-4`이므로 alpha가 거의 1(완전 불투명). 반대로 Gaussian이 없어 루프를 다 돈 하늘 픽셀은 `T ≈ 1`, alpha ≈ 0이 되고, 배경색은 남은 `T`만큼 섞인다.
- **`last_ids = cur_idx`** — 이 픽셀에 **마지막으로 기여한** Gaussian의 인덱스. `cur_idx`는 블렌딩에 성공한 순간에만 갱신되므로, 종료를 유발한 Gaussian은 여기에 안 잡힌다.

## 6. backward도 같이 짧아진다

`last_ids`가 backward의 루프 상한이 된다 (`RasterizeToPixels3DGSSerialBatchBwd.cu`).

```cpp
float T_final           = 1.0f - render_alphas[pix_id];
float T                 = T_final;
const int32_t bin_final = inside ? last_ids[pix_id] : 0;
const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
...
for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t)
{
    if (batch_end - t > bin_final) continue;   // 기여 안 한 Gaussian은 gradient 계산 생략
    ...
}
```

backward는 뒤→앞으로 걸으면서 `T *= 1/(1-alpha)`로 `T`를 되감는다. `bin_final` 뒤쪽 Gaussian은 forward에서 색에 기여하지 않았으니 gradient도 0이고, 계산을 건너뛴다. `warp_bin_final`은 warp 안 픽셀들의 최대값이라 warp divergence를 줄이려는 장치다.

여기서 4절의 `MIN_ONE_MINUS_ALPHA = 1e-6` 같은 하한과 `TRANSMITTANCE_THRESHOLD` 설계가 왜 얽혀 있는지 보인다. `T`를 너무 작게 내려가게 허용하면 backward에서 `1/(1-alpha)`로 되감을 때 float32 상대오차가 폭발한다. 즉 임계값은 **속도 장치이자 수치 안정성 장치**다.

## 7. 같은 규칙을 쓰는 다른 커널들

`TRANSMITTANCE_THRESHOLD` 비교는 3DGS dense forward만의 것이 아니다.

- `RasterizeToPixels2DGSSerialBatchFwd.cu` — 2DGS도 동일한 `next_T <= TRANSMITTANCE_THRESHOLD` 조건.
- `RasterizeToIndices3DGSSerialBatch.cu` / `...2DGS...` — "어떤 Gaussian이 어떤 픽셀에 기여했는지" 목록을 뽑는 커널. 여기서도 같은 조건으로 `break`하기 때문에, **인덱스 목록이 forward 색과 정확히 같은 집합**을 갖는다. 조기 종료 정의가 어긋나면 이 대응이 깨진다.
- `RasterizeContributingCommon.cuh`, `RasterizeContributingCommonSparse.cuh` — top-K 기여 Gaussian 추출.
- `RasterizeToPixelsFromWorld3DGS*` — world-space 래스터라이저는 임계값을 **픽셀별 배열**(`transmittance_threshold[p]`)로 받는다. parallel-batch 경로에서는 배치를 병렬로 부분 합성한 뒤 접기 때문에, 부분 결과의 임계값이 `TRANSMITTANCE_THRESHOLD / T_init`로 조정된다("priming-tightened bound"). 목표는 **순차 커널의 truncation 지점을 bit-for-bit 재현**하는 것 — 즉 조기 종료 지점은 최적화 여부와 무관하게 유지되어야 하는 계약이다.

## 8. 한 줄 정리 & 자주 틀리는 지점

> `T ← T·(1-α)`로 감소하는 누적 투과율이 `1e-4` 이하가 되는 순간, 그 Gaussian은 **빼고** 픽셀 루프를 끝낸다. 타일의 모든 픽셀이 끝나면 남은 Gaussian은 메모리에서 읽지도 않는다.

- ✗ "alpha가 작으면 조기 종료" → 그건 `ALPHA_THRESHOLD` 개별 skip이다.
- ✗ "임계값을 넘긴 Gaussian까지 포함해서 끝난다" → exclusive. 포함하지 않는다.
- ✗ "픽셀 하나가 끝나면 타일이 끝난다" → 타일 break는 **모든** 픽셀이 끝나야 한다.
- ✗ "근사라서 화질이 떨어진다" → `1e-4`는 8비트 색 한 단계보다 작아 사실상 무손실이고, 정렬 순서가 앞→뒤이므로 버리는 것은 항상 가장 안 보이는 것이다.
- ✓ 앞→뒤 정렬이 전제다. 3단계 `isect_tiles`가 (tile_id, depth) 키로 정렬해 두기 때문에 "뒤쪽은 안 보인다"는 판단이 성립한다.

## 참고 위치

- `/home/sungwoo/projects/swcho/gsplat/fm/training/.fm/assets/training_walkthrough.py` (L206–225, 4단계 설명)
- `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/include/Common.h` (L97–115, 임계값 상수와 설계 근거)
- `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSDevice.cuh` (L57–97, 공유 블렌딩 함수)
- `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu` (L222–300, done_mask / 타일 break / 출력)
- `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu` (L135–190, last_ids 기반 backward 단축)
- `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/csrc/RasterizeToIndices3DGSSerialBatch.cu` (L169, 동일 조건)
