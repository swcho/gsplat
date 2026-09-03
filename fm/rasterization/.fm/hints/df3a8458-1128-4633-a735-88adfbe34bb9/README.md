# 타일별 Gaussian 수 분포를 왜 보는가

**Q.** 타일별 Gaussian 수 분포를 시각화하는 것이 왜 의미가 있는가?

**A.** CUDA 블록 하나가 타일 하나를 맡으므로 이 분포가 곧 블록별 작업량 불균형이기 때문이다. 밀도가 높은 씬에서 래스터화 시간이 어디서 나오는지 보여준다.

---

## 1. "블록 하나 = 타일 하나"는 비유가 아니라 커널 런치 그대로다

`rasterize_to_pixels` 커널의 런치 설정을 보면 그리드가 문자 그대로 타일 격자다
(`gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu`).

```cpp
const uint32_t n_tiles = I * grid_h * grid_w;   // 이미지 수 × 타일 세로 × 타일 가로
const dim3 grid        = {n_tiles, 1, 1};
const dim3 threads     = dim3{CTA_SIZE, 1, 1};  // tile_size=16 → CTA_SIZE=256
```

커널 안에서 블록 인덱스를 다시 (이미지, 타일 y, 타일 x)로 푼다.

```cpp
const uint32_t linear_block_index = blockIdx.x + block_offset;
const int32_t  image_id           = linear_block_index / tiles_per_image;
const uint32_t tile_linear        = linear_block_index % tiles_per_image;
const uint32_t tile_x = tile_linear % grid_width;
const uint32_t tile_y = tile_linear / grid_width;
```

그리고 그 블록이 처리할 Gaussian 목록은 오직 `isect_offsets` 두 칸의 차이로 정해진다.

```cpp
const int32_t range_start = isect_offsets[tile_id];
const int32_t range_end   = (마지막 타일이면 n_isects : isect_offsets[tile_id + 1]);
const uint32_t num_batches = (range_end - range_start + BATCH_SIZE - 1) / BATCH_SIZE; // BATCH_SIZE = 256
```

즉 **블록의 일감 크기 = `range_end - range_start` = 그 타일의 Gaussian 수**다.
"타일별 Gaussian 수 히트맵"은 사실상 **CUDA 그리드를 그대로 그린 작업량 지도**다.
다른 시각화(σ 등고선, 투영 타원)는 "무엇이 그려지는가"를 보여주지만, 이 히트맵만이
"GPU가 어디에서 시간을 쓰는가"를 보여준다.

## 2. 블록 실행 시간의 대략적 모델

블록 하나의 구조는 이렇다.

```
for batch in range(0, count, 256):          # num_batches = ceil(count / 256)
    스레드 256개가 Gaussian 1개씩 shared memory로 적재 (id, xy, opacity, conic)
    __syncthreads()
    for t in range(batch_size):             # 최대 256회, 직렬
        σ, α 계산 → skip / 누적 / 종료 판정
    __syncthreads_count(done)               # 블록 전체 종료 검사
```

그래서

```
블록 실행 시간 ≈ ceil(타일의 Gaussian 수 / 256) × (배치당 순회 비용)
배치당 순회 비용 ≈ 글로벌 로드 1회 + __syncthreads 2회 + 최대 256번의 직렬 σ/α 평가
```

핵심은 **직렬 루프**라는 점이다. 배치 안 256개 Gaussian은 앞→뒤 순서를 지켜야 하므로
병렬로 처리할 수 없다. 타일 안의 256개 픽셀(스레드)이 "같은 Gaussian"을 동시에 평가하며
루프를 함께 도는 구조라서, 타일의 Gaussian 수가 2배가 되면 그 블록의 시간도 대략 2배가 된다.

숫자로 감을 잡으면: 어떤 타일이 40개면 배치 1개, 5,000개면 배치 20개.
같은 이미지 안에서 블록 간 20배 차이가 흔히 난다.

> 참고: `tile_size=16, CTA_SIZE=256`이면 `PIXELS_PER_THREAD = 16*16/256 = 1`,
> 즉 스레드 1개 = 픽셀 1개다. `tile_size=4`(고채널용 예외)에서는 `CTA_SIZE=16`이라
> 배치 크기도 16으로 줄어든다 — 아래 "타일 크기 조정"과 연결되는 지점.

## 3. 조기 종료 — 실제 순회 수는 Gaussian 수보다 적을 수 있다

히트맵은 **상한**에 가깝지, 정확한 작업량은 아니다. 포워드에는 두 층의 조기 종료가 있다.

**픽셀 단위** — 투과율 T가 임계값 아래로 떨어지면 그 픽셀은 끝난다.

```cpp
const float next_T = T[p] * (1.0f - alpha);
if (next_T <= TRANSMITTANCE_THRESHOLD) {   // 1e-4
    done_mask |= (1u << p);                // 이 Gaussian은 제외하고 종료
    continue;
}
```

**블록 단위** — 배치 경계에서 타일의 모든 픽셀이 끝났는지 세고, 그렇다면 루프를 깬다.

```cpp
if (__syncthreads_count(done_mask == ALL_DONE) >= BATCH_SIZE) {
    break;   // 남은 배치를 전부 건너뛴다
}
```

임계값들은 `gsplat/cuda/include/Common.h`에 있다: `ALPHA_THRESHOLD = 1/255`,
`MAX_ALPHA = 0.99`, `TRANSMITTANCE_THRESHOLD = 1e-4`
(주석에 따르면 `TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2` — 최대 불투명도 Gaussian
두 장이면 포화되도록 잡은 값).

그래서 히트맵을 읽을 때는 이렇게 보정해서 본다.

| 타일 종류 | Gaussian 수 | 실제 순회 |
|---|---|---|
| 불투명한 표면 정면 (벽, 테이블) | 많음 | **적음** — 앞쪽 몇십 개에서 T가 포화해 조기 종료 |
| 반투명·희박 (하늘, 식물 잎, 안개) | 많음 | **거의 전부** — T가 안 떨어져 끝까지 순회 |
| 배경/빈 타일 | 0~수십 | 배치 1개 이하 |

조기 종료는 "블록 전체가 포화"해야 발동한다는 점도 중요하다. 타일 256픽셀 중 하나라도
하늘이 보이는 틈이 있으면 그 블록은 목록 끝까지 간다. 실루엣 경계 타일이 이래서 비싸다.
또 종료 판정이 **배치(256개) 경계**에서만 일어나므로, 3개째 Gaussian에서 포화해도
그 배치의 256개는 다 훑고 나서야 빠져나온다.

## 4. 왜 "불균형"이 곧 시간인가 — tail latency

CUDA 스케줄러는 블록을 SM에 **동적으로** 배분한다. 프로그래머는 어떤 블록이 어느 SM에
갈지 정하지 않는다. SM 하나가 비면 대기열의 다음 블록이 들어간다. 이 방식은 블록들의
비용이 고만고만할 때 자연스럽게 부하를 고르게 만든다.

문제는 분포의 꼬리다. 예를 들어 1920×1080이면 타일이 120×68 ≈ 8,160개인데, GPU는
SM이 수십 개 × SM당 동시 상주 블록 몇 개 수준이라 수백 개 정도만 동시에 돈다.
그러면 이런 일이 벌어진다.

- 가벼운 블록 수천 개는 빠르게 소진된다.
- **가장 무거운 소수의 블록**이 마지막까지 남는다. 어떤 블록이 5,000개짜리인데
  하필 대기열 뒤쪽에서 뽑히면, 그 블록이 끝날 때까지 커널이 끝나지 않는다.
- 그 동안 다른 SM들은 줄 일감이 없어 놀고 있다 (tail 구간의 낮은 occupancy).

즉 **커널 시간 ≈ 평균이 아니라 "가장 무거운 타일들이 만드는 꼬리"에 지배된다.**
`grid.x`를 늘려 봐야(= 해상도를 키워도) 무거운 블록 하나의 직렬 루프는 짧아지지 않는다.
게다가 블록 안에서도 워프 다이버전스가 겹친다 — 같은 블록의 어떤 워프는 이미 포화(done)이고
어떤 워프는 아직 순회 중인데, 배치 경계의 `__syncthreads`가 이들을 계속 묶어 둔다.

이래서 "평균 타일에 몇 개"보다 **"상위 1% 타일에 몇 개"**가 훨씬 유용한 지표다.

## 5. 노트북에서 분포를 얻는 법 — `isect_offsets` 차분

`isect_offsets`는 타일별 **시작 인덱스**만 담은 `[C, tile_h, tile_w]` 텐서다
(`isect_offset_encode`가 정렬된 키에서 타일이 바뀌는 지점을 찾아 만든다).
타일 t의 Gaussian 목록은 `flatten_ids[offsets[t] : offsets[t+1]]`이고, 마지막 타일은
`n_isects`까지다. 그러니 **개수 = 인접 오프셋의 차분**이고, 끝에 `n_isects`를 붙여 주면
한 줄로 끝난다.

```python
th, tw = m_fused["isect_offsets"].shape[-2:]
flat = torch.cat([
    m_fused["isect_offsets"].flatten(),
    m_fused["isect_offsets"].new_tensor([m_fused["isect_ids"].numel()]),   # 마지막 경계 = n_isects
])
per_tile = (flat[1:] - flat[:-1]).reshape(C, th, tw)
```

노트북(`.fm/assets/rasterization_walkthrough.py` 8절 "타일 부하 시각화")은 이걸
세 장으로 나란히 그린다.

1. `r_fused[0]` — 렌더 이미지 (어느 영역인지 눈으로 대조하려고)
2. `imshow(per_tile[0], cmap="magma")` — 타일별 Gaussian 수 히트맵
3. `hist(tiles_per_gauss[radii>0], bins=30, log=True)` — Gaussian **하나가 덮는 타일 수** 히스토그램

2번과 3번은 같은 `n_isects`를 서로 다른 축으로 자른 것이다.
`n_isects = Σ_tile (타일별 개수) = Σ_gaussian (덮는 타일 수)`.
2번은 "어느 블록이 무거운가"(부하 지도), 3번은 "누가 그걸 만들었나"(범인 찾기)를 답한다.
히스토그램의 y축을 `log`로 두는 것부터가 이 분포가 heavy-tailed임을 전제한 것이다.

장난감 씬 절에서도 같은 차분을 4×3 격자에 대해 그대로 출력해서, 이 계산이
`isect_offsets`의 정의 그 자체임을 보여준다.

```python
flat = torch.cat([isect_offsets.flatten(), isect_offsets.new_tensor([isect_ids.numel()])])
print("타일별 Gaussian 수:\n", (flat[1:] - flat[:-1]).reshape(tile_h, tile_w).cpu().numpy())
```

빈 타일은 이전 오프셋을 그대로 물려받으므로 차분이 자연히 0이 된다.

## 6. 왜 heavy-tailed인가

실제 씬에서 히트맵은 결코 균일하지 않고, 히스토그램은 로그 스케일이 필요할 만큼 꼬리가 길다.
대부분의 타일은 수십 개인데, 소수의 타일은 수천 개다. 이유는 겹쳐서 작동한다.

- **투영 크기의 1/z² 스케일.** Gaussian이 덮는 화면 면적은 깊이의 제곱에 반비례한다.
  카메라 바로 앞의 Gaussian 하나가 화면 수백 타일을 덮는 일이 생긴다.
  덮는 타일 수가 그 Gaussian이 만드는 (Gaussian, 타일) 쌍의 수이므로, 이 한 개가
  수백 개의 일감을 흩뿌린다. 3번 히스토그램의 오른쪽 꼬리가 정확히 이들이다.
- **큰 Gaussian들이 같은 곳에 몰린다.** 카메라 근처의 큰 splat들은 서로 겹치는 타일이
  거의 같다. 개별 기여가 곱이 아니라 합으로 쌓이므로, 그 몇 개 타일의 카운트가 폭증한다.
- **기하학적 밀도 자체가 불균일하다.** 재구성이 잘 된 디테일 영역(garden의 식탁, 화분 잎)에
  밀도화(densification)가 Gaussian을 집중시킨다. 학습이 진행될수록 이 편중은 심해진다.
- **하늘/배경은 거의 비어 있다.** 화면 면적의 상당 부분이 카운트 0~수십이라 평균을 끌어내리고,
  결과적으로 평균과 최댓값의 격차를 더 벌린다.
- **정렬 자체는 균등하다.** radix sort는 `n_isects` 전체에 대해 도는 전역 연산이라
  불균형과 무관하다. 불균형이 아픈 곳은 블렌딩 커널이다. 다만 `n_isects` 자체가 커지면
  정렬 시간과 메모리(`isect_ids` 64비트 + `flatten_ids` 32비트 × n_isects)도 같이 커진다.

## 7. 히트맵이 알려 주는 최적화 방향

히트맵이 "여기가 무겁다"라고 말하면, 대응 수단은 대체로 **`n_isects`를 줄이거나
꼬리를 깎는 것**이다. 노트북에 나오는 것부터 순서대로.

**AccuTile (`isect_tiles`에 `conics`/`opacities` 전달).**
기본 AABB 모드는 `radii` 사각형이 겹치는 타일을 전부 잡는다. AccuTile은 사각형 대신
**알파가 1/255 이상인 실제 타원**과 겹치는 타일만 고른다(SpeedySplat, `IntersectTile.cu`의
`accutile_process_tiles`). 결과 이미지는 같고(잘린 타일의 기여는 어차피 임계값 아래)
정렬·블렌딩 비용만 줄어든다. 길쭉하게 기울어진 Gaussian일수록 사각형과 타원의 넓이 차가
커서 이득이 크다. 노트북은 같은 씬에서 `tiles_per_gauss` AABB vs AccuTile을 나란히 찍어
차이를 보여준다.

**불투명도 인지 반경 (투영 커널에 `opacities` 전달).**
반경이 고정 3.33σ 대신 `min(3.33, sqrt(2·ln(α / (1/255)))) · σ`로 줄어든다.
알파가 1/255 아래로 떨어지는 지점 밖은 어차피 안 그리므로, 투명한 Gaussian은 애초에
더 작은 사각형만 타일링한다. 학습 중 반투명 Gaussian이 많은 구간에서 특히 효과적이다.

**Gaussian pruning / opacity reset.** 기여가 거의 없는 Gaussian을 제거하면 무거운
타일에서 직접 개수가 빠진다. 히트맵으로 "어느 영역이 과밀한가"를 보고 pruning 강도를
정할 수 있다.

**타일 크기 조정.** 타일을 작게 하면 타일당 Gaussian 수의 분산이 줄고 조기 종료가
더 국소적으로 먹지만, 같은 Gaussian이 더 많은 타일에 중복 등록되어 `n_isects`와
정렬 비용이 커진다. 또 배치 크기가 CTA 크기와 묶여 있어(`BATCH_SIZE = CTA_SIZE`)
shared memory 재사용 효율도 바뀐다. gsplat이 지원하는 값은 16(주 경로)과
4(고채널 예외)뿐이니, 실제로는 튜닝 노브라기보다 트레이드오프를 이해하는 용도다.

**밀도화(densification) 억제.** DefaultStrategy의 split/duplicate 임계값을 올리거나
최대 Gaussian 수에 상한을 두면 꼬리가 자란다. 특히 **화면 크기 기준 split**(큰 화면
Gaussian을 쪼개기)은 꼬리를 직접 겨냥한 장치다.

**대안적 정리.** 카메라 근처의 거대 splat 한두 개가 원인이면, scale 상한이나
near-plane 조정 같은 국소 처방이 전역 pruning보다 잘 듣는다. 히트맵 → 히스토그램 →
`tiles_per_gauss` argmax로 내려가면 그 개체를 특정할 수 있다.

## 8. 실무적으로: 히트맵은 렌더 시간과 함께 본다

프로파일링에서 이 그림이 유용한 이유는 **인과의 방향을 보여주기 때문**이다.
노트북의 `bench()`가 주는 것은 "forward N ms"라는 스칼라 하나뿐인데, 이 숫자만으로는
느린 이유가 Gaussian이 많아서인지, 해상도 때문인지, 특정 뷰의 몇 타일 때문인지 알 수 없다.
히트맵을 나란히 두면 다음이 구분된다.

- **뷰 간 시간 차** — 카메라 0은 8ms, 카메라 1은 25ms일 때, 히트맵을 보면 카메라 1이
  과밀 영역을 정면으로 보고 있다는 게 즉시 보인다. 씬의 문제가 아니라 뷰의 문제다.
- **최적화가 실제로 먹었는지** — AccuTile을 켠 전후로 히트맵을 찍으면, 밝은 지점이
  얼마나 어두워졌는지 = 줄어든 일감이 보인다. 시간이 안 줄었다면 병목이 블렌딩이 아니라
  정렬이나 투영에 있다는 뜻이다.
- **학습 중 성능 회귀** — 스텝이 진행되며 렌더 시간이 서서히 늘어날 때, 히트맵의
  최댓값과 `n_isects`를 같이 로깅해 두면 밀도화가 원인인지 바로 확인된다.
- **불균형인지 총량인지** — 총량이 문제면 히트맵이 전반적으로 밝아진다(pruning이 답).
  꼬리가 문제면 몇 점만 극단적으로 밝다(그 개체를 잡는 게 답). 처방이 다르다.

같이 보면 좋은 값들: `n_isects`(= `flatten_ids.numel()`), `per_tile.max()`,
`per_tile.float().mean()`, 그리고 `per_tile.max() / per_tile.float().mean()` 같은
불균형 비율. 마지막 값이 곧 "커널 시간이 평균 대비 얼마나 꼬리에 끌려가는가"의 대리 지표다.
Nsight Compute에서 보는 SM 활용률의 tail 구간 하락과 이 값이 같은 현상의 양면이다.

## 9. Backward는 조기 종료가 없다 — 분포의 영향이 더 크다

`RasterizeToPixels3DGSSerialBatchBwd.cu`의 구조가 결정적이다. 백워드는 포워드가 저장한
최종 T와 `last_ids`(픽셀별 마지막 기여 Gaussian)에서 출발해 **뒤→앞으로** 순회하며
`T /= (1-α)`로 투과율을 복원한다. 그런데 배치 루프에 `break`가 없다.

```cpp
const int32_t bin_final     = inside ? last_ids[pix_id] : 0;
const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
for (uint32_t b = 0; b < num_batches; ++b) {        // ← 끝까지 전부 돈다
    ...
    for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
        bool valid = inside;
        if (batch_end - t > bin_final) { valid = 0; }   // 개별 픽셀만 건너뜀
        ...
    }
}
```

포워드의 `if (__syncthreads_count(...) >= BATCH_SIZE) break;`에 대응하는 것이 없다.
`warp_bin_final`은 워프 안 최댓값이라 **워프에서 가장 깊게 간 픽셀**을 기준으로 시작점을
정하고, `bin_final` 뒤쪽 Gaussian은 `valid=0`으로 표시만 될 뿐 루프는 계속 돈다.
결과적으로 백워드 블록은 `range_start`까지, 즉 **타일 목록의 앞쪽 끝까지 반드시 도달한다.**

여기에 백워드 고유의 추가 비용이 붙는다.

- 워프 단위 리듀스 후 `atomicAdd`로 Gaussian별 grad를 모은다. 같은 Gaussian이 여러 픽셀에서
  갱신되므로, 무거운 타일일수록 **atomic 경합**도 같이 늘어난다.
- shared memory에 색(`rgbs_batch[block_size * CDIM]`)까지 올려서 배치당 shared 사용량이
  포워드보다 크고, 그만큼 SM당 상주 블록 수(occupancy)가 낮아진다 — 무거운 블록이 남았을 때
  겹칠 다른 블록도 적다는 뜻이다.
- `absgrad=True`면 ∂L/∂means2d의 절댓값 합도 따로 누적한다(AbsGS 밀도화 기준).

그래서 **히트맵의 카운트는 포워드에서는 상한이지만 백워드에서는 실제 비용에 훨씬 가깝다.**
학습 루프는 forward + backward를 매 스텝 도는데, backward가 보통 더 오래 걸리고
불균형에 더 민감하다. "추론은 견딜 만한데 학습이 유독 느리다"는 상황이 여기서 나온다.
그러니 이 히트맵은 렌더링 튜닝용 도구이기 이전에 **학습 속도 진단 도구**다.

---

## 한 줄 요약

`per_tile = diff(isect_offsets)`는 단순한 카운트가 아니라 **CUDA 그리드의 작업량 지도**다.
블록 시간이 `ceil(count/256) × 배치 비용`에 비례하고, 스케줄러가 블록을 동적으로 배분하는
이상 커널 시간은 꼬리가 결정한다. 포워드는 조기 종료로 꼬리가 조금 깎이지만 백워드는
그마저 없다. 그래서 이 한 장이 "래스터화 시간이 어디서 나오는가"에 대한 가장 직접적인 답이다.

## 관련 파일

- `fm/rasterization/.fm/assets/rasterization_walkthrough.py` — 8절 "타일 부하 시각화"
- `gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu` — 블록↔타일 매핑, 배치 루프, 조기 종료 `break`
- `gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu` — `bin_final`/`warp_bin_final`, `break` 없는 배치 루프
- `gsplat/cuda/csrc/IntersectTile.cu` — `intersect_tile_kernel`(2패스), `accutile_process_tiles`, `intersect_offset_kernel`
- `gsplat/cuda/include/Common.h` — `ALPHA_THRESHOLD`, `MAX_ALPHA`, `TRANSMITTANCE_THRESHOLD`, `GAUSSIAN_EXTEND`
- `gsplat/cuda/_torch_impl.py` — `_isect_tiles`, `_isect_offset_encode`, `_rasterize_to_pixels` (순수 PyTorch 참조)
