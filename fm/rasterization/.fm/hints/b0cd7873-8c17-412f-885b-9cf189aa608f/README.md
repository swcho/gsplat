# 래스터화 커널의 shared memory 사용법

**Q.** 래스터화 커널이 shared memory를 쓰는 방식은?

**A.** 타일의 Gaussian 목록을 256개씩 배치로 나눠, 각 스레드가 Gaussian 1개씩(id, xy, opacity, conic) shared memory에 적재하고 `__syncthreads()`한다. 그 뒤 모든 스레드가 shared memory의 Gaussian들을 앞→뒤로 순회한다.

---

## 0. 한눈에 보는 구조

`.fm/assets/rasterization_walkthrough.py` 6절이 CUDA 커널 구조를 이렇게 요약한다.

```
블록(blockIdx)  = 타일 하나          →  range = [isect_offsets[tile], isect_offsets[tile+1])
스레드(tid)     = 타일 안 픽셀 하나   →  T=1, pix_out=0
for batch in range(0, len(range), 256):
    각 스레드가 Gaussian 1개씩 shared memory에 적재 (id, xy, opacity, conic)   __syncthreads()
    for g in batch(앞→뒤):  σ, α 계산 → 건너뛰기/누적/종료 판정
    __syncthreads_count(done) == 256 이면 타일 전체 조기 종료
pix_out += background * T ;  render_alpha = 1 - T ;  last_ids = 마지막 기여 Gaussian (backward용)
```

이 문서는 이 3줄짜리 배치 루프가 **왜** 그렇게 생겼는지를, 실제 소스
`gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu`
(`rasterize_to_pixels_3dgs_fwd_kernel<CDIM, TILE_SIZE, CTA_SIZE>`)를 따라가며 설명한다.

---

## 1. 먼저 CUDA 메모리 계층 — 왜 이런 짓을 하나

GPU에서 "연산"은 싸고 "메모리 접근"은 비싸다. 계층은 대략 이렇다(수치는 아키텍처마다 다르므로 자릿수 감각용).

| 계층 | 범위 | 지연시간(대략) | 대역폭(대략) | 크기 |
|---|---|---|---|---|
| 레지스터 | 스레드 1개 전용 | ~0 사이클(파이프라인 흡수) | 사실상 무제한 | 스레드당 수십~255개 |
| **shared memory** | **블록(CTA) 전체 공유** | ~20–30 사이클 | SM당 수 TB/s급 (L1과 동급 물리 SRAM) | SM당 수십~200 KB (블록당 배분) |
| L2 캐시 | GPU 전체 공유 | ~200 사이클 | 수 TB/s | 수 MB~수십 MB |
| global memory (DRAM) | GPU 전체 | ~300–800 사이클 | 수백 GB/s ~ 수 TB/s | 수십 GB |

핵심은 shared memory가 **L1 캐시와 같은 물리 SRAM을 프로그래머가 직접 관리하는 형태**라는 점이다.
캐시는 하드웨어가 알아서 채워 주지만 "언제 evict될지" 보장이 없다. shared memory는
**내가 넣은 것이 내가 지울 때까지 그대로 있다**는 보장을 준다. 대신 블록이 끝나면 사라지고,
동기화(`__syncthreads()`)를 직접 해 줘야 한다.

### 그래서 래스터화에 왜 필요한가 — 256배 재사용

타일 기반 래스터화의 핵심 관찰:

> **같은 타일 안의 256개 픽셀은 전부 같은 Gaussian 목록을 읽는다.**

`isect_offsets[tile] .. isect_offsets[tile+1]` 구간이 그 타일의 목록이고, 타일 안 어느 픽셀이든
이 목록 전체를 앞→뒤로 순회해야 한다. 픽셀마다 다른 건 자기 좌표 `(px, py)`와 누적기 `T`, `pix_out`뿐이다.

shared memory 없이 각 스레드가 순진하게 global에서 읽으면, Gaussian 하나당:

- `flatten_ids[idx]` : int32 = 4 B
- `means2d[g]` : vec2 = 8 B
- `opacities[g]` : float = 4 B
- `conics[g]` : vec3 = 12 B

총 **28 B를 256개 스레드가 각각** 읽는다 → Gaussian 1개당 256 × 28 = **7168 B의 global 트래픽**.

shared memory를 쓰면 **256개 스레드가 나눠서 딱 한 번씩** 읽어 온다 → Gaussian 1개당 28 B.
곧 **global 읽기 요청이 원리상 256배 줄어든다.**

> 현실적으로는 L1/L2 캐시가 중복 읽기의 상당 부분을 흡수하고, 같은 워프 안에서 같은 주소를
> 읽으면 하드웨어가 브로드캐스트해 준다. 그래서 실측 이득이 정확히 256배는 아니다.
> 하지만 (a) 캐시 히트도 shared보다 느리고 (b) 히트가 보장되지 않으며 (c) 서로 다른
> 타일 블록들이 SM을 공유하며 캐시를 서로 밀어내기 때문에, "명시적으로 stage 해 두는" 쪽이
> 훨씬 예측 가능하고 빠르다. 3DGS의 렌더 커널은 이 안쪽 루프가 전체 시간의 대부분을 먹기 때문에
> 여기서의 상수 배 이득이 곧 전체 성능이다.

---

## 2. 실제 소스: shared 배열 선언과 크기 계산

`RasterizeToPixels3DGSSerialBatchFwd.cu`:

```cuda
constexpr uint32_t BATCH_SIZE = CTA_SIZE;
...
const uint32_t num_batches = (range_end - range_start + BATCH_SIZE - 1) / BATCH_SIZE;

extern __shared__ int s[];
int32_t *id_batch      = (int32_t *)s;                                            // [BATCH_SIZE]
vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[BATCH_SIZE]);         // [BATCH_SIZE]
vec3 *conic_batch      = reinterpret_cast<vec3 *>(&xy_opacity_batch[BATCH_SIZE]); // [BATCH_SIZE]
```

### `extern __shared__`가 뭔가

`__shared__ int s[256];`처럼 크기를 컴파일 타임에 박으면 **정적(static) shared memory**다.
반면 `extern __shared__ int s[];`는 **동적(dynamic) shared memory** 선언으로, 실제 바이트 수를
커널 런치 시 세 번째 `<<<grid, block, shmem_size, stream>>>` 인자로 준다.

여기서 동적을 쓰는 이유: `BATCH_SIZE`가 템플릿 인자 `CTA_SIZE`에 따라 달라지고(256 또는 16),
호스트 쪽에서 크기를 계산해 `cudaFuncSetAttribute`로 상한을 올려 줘야 하기 때문이다.
동적 shared memory는 블록당 **하나의 연속된 블록**만 주어지므로, 세 배열을
**같은 버퍼 안에서 포인터 산술로 잘라 쓰는** 위 패턴이 관용구다.

메모리 레이아웃(BATCH_SIZE = 256 기준):

```
s ──────────────────────────────────────────────────────────────────────┐
│ id_batch[0..255]        int32   × 256 = 1024 B                        │
│ xy_opacity_batch[0..255] vec3   × 256 = 3072 B   (x, y, opacity)      │
│ conic_batch[0..255]      vec3   × 256 = 3072 B   (a, b, c)            │
└──────────────────────────────────────────── 합계 7168 B = 7 KB ───────┘
```

호스트 쪽 계산이 정확히 이 식이다.

```cpp
const dim3 threads       = dim3{CTA_SIZE, 1, 1};
const int64_t shmem_size = CTA_SIZE * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3));
```

`vec3 = glm::vec<3, float>` (Common.h) → **12 B**(16이 아니다. `float4` 정렬 패딩이 없다).
따라서 스레드당 4 + 12 + 12 = **28 B**, 블록당 256 × 28 = **7168 B ≈ 7 KB**.

- `tile_size == 16` → `launch_variant<16, 256>()` → 7168 B
- `tile_size == 4` → `launch_variant<4, 16>()` → 16 × 28 = **448 B**

7 KB는 SM당 shared memory 예산(아키텍처에 따라 64–228 KB)에 비해 작아서, 보통은
**shared memory가 occupancy의 병목이 되지 않는다**. 실제 이 커널의 occupancy를 결정하는 건
`pix_out[PIXELS_PER_THREAD][CDIM]` 누적기가 잡아먹는 **레지스터** 쪽이다 —
소스 주석이 CTA=64(PPT=4)를 쓰면 레지스터가 `4*CDIM`으로 늘어 high channel count에서
local memory로 spill하며 ~10배 느려진다고 명시한다. 그래서 tile 16에서는 CTA=256, PPT=1
(**스레드 1개 = 픽셀 1개**)을 고른다.

### 왜 `xy`와 `opacity`를 한 `vec3`로 묶었나

`means2d[g]`(2 float)와 `opacities[g]`(1 float)는 원본 global 텐서에서 서로 다른 배열이지만,
shared에 넣을 때는 `{xy.x, xy.y, opac}` 하나의 `vec3`로 **packing**한다.
소비 측에서 `const vec3 xy_opac = xy_opacity_batch[t];` 한 번의 12 B 읽기로 세 값을 다 얻는다.
접근 횟수를 줄이고 뱅크 사용을 정돈하는 흔한 최적화다.

### 왜 `colors`는 shared에 안 넣나

안쪽 루프에서 색은 여전히 global에서 직접 읽는다.

```cuda
const int32_t g    = id_batch[t];
const float *c_ptr = colors + g * CDIM;
for(uint32_t k = 0; k < CDIM; ++k) { pix_out[p][k] += c_ptr[k] * vis; }
```

이유:

1. **크기.** 색은 Gaussian당 `CDIM` float다. CDIM=3이어도 256 × 12 B = 3 KB가 더 붙고,
   gsplat이 지원하는 CDIM=128 같은 경우엔 256 × 512 B = **128 KB**로 shared 예산을 통째로 날린다.
2. **필요할 때만 읽는다.** 색은 `gw.valid`이고 포화도 안 된 Gaussian에 대해서만 읽힌다.
   σ<0이거나 α<1/255라 건너뛰는 Gaussian이 많으므로, 무조건 stage 하면 오히려 낭비다.
3. **캐시가 잘 먹힌다.** 같은 `t`에 대해 블록의 모든 활성 스레드가 **같은 주소** `colors + g*CDIM`을
   읽는다 → 워프 내 브로드캐스트 + L1/L2 히트. shared로 옮겨도 이득이 적다.

즉 shared에 올리는 건 "**전 픽셀이 반드시, 매번, 똑같이 읽는 작은 메타데이터**"뿐이다.
이것이 stage 대상을 고르는 일반 원칙이다.

---

## 3. 협력 로딩(cooperative / coalesced loading) 패턴

배치 루프의 적재부:

```cuda
#pragma unroll 1
for(uint32_t b = 0; b < num_batches; ++b)
{
    // each thread fetch 1 gaussian from front to back
    const uint32_t batch_start = range_start + BATCH_SIZE * b;
    const uint32_t idx         = batch_start + tid;   // index of gaussian to load
    if(idx < range_end)
    {
        const int32_t g       = flatten_ids[idx];
        id_batch[tid]         = g;
        const vec2 xy         = means2d[g];
        const float opac      = opacities[g];
        xy_opacity_batch[tid] = {xy.x, xy.y, opac};
        conic_batch[tid]      = conics[g];
    }
    ...
```

여기서 `tid`는 커널 앞부분에서 잡아 둔 `const uint32_t tid = threadIdx.x;` 다.
(질문/답변에서 말하는 `tr`은 gsplat 이전 버전 및 원본 3DGS 코드에서 쓰던 이름 —
"thread rank" — 으로, 이 리팩터링된 소스에서는 `tid`로 바뀌었다. 역할은 동일하게
**블록 안 스레드 번호 = shared 배열의 슬롯 번호**다.)

패턴의 요점:

- **슬롯 = 스레드 번호.** 스레드 `tid`는 항상 슬롯 `tid`에만 쓴다. 그래서 쓰기 충돌(race)이 없고,
  동기화가 배리어 하나로 끝난다. `id_batch[tid] = ...` 처럼 인덱스가 `tid` 그 자체라는 게 핵심.
- **읽는 위치도 `tid` 오프셋.** `idx = batch_start + tid` → 인접 스레드가 `flatten_ids`의
  **인접 원소**를 읽는다. 워프 32개 스레드가 연속된 32 × 4 B = 128 B를 요청하므로
  하드웨어가 이를 **하나의 coalesced 트랜잭션**으로 묶는다. 이게 global 읽기에서 가장 중요한 조건이다.
  (`means2d[g]` 등은 `g`가 정렬 키에 따라 흩어져 있어 완전히 coalesced 되진 않지만,
  적어도 "한 번만" 읽는다.)
- **256번 재사용.** 이렇게 각 스레드가 **1개씩만** 가져오면 블록 전체로는 한 배치 = 256개
  Gaussian이 shared에 올라오고, 이후 **256개 스레드 전부가 그 256개를 각자 다 읽는다.**
  적재는 O(256), 소비는 O(256 × 256). 여기가 이득이 발생하는 지점이다.

읽기 쪽(안쪽 루프)에서는 **모든 스레드가 같은 인덱스 `t`를 읽는다.**

```cuda
const vec3 conic   = conic_batch[t];
const vec3 xy_opac = xy_opacity_batch[t];
```

shared memory에서 같은 주소를 여러 스레드가 읽는 건 **뱅크 충돌이 아니라 브로드캐스트**로
처리된다(하드웨어가 지원하는 최적 경로). 그래서 이 접근 패턴은 shared 관점에서도 이상적이다.

---

## 4. `__syncthreads()`의 두 역할과 배리어 위치

```cuda
        // wait for other threads to collect the gaussians in batch. A CTA
        // with <= 32 threads is a single warp, so a warp-level sync suffices.
        if constexpr(CTA_SIZE <= 32) { __syncwarp(); }
        else                         { __syncthreads(); }

        // process gaussians in the current batch
        const uint32_t batch_size = min(BATCH_SIZE, (uint32_t)range_end - batch_start);
        for(uint32_t t = 0; (t < batch_size) && (done_mask != ALL_DONE); ++t)
        { ... }

        // resync all threads before beginning next batch
        // end early if entire tile is done
        if(__syncthreads_count(done_mask == ALL_DONE) >= BATCH_SIZE) { break; }
    } // for b
```

배리어는 **루프 몸통 안에 두 개**가 있고, 각각 다른 위험(hazard)을 막는다.

**① 적재 완료 보장 (write → read, WAR의 반대인 RAW)**
`__syncthreads()` 앞에서는 스레드 t0이 슬롯 0만 채웠다. 배리어 없이 t0이 `conic_batch[7]`을
읽으면 t7이 아직 안 썼을 수도 있다(워프가 다르면 실행 순서가 자유). 배리어가
"**블록의 모든 쓰기가 끝났고, 그 결과가 다른 스레드에게 보인다**"를 보장한다.
`__syncthreads()`는 실행 배리어인 동시에 **메모리 배리어**라는 점이 중요하다.

**② 다음 배치 덮어쓰기 전 순회 완료 보장 (read → write, WAR)**
다음 반복(`b+1`)에서 같은 shared 버퍼를 **재사용**해 덮어쓴다. 느린 스레드가 아직
배치 `b`의 `t=200`을 읽고 있는데 빠른 스레드가 벌써 `id_batch[tid] = ...`로 배치 `b+1`을
써 버리면 데이터가 깨진다. 이걸 막는 배리어가 루프 **끝**의 것이다.

여기서 소스의 묘미: 그 두 번째 배리어를 별도 `__syncthreads()`로 두지 않고
**`__syncthreads_count(...)`가 겸한다.** `__syncthreads_count(pred)`는

- 블록의 모든 스레드를 만나게 하고(= 완전한 `__syncthreads()` 의미론),
- 그중 `pred != 0`인 스레드 수를 세어 **모든 스레드에게 같은 값을 돌려준다.**

즉 "다음 배치 전 재동기화"와 "타일 전체 조기 종료 판정"을 **배리어 하나로 합쳤다**.
주석 `// resync all threads before beginning next batch` / `// end early if entire tile is done`
두 줄이 바로 그 두 역할을 말한다.

> `CTA_SIZE <= 32`(tile 4, CTA 16)일 때는 블록이 단일 워프라 워프가 lockstep으로 움직이므로
> ① 자리에는 더 가벼운 `__syncwarp()`면 충분하다. 다만 ② 자리의 `__syncthreads_count`는
> 값을 세어야 하므로 그대로 쓴다.

---

## 5. 배치 크기가 왜 하필 블록 스레드 수(256)인가

```cuda
constexpr uint32_t BATCH_SIZE = CTA_SIZE;
```

`BATCH_SIZE`를 `CTA_SIZE`와 **같게** 두는 이유는 "각 스레드가 정확히 1개씩 적재"라는
가장 단순하고 균형 잡힌 협력 로딩이 나오기 때문이다.

- `BATCH_SIZE < CTA_SIZE`라면 일부 스레드는 적재에서 놀고(idle lane), shared 재사용 배수가 떨어지며,
  배치 수가 늘어 배리어 횟수가 증가한다. 배리어는 공짜가 아니다.
- `BATCH_SIZE > CTA_SIZE`라면 적재부에 또 하나의 내부 루프(`for i in 0..K: slot = tid + i*CTA`)가 필요해
  코드가 복잡해지고, shared 사용량이 늘어 occupancy를 압박한다.
- `BATCH_SIZE == CTA_SIZE`면 인덱싱이 `id_batch[tid] = ...` 한 줄로 끝나고, 뒤에 나오는
  `__syncthreads_count(...) >= BATCH_SIZE` 조기 종료 조건도 "모든 스레드가 done"과 자연스럽게 일치한다.

그리고 `CTA_SIZE = 256`은 **타일이 16×16 = 256 픽셀**이라서 나온 수다.

```cuda
constexpr uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE;
// (TILE, CTA) = (16,64)->4, (8,32)->2, (4,16)->1
```

tile 16 / CTA 256이면 `PIXELS_PER_THREAD == 1` → **1 스레드 = 1 픽셀 = 1 Gaussian 적재 슬롯**.
"타일 픽셀 수 = 블록 스레드 수 = 배치 크기"라는 세 수가 전부 256으로 맞아떨어진다.

> 참고: `PIXELS_PER_THREAD > 1` 구성(예: CTA=16, tile 4에서는 PPT=1이지만 (16,64)면 PPT=4)에서도
> 배치 크기는 여전히 `CTA_SIZE`다. 즉 "픽셀 수"가 아니라 **"스레드 수"**가 배치 크기를 정한다.
> 협력 로딩의 단위는 스레드지 픽셀이 아니기 때문이다.

---

## 6. 마지막 배치가 256개 미만일 때

타일의 Gaussian 개수 `range_end - range_start`가 256의 배수라는 보장은 전혀 없다.
소스는 두 곳에서 이를 처리한다.

**적재 쪽 — 범위 밖 스레드는 아무것도 쓰지 않는다.**

```cuda
const uint32_t idx = batch_start + tid;
if(idx < range_end) { ...적재... }
```

`if` 밖으로 나가는 게 아니라 **적재만 건너뛴다**는 점을 보라. `return`이나 `continue`가 아니다.
이 스레드도 바로 뒤의 `__syncthreads()`에 반드시 도착해야 하기 때문이다(§7).
해당 슬롯의 shared 값은 **이전 배치의 쓰레기 값**으로 남지만, 소비 쪽에서 절대 읽지 않으므로 문제없다.

**소비 쪽 — 유효 개수만큼만 순회한다.**

```cuda
const uint32_t batch_size = min(BATCH_SIZE, (uint32_t)range_end - batch_start);
for(uint32_t t = 0; (t < batch_size) && (done_mask != ALL_DONE); ++t)
```

`batch_size`(소문자, 런타임 값)가 `BATCH_SIZE`(대문자, 컴파일 타임 상수 256)와 다를 수 있다.
마지막 배치에서 `range_end - batch_start`가 예를 들어 37이면 `t`는 0..36만 돈다 →
쓰레기 슬롯 37..255는 건드리지 않는다. **적재의 `if`와 소비의 `min`이 정확히 짝을 이룬다.**

배치 개수 자체는 올림 나눗셈:

```cuda
const uint32_t num_batches = (range_end - range_start + BATCH_SIZE - 1) / BATCH_SIZE;
```

---

## 7. 이미지 밖 픽셀(`inside == false`)도 적재에 참여해야 하는 이유

이게 초심자가 가장 많이 틀리는 부분이다. **"내 픽셀이 이미지 밖이면 그냥 `return` 하면 되지 않나?"**
→ **안 된다.** `__syncthreads()`는 **블록의 모든 스레드가 도달해야** 풀리는 배리어다.
일부 스레드가 먼저 `return` 해 버리면 남은 스레드들은 영원히 기다리거나(deadlock),
정의되지 않은 동작에 빠진다. CUDA는 배리어가 **divergent 하지 않을 것**을 요구한다.

소스가 이 문제를 어떻게 다루는지, 주석까지 그대로 옮기면:

```cuda
// Evaluate other early exist criteria. We can't directly OOB return
// because __syncthreads_count evaluates predicates for all threads
// in the block and will block until all threads have evaluated the
// predicate.
uint32_t done_mask = (out_x >= image_width) ? ALL_DONE : 0;
#pragma unroll
for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
{
    if(out_y[p] >= image_height) { done_mask |= (1u << p); }
}
```

즉 **경계 밖 픽셀은 "return"이 아니라 "처음부터 done 상태"로 표시**한다.
(`ALL_DONE = (1u << PIXELS_PER_THREAD) - 1u`, PPT=1이면 그냥 `1`.)
그 결과 이 스레드는:

- ✅ 배치 루프에 **계속 참여**하고,
- ✅ `if(idx < range_end)` 안의 **협력 로딩을 정상 수행**하며,
- ✅ 두 배리어에 **모두 도달**하고,
- ❌ 안쪽 순회 루프는 `done_mask != ALL_DONE` 조건에서 즉시 빠져나가 **블렌딩 연산은 안 한다**,
- ❌ 마지막에 `if(out_x < image_width)` / `if(out_y[p] < image_height)` 가드로 **쓰기도 안 한다.**

여기엔 성능상 보너스도 있다. 이미지 가장자리에서 타일이 잘려 나가 스레드의 절반이 OOB여도,
그 절반은 **놀지 않고 적재 일을 계속 한다.** 협력 로딩의 짝수 분할이 유지되므로
경계 타일에서도 적재 성능이 떨어지지 않는다.

같은 논리가 **알파 포화로 일찍 끝난(done) 스레드**에도 적용된다. `T <= 1e-4`가 되어
`done_mask |= (1u << p)`가 켜진 스레드도 여전히 배치 루프를 돌며 적재와 배리어에 참여한다.
"내 픽셀은 끝났으니 나 갈게"가 불가능하다. **타일 전체가 끝나야 같이 나간다.**

> `masks[tile_id] == false`로 통째로 건너뛰는 경우만 예외적으로 `return` 한다.
> 이건 **블록의 모든 스레드가 동일하게** 참으로 평가하는 조건(타일 단위 마스크)이라
> divergence가 없고, 배치 루프에 진입하기 **전**이라 배리어와도 무관하다.

---

## 8. 조기 종료 `__syncthreads_count`와의 관계

```cuda
if(__syncthreads_count(done_mask == ALL_DONE) >= BATCH_SIZE) { break; }
```

`BATCH_SIZE == CTA_SIZE == blockDim.x`이므로 이 조건은 사실상
**"블록의 스레드가 하나도 빠짐없이 done"** 을 뜻한다. 그때만 배치 루프를 벗어난다.

왜 이런 게 필요한가: 앞쪽 Gaussian 몇 개로 이미 불투명해진 픽셀은 뒤쪽 Gaussian을
읽어 봐야 기여가 0이다. 목록이 수천 개인 타일에서 이걸 다 도는 건 순수 낭비다.
하지만 **개별 스레드가 혼자 탈출할 수는 없으므로**(§7), "전원 done"을 매 배치마다
집계해서 **블록 단위로 함께 탈출**한다. 이것이 배리어를 어차피 쳐야 하는 자리에
집계를 얹은 이유이기도 하다 — 조기 종료 판정이 **공짜로 딸려 온다.**

점검 주기가 "매 배치(=256 Gaussian)마다"인 것도 설계 선택이다. 매 Gaussian마다 확인하면
배리어 비용이 순회 비용을 압도한다. 원 3DGS 논문이 "at regular intervals, threads in a tile
are queried"라고 쓴 그 "regular interval"이 여기서는 **배치 경계**다.

전체 흐름을 시간축으로 그리면:

```
        ┌──────────────── 배치 b ────────────────┐┌─────────── 배치 b+1 ───────────┐
t0  │ load slot0 │ B │ read t=0..255 (또는 done) │ C │ load slot0 │ B │ read ... │ C
t1  │ load slot1 │ B │ read t=0..255            │ C │ load slot1 │ B │ read ... │ C
... │            │   │                          │   │            │   │          │
t255│ load s255  │ B │ read t=0..255            │ C │ load s255  │ B │ read ... │ C
                   ▲                              ▲
                   │                              └─ C: __syncthreads_count
                   │                                  = 재동기화(덮어쓰기 방지) + 조기 종료 집계
                   └─ B: __syncthreads()  = 적재 완료 보장
```

---

## 9. 원 3DGS 논문(Sec. 6)과의 대응

Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH 2023)의
Sec. 6 "Fast Differentiable Rasterizer for Gaussians"가 이 커널을 그대로 서술한다.

| 논문 서술 | 소스 대응 |
|---|---|
| "we launch one thread block for each tile" | `blockIdx` ↔ 타일, `grid = {n_tiles, 1, 1}` |
| "Each block first **collaboratively loads packets of Gaussians into shared memory**" | `id_batch[tid] / xy_opacity_batch[tid] / conic_batch[tid]` 적재 + `__syncthreads()` |
| "packets" | `BATCH_SIZE = 256` 단위의 배치, `for(b = 0; b < num_batches; ++b)` |
| "for a given pixel, accumulates color and α values by **traversing the lists front-to-back**" | `for(t = 0; t < batch_size; ++t)` — 정렬된 `flatten_ids` 순서 = 깊이 앞→뒤 |
| "maximizing the gain for both parallelism for data loading/sharing and processing" | 적재 O(N)·소비 O(N×256)의 비대칭이 곧 그 gain |
| "when a target saturation of α is reached, the corresponding thread stops" | `next_T <= TRANSMITTANCE_THRESHOLD` → `done_mask |= (1u << p)` |
| "At regular intervals, threads in a tile are queried and the processing of the entire tile terminates when all pixels have saturated" | 배치 끝의 `__syncthreads_count(done_mask == ALL_DONE) >= BATCH_SIZE` |

논문에서 딱 몇 문장인 이 서술이 실제로는 위 §2–§8의 디테일 전부를 함축하고 있다.

---

## 10. 그래서 "타일 기반"의 진짜 이득이 뭔가

타일 분할을 처음 보면 "화면을 잘라서 병렬화하려고" 정도로 이해하기 쉽다. 그건 절반만 맞다.
GPU는 어차피 픽셀 단위로 병렬화할 수 있다. 타일이 진짜로 사 오는 것은 **재사용 가능한 지역성(locality)** 이다.

타일 기반 설계가 만들어 내는 것을 정리하면:

1. **공유할 "작업 목록"이 생긴다.** 타일 하나에 Gaussian 목록 하나(`isect_offsets`가 정의하는 구간).
   256 픽셀이 **완전히 동일한** 목록을 소비하므로, 한 번 stage 해서 256번 재사용할 수 있다.
   → 이게 shared memory가 성립하는 전제다. 타일이 없으면 픽셀마다 목록이 달라 공유할 게 없다.
2. **정렬을 한 번만 한다.** 깊이 정렬을 픽셀당(수백만 번)이 아니라 타일당(수천 번) 한다.
   그리고 그 정렬 결과가 곧 shared에 앞→뒤로 실리는 순서다.
3. **분기 수렴(convergence).** 같은 워프의 스레드들이 같은 Gaussian `t`를 같은 시점에 처리한다 →
   warp divergence가 최소화되고, `colors + g*CDIM` 같은 global 접근도 브로드캐스트로 처리된다.
4. **조기 종료를 집단으로 할 수 있다.** 픽셀들이 블록 안에 모여 있으니 `__syncthreads_count` 한 번으로
   "이 타일 전체 끝"을 알 수 있다. 인접 픽셀들은 대개 비슷한 시점에 포화하므로(공간적 상관) 실제로 잘 맞는다.
5. **작업량이 균등해진다.** 블록 = 타일이므로 GPU 스케줄러가 타일 단위로 SM에 뿌린다.

정리하면 — **타일 = shared memory에 올릴 만한 크기의, 공유 가능한 작업 단위를 만들어 내는 장치**다.
"화면을 나눈다"가 아니라 "**global memory 트래픽을 256분의 1로 줄일 수 있는 데이터 공유 구조를 만든다**"가
타일 기반 래스터화의 본질이고, `extern __shared__` 세 배열과 배치 루프가 그 이득을 실제로 실현하는 코드다.

---

## 11. 자기 점검 질문

1. `extern __shared__ int s[];` 한 개 선언에서 배열 3개를 잘라 쓰는 이유는? 크기는 어디서 정해지나?
2. `CTA_SIZE=256`일 때 블록당 shared memory는 몇 바이트인가? (`vec3`가 12 B임에 주의)
3. `id_batch[tid] = g`에서 인덱스가 `tid`인 것이 왜 race를 없애는가?
4. 루프 안의 배리어 두 개는 각각 어떤 hazard를 막는가? 두 번째는 왜 `__syncthreads_count`로 대체되었나?
5. `BATCH_SIZE`(상수 256)와 `batch_size`(런타임 값)는 왜 둘 다 필요한가?
6. 픽셀이 이미지 밖일 때 `return` 대신 `done_mask = ALL_DONE`을 쓰는 이유는?
7. `colors`는 왜 shared에 올리지 않는가? 세 가지 이유를 대 보라.
8. 조기 종료 판정을 배치마다가 아니라 Gaussian마다 하면 왜 손해인가?
