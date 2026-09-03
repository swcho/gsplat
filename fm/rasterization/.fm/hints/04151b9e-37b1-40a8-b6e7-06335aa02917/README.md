# backward에서 Gaussian별 grad를 모으는 방법 — warp reduction + atomicAdd

> **Q.** backward에서 Gaussian별 grad를 모으는 방법은?
>
> **A.** Gaussian 하나에 대한 기여가 여러 픽셀(스레드)에 흩어져 있으므로 **warp 단위로 합친 뒤 `atomicAdd`로 모은다.** 이렇게 atomic 연산 횟수를 줄인다.

---

## 1. 문제의 구조 — forward는 gather, backward는 scatter

래스터화 forward 커널의 데이터 흐름은 **"픽셀 ← 여러 Gaussian"** 이다.

```
픽셀 (i, j)  ←  g0, g1, g2, ..., gk   (타일 목록을 앞→뒤로 순회하며 알파 블렌딩)
```

한 스레드가 한 픽셀을 담당하고, 자기 픽셀의 색을 **자기 레지스터에** 누적한 뒤 마지막에 `render_colors[pix_id]`에 **한 번 쓴다.** 출력 주소가 스레드마다 유일하므로 경쟁 조건이 아예 없다. 전형적인 **gather** 패턴이다.

backward는 이 화살표가 뒤집힌다. **"Gaussian ← 여러 픽셀"** 이다.

```
∂L/∂(g의 파라미터)  =  Σ over 픽셀   (g가 덮은 모든 픽셀의 기여를 더함)
```

Gaussian 하나는 보통 수십~수천 픽셀에 걸쳐 있고, 그 픽셀들은 **서로 다른 스레드, 서로 다른 블록(타일), 심지어 서로 다른 카메라**에 흩어져 있다. 각 스레드가 계산한 `v_xy_local`, `v_conic_local` 등은 전체 grad의 **부분합**일 뿐이고, 이것들이 전부 같은 전역 주소 `v_means2d[g]`, `v_conics[g]`로 더해져야 한다.

즉 backward의 출력은 **여러 스레드가 같은 주소에 write**하는 **scatter** 패턴이다. 그냥 `v_means2d[g] += ...`로 쓰면 read-modify-write가 인터리브되어 값을 잃는다(lost update). 그래서 **atomic 연산**이 필요하다.

| | forward | backward |
|---|---|---|
| 방향 | 픽셀 ← 여러 Gaussian (gather) | Gaussian ← 여러 픽셀 (scatter) |
| 출력 주소 | 스레드마다 유일 (`pix_id`) | 여러 스레드가 공유 (`g`) |
| 필요한 동기화 | 없음 (그냥 store) | atomic 필요 |

---

## 2. `atomicAdd`의 의미와 비용

`atomicAdd(ptr, val)`은 "읽고 → 더하고 → 쓰는" 세 동작을 **다른 스레드가 끼어들 수 없는 하나의 단위**로 실행한다. GPU에서는 보통 L2 캐시 라인 단위로 락을 잡는 식으로 구현된다.

문제는 **같은 주소에 여러 스레드가 몰릴 때**다. 하드웨어는 그 요청들을 **직렬화**한다. 32개 스레드가 같은 주소에 atomicAdd를 하면 32번의 순차 처리가 된다. 이것이 backward 커널의 대표적 병목이다.

- **큰 Gaussian일수록 심하다.** 화면에서 넓게 퍼진 Gaussian은 그만큼 많은 픽셀·스레드가 동시에 같은 `v_means2d[g]`를 두드린다. 큰 splat 몇 개가 backward 시간을 지배하는 현상은 3DGS 학습에서 흔히 관찰된다.
- 게다가 이 주소들은 **전역 메모리(L2)** 에 있다. 워프 안 32개가 각기 atomic을 발행하면 메모리 트랜잭션도 32번 나간다.

핵심 통찰: **atomic의 총량 자체를 줄이자.** 어차피 더할 값이라면, 전역 메모리에 가기 **전에** 가능한 만큼 미리 합쳐 놓으면 된다.

---

## 3. warp란 무엇인가

CUDA에서 블록 안의 스레드는 **32개씩 묶인 warp** 단위로 실행된다(NVIDIA GPU에서 warp size = 32은 고정). 같은 warp의 32개 스레드(각각을 **lane**이라 부른다)는:

- **같은 명령어를 동시에** 실행한다 (SIMT). 분기가 갈리면 divergence가 생겨 양쪽을 순차 실행한다.
- **레지스터를 서로 직접 교환할 수 있다.** 이게 결정적이다 — 공유 메모리도, 전역 메모리도, `__syncthreads()`도 거치지 않고, `__shfl_*_sync` 계열 warp shuffle 명령 하나로 옆 lane의 레지스터 값을 읽어올 수 있다.

이 래스터화 커널은 블록이 `dim3 threads = {tile_size, tile_size, 1}`, 즉 16×16 = **256 스레드 = 8개 warp**로 뜬다(`RasterizeToPixels3DGSSerialBatchBwd.cu`의 launch 함수).

---

## 4. warp reduction — `__shfl_down_sync` 기반 로그 단계 합산

32개 lane이 각자 값 하나씩 가지고 있을 때, 이들의 합을 구하는 고전적 패턴:

```cuda
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
// 이제 lane 0의 val이 32개 전체의 합
```

동작 원리:

```
단계 0 (offset=16):  lane i  +=  lane (i+16)   →  앞 16개 lane에 부분합
단계 1 (offset=8) :  lane i  +=  lane (i+8)    →  앞  8개
단계 2 (offset=4) :  lane i  +=  lane (i+4)    →  앞  4개
단계 3 (offset=2) :  lane i  +=  lane (i+2)    →  앞  2개
단계 4 (offset=1) :  lane 0  +=  lane 1        →  lane 0에 총합
```

**log₂(32) = 5단계**만에 끝난다. 그리고 이 5단계는 전부 **레지스터 ↔ 레지스터 교환**이다. 메모리 접근이 0회, 배리어가 0회. warp shuffle은 사실상 ALU 명령 수준의 비용이라 극도로 싸다.

CUDA의 **cooperative groups**(`cg`)는 이 패턴을 `cg::reduce`로 추상화해 준다. gsplat은 `cg::reduce`를 쓴다 — 컴파일러가 아키텍처에 맞게 shuffle 시퀀스(혹은 Ampere 이상의 `redux.sync` 하드웨어 명령)로 낮춰 준다.

---

## 5. 실제 소스 — `warpSum` 헬퍼

`gsplat/cuda/include/Utils.cuh` (Reduce 섹션):

```cuda
template<uint32_t DIM, class WarpT>
inline __device__ void warpSum(float *val, WarpT &warp)
{
#pragma unroll
    for (uint32_t i = 0; i < DIM; i++)
    {
        val[i] = cg::reduce(warp, val[i], cg::plus<float>());
    }
}

template<class WarpT>
inline __device__ void warpSum(float &val, WarpT &warp)
{
    val = cg::reduce(warp, val, cg::plus<float>());
}

template<class WarpT>
inline __device__ void warpSum(vec3 &val, WarpT &warp)
{
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
    val.z = cg::reduce(warp, val.z, cg::plus<float>());
}
```

`vec2`, `vec4`, `mat2/3/4` 오버로드도 있는데, 전부 성분별로 `cg::reduce`를 부르는 얇은 래퍼일 뿐이다. `warpMax`(`cg::greater`)도 같은 파일에 있다.

> **주의:** `cg::reduce`는 warp 내 **모든 활성 lane이 함께 호출해야 하는 collective**다. 그래서 `warpSum` 호출은 반드시 `if (valid)` 블록 **바깥**에 있어야 한다 — 아래 커널 코드에서 확인된다.

---

## 6. 실제 소스 — backward 커널의 조립 지점

`gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu`.

**(a) warp 타일 만들기** (배치 루프 진입 직전):

```cuda
namespace cg = cooperative_groups;
...
const uint32_t tr              = block.thread_rank();
cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
const int32_t warp_bin_final   = cg::reduce(warp, bin_final, cg::greater<int>());
```

`cg::tiled_partition<32>(block)`은 256스레드 블록을 32개짜리 warp 타일 8개로 정적 분할한다. 바로 다음 줄의 `warp_bin_final`도 같은 reduce 기법의 다른 용도다 — warp 안 32픽셀 중 **가장 늦게까지 기여한 Gaussian 인덱스**를 구해서, warp 전체가 그 지점부터 루프를 시작하도록 **불필요한 반복을 건너뛰는** 최적화다.

**(b) 각 스레드가 자기 픽셀의 부분 grad를 레지스터에 계산:**

```cuda
float v_rgb_local[CDIM] = {0.f};
vec3  v_conic_local     = {0.f, 0.f, 0.f};
vec2  v_xy_local        = {0.f, 0.f};
vec2  v_xy_abs_local    = {0.f, 0.f};
float v_opacity_local   = 0.f;
// initialize everything to 0, only set if the lane is valid
if (valid)
{
    ...  // ∂L/∂color, ∂L/∂α → ∂L/∂conic, ∂L/∂means2d, ∂L/∂opacity
}
```

`_local` 접미사가 "이건 아직 **이 스레드 한 픽셀의 몫**"이라는 표시다. `valid`가 아닌 lane은 **0으로 남는다** — 그래야 합산에 참여시켜도 결과가 안 바뀐다. (그래서 선언 시점에 무조건 0 초기화한다.)

**(c) warp 합산 → lane 0만 atomic:**

```cuda
warpSum<CDIM>(v_rgb_local, warp);
warpSum(v_conic_local, warp);
warpSum(v_xy_local, warp);
if (v_means2d_abs != nullptr)
{
    warpSum(v_xy_abs_local, warp);
}
warpSum(v_opacity_local, warp);
if (warp.thread_rank() == 0)
{
    int32_t g        = id_batch[t];
    float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k)
    {
        atomicAdd_system(v_rgb_ptr + k, v_rgb_local[k]);
    }

    float *v_conic_ptr = (float *)(v_conics) + 3 * g;
    atomicAdd_system(v_conic_ptr,     v_conic_local.x);
    atomicAdd_system(v_conic_ptr + 1, v_conic_local.y);
    atomicAdd_system(v_conic_ptr + 2, v_conic_local.z);

    float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
    atomicAdd_system(v_xy_ptr,     v_xy_local.x);
    atomicAdd_system(v_xy_ptr + 1, v_xy_local.y);

    if (v_means2d_abs != nullptr)
    {
        float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
        atomicAdd_system(v_xy_abs_ptr,     v_xy_abs_local.x);
        atomicAdd_system(v_xy_abs_ptr + 1, v_xy_abs_local.y);
    }

    atomicAdd_system(v_opacities + g, v_opacity_local);
}
```

`warp.thread_rank() == 0` — **lane 0만이** 전역 메모리를 건드린다. 나머지 31개 lane은 자기 값을 이미 lane 0에게 넘겼으므로 할 일이 없다.

> **이름에 대한 메모:** 이 트리에서는 `atomicAdd_system`을 쓴다. 업스트림 gsplat 및 다른 PyTorch 확장에서는 보통 `gpuAtomicAdd`(ATen의 타입 제네릭 래퍼, half/bfloat16까지 커버)를 쓴다. 의미는 같고, `_system` 접미사는 멀티-GPU/UVA 스코프까지 보장하는 변형이다. 어느 쪽이든 **핵심은 "lane 0 한 명만, warp당 한 번"** 이다.

---

## 7. 효과 — atomic 횟수 1/32

Gaussian 하나·warp 하나당:

| | atomic 호출 수 | 부가 비용 |
|---|---|---|
| 순진한 구현 | 32회 (lane마다) | 같은 주소 → 32-way 직렬화 |
| warpSum + lane 0 | **1회** | shuffle 5단계 × 성분 수 (레지스터 연산) |

전역 atomic 트래픽이 **1/32**로 떨어진다. 대신 붙는 비용은 레지스터 shuffle 5단계뿐인데, 이는 전역 메모리 atomic 한 번보다도 훨씬 싸다. 명백한 win이다.

CDIM=3(RGB) + conic 3 + means2d 2 + opacity 1 = 픽셀당 9개 float grad가 이 경로를 탄다(absgrad 켜면 11개).

---

## 8. 어떤 grad들이 이렇게 모이는가

| 텐서 | 모양 | 의미 | `warpSum` 호출 |
|---|---|---|---|
| `v_colors` | `[..., N, CDIM]` | ∂L/∂(Gaussian 색) | `warpSum<CDIM>(v_rgb_local, warp)` |
| `v_conics` | `[..., N, 3]` | ∂L/∂(2D 역공분산 상삼각 3성분) | `warpSum(v_conic_local, warp)` |
| `v_means2d` | `[..., N, 2]` | ∂L/∂(화면상 중심 xy) | `warpSum(v_xy_local, warp)` |
| `v_opacities` | `[..., N]` | ∂L/∂(불투명도) | `warpSum(v_opacity_local, warp)` |
| `v_means2d_abs` | `[..., N, 2]` | **absgrad** — ∂L/∂means2d의 **절댓값 합** | `warpSum(v_xy_abs_local, warp)` (포인터가 non-null일 때만) |

`v_means2d_abs`(= `absgrad=True`)는 AbsGS 논문의 밀도화 기준이다. 부호가 상쇄되어 작아지는 일반 grad와 달리 `abs(v_xy_local)`을 누적하므로, "여러 방향으로 잡아당겨지는(= 쪼개야 하는) Gaussian"을 잘 골라낸다. 값의 성격만 다를 뿐 **모으는 메커니즘은 동일**하다.

`v_conics` → `v_covars2d` → `v_means3d`/`v_quats`/`v_scales`로 이어지는 뒷단은 **투영 커널의 backward**(`ProjectionEWA3DGSFused/Packed`)가 담당하며, 거기서도 같은 기법이 쓰인다(§10 참고).

---

## 9. 전제 조건 — 왜 warp 안 32스레드가 "같은" Gaussian을 보고 있는가

`warpSum` 후 lane 0이 `id_batch[t]` 하나로 주소를 계산할 수 있는 이유는, **warp의 32개 lane이 그 순간 전부 동일한 Gaussian `t`를 처리 중**이기 때문이다. 이 전제는 커널의 **shared memory 배치 구조**에서 나온다.

```cuda
extern __shared__ int s[];
int32_t *id_batch      = (int32_t *)s;                                       // [block_size]
vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[block_size]);    // [block_size]
vec3 *conic_batch      = reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]);
float *rgbs_batch      = (float *)&conic_batch[block_size];                  // [block_size * CDIM]
```

흐름:

1. **협력 적재.** 블록의 256 스레드가 각자 Gaussian 하나씩(`id_batch[tr] = flatten_ids[idx]`) 전역 메모리에서 읽어 shared memory에 채운다. → 한 배치 = 256개 Gaussian.
2. `block.sync()` — 배치가 다 찰 때까지 대기.
3. **동기 순회.** `for (uint32_t t = ...; t < batch_size; ++t)` — **모든 스레드가 같은 `t`를 같은 순서로** 돈다. 즉 어느 시점이든 블록 전체(따라서 warp 전체)가 **같은 Gaussian**을 보고 있고, 다만 **각자 다른 픽셀**에 대해 평가할 뿐이다.
4. 그러니 lane 0이 `id_batch[t]` 하나만 읽어 주소를 만들면 warp 전체의 합에 정확히 맞는다.

이것이 "블록 전체가 같은 Gaussian 배치를 동시에 처리한다"는 설계가 warp reduction을 **가능하게** 만드는 지점이다. 주석도 그렇게 말한다: `// have all threads in tile process the same gaussians in batches`.

같은 전제 덕분에 divergence를 줄이는 최적화도 붙는다:

```cuda
// if all threads are inactive in this warp, skip this loop
if (!warp.any(valid))
{
    continue;
}
```

warp 안 아무도 이 Gaussian에 기여하지 않으면 reduce도 atomic도 통째로 건너뛴다.

---

## 10. 대비 — warp 안 lane마다 Gaussian이 **다른** 경우

투영 커널의 packed backward(`ProjectionEWA3DGSPacked.cu`)는 상황이 다르다. 스레드 하나가 (카메라, Gaussian) 쌍 하나를 맡으므로 **한 warp 안에 서로 다른 Gaussian이 섞여 있다.** 그래서 `tiled_partition` 대신 **`labeled_partition`** 을 쓴다:

```cuda
auto warp_group_g = cg::labeled_partition(warp, gid);
...
warpSum(v_mean, warp_group_g);
if (warp_group_g.thread_rank() == 0) { /* atomicAdd */ }
```

`labeled_partition(warp, gid)`은 같은 라벨(`gid`)을 가진 lane끼리 **동적으로** 부분그룹을 만든다. packed 모드에서는 같은 Gaussian이 여러 카메라에 걸쳐 연속 배치되므로 warp 안에 같은 `gid`가 여럿 있을 수 있고, 그 그룹 안에서만 합쳐 대표 lane이 atomic을 한다. 카메라 파라미터 grad(`v_R`, `v_t`)는 `labeled_partition(warp, cid)`으로 카메라 id 기준 그룹을 만든다.

정리하면 **"reduce → 대표 lane만 atomic"** 이라는 패턴은 동일하고, **그룹을 어떻게 정의하느냐**만 커널 구조에 따라 달라진다.

---

## 11. 원 3DGS 구현과의 차이

INRIA 원본 `diff-gaussian-rasterization`의 backward는 각 픽셀 스레드가 **자기 값으로 곧장 `atomicAdd`** 를 호출한다:

```cuda
// 원본 계열의 개념 코드
atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
```

warp reduction 단계가 없다. 즉 **warp당 32번**의 atomic이 나가고, 같은 Gaussian을 보는 32 lane이 같은 주소에서 직렬화된다. gsplat이 `warpSum`을 한 겹 끼워 넣은 것이 backward 속도 차이의 큰 축 하나다(다른 축은 tile-batch/shared-mem 구조, per-Gaussian 조기 종료 등).

일반화하면 **"atomic을 없애는 게 아니라, atomic에 도달하기 전에 최대한 미리 합쳐서 횟수를 줄인다"** 는 GPU 최적화의 표준 레시피다. 히스토그램 커널이 shared-memory 지역 히스토그램을 만든 뒤 전역에 한 번만 합치는 것과 정확히 같은 발상이다.

---

## 12. 부작용 — 부동소수 합의 비결정성

atomicAdd는 **원자성은 보장하지만 순서는 보장하지 않는다.** 어떤 warp/블록이 먼저 도착하느냐는 매 실행마다 스케줄링에 따라 달라진다.

부동소수점 덧셈은 **결합법칙이 성립하지 않으므로**(`(a+b)+c ≠ a+(b+c)`, 반올림 때문에) 더하는 **순서가 바뀌면 결과가 미세하게 달라진다.** 그래서:

- 같은 입력·같은 시드로 두 번 돌려도 grad의 마지막 몇 비트가 다를 수 있다.
- 그 차이가 optimizer를 타고 증폭되어, 긴 학습에서는 재현이 정확히 안 된다.
- `torch.use_deterministic_algorithms(True)`로도 커스텀 CUDA atomic까지는 강제하지 못한다.

warpSum을 넣으면 warp 내부 32개의 합산 순서는 **고정**되므로 비결정성의 원천이 32배 줄기는 하지만, **warp들 사이의 순서는 여전히 비결정적**이라 근본 해결은 아니다. 실무에서는 학습 자체가 확률적이라 대체로 용인하되, 디버깅이나 회귀 테스트에서 `allclose`를 쓸 때는 `atol/rtol`을 넉넉히 잡아야 한다.

## 13. 대안들 (참고)

- **블록 단위 reduce.** 블록의 8개 warp 결과를 shared memory에 모아 한 번 더 줄이면 atomic이 블록당 1회(1/256)가 된다. 하지만 `__syncthreads()`와 shared memory가 추가로 필요하고, 이 커널은 이미 shared memory를 배치 캐시로 꽉 쓰고 있어서 occupancy가 떨어진다. warp 레벨이 비용/이득의 스위트 스팟이다.
- **정렬 후 segmented reduce.** (gaussian_id, grad) 쌍을 전부 쏟아낸 뒤 gaussian_id로 정렬해 구간별 합을 구하면 atomic이 아예 사라지고 **완전히 결정적**이 된다. 대신 거대한 중간 버퍼와 정렬 비용이 든다 — 결정성이 꼭 필요한 상황용.
- **sparse_grad.** gsplat의 `sparse_grad=True`는 packed 표현에서 grad를 sparse 텐서로 되돌려, 보이지 않는 Gaussian의 dense 0 grad 자체를 만들지 않는다. 경쟁을 줄이는 게 아니라 **일의 총량**을 줄이는 직교하는 축이다.
- **하드웨어 지원.** Ampere 이상은 `redux.sync` 로 warp reduce를 한 명령에 하고, 최근 아키텍처의 L2는 같은 주소 atomic을 하드웨어에서 병합해 준다. 그래도 소프트웨어 reduce가 여전히 유리하다.

---

## 한 줄 요약

backward는 **Gaussian ← 여러 픽셀**의 scatter라 여러 스레드가 같은 주소에 더해야 한다. 블록 전체가 shared memory 배치를 통해 **동시에 같은 Gaussian**을 보고 있다는 점을 이용해, `cg::reduce`(=`__shfl_down_sync` 5단계) 기반 `warpSum`으로 warp 32개 lane의 부분 grad를 **레지스터만으로** 합치고, `warp.thread_rank() == 0`인 lane 하나만 `atomicAdd`를 호출한다. atomic 횟수가 **1/32**로 줄어든다. 대상은 `v_colors`, `v_conics`, `v_means2d`, `v_opacities` (+ absgrad의 `v_means2d_abs`).

---

### 소스 위치

- `gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu` — `cg::tiled_partition<32>` / `warp_bin_final` (L154–155), shared memory 배치 (L128–133), `warpSum` + `thread_rank() == 0` + `atomicAdd_system` 블록 (L282–312)
- `gsplat/cuda/include/Utils.cuh` — `warpSum` / `warpMax` 오버로드 (Reduce 섹션, L153~)
- `gsplat/cuda/csrc/ProjectionEWA3DGSPacked.cu` — `cg::labeled_partition` 변형 (L611, L666)
- `gsplat/cuda/csrc/RasterizeToPixels2DGSSerialBatchBwd.cu`, `RasterizeToPixelsSparseBwd.cu` — 동일 패턴의 다른 변형
