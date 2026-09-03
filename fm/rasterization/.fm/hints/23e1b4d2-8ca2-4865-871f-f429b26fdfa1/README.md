# `packed=True` — 투영 결과를 COO 희소 형식으로 반환한다

> **Q.** `packed=True`는 투영 결과를 어떤 형태로 반환하는가?
>
> **A.** `[C,N,...]` 밀집 텐서 대신 radii>0인 (카메라, Gaussian) 쌍만 `[nnz,...]`로 반환한다.
> 어느 카메라·어느 Gaussian인지는 `camera_ids`/`gaussian_ids` COO 인덱스로 알려 준다.

워크스루의 9절(`## 9. packed=True — 가시 Gaussian만 남기는 희소 표현`)에 해당하는 내용이다.
`rasterization()`의 기본값은 `packed=True`이고, 워크스루 본문은 설명을 쉽게 하려고 일부러
`packed=False`(밀집 `[C,N,...]`)로 진행했다는 점을 기억해 두면 좋다.

---

## 1. 밀집 `[C, N, ...]` 레이아웃이 왜 낭비인가

투영 단계(`fully_fused_projection`)는 "카메라 C대 × Gaussian N개"의 모든 조합에 대해
2D 평균·깊이·conic·radii를 계산할 수 있는 구조다. `packed=False`면 결과를 말 그대로
`[C, N, ...]` 격자로 받는다.

```
        Gaussian 0  1  2  3  4  5  ...  N-1
cam 0 :     ·  ✓  ·  ·  ✓  ·  ...   ·
cam 1 :     ✓  ·  ·  ✓  ·  ·  ...   ✓
cam 2 :     ·  ·  ·  ·  ·  ✓  ...   ·
```

문제는 **✓(= radii > 0, 실제로 화면 안에 보이는 쌍)가 극소수**라는 것이다. 이유는 두 가지다.

- **각 카메라는 씬의 일부만 본다.** 큰 실외 씬을 여러 대의 카메라로 찍으면, 카메라 하나의
  frustum 안에 들어오는 Gaussian은 전체의 몇 % 수준일 수 있다.
- **프러스텀 컬링·크기 컬링이 더 걷어낸다.** `near_plane`/`far_plane` 밖, 이미지 사각형 밖,
  `radius_clip`보다 작은 splat은 전부 무효 처리된다.

밀집 모드에서는 이 무효 원소들도 **메모리를 그대로 차지하고**(`radii == 0`이 "무시하라"는
표시 역할만 함), forward에서 쓰기 대역폭을, backward에서 0-grad 저장 공간을 낭비한다.
`_wrapper.py`의 docstring도 정확히 이 점을 말한다.

> During projection, we ignore the Gaussians that are outside of the camera frustum.
> So not all the elements in the output tensors are valid. The output `radii` could serve as
> an indicator, in which zero radii means the corresponding elements are invalid...
> If `packed=True`, the output tensors will be packed into a flattened tensor,
> **in which all elements are valid.**

### 메모리 절감 예시 계산

C = 8대, N = 1,000,000개, 가시 비율 30%(즉 nnz = 2,400,000)라고 하자.
투영 결과 하나의 (카메라, Gaussian) 쌍이 들고 다니는 float 개수는 대략
`means2d` 2 + `depths` 1 + `conics` 3 = **6 floats** = 24 B (fp32), 여기에 `radii`(int32 ×2 = 8 B)를 더하면 32 B.

| | 원소 수 | 크기 |
|---|---|---|
| 밀집 `[C,N,...]` | C·N = 8,000,000 | 8M × 32 B ≈ **256 MB** |
| packed `[nnz,...]` | nnz = 2,400,000 | 2.4M × 32 B ≈ **77 MB** |
| packed 인덱스 추가분 | nnz × (batch/camera/gaussian_ids, int64 ×3 = 24 B) | ≈ **58 MB** |
| packed 합계 | | ≈ **135 MB** (약 47%) |

가시 비율이 낮을수록(카메라가 많고 씬이 클수록) 이득이 커진다. 반대로 가시 비율이
거의 100%에 가까운 작은 씬·단일 카메라라면 인덱스 3개(24 B/원소)를 추가로 들고 다니는
만큼 오히려 손해일 수도 있다. 그래서 `rasterization()` docstring이 이걸
**memory-speed trade-off**라고 부른다.

> **Memory-Speed Trade-off**: If `packed` is True, the intermediate results are packed into
> sparse tensors, which is more memory efficient but might be slightly slower. This is
> especially helpful when the scene is large and **each camera sees only a small portion of the scene.**

---

## 2. COO(coordinate) 희소 형식이란

희소 행렬을 저장하는 가장 직관적인 형식이다. **"값 배열 + 그 값이 어느 좌표에 있었는지를 담은
좌표 배열"** 로 구성된다. `nnz`는 non-zero의 개수를 뜻하는 관용어다.

일반적인 2D 희소 행렬 COO:

```
values  = [v0, v1, v2, ...]      # [nnz]
row_ids = [ 1,  0,  2, ...]      # [nnz]
col_ids = [ 3,  1,  5, ...]      # [nnz]
```

gsplat의 packed 투영 출력은 정확히 이 구조다. 행 = 카메라, 열 = Gaussian.

```
means2d      = [[...], [...], ...]   # [nnz, 2]   ← 값
depths       = [ ... ]               # [nnz]      ← 값
conics       = [[...], ...]          # [nnz, 3]   ← 값
radii        = [[...], ...]          # [nnz, 2]   ← 값
camera_ids   = [ ... ]               # [nnz] int  ← 좌표 (행)
gaussian_ids = [ ... ]               # [nnz] int  ← 좌표 (열)
batch_ids    = [ ... ]               # [nnz] int  ← 좌표 (배치 차원)
```

즉 `k`번째 packed 원소는 "배치 `batch_ids[k]`의 카메라 `camera_ids[k]`가 본 Gaussian
`gaussian_ids[k]`의 투영 결과"이고, 밀집 모드였다면 `means2d[camera_ids[k], gaussian_ids[k]]`
자리에 있었을 값이다.

워크스루의 확인 코드:

```python
with torch.no_grad():
    _, _, meta_p = rasterization(means, quats, scales, opacities, sh_coeffs,
                                 viewmats, Ks, W, H, sh_degree=SH_DEGREE, packed=True)
nnz = meta_p["means2d"].shape[0]
print(f"dense means2d {tuple(m_fused['means2d'].shape)} → "
      f"packed means2d {tuple(meta_p['means2d'].shape)} (nnz/CN = {nnz/(C*N):.1%})")
print("camera_ids  [:8] =", meta_p["camera_ids"][:8].tolist())
print("gaussian_ids[:8] =", meta_p["gaussian_ids"][:8].tolist())
```

`camera_ids`는 정렬되어 있다(카메라 0의 쌍이 전부 나온 뒤 카메라 1 …). 커널이
(카메라 행, Gaussian 열) 순서로 스캔하며 쓰기 때문이다. 그래서 커널은 덤으로
**CSR 스타일의 `indptr` `[B*C+1]`** 도 함께 내놓는다 — `indptr[i]..indptr[i+1]`이
i번째 (배치, 카메라)가 차지하는 packed 구간이다. COO에 CSR 인덱스 포인터가 얹힌 셈.

---

## 3. packed 커널의 2패스 구조

핵심 딜레마: **`nnz`를 미리 모른다.** 각 스레드가 자기 Gaussian이 보이는지 계산하기 전에는
출력 버퍼 크기도, 자기가 몇 번째 슬롯에 써야 하는지도 알 수 없다. GPU 스레드끼리 순서를
합의하려면 `atomicAdd` 카운터로 하는 방법도 있지만 그러면 **출력 순서가 비결정적**이 된다.

gsplat은 **타일 교차(`isect_tiles`)와 완전히 같은 2패스 + prefix sum 패턴**을 쓴다.
워크스루 5절에 나온 그 패턴이다("1차: Gaussian별 타일 수 세기 → prefix sum → 2차: 키 채우기").

### 1패스 — 블록별 개수 세기

`ProjectionEWA3DGSPacked.cu`, 스레드 블록당 256개 Gaussian(`N_THREADS_PACKED 256`)을 맡는다.
각 스레드가 투영을 계산해 `valid` 여부를 정하고, `cub::BlockReduce`로 블록 안의 valid 개수를 합산해
`block_cnts[block_idx]` 하나만 기록한다. 출력 텐서에는 아무것도 쓰지 않는다.

```cpp
int32_t thread_data = static_cast<int32_t>(valid);
if (block_cnts != nullptr) {
    // First pass: compute the block-wide sum
    typedef cub::BlockReduce<int32_t, N_THREADS_PACKED> BlockReduce;
    aggregate = BlockReduce(temp_storage).Sum(thread_data);
    if (threadIdx.x == 0) block_cnts[block_idx] = aggregate;
}
```

### 사이 단계 — prefix sum(누적합)으로 각 블록의 쓰기 시작 위치 결정

호스트(`Projection.cpp`)에서:

```cpp
block_accum = at::cumsum(block_cnts, 0, at::kInt);
nnz         = block_accum[-1].item<int32_t>();     // ← 여기서 nnz가 처음 정해진다
```

누적합의 마지막 값이 곧 전체 `nnz`이고, `block_accum[block_idx - 1]`이 그 블록의
**출력 시작 오프셋**이다. 이제 정확한 크기로 출력 버퍼를 잡는다.

```cpp
at::Tensor batch_ids    = at::empty({nnz}, opt.dtype(at::kLong));
at::Tensor camera_ids   = at::empty({nnz}, opt.dtype(at::kLong));
at::Tensor gaussian_ids = at::empty({nnz}, opt.dtype(at::kLong));
at::Tensor radii        = at::empty({nnz, 2}, opt.dtype(at::kInt));
at::Tensor means2d      = at::empty({nnz, 2}, opt);
at::Tensor depths       = at::empty({nnz}, opt);
at::Tensor conics       = at::empty({nnz, 3}, opt);
```

### 2패스 — 실제 쓰기

같은 커널을 `block_cnts = nullptr`, `block_accum = <누적합>`으로 다시 돌린다.
투영을 **한 번 더 계산**하고(재계산이 저장보다 싸다), 블록 안에서는 `cub::BlockScan`의
exclusive sum으로 스레드별 로컬 오프셋을 얻은 뒤 블록 오프셋을 더해 최종 슬롯을 정한다.

```cpp
// Second pass: write out the indices of the non zero elements
BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
if (valid) {
    if (block_idx > 0) thread_data += block_accum[block_idx - 1];
    batch_ids[thread_data]       = bid;
    camera_ids[thread_data]      = cid;
    gaussian_ids[thread_data]    = gid;
    radii[thread_data * 2]       = (int32_t)radius_x;
    ...
    means2d[thread_data * 2]     = mean2d.x;
    depths[thread_data]          = mean_c.z;
    conics[thread_data * 3]      = covar2d_inv[0][0];
    ...
}
// lane 0 of the first block in each row writes the indptr
```

`atomicAdd` 없이 결정적(deterministic)이고 압축된 출력이 나온다. 대가는 투영 계산 2회다.

---

## 4. `nnz`가 런타임에야 정해진다 — 동적 출력 크기의 대가

위 흐름에서 보이듯 `nnz`는 **1패스 결과를 GPU→CPU로 가져와야(`.item<int32_t>()`) 알 수 있다.**
여기서 두 가지 부작용이 생긴다.

- **동기화 지점.** `block_accum[-1].item()`은 device→host 복사이므로 스트림을 동기화한다.
  밀집 모드에는 없는 파이프라인 버블이다.
- **출력 shape가 데이터 의존적(data-dependent).** 매 iteration마다 `means2d`의 shape가 달라진다.
  이 때문에 `torch.compile`은 이 지점에서 graph break를 내거나 dynamic shape로 재컴파일하기 쉽고,
  **CUDA Graph 캡처는 사실상 불가능**하다(캡처는 고정 shape와 동기화 없는 실행을 요구한다).
  고정 배치 shape를 원하는 최적화 경로에서는 `packed=False`가 오히려 유리할 수 있다.

밀집 모드는 shape가 `[C, N, ...]`으로 항상 같기 때문에 이런 문제가 없다. "packed가
메모리는 아끼지만 *might be slightly slower*"라는 문서 문구의 실체 중 하나가 이것이다.

---

## 5. 반환되는 텐서 목록이 어떻게 바뀌는가

`fully_fused_projection()`의 반환값 비교 (`gsplat/cuda/_wrapper.py`):

| | `packed=False` | `packed=True` |
|---|---|---|
| `batch_ids` | — | `[nnz]` int32 **(추가)** |
| `camera_ids` | — | `[nnz]` int32 **(추가)** |
| `gaussian_ids` | — | `[nnz]` int32 **(추가)** |
| `indptr` | — | `[B*C+1]` int32 **(추가, CSR 스타일)** |
| `radii` | `[..., C, N, 2]` | `[nnz, 2]` |
| `means2d` | `[..., C, N, 2]` | `[nnz, 2]` |
| `depths` | `[..., C, N]` | `[nnz]` |
| `conics` | `[..., C, N, 3]` | `[nnz, 3]` |
| `compensations` | `[..., C, N]` | `[nnz]` |

`rasterization()`의 `meta` 딕셔너리도 마찬가지다. `packed=True`면 `meta["batch_ids"]`,
`meta["camera_ids"]`, `meta["gaussian_ids"]`가 실제 텐서로 채워지고, `packed=False`면
`rendering.py`가 셋 다 명시적으로 `None`으로 만든다.

```python
if not packed:
    batch_ids = None
    camera_ids = None
    gaussian_ids = None
```

**규칙: 차원 수로 모드를 판별할 수 있다.** `means2d.dim() == 2`(즉 `[nnz, 2]`)이면 packed다.
실제로 `isect_tiles_lidar`류의 일부 래퍼는 `packed = means2d.dim() == 2`로 자동 판정한다.

---

## 6. 이후 단계는 packed 입력을 그대로 받는다

packed는 투영 한 단계만의 최적화가 아니다. 파이프라인 뒤쪽이 전부 같은 `[nnz, ...]`
인터페이스를 지원해서 **밀집으로 되돌리지 않고 끝까지 간다.**

- **④ SH 평가 (`spherical_harmonics`)** — `coeffs`는 `[N, K, D]`(dense) 또는 `[nnz, K, D]`(packed).
  docstring: *"In packed mode, callers pre-gather coefficients by `gaussian_ids` so the first
  coefficient dimension is `nnz`."* 즉 호출자가 `coeffs[gaussian_ids]`로 미리 gather해 넘긴다.
  방향 벡터도 `camera_ids`로 카메라 위치를 골라 계산한다. masks도 `[nnz]`.
- **⑤ 타일 교차 (`isect_tiles`)** — `packed=True`, `n_images`, `image_ids`, `gaussian_ids`를 함께 받는다.
  `flatten_ids`가 가리키는 인덱스 공간이 `[I*N]`에서 `[nnz]`로 바뀐다.
- **⑦ 블렌딩 (`rasterize_to_pixels`)** — `means2d`/`conics`/`colors`/`opacities` 모두
  `[nnz, ...]` 형태를 그대로 받는다.

그리고 워크스루가 확인하듯 **결과 이미지는 밀집 모드와 동일하다.**

```python
print("render 동일:", torch.allclose(
    rasterization(..., packed=True)[0], r_fused, atol=1e-5))
```

---

## 7. `sparse_grad=True`와의 결합 — 희소 gradient

`sparse_grad`는 **`packed=True`일 때만** 쓸 수 있다(`assert packed, "sparse_grad is only
supported when packed is True"`). 켜면 backward에서 `means`/`quats`/`scales`(또는 `covars`)의
gradient가 밀집 `[N, 3]`이 아니라 **`torch.sparse_coo_tensor`** 로 돌아온다.

`Projection.cpp`:

```cpp
// When sparse_grad is set, the kernel writes per-nnz dense gradients indexed
// by gaussian_ids; wrap them as sparse COO over the full per-input shapes so
// the optimizer can scatter-update only the touched gaussians. Coalesced
// when there is a single batch (each gaussian_id appears once).
if (sparse_grad) {
    const bool is_coalesced = viewmats.size(0) == 1;
    const at::Tensor sparse_grad_indices = gaussian_ids.unsqueeze(0);   // [1, nnz]
    v_means = make_sparse_coo_grad(sparse_grad_indices, v_means, means.sizes(), is_coalesced);
    ...
}
```

즉 gradient도 **인덱스 = `gaussian_ids`, 값 = `[nnz, 3]`, 논리적 크기 = `[N, 3]`** 인 COO 텐서다.
투영 출력의 희소성이 그대로 gradient의 희소성으로 이어진다.

**옵티마이저는 이걸 이해해야 한다.** 밀집 Adam은 sparse grad를 못 받으므로
`torch.optim.SparseAdam`을 쓴다(`examples/simple_trainer.py`).

```python
optimizer_class = torch.optim.SparseAdam if sparse_grad else torch.optim.Adam
...
# Turn Gradients into Sparse Tensor before running optimizer
if cfg.sparse_grad:
    assert cfg.packed, "Sparse gradients only work with packed mode."
    gaussian_ids = info["gaussian_ids"]
    for k in self.splats.keys():
        grad = self.splats[k].grad
        if grad is None or grad.is_sparse:
            continue
        self.splats[k].grad = torch.sparse_coo_tensor(
            indices=gaussian_ids[None],     # [1, nnz]
            values=grad[gaussian_ids],      # [nnz, ...]
            size=self.splats[k].size(),     # [N, ...]
            is_coalesced=len(Ks) == 1,
        )
```

이렇게 하면 옵티마이저가 **이번 iteration에 실제로 보인 Gaussian만** 업데이트한다.
Gaussian이 수백만 개인 대형 씬에서 Adam 상태 갱신 비용과 grad 버퍼가 크게 줄어든다.
`is_coalesced`가 카메라 1대일 때만 `True`인 이유는, 카메라가 여러 대면 같은 `gaussian_id`가
여러 번 등장해 중복 인덱스가 생기기 때문이다(중복은 합산되어야 하므로 coalesce 필요).

참고로 `sparse_grad`는 배치 차원을 지원하지 않는다
(`TORCH_CHECK(!sparse_grad || means.dim() == 2, "sparse_grad does not support batch dimensions")`).

---

## 8. 트레이드오프 정리

**얻는 것**
- 중간 텐서 메모리 대폭 절감 (가시 비율이 낮을수록 이득 ↑). 카메라 수가 많거나 씬이 클수록 결정적.
- 무효 원소에 대한 쓰기 대역폭·downstream 연산 낭비 제거.
- `sparse_grad`로 gradient/옵티마이저 비용까지 희소화 가능.

**치르는 비용**
- **투영 계산 2회** (1패스 카운트 + 2패스 쓰기). 워크로드에 따라 forward가 느려질 수 있다.
- **인덱싱 gather 비용.** SH 계수를 `coeffs[gaussian_ids]`로, opacity를 `opacities[gaussian_ids]`로
  뽑아야 한다. 밀집 모드의 broadcast는 공짜지만 gather는 랜덤 접근이라 캐시 친화적이지 않다.
- **동기화·동적 shape.** `nnz`가 런타임 결정 → D2H 동기화, `torch.compile` 재컴파일/graph break,
  CUDA Graph 캡처 불가.
- **원소 순서가 밀집 모드와 다르다.** packed 배열은 (카메라, Gaussian) 스캔 순서로 압축되어 있고,
  이후 타일 정렬의 `flatten_ids`도 `[nnz]` 인덱스 공간을 가리킨다. 인덱스를 직접 만지는 코드
  (densification 통계, absgrad 집계 등)는 모드에 따라 분기해야 한다 — 예: `simple_trainer.py`가
  visibility mask를 만들 때 `if cfg.packed:` 로 갈라진다.
- 부동소수점 누적 순서가 달라져 밀집 모드와 **bit-exact하지는 않다**(워크스루도 `atol=1e-5`로 비교).

**언제 무엇을 쓰나**
- 큰 실외 씬 + 카메라 다수 + 메모리 압박 → `packed=True` (기본값), 필요하면 `sparse_grad=True`.
- 작은 씬 / 카메라 1대 / 대부분의 Gaussian이 보임 / 고정 shape 컴파일 최적화 → `packed=False`.
- **설명·디버깅 목적** → `packed=False`. 워크스루가 본문 내내 밀집을 쓴 이유가 그것이다
  (`[C, N, ...]`은 인덱스가 그대로 (카메라, Gaussian)이라 눈으로 따라가기 쉽다).

---

## 관련 소스

| 대상 | 위치 |
|---|---|
| 워크스루 9절 | `.fm/assets/rasterization_walkthrough.py` |
| Python 래퍼 / docstring | `gsplat/cuda/_wrapper.py` — `fully_fused_projection`, `RegisterProjectionEWA3DGSPacked` |
| `packed` 분기, meta 구성 | `gsplat/rendering.py` — `rasterization()` |
| 2패스 호스트 드라이버 (`cumsum` → `nnz`) | `gsplat/cuda/csrc/Projection.cpp` |
| packed 커널 (BlockReduce / BlockScan) | `gsplat/cuda/csrc/ProjectionEWA3DGSPacked.cu` |
| `N_THREADS_PACKED 256` | `gsplat/cuda/include/Common.h` |
| `sparse_grad` + `SparseAdam` 사용 예 | `examples/simple_trainer.py` |
