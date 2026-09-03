# `quats` 파라미터의 정규화 위치

**Q.** `quats` 파라미터는 정규화를 어디서 하는가?
**A.** 미정규화 상태로 저장하고 rasterization 내부에서 normalize한다. 초기값은 랜덤이다.

---

## 1. 한눈에 보는 구조

```
splats["quats"]              torch.rand(N, 4)  ← 미정규화, 노름 ≈ 1.15, 자유롭게 표류
        │                    (Adam이 4개 성분을 제약 없이 업데이트)
        ▼  그대로 전달 (activation 없음!)
rasterization(quats=...)     docstring: "It's not required to be normalized"
        ▼
CUDA 커널 quat_to_rotmat()   inv_norm = rsqrt(w²+x²+y²+z²)  ← 여기서 정규화
        ▼
R (3×3 회전행렬)  →  Σ = R S Sᵗ Rᵗ
```

워크스루의 파라미터 표(`training_walkthrough.py` 2단계)가 이 열을 "저장 공간 → 활성화"로
적어 놓은 이유가 여기 있다. 다른 파라미터는 파이썬 쪽에서 활성화 함수를 통과한다.

| 파라미터 | 저장 공간 | 활성화가 일어나는 곳 |
|---|---|---|
| `scales` | log | **파이썬**: `torch.exp(splats["scales"])` |
| `opacities` | logit | **파이썬**: `torch.sigmoid(splats["opacities"])` |
| `quats` | 미정규화 4-벡터 | **CUDA 커널 내부**: `quat_to_rotmat()` |

`quats`만 파이썬 단계에서 아무 변환도 하지 않는다는 점이 이 카드의 핵심이다.

## 2. 코드로 확인하기

### 저장 — 미정규화, 랜덤 초기값

`examples/simple_trainer.py:331` (`create_splats_with_optimizers`):

```python
quats = torch.rand((N, 4))  # [N, 4]
...
("quats", torch.nn.Parameter(quats), quats_lr),   # quats_lr = 1e-3
```

`F.normalize`가 없다. 워크스루의 `init_splats_with_optimizers`도 동일하게
`quats = torch.rand(N, 4, device=device)`만 쓴다.

### 전달 — 파이썬 쪽에서 손대지 않음

`examples/simple_trainer.py:666-668`에 주석으로 의도가 명시돼 있다:

```python
# quats = F.normalize(splats["quats"], dim=-1)  # [N, 4]
# rasterization does normalization internally
quats = splats["quats"]  # [N, 4]
```

한 줄이 주석 처리된 채 남아 있는 것 자체가 "일부러 안 한다"는 표시다.
`gsplat/rendering.py:400`의 docstring도 같은 계약을 선언한다:

> `quats`: The quaternions of the Gaussians (wxyz convension). **It's not required to be normalized.** `[..., N, 4]`

### 정규화 — CUDA 커널 안

`gsplat/cuda/include/Utils.cuh:228`:

```cpp
inline __device__ mat3 quat_to_rotmat(const vec4 quat)
{
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    float inv_norm = rsqrt_parse_safe(x*x + y*y + z*z + w*w);
    x *= inv_norm; y *= inv_norm; z *= inv_norm; w *= inv_norm;
    ...
}
```

이 함수가 `quat_scale_to_covar_preci()`(같은 헤더) → `ProjectionEWA3DGSFused.cu:129`,
`Projection2DGSFused.cu:166` 등 투영 커널 전부에서 호출된다. 즉 **정규화는 forward의
가장 안쪽, 회전행렬을 만드는 순간에 딱 한 번** 일어난다.

파이썬 참조 구현(`gsplat/cuda/_math.py:655` `_quat_to_rotmat`)도 같은 자리에서
`quats = F.normalize(quats, p=2, dim=-1)`을 한다 — CUDA와 참조 구현의 계약이 일치한다.

## 3. 왜 미리 정규화하지 않는가

### (a) 제약 없는 4차원 공간에서 Adam을 그냥 돌릴 수 있다

단위 quaternion은 S³(3차원 다양체)이다. 이걸 제대로 최적화하려면 리만 경사법이나
매 스텝 후 재투영(re-normalization)이 필요하다. gsplat은 대신 **4개 성분을 그냥
자유 파라미터로 두고**, 정규화를 forward 안으로 밀어 넣어 문제를 없앤다.
학습 루프에는 `quats`를 다시 정규화하는 후처리 스텝이 아예 없다.

### (b) backward가 정규화의 야코비안을 정확히 포함한다

`Utils.cuh:253` `quat_to_rotmat_vjp()`의 마지막 줄:

```cpp
vec4 quat_n = vec4(w, x, y, z);                                   // 정규화된 q
v_quat += (v_quat_n - glm::dot(v_quat_n, quat_n) * quat_n) * inv_norm;
```

`g - (g·q̂)q̂`는 gradient를 **구면 접평면으로 투영**하는 연산이다(반경 방향 성분 제거).
결과적으로:

- 노름을 키우거나 줄이는 방향의 gradient는 0이다 → `|q|`는 회전에 아무 영향이 없고,
  gradient도 노름을 밀지 않으므로 학습 중 노름이 거의 그대로 유지된다.
- 전체가 `* inv_norm`으로 스케일된다 → **유효 학습률이 `1/|q|`에 비례**한다.
  노름이 큰 Gaussian은 회전이 더 느리게 갱신된다.

정규화를 파이썬에서 미리 해버려도 autograd가 같은 야코비안을 만들어 주므로 수치적으로는
등가지만, 커널 안에서 처리하면 `[N,4]` 중간 텐서 하나와 그 backward 그래프가 없어진다.

### (c) 옵티마이저 상태가 흔들리지 않는다

`param.data`를 매 스텝 재정규화하면 파라미터가 Adam의 1·2차 모멘트와 어긋난 값으로
점프한다. 미정규화로 두면 그런 불일치가 없다.

### (d) 노름 방향 자유도는 그냥 남아 있다

`|q|`는 gauge(무의미한 자유도)다. 최적화가 이 방향으로 표류해도 렌더 결과는 불변이고,
`q`와 `-q`가 같은 회전을 나타내는 이중 덮개 문제도 자동으로 흡수된다.

## 4. 초기값이 `torch.rand`인데 왜 괜찮은가

`torch.rand((N, 4))`는 각 성분이 `[0, 1)` 균등이다. 이건 **단위 구면 위 균등분포가 아니다**
(성분이 모두 양수이므로 4차원 초구의 한 옥탄트에 몰려 있고, `w > 0` 쪽으로 치우친다).
그런데도 문제가 없는 이유는 **초기 scale이 등방(isotropic)**이기 때문이다:

```python
dist_avg = knn_mean_dist(points, k=3)
scales = torch.log(dist_avg)[:, None].repeat(1, 3)   # 세 축이 모두 같은 값
```

세 축의 scale이 같으면 `S = s·I`이므로

$$\Sigma = R\,S S^\top R^\top = s^2 R R^\top = s^2 I$$

**회전 `R`이 수식에서 완전히 사라진다.** 초기 Gaussian은 완벽한 구이고, 구는 어떻게
돌려도 같은 구다. 그래서 초기 회전값은 무엇이든(랜덤이든 항등이든) 렌더 결과가 동일하다.
`scales`가 비등방으로 갈라지기 시작한 뒤부터 `quats`의 gradient가 의미를 갖는다.
`quats_lr = 1e-3`이 다른 파라미터보다 작은 것도 이 순서와 맞는다.

참고로 진짜 균등한 랜덤 회전이 필요한 곳에서는 다르게 쓴다 —
`gsplat/_helper.py:99`는 `F.normalize(torch.randn((N, 4)), dim=-1)`를 사용한다
(가우시안을 정규화해야 구면 균등분포가 된다).

## 5. 예외 — 명시적으로 `F.normalize`가 필요한 곳

"내부에서 정규화한다"는 계약은 **gsplat 자체 rasterization 경로에만** 적용된다.
파이썬에서 회전행렬을 직접 써야 하거나 외부 커널로 넘길 때는 직접 정규화해야 한다.

| 위치 | 코드 | 이유 |
|---|---|---|
| `gsplat/strategy/ops.py:197` | `quats = F.normalize(params["quats"][sel], dim=-1)` | split에서 자식 Gaussian 오프셋을 뽑으려면 파이썬에서 `rotmats`가 필요 |
| `gsplat/rendering.py:1283` | `quats = F.normalize(quats, dim=-1)` | `rasterization_inria_wrapper` — 주석: "rasterization from inria does **not** do normalization internally" |
| `gsplat/rendering.py:1608` | 같음 | `rasterization_2dgs_inria_wrapper` — 동일한 이유 |
| `gsplat/compression/png_compression.py:100` | `splats["quats"] = F.normalize(...)` | 양자화 전에 값 범위를 고정해야 압축이 됨 |
| `gsplat/scene/.../gaussian_inference_scene.py:325` | `quats = F.normalize(splats["quats"], dim=-1)` | 추론 씬은 unit-norm을 요구하고 **검증까지 한다** |

마지막 항목의 검증 코드가 계약 경계를 잘 보여준다:

```python
norms = quats.norm(dim=-1)
bad_norms = (norms - 1).abs() > 1e-3
if bad_norms.any():
    raise ValueError(f"quats are not unit-norm ...; did you forget F.normalize(quats, dim=-1)?")
```

즉 **학습 경로는 미정규화, 추론/압축/직렬화 경로는 정규화**라는 두 세계가 있다.
`gsplat/exporter.py:455`의 docstring도 PLY에서 읽어온 값이
"`quats`: (N, 4) quaternions **as stored (unnormalized)**"라고 명시한다.

## 6. 흔한 함정

- **밀도화 때 새 Gaussian의 quats는 정규화되지 않은 채 복사된다.**
  `ops.py`의 `split`은 `means`/`scales`/`opacities`만 특별 처리하고 나머지는
  `p[sel].repeat(repeats)`로 그대로 복제한다. `quats`도 그 나머지에 들어가므로
  부모의 미정규화 노름이 그대로 상속된다 — 의도된 동작이다(노름은 무의미하므로).
- **`|q| = 0`은 보호되지 않는다.** `rsqrt_parse_safe`의 "safe"는 host/device 분기용
  이름일 뿐(`__CUDA_ARCH__` 유무에 따라 `rsqrt` / `1/std::sqrt`), 0 입력을 막지 않는다.
  `torch.rand` 초기화 + 접평면 gradient 조합에서는 실질적으로 발생하지 않는다.
- **체크포인트를 다른 뷰어에 넣을 때** 노름이 1이 아니라 놀랄 수 있다. 정상이다.
  INRIA 원본 포맷과 호환시키려면 `F.normalize`를 한 번 통과시켜라.
- **`quats` 값 자체를 로깅해 회전이 학습되는지 보려 할 때**, 초기 구간에서는 등방 scale
  때문에 gradient가 사실상 0에 가깝다. `scales`가 갈라진 뒤를 봐야 한다.

## 7. 자기 점검

1. `splats["quats"]`의 노름을 학습 1000 스텝 뒤에 재면 1에 가까울까? (→ 초기 노름
   근처에 머문다. 1이 아니고, gradient가 노름을 밀지 않으므로 크게 변하지도 않는다)
2. 학습 루프의 `rasterize_splats`에서 `quats`에 `F.normalize`를 추가하면 결과가 달라질까?
   (→ 렌더 결과는 같다. 다만 유효 학습률의 `1/|q|` 스케일이 사라져 갱신 속도가 미묘하게
   달라지고, 중간 텐서 하나가 추가된다)
3. 초기 `quats`를 전부 `[1,0,0,0]`(항등 회전)으로 바꾸면 첫 렌더 이미지가 달라질까?
   (→ 달라지지 않는다. 초기 scale이 등방이라 `Σ = s²I`로 `R`이 소거된다)
