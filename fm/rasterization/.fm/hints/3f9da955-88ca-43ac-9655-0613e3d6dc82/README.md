# `_persp_proj`의 FOV clamp 트릭

> **Q.** 참조 구현 `_persp_proj`가 Jacobian 계산 전에 하는 clamp 트릭은?
> **A.** x/z와 y/z를 시야각의 1.3배 안으로 clamp한다. 화면 밖 멀리 있는 Gaussian의 Jacobian이 폭주하는 것을 막는 Inria 원본의 트릭이다.

---

## 1. 위치: 파이프라인 어디에 있나

EWA splatting 3단계(walkthrough 3장 "②③ 카메라 변환 + 원근 투영")에서 2D 공분산은

$$\Sigma_{2D} = J\,\Sigma_c\,J^\top + \epsilon I,\qquad
J = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix}$$

로 만들어진다. clamp는 **이 J를 조립하기 직전** 한 줄로 들어간다. 즉 카메라 좌표 변환($\mu_c = R\mu+t$)과 J 계산 사이가 유일한 삽입 지점이다.

---

## 2. 정확한 코드

### gsplat — PyTorch 참조 구현

`/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/_torch_impl.py` (`_persp_proj`, L88–L103):

```python
tan_fovx = 0.5 * width / fx            # [..., C, 1]
tan_fovy = 0.5 * height / fy

lim_x_pos = (width  - cx) / fx + 0.3 * tan_fovx
lim_x_neg =           cx  / fx + 0.3 * tan_fovx
lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
lim_y_neg =           cy  / fy + 0.3 * tan_fovy

tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)   # ← 핵심
ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

O = torch.zeros(...)
J = torch.stack([fx / tz, O, -fx * tx / tz2,
                 O, fy / tz, -fy * ty / tz2], dim=-1).reshape(..., 2, 3)

cov2d   = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
means2d = torch.einsum("...ij,...nj->...ni", Ks[..., :2, :3], means)
means2d = means2d / tz[..., None]      # ← 원본 means를 씀. clamp된 tx/ty가 아니다!
```

### gsplat — CUDA 커널

`/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/include/Utils.cuh` (`persp_proj`, L586–L594) — 완전히 동일한 식:

```cpp
float lim_x_pos = (width  - cx) / fx + 0.3f * tan_fovx;
float lim_x_neg =           cx  / fx + 0.3f * tan_fovx;
float rz = 1.f / z;
float tx = z * glm::min(lim_x_pos, glm::max(-lim_x_neg, x * rz));
float ty = z * glm::min(lim_y_pos, glm::max(-lim_y_neg, y * rz));
...
mat3x2 J = mat3x2(fx*rz, 0.f,  0.f, fy*rz,  -fx*tx*rz2, -fy*ty*rz2);
cov2d  = J * cov3d * glm::transpose(J);
mean2d = vec2({fx * x * rz + cx, fy * y * rz + cy});   // 원본 x, y
```

이 헬퍼는 `ProjectionEWA3DGSFused.cu`, `ProjectionEWA3DGSPacked.cu`, `ProjectionEWASimple.cu` 세 커널이 공유한다.

### Inria 원본 (diff-gaussian-rasterization `forward.cu`, `computeCov2D`)

```cpp
const float limx = 1.3f * tan_fovx;
const float limy = 1.3f * tan_fovy;
const float txtz = t.x / t.z;
const float tytz = t.y / t.z;
t.x = min(limx, max(-limx, txtz)) * t.z;
t.y = min(limy, max(-limy, tytz)) * t.z;
```

여기서 `tan_fovx = tan(fovx * 0.5)`이고, 대칭 카메라라면 `= 0.5 * W / fx`다.

---

## 3. gsplat의 개선점 — 비대칭 principal point 대응

Inria는 **±1.3·tan_fov 하나의 대칭 구간**을 쓴다. 이건 principal point가 이미지 정중앙(`cx = W/2`)이라는 암묵적 가정이다. gsplat은 이걸 **좌/우(상/하)로 쪼갠다**:

| | 오른쪽 한계 | 왼쪽 한계 |
|---|---|---|
| 화면 경계의 실제 x/z | `(W - cx) / fx` | `-cx / fx` |
| + 여유 | `+ 0.3·tan_fovx` | `- 0.3·tan_fovx` |
| = gsplat | `lim_x_pos` | `-lim_x_neg` |

**Inria와 정확히 일치함을 확인** — `cx = W/2`이면

```
lim_x_pos = (W - W/2)/fx + 0.3·(0.5W/fx) = 0.5W/fx + 0.3·(0.5W/fx) = 1.3 · tan_fovx
```

즉 `0.3 · tan_fov`라는 여유항의 존재 이유가 여기서 드러난다. gsplat은 Inria의 `1.3`이라는 마법의 숫자를 **"화면 경계 + FOV의 15%(= 0.3 × 반각) 여유"**로 분해해서, 경계 부분만 실제 `cx`/`cy`로 바꿔 끼운 것이다.

toy 카메라(`W=64, H=48, fx=fy=60, cx=32, cy=24`)로 확인:

```
tan_fovx = 0.5333,  lim_x_pos = lim_x_neg = 0.6933 = 1.3 × 0.5333   ← Inria와 동일
```

principal point를 왼쪽으로 옮기면(`cx = 8`) 비로소 갈라진다:

```
cx =  8.0 -> lim_x_pos = 1.0933,  lim_x_neg = 0.2933   (Inria는 양쪽 다 0.6933)
```

`cx=8`이면 화면은 카메라 축 기준 **오른쪽으로 치우쳐** 있다. Inria 식이었다면 왼쪽으로 0.6933까지 허용해 필요 없는 영역을 열어주고, 오른쪽은 0.6933에서 잘려 **화면 안에 있는(x/z가 0.69~1.09인) Gaussian까지 clamp**되어 모양이 왜곡된다. gsplat 버전은 이 두 오류를 모두 없앤다. 크롭된 이미지, 스테레오 리그, 자율주행/멀티카메라 rig처럼 `cx ≠ W/2`인 실제 캘리브레이션에서 의미가 있다.

---

## 4. 왜 Jacobian이 폭주하나

J의 3열 항이 문제다:

$$J_{02} = -\frac{f_x x}{z^2} = -\frac{f_x}{z}\cdot\frac{x}{z}$$

**x/z에 정비례한다.** 그리고 $\Sigma_{2D} = J\Sigma_c J^\top$는 J에 대해 **2차**이므로, x/z가 k배가 되면 $\Sigma_{2D}$의 해당 성분은 대략 $k^2$배가 된다. radii는 `ceil(3.33·√diag(Σ₂D))`이니 반경은 **x/z에 선형으로** 커진다.

toy 카메라, z=5, 등방 σ=0.1인 Gaussian으로 계산한 실제 수치:

| x/z | J₀₂ (clamp 없음) | J₀₂ (clamp) | radii_x (clamp 없음) | radii_x (clamp) |
|---:|---:|---:|---:|---:|
| 0.50 (화면 안) | −6.0 | −6.0 | 5 | 5 |
| 0.69 (limit 위) | −8.28 | −8.28 | 6 | 6 |
| 2.0 | −24.0 | −8.32 | 10 | 6 |
| 10.0 | −120.0 | −8.32 | **41** | 6 |
| 50.0 | −600.0 | −8.32 | **200** | 6 |

이게 왜 재앙인지:

- **카메라 옆·뒤쪽 Gaussian**: 시야각을 크게 벗어나면 x/z가 수십~수백이 된다. 반경 200px짜리 축정렬 사각형은 16×16 타일 기준 **625개 타일**과 교차한다. `isect_tiles`가 만드는 (tile, gaussian) 페어 수가 이런 소수의 Gaussian 때문에 수십~수백 배로 부풀고, 정렬 비용과 intersection 버퍼 메모리가 폭발한다. 최악의 경우 OOM이다.
- **z → 0 근처**: 근평면 바로 앞의 Gaussian은 `fx/z`도 `x/z`도 동시에 커져 $J_{02} \propto 1/z^2$로 발산한다.

  ```
  z=1.00, x/z=3:  J₀₂ = −180      (clamp 후 −41.6)
  z=0.10, x/z=3:  J₀₂ = −1800     (clamp 후 −416)
  z=0.02, x/z=3:  J₀₂ = −9000     (clamp 후 −2080)
  ```

  clamp가 `1/z²` 발산 자체를 없애주지는 않는다(그건 near-plane 컬링의 몫이다). 다만 **x/z 방향의 배수를 최대 `lim` 배로 묶어** 폭주의 한 축을 잘라낸다.
- 게다가 fp32에서 $J\Sigma J^\top$의 원소가 $10^6$ 규모가 되면 det 계산과 conic 역행렬에서 정밀도가 무너져 NaN이 나올 수 있다.

---

## 5. 핵심 — clamp되는 건 μ가 아니라 "J에 들어가는 x/z"뿐

가장 자주 오해하는 지점이다. 위 코드에서 `tx`, `ty`는 **재대입된 로컬 변수**이고, 이후 J를 조립하는 데만 쓰인다. 투영 위치는

```python
means2d = Ks[..., :2, :3] @ means / tz     # ← clamp 안 된 원본 means
```

로 **원본 `means`**에서 계산된다. CUDA 쪽도 `mean2d = {fx * x * rz + cx, ...}`로 원본 `x, y`를 쓴다.

따라서:

- **Gaussian이 화면 쪽으로 끌려오지 않는다.** means2d는 여전히 화면 밖 좌표(예: x=−3000px)를 가리키므로, 뒤에 오는 `inside` 컬링이 정상 동작한다.
- 바뀌는 건 오직 **"어느 방향에서 본 것처럼 찌그러뜨릴지"**라는 모양(공분산)뿐이다. 화면 밖 Gaussian의 2D 모양은 어차피 안 보이므로 부정확해도 무해하다.
- 만약 μ까지 clamp했다면 화면 밖 Gaussian이 전부 화면 가장자리에 몰려 렌더링되는 심각한 버그가 된다.

한 줄 요약: **clamp는 위치가 아니라 Jacobian의 "시점 기울기"를 제한한다.**

---

## 6. `1.3`이라는 배수의 의미

왜 딱 화면 경계(`1.0×`)에서 자르지 않는가?

원근 투영의 1차 근사는 μ 위치에서 접평면을 잡는 것이다. Gaussian은 점이 아니라 퍼져 있어서, **중심이 화면 경계 바로 안에 있어도 꼬리는 경계 밖까지 뻗는다.** 만약 `1.0×`에서 clamp하면 화면 가장자리에 걸친(그리고 실제로 픽셀에 기여하는) Gaussian들의 J가 잘려서, 모양이 눈에 띄게 왜곡되고 화면 테두리를 따라 아티팩트 띠가 생긴다.

`0.3 · tan_fov` (= FOV 반각의 30%, 전체 FOV의 15%)의 여유는 **"기여하는 Gaussian은 전부 clamp 밖에 두고, 확실히 안 보이는 것만 자른다"**는 안전 마진이다. 정확도(clamp를 늦게)와 폭주 억제(clamp를 일찍) 사이의 실용적 타협일 뿐, 유도된 값은 아니다. Inria가 경험적으로 고른 상수를 gsplat이 그대로 계승했다.

---

## 7. 순서 논리 — "어차피 컬링될 텐데 왜 막나"

당연한 반문이다. 화면 밖 Gaussian은 아래 컬링에서 걸러진다 (`_torch_impl.py` `_fully_fused_projection`):

```python
valid = (depths > near_plane) & (depths < far_plane)
radius[~valid] = 0.0

inside = ((means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < width)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < height))
radius[~inside] = 0.0
```

문제는 **`inside` 판정 자체가 `radius`에 의존한다**는 것이다. 순서가 이렇게 잠긴다:

```
Σ_c ─► J ─► Σ_2D ─► radii ─► inside 판정 ─► 컬링
        ↑                       ↑
     여기서 폭주            radii를 필요로 함
```

컬링을 하려면 radii가 있어야 하고, radii를 얻으려면 이미 폭주한 Σ₂D를 계산해야 한다. 게다가 radii가 200px로 부풀면 `means2d - radius < width` 조건이 통과해버려서, **원래 컬링되었어야 할 Gaussian이 "화면에 걸친다"고 판정되어 살아남는다.** 즉 clamp 없이는 컬링이 헐거워지고, 살아남은 거대 사각형이 그대로 `isect_tiles`로 흘러간다.

그래서 **막는 것이 먼저, 컬링은 나중**이다. 원본 논문 구현이 clamp를 J 조립 직전에 둔 이유가 이것이다.

(참고: near-plane 컬링 `depths > near_plane`은 radii와 무관하므로 카메라 **정확히 뒤쪽**(z ≤ near)은 clamp 없이도 안전하게 잡힌다. walkthrough의 toy 씬에서 `z=-2` Gaussian이 `radii=0`으로 컬링되는 게 그 경우다. clamp가 진짜로 필요한 건 **z는 양수인데 옆으로 크게 벗어난** Gaussian이다.)

---

## 8. 부작용 — 화면 밖 Gaussian의 gradient

`clamp`는 포화 구간에서 도함수가 0이다. gsplat의 backward는 이걸 **명시적으로** 분기 처리한다 (`Utils.cuh` `persp_proj_vjp`, L669–L685):

```cpp
// fov clipping
if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
    v_mean3d.x += -fx * rz2 * v_J[2][0];        // 정상: ∂J₀₂/∂x = -fx/z²
} else {
    v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;   // 포화: x로 가는 경로가 통째로 사라지고
}                                               //       tx = z·lim 이므로 z로 재라우팅
if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
    v_mean3d.y += -fy * rz2 * v_J[2][1];
} else {
    v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
}
v_mean3d.z += -fx*rz2*v_J[0][0] - fy*rz2*v_J[1][1]
            + 2.f*fx*tx*rz3*v_J[2][0] + 2.f*fy*ty*rz3*v_J[2][1];
```

읽는 법:

- **포화된 경우 `∂Σ₂D/∂x`는 정확히 0이다.** clamp 밖 Gaussian은 "2D 모양이 이상하다"는 신호로 **옆으로 움직일 수 없다**.
- 대신 `tx = z · lim`이 되어 tx가 z의 함수가 되므로, 그 미분이 `v_mean3d.z`로 흘러간다(`else` 가지). 마지막 무조건 항의 `+2·fx·tx·rz3` 와 합쳐져 순 효과는 `+fx·tx·rz3·v_J[2][0]`이다. 즉 **기울기가 사라지는 게 아니라 x축에서 z축으로 옮겨간다.**
- x, y로 가는 gradient가 완전히 0이 되는 건 아니다. `means2d` 경로(`v_mean3d += (fx·rz·v_mean2d.x, fy·rz·v_mean2d.y, ...)`)는 clamp를 거치지 않으므로 살아 있다.

실무적 영향은 **거의 없다**. 이유:

1. clamp가 걸릴 정도로 벗어난 Gaussian은 §7의 컬링에서 `radii = 0`이 되어 어떤 타일에도 들어가지 않는다. 픽셀 손실에 기여하지 않으니 `v_cov2d`, `v_mean2d` 자체가 0이다. 즉 대부분의 경우 gradient가 0인 것은 clamp 때문이 아니라 **애초에 안 보이기 때문**이다.
2. clamp 경계(`0.3·tan_fov` 여유) 안쪽, 즉 화면에 실제로 기여하는 영역에서는 clamp가 걸리지 않아 gradient가 온전하다.

그래도 남는 이론적 흠은 있다: **여러 카메라를 동시에 학습할 때, 어떤 뷰에서 화면 밖인 Gaussian은 그 뷰로부터 모양 gradient를 받지 못한다.** 그 Gaussian이 다른 뷰에서는 보인다면 그쪽 gradient로 학습되므로 실전에서 문제가 되지 않는다. 3DGS의 densification/pruning이 결국 안 보이는 Gaussian을 정리하기도 한다.

---

## 9. walkthrough의 `project_manually`와의 차이

노트북의 교육용 구현은 clamp를 **일부러 생략**했다:

```python
J = torch.stack([fx / z, O, -fx * x / z**2, O, fy / z, -fy * y / z**2], -1).reshape(-1, 2, 3)
```

그래서 노트북에도 다음 주석이 붙어 있다:

> 참고: 참조 구현 `_persp_proj`는 J를 만들기 전에 x/z, y/z를 시야각의 1.3배 안으로 clamp한다(화면 밖 멀리 있는 Gaussian의 Jacobian이 폭주하는 것을 막는 Inria 원본의 트릭). **장난감 씬은 모두 시야 안이라 결과가 같다.**

manual / `_fully_fused_projection` / CUDA 세 구현의 `maxdiff`가 0으로 맞아떨어지는 건 toy 씬의 모든 Gaussian이 clamp 구간 **안**에 있기 때문이다. 큰 실제 씬(수백만 Gaussian, 대부분이 시야 밖)에서는 이 두 구현이 갈라지고, clamp 없는 버전은 메모리에서 터진다.

---

## 10. 30초 요약

| 항목 | 내용 |
|---|---|
| 무엇을 | `x/z`, `y/z`를 `[-lim_neg, +lim_pos]`로 clamp |
| 언제 | Jacobian J를 조립하기 **직전** |
| 얼마나 | 대칭 카메라 기준 `±1.3 · tan(fov/2)` = 화면 경계 + FOV 반각의 30% 여유 |
| 무엇에 영향 | `Σ_2D = JΣJᵀ` **만**. `means2d`(투영 위치)는 원본 값 유지 |
| 왜 | `J₀₂ = -f_x·(x/z)/z`가 x/z에 비례 → Σ₂D는 그 제곱 → radii 폭발 → 타일 교차 수 폭발 (메모리/속도) |
| 순서 이유 | `inside` 컬링이 radii에 의존 → 폭주한 radii는 컬링을 통과해버림 → 먼저 막아야 함 |
| gsplat 개선 | Inria의 대칭 `±1.3·tan_fov`를 `cx`/`cy` 기반 비대칭 `lim_pos`/`lim_neg`로 분해 (`cx = W/2`면 원본과 동일) |
| 부작용 | 포화 구간에서 `∂Σ₂D/∂x = 0` (gradient가 z축으로 재라우팅). 어차피 컬링되는 Gaussian이라 실무 영향 미미 |

---

## 참고 파일

- `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/_torch_impl.py` — `_persp_proj` (L53–L209), `_fully_fused_projection` 컬링 (L341–L348)
- `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/include/Utils.cuh` — `persp_proj` (L567–L606), `persp_proj_vjp` (L608–L689)
- `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu` — 호출부 (L141)
- `/home/sungwoo/projects/swcho/gsplat/fm/rasterization/.fm/assets/rasterization_walkthrough.py` — 3장 `project_manually` (L233~) 및 clamp 주석 (L275)
- Inria 원본: `diff-gaussian-rasterization/cuda_rasterizer/forward.cu` → `computeCov2D`
