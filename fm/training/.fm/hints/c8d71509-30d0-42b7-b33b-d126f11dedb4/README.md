# SH 차수 스케줄 (SH degree warm-up)

> **Q.** 학습 루프에서의 SH 차수 스케줄은 무엇이며 왜 쓰는가?
> **A.** `sh_degree_to_use = min(step // 1000, SH_DEGREE)`로 1000스텝마다 한 차수씩 올린다. 처음엔 DC만 학습해 색부터 안정화시키는 효과가 있다.

---

## 1. 코드에서의 위치

워크스루(`training_walkthrough.py:365`), 학습 루프 안 forward 직전 한 줄이다.

```python
for step in pbar:
    ...
    # (2) SH 차수 스케줄
    sh_degree_to_use = min(step // 1000, SH_DEGREE)   # SH_DEGREE = 3

    # (3) forward
    renders, alphas, info = rasterize_splats(
        splats, camtoworlds, Ks, width, height, sh_degree=sh_degree_to_use
    )
```

원본은 `examples/simple_trainer.py:906`이고, 1000이라는 숫자는 하드코딩이 아니라 config 필드다.

```python
# examples/simple_trainer.py
sh_degree: int = 3            # 최대 차수
sh_degree_interval: int = 1000  # "Turn on another SH degree every this steps"
...
sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
```

`Config.adjust_steps(factor)`(`simple_trainer.py:268`)에서 `sh_degree_interval`도 `max_steps`와 **같은 배율로 함께 스케일**된다. 즉 "전체 학습의 앞 1/30 구간은 DC만" 이라는 비율이 유지되도록 설계돼 있다.

## 2. 그래서 실제로 무슨 일이 일어나는가

`SH_DEGREE = 3`, 30k 스텝 기준 타임라인:

| 스텝 구간 | `sh_degree_to_use` | 실제로 쓰이는 계수 `k` | 계수 개수 |
|---|---|---|---|
| 0 – 999 | 0 | `k=0` (DC) | 1 |
| 1000 – 1999 | 1 | `k=0..3` | 4 |
| 2000 – 2999 | 2 | `k=0..8` | 9 |
| 3000 – 30000 | 3 | `k=0..15` (전부) | 16 |

`sh_degree=d`를 넘기면 `rasterization()`은 `_maybe_evaluate_sh()`(`gsplat/rendering.py:693`)를 거쳐 `spherical_harmonics(d, means, viewmats, coeffs)`를 호출하고, 커널은 **`(d+1)^2`개 계수만 읽어** 색을 만든다. 나머지 고차 계수는 forward에 아예 참여하지 않는다.

- `d = 0` 구간의 색은 $c = C_0 \cdot \text{sh0} + 0.5$ (여기서 $C_0 = Y_0^0 = \frac{1}{2\sqrt{\pi}} = 0.2820948$). 방향 $\mathbf{d}$가 식에 없다 → **완전한 Lambertian(뷰 무관) 렌더**.
- $d \ge 1$부터 시선 방향이 들어오면서 뷰마다 색이 달라질 수 있게 된다.

## 3. 고차 계수는 "고정"되는 게 아니라 grad가 정확히 0이다

흔한 오해: "쓰지 않는 계수는 `requires_grad=False`로 얼려야 하는 거 아닌가?" — 필요 없다.

SH backward 커널(`gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu:786~`)은 `DEGREE`를 **템플릿 파라미터**로 받아 `MAX_K = (DEGREE+1)^2`개의 레지스터 누산기만 두고, 그 밖의 슬롯은 명시적으로 0을 쓴다.

```cpp
constexpr int MAX_K = (DEGREE + 1) * (DEGREE + 1);
std::array<opmath_t, MAX_K> acc{};
...
for (int k = 0; k < MAX_K; ++k)            // 활성 차수만 누적값 기록
    if (k < K) v_coeffs_ptr[k * D + c] = acc[k];
// Padded bases may allocate K > MAX_K; zero the tail so at::empty outputs are defined.
for (uint32_t k = MAX_K; k < K; ++k)       // 비활성 차수는 정확히 0
    v_coeffs_ptr[k * D + c] = 0.f;
```

여기에 두 가지 조건이 겹쳐 **비활성 계수가 진짜로 움직이지 않는다**:

1. `shN`은 초기화 시 **정확히 0**이다 (`init_splats_with_optimizers`: `colors`를 `zeros`로 만들고 `colors[:, 0, :]`에만 SfM 색을 넣음).
2. Adam은 weight decay 없이 쓰이고, grad가 계속 0이면 `exp_avg`/`exp_avg_sq`도 0에서 시작해 0으로 남는다 → 업데이트량 0.

결과적으로 스케줄은 "점진적 unfreezing"과 동일한 효과를 **한 줄의 `min()`으로** 얻는다.

## 4. 왜 이렇게 하는가 — 세 가지 이유

### (1) 색–기하 모호성 제거: 저주파부터 맞춘다

Gaussian 하나가 사진마다 다른 색을 낼 수 있으면, "이 Gaussian은 위치가 틀렸다"는 오차를 **위치를 고치는 대신 뷰별 색으로 설명해 버릴 수 있다**. 이건 재구성 손실을 줄이지만 기하는 전혀 개선하지 않는 지름길이다.

DC만 켜두면 그 지름길이 막힌다. 한 Gaussian의 색은 모든 뷰에서 하나뿐이라, 여러 뷰의 L1/SSIM을 동시에 줄이는 유일한 방법이 **means / scales / quats / opacities를 실제로 옳게 만드는 것**뿐이다. 스케줄의 본질은 "색을 안정화한다"보다 한 걸음 더 나아가 **초기 구간의 gradient 예산을 기하 쪽으로 강제 배분한다**에 가깝다.

### (2) 파라미터 수의 압도적 비대칭

Gaussian 하나의 파라미터를 세어 보면:

| 파라미터 | 개수 |
|---|---|
| `means` | 3 |
| `scales` | 3 |
| `quats` | 4 |
| `opacities` | 1 |
| `sh0` (DC) | 1×3 = 3 |
| `shN` (1~3차) | 15×3 = 45 |
| **합계** | **59** |

**48/59 ≈ 81%가 색 계수**이고, 그중 45개가 고차항이다. 처음부터 전부 풀어 두면 자유도 대부분이 뷰 의존 색에 쏠려 노이즈·플로터(floater)를 외우기 딱 좋다. 스케줄은 초기에 활성 파라미터를 59개 → 14개로 줄여 최적화 문제 자체를 쉽게 만든다.

### (3) 밀도화 신호의 품질

`DefaultStrategy`는 `info["means2d"]`의 **화면공간 gradient 크기**로 "이 Gaussian을 쪼갤/복제할까"를 판단한다(기본 `refine_start_iter=500`, `refine_every=100`). 이 신호는 "색이 안 맞는다"가 아니라 "위치가 안 맞는다"여야 의미가 있다. 고차 SH가 색 오차를 흡수해 버리면 means2d grad가 작아져 **정작 쪼개야 할 Gaussian이 안 쪼개진다**. 스케줄 덕분에 밀도화가 시작되는 500스텝 시점에는 여전히 DC-only라, refine이 깨끗한 기하 신호를 받는다.

부수적으로, 원 3DGS 논문(Kerbl et al. 2023)도 같은 이유로 "0차에서 시작해 1000 iteration마다 한 차수씩" 올린다고 명시한다. gsplat의 이 한 줄은 그 레시피를 그대로 옮긴 것이다.

## 5. 학습률과의 조합

스케줄만으로 끝이 아니고, lr도 고차항을 억누른다 (`init_splats_with_optimizers`):

```python
"sh0": 2.5e-3,
"shN": 2.5e-3 / 20,   # 고차 SH는 천천히
```

즉 **차수 게이팅(스케줄) + lr 1/20 감쇠**의 이중 안전장치다. 차수가 켜진 뒤에도 고차항은 DC보다 20배 느리게 움직이며 "잔여 뷰 의존 성분"만 조금씩 담당한다.

## 6. 워크스루 데모에서의 구체적 결과

데모는 `MAX_STEPS = 2_000`이므로 스케줄이 도달하는 차수는:

- 스텝 0–999 → `sh_degree_to_use = 0`
- 스텝 1000–1999 → `sh_degree_to_use = 1`
- **2차·3차는 한 번도 켜지지 않는다** → 해당 계수는 초기값 0 그대로.

그래서 진행바에 찍히는 `sh=0` → `sh=1` 전환이 1000스텝에서 정확히 한 번 보인다.

```python
pbar.set_description(f"loss={loss.item():.3f} | GS={len(splats['means']):,} | sh={sh_degree_to_use}")
```

한편 평가 함수 `eval_psnr()`은 스케줄을 무시하고 항상 최대 차수로 렌더한다.

```python
render, _, _ = rasterize_splats(..., sh_degree=SH_DEGREE)   # 3
```

이래도 결과가 달라지지 않는다 — 2·3차 계수가 정확히 0이라 SH 합에 0을 더하는 것과 같기 때문이다. (계수가 0이 아닐 때 학습/평가 차수를 다르게 두면 값이 달라지므로, 일반적으로는 평가에 최대 차수를 쓰는 게 맞다.)

## 7. 체크포인트

- `min(step // interval, sh_degree)`이므로 **`interval * sh_degree` 스텝 이후에는 스케줄이 상수**가 되어 아무 일도 하지 않는다 (30k 기준 3000스텝 이후, 즉 학습의 90%는 풀 차수).
- 차수를 낮추는 것은 계수를 **버리는 게 아니라 forward에서 읽지 않는 것**이다. 나중에 켜면 그때까지 누적된(=0인) 값에서 이어서 학습한다.
- `rasterization()`은 `(sh_degree + 1)**2 <= colors.shape[-2]`를 assert한다(`gsplat/rendering.py:823`). `colors`는 항상 `[N, 16, 3]` 전체를 넘기고 **차수만 인자로 제한**하는 구조라, 스텝마다 텐서를 슬라이싱할 필요가 없다.
- 밀도화(split/duplicate)로 Gaussian이 늘어날 때 `shN`도 함께 복제되지만, 비활성 구간에서는 값이 모두 0이므로 복제해도 영향이 없다.
- `sh_degree=None`으로 부르면 SH 평가 자체를 건너뛰고 `colors`를 이미 활성화된 RGB(또는 임의 N-D feature)로 취급한다 — 스케줄과는 별개의 경로다.

## 8. 한 문장 요약

> 뷰 의존 색이라는 강력한 자유도를 **처음부터 주지 않음으로써**, 초기 최적화가 색으로 오차를 덮는 대신 기하(위치·크기·회전·불투명도)를 맞추도록 강제하는 커리큘럼 학습이다.
