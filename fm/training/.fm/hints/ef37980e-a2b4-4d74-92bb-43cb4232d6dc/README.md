# prune 연산의 조건은?

## 한 줄 답

두 가지 OR 조건이다.

1. **투명함** — `sigmoid(opacities) < prune_opa`(기본 `0.005`)
2. **비대함** — `exp(scales).max(-1) > prune_scale3d * scene_scale`(기본 `0.1` → 10%·scene_scale)

앞은 "화면에 아무 기여도 못 하는 것", 뒤는 "혼자 씬을 뒤덮어 디테일을 가리는 것"을
치운다. duplicate/split이 **개수를 늘리는** 쪽이라면 prune은 **되돌리는** 쪽이고, 둘이
같은 refine 호출 안에서 연달아 돈다.

## 실제 코드

`gsplat/strategy/default.py`의 `DefaultStrategy._prune_gs()` — 전부 이 열 줄이다.

```python
def _prune_gs(self, params, optimizers, state, step, scene=None) -> int:
    is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
    if step > self.reset_every:
        is_too_big = (
            torch.exp(params["scales"]).max(dim=-1).values
            > self.prune_scale3d * state["scene_scale"]
        )
        # The official code also implements sreen-size pruning but
        # it's actually not being used due to a bug: ...issues/123
        if step < self.refine_scale2d_stop_iter:
            is_too_big |= state["radii"] > self.prune_scale2d
        is_prune = is_prune | is_too_big

    n_prune = is_prune.sum().item()
    if n_prune > 0:
        remove(params=params, optimizers=optimizers, state=state,
               mask=is_prune, scene=scene)
    return n_prune
```

관련 기본값(dataclass 필드):

| 필드 | 기본값 | 의미 |
|---|---|---|
| `prune_opa` | `0.005` | 이 **불투명도 미만**이면 제거 |
| `prune_scale3d` | `0.1` | 최대축 크기가 이 비율×`scene_scale` **초과**면 제거 |
| `prune_scale2d` | `0.15` | 화면상 반지름 비율 기준 (기본 비활성) |
| `refine_scale2d_stop_iter` | `0` | 0이면 2D 크기 기준 prune 자체가 꺼짐 |
| `reset_every` | `3000` | opacity 리셋 주기 = 크기 prune이 켜지는 시점 |

## 조건 1: `sigmoid(opacities) < 0.005`

핵심은 **`params["opacities"]`가 불투명도가 아니라 logit**이라는 점이다. 초기화도
`torch.logit(torch.full((N,), 0.1))`로 하고(워크스루 `init_splats_with_optimizers`),
`rasterization()` 내부에서 `sigmoid`를 씌워 쓴다. 그래서 비교 전에 `sigmoid`를 통과시킨다.
logit 공간으로 옮겨보면 임계값은

$$\mathrm{logit}(0.005)=\ln\frac{0.005}{0.995}\approx -5.293$$

즉 raw 파라미터가 `-5.29` 아래로 내려간 Gaussian이 제거 대상이다. 왜 하필 0.005인가 —
알파 블렌딩에서 $\alpha_i=o_i\cdot G_i(\mathbf{x})$ 이고 $G_i\le 1$ 이므로 불투명도 0.005짜리는
**중심 픽셀에서조차 최대 0.5% 기여**한다. 8비트 색으로 치면 1~2 LSB. 학습이
"이 Gaussian은 없는 게 낫다"고 판단해 opacity를 밀어내린 결과이니, 그 판단을 실제
메모리·렌더 비용 절감으로 확정해 주는 단계다.

### opacity reset과 짝을 이룬다

`step_post_backward()`의 마지막 블록:

```python
if step % self.reset_every == 0 and step > 0:
    reset_opa(params=params, optimizers=optimizers, state=state,
              value=self.prune_opa * 2.0)     # ← 0.01 = 0.005 × 2
```

리셋값이 상수 0.01이 아니라 **`prune_opa * 2.0`** 이라는 게 포인트다. 즉 3000스텝마다
모든 Gaussian의 불투명도를 "prune 임계값의 딱 두 배"로 눌러 놓고
(`reset_opa`는 정확히는 `torch.clamp(p, max=logit(0.01))` — 이미 더 낮은 것은 건드리지
않는 상한 clamp다), 이후 100스텝마다 도는 refine에서 **다시 올라오지 못한 것만** prune이
걷어낸다. 진짜 필요한 Gaussian은 gradient가 opacity를 곧 밀어올리고, 카메라 근처를 떠도는
floater는 그대로 0.005 아래에 머물다 사라진다. 이 "리셋 → 회복 못 하면 제거" 루프가
3DGS 논문의 floater 제거 메커니즘이고, prune 조건이 그 판정 기준 역할을 한다.

## 조건 2: `exp(scales).max(-1) > 0.1 * scene_scale`

`scales`도 **log 공간**이다(초기화가 `torch.log(dist_avg)`). 그래서 `exp`로 되돌리고,
`.max(dim=-1)`로 **세 축 중 가장 긴 축**을 본다 — 부피나 평균이 아니라 최대축이므로
"납작하지만 한 방향으로 길게 늘어난" 판자형 Gaussian도 걸린다.

`scene_scale`은 카메라 위치들의 중심에서 가장 먼 카메라까지의 거리다.

```python
# examples/datasets/colmap.py
camera_locations = camtoworlds[:, :3, 3]
scene_center = np.mean(camera_locations, axis=0)
dists = np.linalg.norm(camera_locations - scene_center, axis=1)
self.scene_scale = np.max(dists)
# 트레이너에서는 parser.scene_scale * 1.1 * global_scale 로 씀
```

임계값을 이 값에 비례시키는 이유는 3DGS의 모든 밀도화 상수가 **씬 크기에 무관해야**
하기 때문이다(같은 이유로 `means`의 lr도 `1.6e-4 * scene_scale`). 씬 반경의 10%를 한
Gaussian이 덮고 있다면 그건 형상을 표현하는 게 아니라 **배경을 뭉개는 얼룩**이다.
split(`grow_scale3d=0.01`, 1%)보다 한 자릿수 위에 걸려 있으니, 정상 흐름은
"커지면 split으로 쪼개진다 → 그래도 계속 자라 10%를 넘으면 포기하고 제거"다.

| 크기(scene_scale 대비) | grad 큼 | 처리 |
|---|---|---|
| ≤ 1% | 예 | **duplicate** (복제) |
| > 1% | 예 | **split** (2개로, 크기 ÷1.6) |
| > 10% | 무관 | **prune** (제거) |

### `step > reset_every` 가드 — 처음 3000스텝은 크기 조건이 꺼져 있다

`is_too_big` 계산이 `if step > self.reset_every:` 안에 들어 있다. 즉 refine이 시작되는
500스텝부터 3000스텝까지는 **opacity 조건만** 적용된다. 초기 Gaussian은 KNN 평균거리로
크기를 잡아 놓은 상태라 희소한 SfM 영역에서는 정당하게 클 수 있고, 아직 회전/크기가
자리를 못 잡았기 때문이다. 첫 opacity 리셋을 지나 학습이 안정된 뒤부터 크기 기준을 켠다.
원 3DGS 구현의 `if iteration > opt.opacity_reset_interval:` 과 같은 조건이다.

### 2D(화면) 크기 기준은 기본적으로 꺼져 있다

`prune_scale2d = 0.15`(화면 긴 변 대비 반지름)도 구현돼 있지만
`refine_scale2d_stop_iter`의 기본값이 `0`이라 `step < 0`이 성립할 수 없어 절대 켜지지
않는다. 주석대로 원 저자 코드에서도 버그로 실질 미사용이었던 규칙이라, gsplat은
"완전성을 위해 구현하되 기본 비활성"을 택했다. 켜려면 양수를 주면 되고, 그때는
`initialize_state()`가 `state["radii"]`를 함께 준비한다.

## 언제 불리나

`step_post_backward()` 안에서, **grow 다음에** 호출된다.

```python
if (step > self.refine_start_iter                      # > 500
    and step % self.refine_every == 0                  # 매 100스텝
    and step % self.reset_every >= self.pause_refine_after_reset):
    n_dupli, n_split = self._grow_gs(...)              # 먼저 늘리고
    n_prune = self._prune_gs(...)                       # 그 다음 걷어낸다
    state["grad2d"].zero_(); state["count"].zero_()     # 통계 리셋
    torch.cuda.empty_cache()
```

- 창(window): `refine_start_iter=500` 초과 ~ `refine_stop_iter=15000` 미만, `refine_every=100`
  마다. 15000스텝 이후에는 `step_post_backward()`가 맨 앞에서 `return`하므로 Gaussian 수가
  고정되고, 남은 스텝은 순수 최적화에만 쓰인다.
- grow → prune 순서라서 방금 split로 태어난 자식도 같은 호출에서 prune 심사를 받는다
  (부모가 이미 10%를 넘겼다면 자식은 ÷1.6로 줄어 살아남을 수 있다).
- `pause_refine_after_reset`(기본 0)을 이미지 수 정도로 주면 리셋 직후 몇 스텝은
  refine을 쉬어, 아직 회복할 기회를 못 받은 Gaussian이 억울하게 잘리는 걸 막는다.

## `remove()`가 실제로 하는 일

조건에 걸린 인덱스를 텐서에서 빼는 것으로 끝이 아니다 — **Adam의 모멘텀 상태와 러닝
통계까지 같이 잘라야** 인덱스 정렬이 유지된다.

```python
# gsplat/strategy/ops.py
sel = torch.where(~mask)[0]
param_fn     = lambda name, p: torch.nn.Parameter(p[sel], requires_grad=p.requires_grad)
optimizer_fn = lambda key, v: v[sel]                 # exp_avg / exp_avg_sq 도 함께
_update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
for k, v in state.items():                            # grad2d, count, radii
    if isinstance(v, torch.Tensor):
        state[k] = v[sel]
```

`means`/`scales`/`quats`/`opacities`/`sh0`/`shN` 여섯 파라미터 텐서가 모두 새 `Parameter`로
교체되므로, 옵티마이저를 파라미터별로 따로 들고 있는 gsplat의 구조
(`{name: Adam([...])}`)가 여기서 값을 한다. 밀도화가 파라미터 개수 N을 매번 바꾸는데도
학습이 이어지는 이유다.

## MCMCStrategy와의 대비

`MCMCStrategy`에는 이런 prune이 **없다**. 같은 `0.005`가 `min_opacity`라는 이름으로
있지만 쓰임이 다르다.

```python
# gsplat/strategy/mcmc.py
dead_mask = opacities <= self.min_opacity
relocate(..., mask=dead_mask, min_opacity=self.min_opacity)
```

죽은 Gaussian을 **지우는 대신 opacity가 높은 곳으로 순간이동(relocate)** 시킨다. MCMC는
`cap_max`로 총 개수 상한을 두고 그 예산을 재배치하는 SGLD 관점이라, "제거"가 아니라
"이주"가 된다. 반대로 `DefaultStrategy`는 개수를 자유롭게 늘리되 prune으로 줄이는
휴리스틱이다.

## 자주 나오는 오해

- ❌ "`opacities < 0.005`를 그대로 비교한다" → `sigmoid`를 먼저 씌운다. raw 값 기준으로는
  `-5.293` 아래다.
- ❌ "크기 조건은 부피나 평균 크기다" → **세 축 중 최대**(`max(dim=-1)`)다.
- ❌ "크기 조건이 500스텝부터 적용된다" → `step > reset_every`(3000) 이후부터다.
- ❌ "화면 크기(2D) prune도 기본으로 돈다" → `refine_scale2d_stop_iter=0`이라 꺼져 있다.
- ❌ "prune은 opacity reset과 무관한 별도 규칙이다" → 리셋값이 `prune_opa * 2.0`으로
  **prune 임계값에서 파생**된다. 두 연산은 한 쌍의 메커니즘이다.
- ❌ "prune이 gradient 신호를 본다" → 보지 않는다. grow만 `grad2d`를 쓰고, prune은
  현재 파라미터 값(opacity, scale)만 본다.

## 손실 곡선에서 보이는 흔적

워크스루의 학습 로그를 보면 `verbose=True`일 때 `Step 3000: N GSs pruned.`가 유독 크게
찍힌다. 직전에 opacity 리셋이 걸렸고, 같은 스텝에서 크기 조건까지 처음 켜지기 때문이다.
그 직후 loss가 한 번 튀었다가 회복하는 패턴이 나타나는데, 이는 리셋+대량 prune으로
장면이 잠깐 옅어졌다가 남은 Gaussian이 불투명도를 되찾는 과정이다.
