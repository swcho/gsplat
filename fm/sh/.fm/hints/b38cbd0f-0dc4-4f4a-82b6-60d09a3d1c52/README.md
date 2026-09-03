# 3DGS의 SH DC 계수 초기화 `sh0 = (rgb − 0.5) / C0` 유도

## 한 줄 요약

3DGS의 색 공식은 "SH 급수의 합 + 0.5"이다. 학습 시작 시 고차 계수를 전부 0으로 두면 합에는 상수항 $c_0 Y_0^0$ 하나만 남으므로, 색은 방향과 무관하게 $c_0 Y_0^0 + 0.5$가 된다. 이 값이 SfM 포인트의 색 $\mathbf{rgb}$와 같아야 한다는 조건을 $c_0$에 대해 풀면 $c_0 = (\mathbf{rgb} - 0.5)/Y_0^0 = (\mathbf{rgb} - 0.5)/C_0$ 이다. 여기서 $C_0 = Y_0^0 = \frac{1}{2\sqrt{\pi}} \approx 0.28209479$.

## 1. 출발점: 3DGS의 색 공식

Gaussian $i$의 색은 시점 방향 $\mathbf d$ (카메라 위치에서 Gaussian 중심을 향하는 단위 벡터)의 함수로 다음과 같이 정의된다.

$$
\mathbf c_i(\mathbf d) = \max\!\Big(0,\ \sum_{k=0}^{15} \mathbf c_{i,k}\,Y_k(\mathbf d) + 0.5\Big)
$$

- $Y_k$: 3차까지의 실수형 Spherical Harmonics 기저 16개 ($k = \ell^2 + \ell + m$ 순서).
- $\mathbf c_{i,k} \in \mathbb R^3$: 학습 대상인 SH 계수 (RGB 채널마다 하나씩, 총 $16\times3 = 48$개).
- $+0.5$: 계수가 전부 0일 때 색이 검정(0)이 아니라 중간 회색(0.5)이 되게 하는 오프셋. 계수가 양·음 대칭으로 움직이면서 색을 어둡게도 밝게도 만들 수 있게 해 준다.
- $\max(0,\cdot)$: 음수 색을 잘라낸다 (gsplat `rendering.py`의 `_maybe_evaluate_sh`에서 `+0.5` 후 `clamp_min(0)`).

## 2. 유도 과정 (단계별)

**Step 1 — 초기 상태의 가정.** 학습 시작 시 고차 계수 $\mathbf c_{i,k}$ ($k \ge 1$, 즉 $\ell \ge 1$)를 모두 0으로 둔다. 시점 의존성(광택, 프레넬 등)은 아직 아무것도 모르므로 "모든 방향에서 같은 색"으로 시작하는 것이 자연스럽다.

**Step 2 — 급수가 상수항 하나로 줄어든다.**

$$
\sum_{k=0}^{15} \mathbf c_{i,k}\,Y_k(\mathbf d) = \mathbf c_{i,0}\,Y_0^0(\mathbf d) + \underbrace{\sum_{k=1}^{15} \mathbf 0 \cdot Y_k(\mathbf d)}_{=\,0} = \mathbf c_{i,0}\,Y_0^0
$$

**Step 3 — $Y_0^0$은 상수다.** 0차 SH는 정의상 방향에 의존하지 않는다.

$$
Y_0^0(\mathbf d) = \frac{1}{2\sqrt\pi} \approx 0.28209479 \equiv C_0
$$

따라서 초기 색은 모든 방향 $\mathbf d$에 대해

$$
\mathbf c_i(\mathbf d) = \mathbf c_{i,0}\,C_0 + 0.5 \qquad (\text{방향 무관})
$$

(0 ≤ rgb ≤ 1 범위라면 이 값은 음수가 될 수 없으므로 $\max(0,\cdot)$은 아무 일도 하지 않는다.)

**Step 4 — SfM 색과 일치시키는 조건.** 이 초기 색이 그 Gaussian의 씨앗이 된 SfM 포인트의 색 $\mathbf{rgb}$와 같아야 한다.

$$
\mathbf c_{i,0}\,C_0 + 0.5 = \mathbf{rgb}
$$

**Step 5 — $\mathbf c_{i,0}$에 대해 푼다.** 양변에서 0.5를 빼고 $C_0$로 나누면

$$
\boxed{\ \mathbf c_{i,0} = \frac{\mathbf{rgb} - 0.5}{C_0}\ }
$$

이것이 `sh0 = (rgb - 0.5) / C0`이다. 채널별로 독립이므로 R, G, B 각각에 같은 식이 적용된다.

## 3. "DC 계수"라는 관점에서 다시 보기

DC(direct current)는 신호처리의 "주파수 0 성분"이다. SH에서는 $\ell = 0$ 항이고, $Y_0^0$이 상수이므로

$$
c_0^0\,Y_0^0 = \frac{1}{4\pi}\int_{S^2} f(\mathbf d)\,d\Omega = \overline f
$$

즉 **DC 항만으로 복원한 값은 함수의 구면 평균**이다. 고차 항($\ell \ge 1$)은 모두 $Y_0^0$과 직교하므로 구면 평균이 0인 "변동"만 담는다. 따라서 초기화 식은 "이 Gaussian의 평균 색을 $\mathbf{rgb}$로 놓고 시점 의존 변동은 0으로 시작한다"는 뜻이기도 하다. 학습이 진행되면 `shN`이 그 위에 방향별 변동을 덧입힌다.

## 4. 실제 코드

### gsplat (`examples/utils.py`)

```python
def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0
```

### 원본 3DGS (`utils/sh_utils.py`)

```python
C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5
```

`SH2RGB`는 `RGB2SH`의 정확한 역함수다: $\text{SH2RGB}(\text{RGB2SH}(x)) = \frac{x-0.5}{C_0}\cdot C_0 + 0.5 = x$.

### gsplat 초기화 (`examples/simple_trainer.py`, `create_splats_with_optimizers`)

```python
if init_type == "sfm" or init_type == "lidar":
    points = torch.from_numpy(parser.points).float()
    rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()   # uint8 → [0, 1]
...
colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))               # [N, 16, 3], 전부 0
colors[:, 0, :] = rgb_to_sh(rgbs)                                # DC(k=0)만 채움
params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))          # 2.5e-3
params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))          # 2.5e-3 / 20
```

`colors`를 0으로 만든 뒤 0번 슬롯만 `rgb_to_sh`로 덮어쓴다 — 위 유도의 Step 1(고차 = 0)과 Step 5가 코드 두 줄에 그대로 대응한다.

## 5. SfM(COLMAP) 포인트 색은 어디서 오는가

COLMAP은 여러 이미지에서 특징점을 매칭해 3D 점을 삼각측량하고, 각 3D 점(`points3D.bin`/`.txt`)에 위치 `xyz`, 재투영 오차 `error`, 색 `rgb`를 기록한다. 이 색은 그 점을 관측한 이미지들의 해당 픽셀 색을 평균한 **0~255 uint8** 값이다. gsplat의 COLMAP 파서(`examples/datasets/colmap.py`)는 이를 그대로 읽어 온다.

```python
points_rgb = np.array(
    [points3D[point3D_id].color for point3D_id in point3D_ids],
    dtype=np.uint8,
).reshape(-1, 3)
```

이것이 `parser.points_rgb`이고, 초기화 시 `/ 255.0`으로 나눠 사용한다.

## 6. 왜 rgb를 0~1로 정규화한 뒤 쓰는가

- 색 공식의 $+0.5$ 오프셋과 $\max(0,\cdot)$ 클램프, 그리고 학습 시 비교 대상인 GT 픽셀(`pixels = data["image"] / 255.0`)이 모두 **0~1 스케일**을 전제한다. 렌더 출력과 GT가 같은 스케일이어야 손실이 의미가 있다.
- 만약 0~255 값을 그대로 넣으면 $(200 - 0.5)/0.282 \approx 707$ 같은 거대한 계수가 되고, 렌더링 결과도 0~1을 훌쩍 넘어서 첫 스텝부터 손실이 폭발한다.
- 0~1 입력에 대해 `sh0`은 $[-1.7725,\ 1.7725]$ 범위($\pm 0.5/C_0$)에 머물러, 다른 파라미터(예: 위치, 스케일의 log)와 비슷한 크기의 수치가 되어 Adam 학습률(`sh0_lr = 2.5e-3`) 설계와도 맞는다.

## 7. 수치 예: 초기 색이 정확히 rgb로 복원된다

| rgb (채널값) | sh0 = (rgb − 0.5)/C0 | 복원 sh0·C0 + 0.5 |
|---|---|---|
| 0.8 | 1.0635 | 1.0635 × 0.28209 + 0.5 = **0.8** |
| 0.3 | −0.7090 | **0.3** |
| 0.1 | −1.4180 | **0.1** |
| 0.5 | 0.0 | **0.5** (회색은 계수 0) |
| 0.0 | −1.7725 | **0.0** |
| 1.0 | +1.7725 | **1.0** |

노트북(`sh_walkthrough.py` §3)에서는 `rgb = [[0.8, 0.3, 0.1], [0.1, 0.6, 0.9]]`를 `sh0`으로 바꾸고, 고차 15개를 0으로 채운 뒤 **임의의 방향**에서 `sh_bases(dir, 3) · coeffs + 0.5`를 평가해 원래 rgb가 그대로 나오는 것을 확인한다. 방향이 무엇이든 고차 기저값에 0이 곱해지므로 결과는 변하지 않는다.

## 8. 초기화가 학습 속도에 미치는 영향

- SfM 색으로 초기화하면 첫 렌더가 이미 대략 맞는 색을 내므로, 초기 그래디언트는 **기하(위치·스케일·불투명도)** 를 고치는 데 쓰인다.
- 반대로 `sh0 = 0`(모든 Gaussian이 회색)이나 랜덤 색으로 시작하면, 첫 수백~수천 스텝 동안 손실의 대부분이 "색 맞추기"에서 나와 색 계수만 크게 움직이고 기하 학습이 뒤로 밀린다. `sh0_lr = 2.5e-3`인 Adam은 한 스텝에 계수를 대략 학습률 크기만큼만 움직이므로, 회색(0)에서 밝은 색(1.06)까지 가려면 수백 스텝이 걸린다.
- 또한 gsplat은 `shN_lr = sh0_lr / 20`으로 고차 계수를 훨씬 느리게 배우고, `sh_degree_interval = 1000`으로 처음 1000스텝은 DC만 사용한다. 이 두 장치는 모두 "기본색이 먼저 정확해야 시점 의존성을 안정적으로 배운다"는 전제 위에 있고, 그 기본색을 첫 스텝부터 맞춰 주는 것이 바로 이 초기화다.
- 부수 효과: 색이 크게 틀린 Gaussian은 그래디언트가 커져 densification(분할/복제) 기준에 걸리기 쉬운데, 색을 먼저 맞춰 두면 초기 densification이 기하 오류가 큰 곳에 집중된다.

## 9. PLY 파일의 `f_dc_*`가 바로 이 sh0이다

gsplat의 `export_splats`(`gsplat/exporter.py`)는 PLY 헤더를 다음처럼 쓴다.

```python
for i, data in enumerate([sh0, shN]):
    prefix = "f_dc" if i == 0 else "f_rest"
    for j in range(data.shape[1]):
        buffer.write(f"property float {prefix}_{j}\n".encode())
```

- `f_dc_0, f_dc_1, f_dc_2` = `sh0`의 R, G, B 계수 (**RGB 값이 아니라 SH 계수**). 초기 상태라면 정확히 $(\mathbf{rgb}-0.5)/C_0$이고, 학습 후에는 갱신된 값이다.
- `f_rest_0 … f_rest_44` = `shN` 15×3개.
- 뷰어(SuperSplat, antimatter15/splat, Polycam 등)는 이 값을 읽어 `SH2RGB`, 즉 `f_dc * C0 + 0.5`로 기본색을 복원한다. 그래서 PLY를 텍스트로 열어 `f_dc_0`이 1.06처럼 1을 넘거나 −1.4처럼 음수인 것을 보고 놀랄 필요가 없다 — 각각 0.8, 0.1의 색이다.
- `simple_trainer.py`는 appearance 모듈을 쓰는 경우(`feature_dim`이 있을 때)에도 최종 RGB를 `rgb_to_sh(rgb)`로 변환해 `sh0` 슬롯에 넣어 저장한다. PLY 규약이 "f_dc는 SH 계수"이기 때문이다.

## 정리

| 항목 | 내용 |
|---|---|
| 색 공식 | $\max(0, \sum_k \mathbf c_k Y_k(\mathbf d) + 0.5)$ |
| 초기 가정 | $\mathbf c_k = 0$ for $k \ge 1$ → 색 = $\mathbf c_0 C_0 + 0.5$ (방향 무관) |
| 조건 | 초기 색 = SfM 포인트 색 $\mathbf{rgb}$ (0~1) |
| 해 | $\mathbf c_0 = (\mathbf{rgb} - 0.5)/C_0$, $C_0 = 1/(2\sqrt\pi)$ |
| 코드 | `rgb_to_sh` / `RGB2SH`, 역함수 `SH2RGB(sh) = sh*C0 + 0.5` |
| 저장 | PLY `f_dc_0..2` = sh0 (SH 계수), 뷰어가 SH2RGB로 복원 |
