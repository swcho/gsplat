# 고차 SH(`shN`)의 학습률을 `sh0`의 1/20로 낮추는 이유

**Q.** 고차 SH(`shN`)의 학습률을 `sh0`의 1/20로 낮추는 이유는?

**A.** 고차 SH는 뷰 의존적 디테일을 담당하므로 천천히 학습시켜야 안정적이다. 색의 기본값(DC)이 먼저 잡히도록 하는 효과가 있다.

---

## 1. 코드에서의 위치

워크스루의 `init_splats_with_optimizers()`가 파라미터별 lr을 딕셔너리로 정의한다
(`fm/training/.fm/assets/training_walkthrough.py:184-191`).

```python
lrs = {
    "means":     1.6e-4 * scene_scale,  # 위치는 씬 크기에 비례
    "scales":    5e-3,
    "quats":     1e-3,
    "opacities": 5e-2,
    "sh0":       2.5e-3,
    "shN":       2.5e-3 / 20,           # 고차 SH는 천천히
}
optimizers = {
    name: torch.optim.Adam([{"params": splats[name], "lr": lr, "name": name}],
                           eps=1e-15, fused=True)
    for name, lr in lrs.items()
}
```

이건 저장소 본체와 동일한 값이다 — `examples/simple_trainer.py:203-205`:

```python
sh0_lr: float = 2.5e-3
shN_lr: float = 2.5e-3 / 20
```

그리고 이 `/20`은 gsplat이 새로 고른 숫자가 아니라 원 3DGS 구현(Kerbl et al. 2023)에서
`f_dc`의 lr을 `feature_lr = 0.0025`로, `f_rest`의 lr을 `feature_lr / 20.0`으로 준 것이
그대로 계승된 값이다. 즉 **경험적으로 굳어진 관행**이고, 그 관행이 왜 잘 동작하는지가
아래 세 가지 이유다.

## 2. `sh0`와 `shN`은 서로 다른 일을 한다

`sh_degree=3`이면 Gaussian 하나의 색은 계수 16개 × 3채널이고, 워크스루는 이걸 두 텐서로
쪼개 둔다 (`training_walkthrough.py:179-180`).

| 텐서 | shape | 의미 |
|---|---|---|
| `sh0` | `[N, 1, 3]` | **DC(l=0)** — 방향과 무관한 "이 Gaussian의 기본 색" |
| `shN` | `[N, 15, 3]` | **l=1,2,3** — 시선 방향에 따라 변하는 성분(스페큘러, 반사, 뷰 의존 음영) |

렌더 시 둘을 다시 붙여 넘긴다 (`training_walkthrough.py:233`):

```python
colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N,16,3]
```

초기값도 비대칭이다 (`training_walkthrough.py:169-171`):

```python
C0 = 0.28209479177387814
colors = torch.zeros(N, (sh_degree + 1) ** 2, 3, device=device)
colors[:, 0, :] = (rgbs - 0.5) / C0     # DC만 SfM 색, 고차항은 0
```

즉 **`sh0`는 SfM 색으로 이미 대략 맞춰진 상태에서 미세조정만 필요**하고,
**`shN`은 0에서 출발해 "필요한 만큼만" 자라야 하는 보정항**이다. 성격이 다른 두 파라미터를
같은 lr로 밀면 보정항이 기본색보다 먼저 요동친다.

## 3. 왜 하필 작게? — SH 기저의 크기가 원래 더 크다

같은 크기의 계수 변화가 렌더 색에 주는 영향은 밴드마다 다르다. SH의 덧셈정리
$\sum_m Y_{lm}(\mathbf{d})^2 = \frac{2l+1}{4\pi}$ 로부터, 밴드별 기저 벡터의 노름은
**방향과 무관한 상수**다:

| 밴드 | 계수 개수 | $\lVert Y_l(\mathbf{d})\rVert_2$ |
|---|---|---|
| l=0 (`sh0`) | 1 | 0.2821 |
| l=1 | 3 | 0.4886 |
| l=2 | 5 | 0.6308 |
| l=3 | 7 | 0.7464 |
| **l=1..3 합 (`shN`)** | **15** | **1.0925** |

(gsplat의 SH 기저 상수는 `gsplat/cuda/_torch_impl.py:987` 이하에 하드코딩되어 있다:
`0.2820947917738781`, `-0.48860251190292`, `-1.092548430592079`, … 위 표의 값과 일치한다.)

$1.0925 / 0.2821 = \sqrt{15} \approx 3.873$.

의미: **`shN`의 15차원 공간에서 크기 $\epsilon$만큼 움직이면 렌더 색은 `sh0`를 같은 크기로
움직였을 때보다 약 3.87배 더 변한다.** 여기가 결정적인데, 옵티마이저가 Adam이므로
스텝 크기가 gradient 크기와 거의 무관하게 $\approx \text{lr}$로 정규화된다
($\Delta\theta \approx -\text{lr}\cdot m/\sqrt{v}$, `eps=1e-15`는 거의 개입하지 않음).
따라서 **lr 비율이 곧 파라미터 이동량 비율**이다.

- lr을 같게 두면: 스텝당 색 변화가 `shN` 쪽이 3.87배 크다 → 보정항이 기본색을 압도.
- 1/20을 걸면: $3.873 / 20 \approx 0.19$ → `shN`이 만드는 색 변화가 `sh0`의 **약 1/5**.

즉 1/20이라는 숫자는 "기저가 3.9배 크고 차원이 15개인 항을, 그래도 DC보다 5배쯤
느리게 만들자"에 해당한다. 답에 있는 *"DC가 먼저 잡히도록"* 이 여기서 정량적으로 확인된다.

## 4. 왜 느려야 안정적인가 — 뷰 의존 항은 과적합하기 쉽다

`shN`은 카메라 방향의 함수이므로, 어떤 Gaussian의 색이 뷰마다 달라 보이는 오차를
**"진짜 스페큘러"로도, "형상이 틀렸다"로도** 설명할 수 있다. 근본적으로 ill-posed다.

학습 초반에는 형상(`means`/`scales`/`opacities`)이 아직 엉망이라 뷰마다 색 오차가 크다.
이때 `shN`이 빠르게 움직이면 손실 감소의 가장 쉬운 경로를 택한다 — **형상을 고치는 대신
"이 뷰에서는 이 색, 저 뷰에서는 저 색"을 계수에 외워버린다.** 결과는 익숙한 3DGS 병증들:

- 학습 뷰에서만 그럴듯하고 새 뷰에서 색이 튀는 **floater / 반투명 얼룩**
- 매트한 표면에 없던 **가짜 스페큘러 하이라이트**
- 형상 오류가 색으로 은폐되어 densification 신호(`info["means2d"]`의 gradient)까지 흐려짐

`shN`을 20배 느리게 두면 초반의 큰 오차는 **형상과 DC가 먼저 흡수**하고, 뷰 의존 성분은
형상이 어느 정도 수렴한 뒤에야 의미 있게 자란다. 낮은 lr은 사실상 **암묵적 정규화**
(`shN`을 0 근처에 오래 붙잡아 두는 것 ≈ 약한 weight decay)로 작동한다.

## 5. 짝을 이루는 두 번째 장치 — SH 차수 워밍업

lr 감쇠는 혼자 쓰이지 않는다. 워크스루의 학습 루프에는 차수 스케줄이 있다
(`training_walkthrough.py:365`, 본체는 `simple_trainer.py:906`).

```python
sh_degree_to_use = min(step // 1000, SH_DEGREE)   # cfg.sh_degree_interval = 1000
```

| step | 활성 차수 | 실제로 학습되는 계수 |
|---|---|---|
| 0–999 | 0 | `sh0`만 (`shN` gradient = 0, 초기값 0에 그대로 머묾) |
| 1000–1999 | 1 | + l=1 (3개) |
| 2000–2999 | 2 | + l=2 (5개) |
| 3000– | 3 | + l=3 (7개), 전체 16개 |

두 장치의 역할이 다르다.

- **차수 워밍업**: 밴드를 *언제* 켤지 결정하는 **on/off 게이트**. 꺼진 밴드에는 gradient가
  아예 흐르지 않는다(`spherical_harmonics(degrees_to_use=...)`가 상위 밴드를 평가하지 않음).
- **lr 1/20**: 켜진 뒤 *얼마나 빨리* 자랄지 결정하는 **속도 제한**.

워밍업만 있으면 step 1000에서 l=1이 큰 lr로 갑자기 튀어 색이 출렁이고,
lr 감쇠만 있으면 형상이 잡히기 전부터 15개 계수가 조금씩이라도 오차를 외우기 시작한다.
초기 렌더를 `sh_degree=0`으로 확인하는 워크스루 코드(`training_walkthrough.py:253`)가
"DC만으로 먼저 씬을 세운다"는 이 순서를 그대로 보여준다.

## 6. 정리 — 한 문장으로

`sh0`는 이미 SfM 색으로 초기화된 **방향 무관 기본색**이고, `shN`은 0에서 출발하는
**15차원 뷰 의존 보정항**이다. 후자는 기저 노름이 $\sqrt{15}\approx3.87$배 커서 같은 lr이면
색을 훨씬 크게 흔들고, 게다가 형상 오류를 "뷰마다 다른 색"으로 외워 과적합하기 쉽다.
그래서 lr을 1/20로 걸어 **DC와 형상이 먼저 수렴하고 뷰 의존 디테일은 나중에 얹히도록**
학습 순서를 강제한다. `sh_degree_interval`의 차수 워밍업이 같은 목적의 게이트 쪽 장치다.

### 관련 파일

- `/home/sungwoo/projects/swcho/gsplat/fm/training/.fm/assets/training_walkthrough.py` — L169–191(초기화·lr), L233(계수 concat), L365(차수 스케줄)
- `/home/sungwoo/projects/swcho/gsplat/examples/simple_trainer.py` — L203–205(`sh0_lr`/`shN_lr`), L160–165(`sh_degree`, `sh_degree_interval`), L299–300·L344–347(`create_splats_with_optimizers`), L906(차수 스케줄)
- `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/_torch_impl.py` — L980 이하 SH 기저 상수
- `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/_wrapper.py` — L436 `spherical_harmonics(degrees_to_use, ...)`, L493 `spherical_harmonics_l0`, L508 `spherical_harmonics_l1_plus`
