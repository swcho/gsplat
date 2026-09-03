# gsplat의 SH 파라미터 분할(`sh0` / `shN`)과 학습률 차이

> **Q.** gsplat은 SH 파라미터를 어떻게 나누어 관리하며, 학습률은 어떻게 다른가?
>
> **A.** `sh0`(DC 계수, 형상 `[N,1,3]`)와 `shN`(나머지 15개 고차 계수, `[N,15,3]`)으로 나눈다.
> `shN`의 학습률은 `sh0`의 **1/20**(2.5e-3 vs 1.25e-4)이다.
> "시점 무관 기본색은 빨리, 시점 의존성은 기본색이 잡힌 뒤 천천히 배우라"는 의도이며,
> 원본 INRIA 3DGS의 `feature_lr` / `feature_lr / 20` 관행을 그대로 따른 것이다.

---

## 1. `simple_trainer.py`의 실제 파라미터 표

`examples/simple_trainer.py`의 `create_splats_with_optimizers()`는 Gaussian 하나를 아래 6개 파라미터로 표현하고,
**파라미터마다 별도의 Adam 옵티마이저**를 만든다.

```python
params = [
    # name, value, lr
    ("means",     torch.nn.Parameter(points),    means_lr * scene_scale),
    ("scales",    torch.nn.Parameter(scales),    scales_lr),
    ("quats",     torch.nn.Parameter(quats),     quats_lr),
    ("opacities", torch.nn.Parameter(opacities), opacities_lr),
]
if feature_dim is None:
    # color is SH coefficients.
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))   # [N, K, 3], K = 16
    colors[:, 0, :] = rgb_to_sh(rgbs)                    # (rgb - 0.5) / C0
    params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
    params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
```

| 이름 | 형상 | 의미 / 활성화 함수 | 기본 lr (`Config`) | 초기값 |
|---|---|---|---|---|
| `means` | `[N,3]` | 중심 위치 (world) | `1.6e-4 × scene_scale` | SfM 포인트 |
| `scales` | `[N,3]` | log-scale, 사용 시 `exp` | `5e-3` | log(3-NN 평균거리 × init_scale) |
| `quats` | `[N,4]` | 회전 쿼터니언(래스터라이저가 내부 정규화) | `1e-3` | 랜덤 |
| `opacities` | `[N]` | logit, 사용 시 `sigmoid` | `5e-2` | logit(0.1) |
| **`sh0`** | **`[N,1,3]`** | **SH 0차(DC) 계수 = 시점 무관 기본색** | **`2.5e-3`** | `(rgb − 0.5) / C0` |
| **`shN`** | **`[N,15,3]`** | **SH 1~3차 계수 = 시점 의존 변동** | **`2.5e-3 / 20 = 1.25e-4`** | 0 |

`Config` 데이터클래스 원문:

```python
# LR for SH band 0 (brightness)
sh0_lr: float = 2.5e-3
# LR for higher-order SH (detail)
shN_lr: float = 2.5e-3 / 20
```

옵티마이저는 이름별로 하나씩이며, 배치 크기 `BS = batch_size × world_size`에 따라 lr을 `√BS` 배로 스케일한다.

```python
optimizers = {
    name: optimizer_class(
        [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
        eps=1e-15 / math.sqrt(BS),
        betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        fused=True,
    )
    for name, _, lr in params
}
```

> **참고**: `means`에만 `× scene_scale`이 붙는다. 위치는 장면 단위(m)에 비례해야 하지만 SH 계수는 색 공간(0~1 부근) 값이므로
> 장면 크기와 무관하다. `sh0`/`shN`의 lr에는 별도 스케줄(지수 감쇠)도 없다 — 원본 3DGS와 동일하게 `means`만 감쇠한다.

### 렌더링 시에는 다시 합쳐진다

파라미터는 둘로 나눠 저장하지만, SH 커널은 `[N, K, 3]` 하나를 받으므로 `rasterize_splats()`에서 매 스텝 `torch.cat`으로 합친다.

```python
# Cast before the cat so both the cat and the SH kernel run on fp16.
if self.cfg.sh_fp16:
    colors = torch.cat([splats["sh0"].half(), splats["shN"].half()], 1)  # [N, K, 3]
else:
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)                # [N, K, 3]
```

`cat`은 미분 가능한 연산이라 역전파 시 gradient가 자동으로 `sh0`(첫 1개)와 `shN`(나머지 15개)으로 다시 갈라진다.
따라서 "나누어 저장 → 합쳐서 연산 → 나뉘어 갱신"이 매 스텝 반복되며, 두 텐서가 **다른 옵티마이저·다른 lr**로 갱신된다는 것 외에
수학적으로는 `[N,16,3]` 텐서 하나를 학습하는 것과 완전히 같다.

---

## 2. 원본 3DGS와 동일한 관행

INRIA 3DGS 원본(`arguments/__init__.py`, `scene/gaussian_model.py`)은 같은 것을 `_features_dc` / `_features_rest`라는 이름으로 나눈다.

| 원본 3DGS | gsplat | lr |
|---|---|---|
| `_features_dc` `[N,1,3]` | `sh0` `[N,1,3]` | `feature_lr = 0.0025` |
| `_features_rest` `[N,15,3]` | `shN` `[N,15,3]` | `feature_lr / 20 = 0.000125` |
| `_xyz` | `means` | `position_lr_init = 0.00016 × spatial_lr_scale` |
| `_scaling` | `scales` | `scaling_lr = 0.005` |
| `_rotation` | `quats` | `rotation_lr = 0.001` |
| `_opacity` | `opacities` | `opacity_lr = 0.05` (v2에서 0.025로 낮아짐) |

```python
# 원본 3DGS gaussian_model.py training_setup()
{'params': [self._features_dc],   'lr': training_args.feature_lr,        'name': 'f_dc'},
{'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, 'name': 'f_rest'},
```

숫자(`2.5e-3`, `/20`, `1.6e-4`, `5e-3`, `1e-3`, `5e-2`)가 모두 일치한다. gsplat은 원본의 결과를 재현하는 것을 목표로 했기 때문에
하이퍼파라미터를 그대로 가져왔고, 이름만 `f_dc/f_rest → sh0/shN`으로 바꿨다.

---

## 3. 왜 둘로 나누는가

`[N,16,3]` 텐서 하나로도 동작은 한다. 그럼에도 나누는 이유는 세 가지다.

### (a) 학습률을 다르게 줄 수 있다

PyTorch 옵티마이저의 lr은 **파라미터 텐서 단위**로 붙는다. 한 텐서 안의 슬라이스(`colors[:, 0]` vs `colors[:, 1:]`)에 다른 lr을
주려면 gradient를 직접 스케일하는 등 우회가 필요하다. 텐서를 둘로 쪼개면 각각 자기 옵티마이저(또는 param group)를 갖게 되어
`sh0_lr`, `shN_lr`을 그냥 지정하면 끝난다. **이것이 나누는 가장 직접적인 이유다.**

### (b) 옵티마이저 상태·densification 처리의 편의

gsplat의 `DefaultStrategy`/`MCMCStrategy`는 학습 중 Gaussian을 복제(split/duplicate)·삭제(prune)하며 N이 계속 바뀐다.
`gsplat/strategy/ops.py`의 `_update_param_with_optimizer()`는 `splats`의 **이름별로** 루프를 돌며

1. 파라미터 텐서를 새 N으로 재구성하고,
2. 대응하는 `optimizers[name]`의 Adam 상태(`exp_avg`, `exp_avg_sq`)를 같은 인덱스로 재구성한다.

```python
for name in names:
    ...
    optimizer = optimizers[name]
    # param과 optimizer.state를 같은 규칙으로 잘라 붙인다
```

"파라미터 하나 ↔ 옵티마이저 하나 ↔ lr 하나"가 1:1로 대응하는 균일한 구조 덕분에, `sh0`과 `shN`이 다른 lr을 가져도
densification 코드는 이름만 순회하면 되고 특수 처리가 필요 없다. 이 규칙은 `means`, `scales` 등 다른 모든 파라미터에도 똑같이 적용된다.

### (c) 저장/로드 형식과의 일치

3DGS 표준 PLY 포맷은 DC와 나머지를 원래부터 다른 속성으로 저장한다.

- `f_dc_0, f_dc_1, f_dc_2` — DC 계수 (RGB 3개)
- `f_rest_0 … f_rest_44` — 고차 계수 45개. **channel-major** 순서(R의 1~15차, 그다음 G, 그다음 B)

`gsplat/exporter.py`의 `splat2ply_bytes()`가 `sh0 → f_dc_*`, `shN → f_rest_*`로 내보내고,
역함수(`ply → sh0/shN`)도 `f_rest`를 `[N,3,15]`로 읽어 `swapaxes(1,2)`해서 `[N,15,3]` `shN`으로 복원한다.
메모리 상의 파라미터 분할이 파일 포맷의 분할과 동일하므로 변환 코드가 단순해지고, 원본 3DGS·다른 뷰어와 상호 호환된다.
또 `sh_degree=0`으로 학습하면 `shN`이 `[N,0,3]`(빈 텐서)가 되어 `f_rest`가 없는 PLY가 자연스럽게 나온다
(`simple_trainer.py`의 PLY 저장부에서 `shN = torch.empty([N, 0, 3])`로 처리하는 분기가 이 경우다).

---

## 4. 왜 1/20인가 — 고차 계수는 천천히

### 문제: 고차 SH는 관측 수에 비해 자유도가 크다

Gaussian 하나의 색은 카메라 위치별로 관측된다. 한 Gaussian을 실제로 보는 카메라는 보통 수 개~수십 개인데,
3차 SH는 채널당 16개(고차 15개) 자유도를 갖는다. `sh_walkthrough.py` 4.2절의 실험이 이를 그대로 보여준다:

- 관측 뷰가 8개일 때 최소제곱으로 L=3(16개)까지 맞추면 **관측 사이 방향에서 색이 요동**(과적합)하고,
  L=0~1이 오히려 전 구면 MSE가 낮다.
- 관측이 300개쯤 되면 L=3이 가장 좋아진다.

즉 고차 계수는 **데이터가 충분히 쌓이기 전에 크게 움직이면 과적합**한다. 이것이 실제 3DGS 렌더에서 나타나는 형태가

- 시점을 움직일 때 색이 튀는 **플리커/깜빡임**,
- 학습 카메라 밖 방향(novel view)에서 나타나는 **색 얼룩·무지개 노이즈**,
- 실제로는 반투명/그림자인 영역을 "특정 방향에서만 밝은 SH"로 억지로 맞추는 **floater**의 색 부분

이다. 고차 SH는 학습 뷰에서의 loss를 낮추는 데는 매우 유연한 도구여서, lr이 크면 기하(위치·크기)가 틀린 것을 색으로 덮어 버리는
"cheating"이 일어나기도 쉽다.

### 해법: 기본색이 먼저 수렴하게 하고, 시점 의존성은 잔차로 천천히

`sh0`은 SfM 색으로 초기화되어 이미 정답 근처에서 시작하며, 어느 방향에서 봐도 동일하게 관측되므로 gradient가 모든 뷰에서 일관된다.
반면 `shN`은 0에서 시작하고, 뷰마다 gradient 방향이 다르다(각 뷰는 자기 방향의 SH 기저값만큼 밀어낸다).
lr을 1/20로 두면

1. 초반에는 `sh0`이 빠르게 평균 색을 잡고,
2. `shN`은 "`sh0`으로 설명 안 되는 잔차(residual)" 중 여러 뷰에서 **일관되게** 나타나는 부분만 천천히 누적한다.

Adam은 gradient를 정규화하므로 lr이 곧 "스텝당 최대 이동량"에 가깝다. 1/20은 스텝당 이동량 자체를 20배 줄여, 노이즈성 gradient가
누적되기 전에 상쇄될 시간을 준다. 값 20 자체는 원본 3DGS 저자가 경험적으로 고른 것이며 정확한 유도가 있는 상수는 아니다 —
같은 "구조적 prior(부드러운 것부터)"를 부여하는 다른 방법으로 SH 계수에 L2 정규화를 걸거나 차수별로 lr을 더 세분화할 수도 있다.

---

## 5. `sh_degree_interval`과 결합되는 커리큘럼

lr 1/20은 "얼마나 빨리"를 제어하고, `sh_degree_interval`은 "언제부터"를 제어한다. 둘이 함께 저차 → 고차 커리큘럼을 만든다.

```python
# Config
# Turn on another SH degree every this steps
sh_degree_interval: int = 1000

# 학습 루프
sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
renders, alphas, info = self.rasterize_splats(..., sh_degree=sh_degree_to_use, ...)
```

| 스텝 | 사용 차수 | 활성 계수 수 | 갱신되는 파라미터 |
|---|---|---|---|
| 0 ~ 999 | 0 | 1 | `sh0`만 (shN은 gradient 0) |
| 1000 ~ 1999 | 1 | 4 | `sh0` + `shN[:, 0:3]` |
| 2000 ~ 2999 | 2 | 9 | `sh0` + `shN[:, 0:8]` |
| 3000 ~ | 3 | 16 | 전부 |

내부적으로 `spherical_harmonics(sh_degree, ...)`는 `coeffs[:, :(sh_degree+1)^2]`만 읽으므로, 쓰이지 않은 고차 계수는
forward에 기여하지 않고 gradient도 0이다. **`shN`의 고차 슬라이스는 처음 1000스텝 동안 아예 갱신되지 않으며**, 그 뒤에도 1/20 lr로 천천히 자란다.

동작 원리는 이미지 피라미드(coarse-to-fine)와 같다.

- 처음 1000스텝은 색이 시점 무관이므로 gradient가 **기하(means/scales/quats/opacities)** 쪽으로 몰린다.
  색으로 덮어쓸 수 없으니 위치와 크기를 제대로 맞춰야 한다 — densification(`refine_start_iter=500`)이 시작되는 시기와도 맞물린다.
- 차수를 하나씩 켤 때 새로 켜진 계수는 0에서 시작하므로 렌더 결과가 갑자기 바뀌지 않는다(연속적인 전환).
- 1000스텝은 30k 스텝 학습의 1/30이고, `adjust_steps(factor)`로 `max_steps`를 줄이면 이 간격도 같은 비율로 줄어든다
  (`self.sh_degree_interval = int(self.sh_degree_interval * factor)`).

`sh_walkthrough.py`의 마지막 Adam 실험(4000스텝, 500스텝마다 차수 +1, lr 2.5e-2 vs 2.5e-2/20)이 이 스케줄의 축소판이며,
loss 곡선에서 차수가 켜지는 지점마다 loss가 한 단계씩 더 떨어지는 것을 볼 수 있다.

---

## 6. 실무 팁

**`shN_lr`을 올리면 생기는 것**
- 학습 PSNR은 조금 오를 수 있지만 test PSNR이 떨어지는 전형적인 과적합 패턴.
- 카메라를 돌릴 때 표면 색이 **번쩍이거나 얼룩처럼 깜빡임**(view-dependent flicker). 특히 관측 뷰가 적은 장면 가장자리·천장·바닥.
- 학습 궤적 밖에서 **채도 높은 무지개색 노이즈**. 고차 SH 다항식은 단위구 위에서 큰 진폭으로 진동하는데, 학습 뷰에서만 제약되어
  나머지 방향에서 폭주한 결과다(`+0.5`, `clamp_min(0)`은 음수만 잘라 주므로 과포화는 막지 못한다).
- 기하 대신 색으로 loss를 줄이는 방향으로 학습이 흘러 Gaussian 개수가 덜 늘거나 floater가 남는다.

**`shN_lr`을 낮추거나 `sh_degree`를 낮추면**
- 광택·프레넬 같은 부드러운 시점 의존 효과가 약해지고 표면이 평평한 diffuse 느낌이 된다. 대신 안정적이고 파일이 작다.
- 야외·drone·큰 실외 장면에서 `sh_degree=1~2`로 낮춰도 PSNR 손실이 작은 경우가 많다 — SH 계수는 Gaussian당 48개 float로
  전체 저장 용량의 대부분(약 60%)을 차지하므로 압축 연구가 가장 먼저 건드리는 부분이다.

**`sh_fp16`** — `cat`과 SH 커널만 fp16으로 돌리고 파라미터·Adam 상태는 fp32로 유지한다. `shN`이 작은 lr로 조금씩 움직이는 만큼
fp16으로 누적하면 정밀도가 부족해지므로, 파라미터는 반드시 fp32에 남긴다.

**`packed` / `sparse_grad`와의 관계** — 이 옵션들은 `sh0`/`shN`의 분할 자체와는 무관하고 *어떤 Gaussian이 갱신되는가*를 바꾼다.
- `packed=True`는 화면에 보이는 Gaussian만 모아(`gaussian_ids`) 래스터라이즈해 메모리를 아낀다.
- `sparse_grad=True`(`packed` 필수)는 각 파라미터의 `.grad`를 보이는 Gaussian만 담은 sparse COO 텐서로 바꾸고 `SparseAdam`을 쓴다.
  파라미터가 **이름별로 분리되어 있기 때문에** `for k in self.splats.keys(): self.splats[k].grad = torch.sparse_coo_tensor(...)`
  처럼 `sh0`, `shN`을 포함한 모든 파라미터에 같은 코드로 적용된다. 보이지 않는 Gaussian의 Adam 모멘트가 갱신되지 않아
  뷰가 적은 영역의 `shN`이 덜 흔들리는 부수 효과도 있다(`visible_adam`/SelectiveAdam도 같은 취지).

---

## 요약

| 항목 | `sh0` | `shN` |
|---|---|---|
| 형상 | `[N,1,3]` | `[N,15,3]` |
| 의미 | DC = 시점 무관 기본색(구면 평균) | 1~3차 = 시점 의존 변동(평균 0) |
| 초기값 | `(rgb − 0.5)/C0` | 0 |
| lr | `2.5e-3` | `2.5e-3 / 20 = 1.25e-4` |
| 활성화 시점 | 0스텝부터 | 1000/2000/3000스텝에 차수별로 |
| PLY 속성 | `f_dc_0..2` | `f_rest_0..44` (channel-major) |
| 원본 3DGS 대응 | `_features_dc`, `feature_lr` | `_features_rest`, `feature_lr/20` |

나누는 이유는 (a) 텐서 단위로 붙는 lr을 다르게 주기 위해, (b) 이름별 옵티마이저 1:1 구조로 densification을 단순하게 하기 위해,
(c) PLY의 `f_dc`/`f_rest` 분리와 맞추기 위해서다. 1/20은 관측 수 대비 자유도가 큰 고차 계수가 과적합·깜빡임을 일으키지 않도록
"기본색 먼저, 시점 의존성은 잔차로 천천히"를 강제하는 장치이고, `sh_degree_interval`의 차수 활성화 스케줄과 합쳐져 coarse-to-fine 커리큘럼을 이룬다.

### 참고 파일
- `/home/sungwoo/projects/swcho/gsplat/examples/simple_trainer.py` — `Config`(lr 기본값, `sh_degree_interval`), `create_splats_with_optimizers()`, `Runner.rasterize_splats()`의 `torch.cat`, 학습 루프의 `sh_degree_to_use`
- `/home/sungwoo/projects/swcho/gsplat/examples/utils.py` — `rgb_to_sh()`
- `/home/sungwoo/projects/swcho/gsplat/gsplat/strategy/ops.py` — `_update_param_with_optimizer()` (이름별 파라미터·옵티마이저 상태 갱신)
- `/home/sungwoo/projects/swcho/gsplat/gsplat/exporter.py` — `f_dc_*`/`f_rest_*` PLY 변환
- `/home/sungwoo/projects/swcho/gsplat/fm/sh/.fm/assets/sh_walkthrough.py` — 3절(DC 계수와 `sh0`/`shN` 분할), 4.1절(`sh_degree_interval`), 4.2절(관측 수 대비 차수 과적합 실험, 두 lr로 Adam 학습)
