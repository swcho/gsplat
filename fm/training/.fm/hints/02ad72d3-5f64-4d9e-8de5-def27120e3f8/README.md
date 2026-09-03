# Gaussian 초기 크기를 주변 점 밀도에 맞추는 이유

**Q.** Gaussian의 초기 크기를 주변 점 밀도에 맞춰 잡는 이유는?

**A.** 씬을 빈틈없이 덮으면서도 과하게 겹치지 않도록 하기 위해서다. 3-최근접 이웃 평균거리를 크기로 쓰면 밀집 영역은 작게, 희소 영역은 크게 초기화된다.

---

## 1. 초기화가 왜 "그냥 상수"면 안 되는가

3DGS 학습의 출발점은 COLMAP이 만든 **sparse 포인트 클라우드**다. 위치(`means`)와
색(`sh0`)은 SfM이 직접 알려주므로 그대로 베끼면 된다. 하지만 **크기(`scales`)에
대응하는 정보는 SfM 출력에 없다.** 그래서 무언가로 채워야 하는데, 모든 Gaussian에
같은 상수를 주면 두 방향 모두 망가진다.

| 초기 크기 | 렌더 결과 | 최적화에 미치는 영향 |
|---|---|---|
| **너무 작다** | 점들이 낱개 스프라이트로 보이고 사이가 배경으로 뚫린다 | Gaussian이 화면에서 1픽셀 미만 → 픽셀에 거의 기여하지 않으므로 **gradient가 사실상 0**. 위치·색·크기 어느 것도 학습 신호를 못 받는다 |
| **너무 크다** | 씬 전체가 뿌옇게 뭉개진다 | 하나의 Gaussian이 서로 다른 물체 수십 개의 픽셀에 동시에 기여 → **서로 상충하는 gradient가 평균되어 상쇄**된다. 게다가 타일마다 겹치는 Gaussian 수가 폭증해 rasterization 비용도 커진다 |

여기에 씬 스케일 문제가 겹친다. 같은 상수 0.01은 방 하나 크기 씬에서는 적당하지만
건물 규모 씬에서는 먼지만 하다. `scene_scale`로 나눠 상대화한다 해도, **하나의 씬
안에서 밀도가 균일하지 않다**는 문제는 남는다. 가까운 벽면은 포인트가 촘촘하고
멀리 있는 하늘·배경은 듬성듬성한 것이 COLMAP 출력의 정상적인 모습이다.

### 왜 "나중에 학습되니까 괜찮다"가 아닌가

`scales`는 **log 공간**에 저장되고 학습률은 `5e-3`이다 (walkthrough의 `lrs["scales"]`,
`simple_trainer.py`의 `scales_lr`). Adam은 스텝 크기가 대략 lr로 정규화되므로 한
iteration에 log 값이 약 0.005 움직인다 = **크기가 회당 0.5%씩** 바뀐다는 뜻이다.
초기값이 정답보다 100배 작으면 log 차이가 `ln 100 ≈ 4.6`이므로 그것만 메우는 데
최소 900 스텝이 필요하다. 그동안 그 Gaussian은 (위 표의 첫 줄처럼) gradient를 거의
못 받으니 실제로는 그보다 훨씬 오래 걸리거나 영영 회복하지 못한다.
**초기 크기는 "곧 학습될 값"이 아니라 학습이 시작될 수 있는지를 결정하는 값이다.**

## 2. 답: 3-최근접 이웃 평균거리

포인트 클라우드는 이미 "이 지역이 얼마나 촘촘한가"를 스스로 알려주고 있다.
어떤 점에서 가장 가까운 이웃까지의 거리가 곧 **그 점이 담당해야 할 공간의 반지름**에
비례한다. 이것을 그대로 크기로 쓴다.

walkthrough의 구현:

```python
def knn_mean_dist(points: torch.Tensor, k: int = 3, chunk: int = 8192) -> torch.Tensor:
    """각 점에서 k-최근접 이웃까지의 평균 거리 (GPU, 청크 처리)."""
    out = []
    for i in range(0, len(points), chunk):
        d = torch.cdist(points[i : i + chunk], points)   # [chunk, N]
        knn_d = d.topk(k + 1, largest=False).values[:, 1:]  # 자기 자신 제외
        out.append(knn_d.mean(dim=-1))
    return torch.cat(out)

# 크기: 주변 점 밀도에 맞춰 초기화 (빈틈없이 덮되 과하게 겹치지 않도록)
dist_avg = knn_mean_dist(points, k=3)
scales = torch.log(dist_avg)[:, None].repeat(1, 3)   # [N,3] log-space
```

읽을 포인트가 세 개 있다.

- `topk(k+1, largest=False)` 에서 **가장 가까운 것은 자기 자신(거리 0)** 이므로
  `[:, 1:]`로 잘라낸다. k=3을 쓰려면 4개를 뽑아야 한다.
- `.repeat(1, 3)` — 세 축에 같은 값을 넣어 **등방(isotropic) 구**로 시작한다.
  방향성(어느 축으로 납작해야 하는가)은 SfM이 알려주지 않으므로 학습에 맡기고,
  크기만 데이터에서 가져온다.
- `torch.log(...)` — `scales`는 렌더 직전 `torch.exp(splats["scales"])`로 활성화된다.
  log 공간에 저장하면 (a) 음수 크기가 원리적으로 불가능하고, (b) 학습이 **곱셈적**으로
  이뤄져 큰 Gaussian과 작은 Gaussian이 같은 lr로 공평하게 상대 변화를 받는다.

### 왜 하필 3개인가

1개(최근접)만 쓰면 COLMAP이 만든 중복점·노이즈 때문에 거리가 우연히 0에 가까워질 수
있고, 그러면 `log`가 발산한다. 반대로 k를 크게 잡으면 국소 밀도가 아니라 광역 평균을
재게 되어 밀집/희소 구분이 흐려진다. **k=3은 노이즈 평활과 국소성의 타협점**이고,
원 3DGS 논문의 "이웃한 세 점까지의 평균 거리를 축으로 하는 등방 Gaussian" 초기화를
그대로 따른 값이다.

## 3. 왜 이 값이 "빈틈없이 덮되 과하게 겹치지 않는" 크기인가

간격 `d`의 규칙적인 격자를 생각하면 3-NN 평균거리 = `d`, 즉 `σ = d`가 된다.
이웃 두 Gaussian의 **중간 지점**(각 중심에서 `d/2`)에서의 밀도는

```
exp(-(d/2)² / (2d²)) = exp(-0.125) ≈ 0.88
```

즉 **중심 밀도의 88%** — 구멍이 나지 않는다. 반대로 `σ = 0.1d`로 잡으면 같은 지점의
값이 `exp(-12.5) ≈ 3.7e-6`으로 사실상 0이고, 렌더 이미지는 점묘화가 된다.
`σ = 10d`면 하나의 Gaussian이 수십 개 이웃의 영역을 덮어 위 표의 두 번째 줄이 된다.
`σ ≈ d`는 **인접 Gaussian끼리 부드럽게 이어질 정도로만 겹치는** 지점이다.

그리고 이 규칙은 **씬 안에서 자동으로 지역별로 달라진다.**

| 영역 | 3-NN 평균거리 | 초기 Gaussian |
|---|---|---|
| 카메라 근처 텍스처 풍부한 벽면 (포인트 촘촘) | 작다 | 작게 → 디테일을 지울 위험 없음 |
| 먼 배경·하늘·특징 없는 바닥 (포인트 듬성) | 크다 | 크게 → 넓은 면적을 소수로 덮어 구멍 방지 |

`scene_scale`로 나누거나 곱하는 정규화가 **필요 없다**는 점도 중요하다. 거리 자체가
씬의 물리 단위로 측정되므로 방 크기 씬이든 도시 규모 씬이든, 나아가
`init_type="random"`으로 `init_extent * scene_scale` 큐브 안에 10만 점을 뿌린
경우까지 같은 코드가 알맞은 크기를 내놓는다.

## 4. gsplat 본체(`simple_trainer.py`)와의 미세한 차이

`examples/simple_trainer.py:288` `create_splats_with_optimizers`의 해당 부분:

```python
# Initialize the GS size to be the average dist of the 3 nearest neighbors
dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)   # [N,]
dist_avg = torch.sqrt(dist2_avg)
scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)   # [N, 3]
```

세 가지가 다르다.

1. **kNN 백엔드**: `examples/utils.py:156`의 `knn()`은 sklearn
   `NearestNeighbors(n_neighbors=K, metric="euclidean")`를 CPU에서 쓴다 (K=4로 부르고
   `[:, 1:]`로 자기 자신 제거 → 이웃 3개). walkthrough는 의존성을 줄이려고
   `torch.cdist` + `topk`를 GPU에서 청크로 돌린다. 결과 이웃 집합은 같다.
   단 `cdist` 방식은 `[chunk, N]` 행렬을 만들므로 N이 매우 크면 청크 크기가 곧
   메모리 상한이 된다.
2. **평균의 종류**: 본체는 거리의 **제곱평균제곱근(RMS)** `sqrt(mean(d²))`,
   walkthrough는 **산술평균** `mean(d)`을 쓴다. RMS ≥ 산술평균이므로 본체 쪽이
   약간 더 크게 초기화되고, 세 이웃 거리가 들쭉날쭉한 곳(밀도 경계)에서 차이가
   커진다. 학습으로 흡수되는 수준의 차이지만 값을 비교할 때는 알아둘 것.
3. **`init_scale`**: 본체에는 초기 크기 전체를 일괄로 배율 조정하는 노브가 있다
   (기본 1.0). 데이터셋 특성상 초기값이 전반적으로 너무 크거나 작을 때 건드리는
   유일한 손잡이다. walkthrough는 이 인자를 생략했다.

## 5. 초기화 이후 — 나머지는 densification이 맡는다

초기 크기는 "정답"이 아니라 **학습이 gradient를 받을 수 있는 출발선**이다.
이후 `DefaultStrategy`(`gsplat/strategy/default.py`)가 크기를 계속 재조정하는데,
그 임계값들이 모두 `scene_scale` 상대값이라는 점이 초기화와 맞물린다.

| 연산 | 조건 (기본값) | 의미 |
|---|---|---|
| **duplicate** | `grow_grad2d = 2e-4` 초과 & 크기 ≤ `grow_scale3d = 0.01` × scene_scale | 작은데 오차가 큰 곳 → 복제 |
| **split** | `grow_grad2d` 초과 & 크기 > `0.01` × scene_scale | 큰데 오차가 큰 곳 → 2개로 쪼개고 크기 ÷1.6 |
| **prune** | `prune_opa = 0.005` 미만, 또는 크기 > `prune_scale3d = 0.1` × scene_scale | 기여 없거나 비대한 것 제거 |

즉 처음부터 `scene_scale`의 10%를 넘게 초기화하면 첫 refine에서 대량 삭제되고,
반대로 지나치게 작게 초기화하면 gradient를 못 받아 duplicate 대상에도 오르지 못한다.
**3-NN 초기화는 대다수 Gaussian을 이 두 임계값 사이의 "학습 가능한 구간"에
정확히 떨어뜨리는 역할을 한다.**

## 6. 실무 함정

- **중복점 → `log(0) = -inf`.** COLMAP 출력에 완전히 겹치는 점이 있으면 3-NN 평균거리가
  0이 되어 `scales`에 `-inf`가 들어가고 첫 forward에서 NaN이 번진다. 포인트 중복 제거를
  먼저 하거나 거리에 하한(예: `clamp_min`)을 두는 것이 안전하다.
- **N² 메모리.** `torch.cdist` 방식은 청크 단위로도 `[chunk, N]`을 만든다. 포인트가
  수백만 개면 청크를 줄이거나 sklearn/KD-tree 경로(본체 방식)로 돌아가는 편이 낫다.
- **좌표 정규화 순서.** kNN 거리는 `means`가 놓인 좌표계에서 측정된다. 포인트/카메라를
  정규화(`parser`의 normalize)한 뒤에 kNN을 돌려야 `scene_scale` 기반 임계값들과
  단위가 맞는다.
- **등방 초기화는 의도된 것.** 처음부터 축별로 다른 크기를 추정하려 해도 sparse
  포인트만으로는 신뢰할 만한 국소 평면 방향이 안 나온다. 납작해지는 것(surfel처럼)은
  학습이 알아서 한다.

## 7. 관련 코드 위치

- `fm/training/.fm/assets/training_walkthrough.py` — `knn_mean_dist`,
  `init_splats_with_optimizers` (2단계: Gaussian 파라미터 초기화)
- `examples/simple_trainer.py:288` — `create_splats_with_optimizers`
- `examples/utils.py:156` — `knn()` (sklearn 기반)
- `gsplat/strategy/default.py` — `grow_scale3d` / `prune_scale3d` 등 크기 관련 임계값
