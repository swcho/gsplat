# 학습 루프의 데이터 샘플링 — `DataLoader` + 무한 순환 iterator

## 카드 요약

```python
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=True, num_workers=4, persistent_workers=True
)
loader_iter = iter(trainloader)

for step in range(MAX_STEPS):
    try:
        data = next(loader_iter)
    except StopIteration:          # epoch 끝
        loader_iter = iter(trainloader)   # 새로 셔플된 iterator
        data = next(loader_iter)
    ...
```

워크스루 `training_walkthrough.py:336-357`, 실제 트레이너 `examples/simple_trainer.py:847-880`
(트레이너 쪽은 `batch_size=cfg.batch_size`, `pin_memory=True`가 더 붙어 있을 뿐 구조가 같다).

---

## 1. 왜 "epoch 루프"가 아니라 "step 루프"인가

3DGS 학습은 **에폭이 아니라 스텝 단위로 정의된다.** 논문 재현은 30,000스텝,
워크스루 데모는 `MAX_STEPS = 2_000`이다. 밀도화 스케줄도 전부 스텝 기준이다
(refine은 스텝 500~15,000 구간에서 100스텝마다, opacity reset은 3,000스텝마다).

그런데 `DataLoader`는 태생적으로 **에폭 단위 iterable**이다. 한 바퀴 다 돌면
`StopIteration`을 던지고 끝난다. garden 씬(`test_every=8`)의 학습 이미지는 약 161장이므로
2,000스텝을 채우려면 데이터셋을 12바퀴 넘게 돌아야 한다.

그래서 "스텝 루프 안에 에폭 iterable을 끼워 넣는" 어댑터가 필요하고, 그게 바로
`try: next(...) / except StopIteration: iterator 재생성` 패턴이다.
바깥에서 보면 **끝나지 않는 무한 스트림**처럼 동작한다.

```
step:   0    1    2  ...  160 | 161  162 ...  321 | 322 ...
        └── epoch 0 (셔플 A) ──┘└── epoch 1 (셔플 B) ─┘└── epoch 2 ...
                              ↑ 여기서 StopIteration → iter() 다시
```

## 2. `shuffle=True`의 정확한 의미 — 복원추출이 아니다

흔한 오해: "매 스텝 랜덤으로 한 장 뽑는다(i.i.d. 샘플링)". **아니다.**

`shuffle=True`는 내부적으로 `RandomSampler`를 쓰고, 이는 `torch.randperm(len(dataset))`,
즉 **한 에폭 = 전체 인덱스의 무작위 순열**이다. 결과적으로:

- **한 에폭 안에서는 같은 이미지가 두 번 나오지 않는다** (비복원추출).
- 모든 학습 뷰가 정확히 같은 횟수만큼(에폭당 1회) 관찰된다 → 특정 시점만 과대표집되어
  그 방향으로만 Gaussian이 과적합되는 일이 없다.
- 에폭 경계마다 `iter()`를 새로 만들기 때문에 **매 에폭 새로운 순열**이 뽑힌다.
  같은 순서가 반복되지 않으므로 뷰 순서에 의한 주기적 편향이 생기지 않는다.

원조 INRIA 3DGS 구현도 같은 성질이다 — `viewpoint_stack.pop(randint(...))`로
스택이 빌 때까지 뽑고, 비면 다시 채운다. 즉 "shuffled 순열 반복"이라는 점은 동일하다.

`torch.manual_seed(42)`가 앞에서 설정되어 있으므로(walkthrough:72) 순열 순서는 재현 가능하다.

## 3. 왜 `batch_size=1`인가

3DGS 학습에서 배치 1은 관행이 아니라 **거의 강제 조건**이다.

| 이유 | 설명 |
|---|---|
| 해상도 불일치 | COLMAP 씬은 카메라가 여러 개일 수 있고 이미지 크기 `(H, W)`가 서로 다르다. 기본 `collate_fn`은 shape가 다른 텐서를 stack할 수 없어 그대로 터진다. `simple_trainer.py:461`은 아예 `num_cameras > 1 and batch_size != 1`이면 `ValueError`를 던진다. |
| 렌더링 단위 | 래스터화는 뷰 하나당 `(width, height)` 를 인자로 받는다. 배치를 섞으려면 뷰마다 다른 캔버스 크기를 다뤄야 한다. |
| 메모리 | 이미지 한 장이 이미 수백만 픽셀이고, Gaussian 수십~수백만 개의 gradient가 따라붙는다. 배치를 키울 실익이 적다. |
| 후처리 제약 | `post_processing == "ppisp"`도 `batch_size=1`을 요구한다 (`simple_trainer.py:466`). |

배치를 키울 때는 learning rate를 그냥 두면 안 되고 `lr * sqrt(batch_size)`로 스케일한다
(`simple_trainer.py:534, 559, 596`). 워크스루는 이 복잡도를 피하려고 1로 고정했다.

그래서 `data`의 텐서들은 **배치 차원 1이 붙은 채로** 나온다:

```python
camtoworlds = data["camtoworld"].to(DEVICE)   # [1,4,4]
Ks          = data["K"].to(DEVICE)            # [1,3,3]
pixels      = data["image"].to(DEVICE) / 255.0  # [1,H,W,3]
height, width = pixels.shape[1:3]
```

`rasterization()`이 요구하는 "카메라 배치" 레이아웃과 그대로 맞아떨어지므로
`[None]`으로 차원을 끼워넣는 수고가 없다 (스냅샷 렌더에서는 `valset[0]`을 직접
쓰기 때문에 `snap_c2w = snap_view["camtoworld"][None]`처럼 수동으로 붙인다).

## 4. `num_workers=4` — 병목이 GPU가 아니라 디스크/CPU라서

`Dataset.__getitem__`(`examples/datasets/colmap.py:466`)이 매번 하는 일을 보면
왜 워커가 필요한지 분명하다.

1. `imageio.imread(...)` — **JPEG 디코딩** (디스크 I/O + CPU)
2. 왜곡 계수가 있으면 `cv2.remap(...)`으로 **undistort**, ROI 크롭
3. `patch_size`가 있으면 랜덤 크롭 + `K`의 principal point 보정
4. numpy → `torch.from_numpy(...).float()` 변환
5. `load_depths=True`면 SfM 포인트를 이미지 평면에 투영해 depth 타깃까지 생성

이 전부가 **순수 CPU 작업**이다. 워커 4개가 별도 프로세스에서 다음 스텝 이미지를
미리 만들어 두므로, GPU가 래스터화/backward를 도는 동안 데이터 준비가 겹쳐 돌아간다.
워커가 0이면 매 스텝 "JPEG 디코딩 → 그다음 렌더링"이 직렬로 붙어 학습이 눈에 띄게 느려진다.

트레이너 쪽은 여기에 `pin_memory=True`까지 붙여 host→device 복사를 비동기 DMA로 만든다.

주의: `image`는 `.float()`이지만 **아직 0~255 스케일**이다. 정규화는 루프 안에서
`/ 255.0`으로 한다. `Dataset` 자체는 스케일을 건드리지 않는다.

## 5. `persistent_workers=True` — 이 패턴에서 특히 중요한 이유

기본값(`False`)이면 **`StopIteration`이 날 때마다 워커 4개가 죽고, `iter()`를 다시 부를 때
4개가 다시 fork된다.** 프로세스 생성 + 데이터셋 객체 pickle 복사 + 시작 오버헤드가
에폭마다 반복된다.

이 루프는 에폭 경계를 **12번 이상** 지나가므로 그 비용이 그대로 학습 시간에 얹힌다.
`persistent_workers=True`는 워커를 살려 둔 채 새 순열만 내려보내서 이 재기동을 없앤다.

즉 **`persistent_workers`는 "iterator를 계속 다시 만든다"는 이 코드 패턴과 짝을 이루는 옵션**이다.
둘 중 하나만 알면 왜 붙었는지 이해가 안 되는 조합이다.
(전제 조건: `num_workers > 0`이어야 한다. 0이면 무시되거나 에러가 난다.)

## 6. 왜 `itertools.cycle`을 안 쓰나

`for data in itertools.cycle(trainloader)` 가 더 짧아 보이지만 **쓰면 안 된다.**

- `cycle`은 첫 바퀴에 나온 원소를 전부 **메모리에 캐시해 두고** 그대로 재생한다.
  → 이미지 텐서 161장이 통째로 램에 쌓이고,
  → 두 번째 바퀴부터는 **셔플이 죽는다** (첫 에폭 순서가 영원히 반복).
- `DataLoader`를 `iter()`로 다시 만들어야 비로소 `RandomSampler`가 새 순열을 뽑는다.

`while True: for data in trainloader: ...` 형태의 중첩 루프도 같은 효과를 내지만,
스텝 카운터/스케줄러/`tqdm` 진행바를 바깥 루프에 두기 불편해진다. `try/except StopIteration`은
**스텝 루프를 평평하게(flat) 유지하면서** 무한 스트림을 얻는 관용구다.

## 7. `data` 딕셔너리에 뭐가 들어 있나

`colmap.py:490` 이후 기준:

| 키 | shape (batch 1 기준) | 설명 |
|---|---|---|
| `K` | `[1,3,3]` | undistort 후의 내부 파라미터 |
| `camtoworld` | `[1,4,4]` | 카메라 포즈 |
| `image` | `[1,H,W,3]` | float, **0~255** |
| `image_id` | `[1]` | 데이터셋 내 인덱스 (pose/appearance 최적화의 임베딩 키) |
| `camera_idx` | `[1]` | 0-based 카메라 인덱스 |
| `mask` | `[1,H,W]` | (있을 때) bool |
| `exposure` | `[1]` | (있을 때) 노출값 |
| `points`, `depths` | `[1,M,2]`, `[1,M]` | `load_depths=True`일 때만, depth loss용 |

워크스루는 `camtoworld` / `K` / `image` 세 개만 쓴다. 트레이너는 `image_id`(pose noise,
appearance optimization), `mask`, `exposure`, `points`/`depths`(depth loss)까지 활용한다.

## 8. train/val 분리는 어디서 되나

샘플링 이전 단계다. `Dataset.__init__`(`colmap.py:457`)이 인덱스를 나눈다.

```python
indices = np.arange(len(self.parser.image_names))
if split == "train":
    self.indices = indices[indices % self.parser.test_every != 0]  # 나머지 전부
else:
    self.indices = indices[indices % self.parser.test_every == 0]  # 매 test_every번째
```

`TEST_EVERY = 8`이므로 8장 중 1장이 val이다. `DataLoader`는 `trainset`만 보므로
검증 이미지는 학습 루프에 절대 들어오지 않는다. 검증용 loader는 별도로
`shuffle=False, num_workers=1`로 만든다 (`simple_trainer.py:1209`) — 평가는 순서가
고정돼야 비교가 되기 때문이다.

워크스루의 고정 시점 스냅샷도 `valset[0]`을 써서 **학습에 쓰이지 않은 뷰**에서
경과를 관찰한다 (`snap_view = valset[0]`).

---

## 한 줄 정리

> `DataLoader(batch_size=1, shuffle=True)`로 **에폭마다 새로 셔플된 순열에서 이미지를 한 장씩**
> 꺼내고, `StopIteration`을 잡아 `iter()`를 다시 만들어 **스텝 기반 학습 루프에 맞는 무한 스트림**으로
> 바꾼다. `num_workers=4`는 JPEG 디코딩/undistort를 GPU 연산과 겹쳐 돌리고,
> `persistent_workers=True`는 그 잦은 iterator 재생성 때문에 워커가 매번 재기동되는 것을 막는다.
