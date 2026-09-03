# GPU 벤치마크를 정확히 재려면 무엇이 필요한가

> **한 줄 답**: 측정 구간의 **앞과 뒤에서** `torch.cuda.synchronize()`를 호출해야 한다.
> CUDA 커널은 비동기로 실행되므로, 동기화 없이 `time.perf_counter()`로 재면 GPU가 실제로 일한 시간이 아니라
> **호스트가 커널을 큐에 밀어 넣는 데 걸린 시간(런치 오버헤드)** 만 재게 된다.

---

## 1. 왜 이런 일이 생기나 — CUDA의 비동기 실행 모델

PyTorch에서 `c = a @ b` 같은 GPU 연산을 호출하면 실제로 벌어지는 일은 이렇다.

```
[CPU 스레드]  커널 인자 준비 → 스트림에 커널 launch → 즉시 return  (수 µs)
                                     │
                                     ▼
[GPU]                          스트림 큐: [K1][K2][K3] ...  → 순서대로 실행 (수 ms)
```

- **스트림(stream)** 은 GPU 작업의 FIFO 큐다. 같은 스트림에 넣은 커널은 넣은 순서대로 실행되고,
  PyTorch는 기본적으로 디바이스당 하나의 "현재 스트림"에 모든 op를 넣는다.
- **커널 launch는 논블로킹**이다. CPU는 "이 일 좀 해 둬"라고 큐에 적어 놓고 곧바로 다음 파이썬 줄로 넘어간다.
  GPU가 그 커널을 언제 시작해서 언제 끝냈는지 CPU는 **묻기 전까지 모른다**.
- 그래서 파이썬 쪽 벽시계(`time.perf_counter()`)는 기본적으로 **"주문서를 쓰는 데 걸린 시간"** 을 잰다.
  요리(GPU 실행)가 끝나기를 기다린 시간이 아니다.

> 식당 비유: 주문서를 10장 써서 주방에 넣는 데 0.7초 걸렸다고 해서 요리가 0.7초 만에 나오는 게 아니다.
> 요리 시간을 재려면 마지막 접시가 나올 때까지 기다린 뒤 시계를 봐야 한다 —
> 그 "기다림"이 `torch.cuda.synchronize()`다.

`torch.cuda.synchronize()`는 **현재 디바이스의 모든 스트림에 큐잉된 작업이 전부 끝날 때까지 CPU를 블록**한다.
그래서 "시작 지점을 깨끗하게 맞추는 용도"(이전 셀에서 남은 작업이 내 측정 구간에 섞이지 않게)와
"끝 지점을 실제 완료 시각에 맞추는 용도" 둘 다에 필요하다. **앞뒤 두 번**인 이유가 이것이다.

---

## 2. 함정 실증 — 같은 코드, 400배 차이

이 힌트를 쓰면서 RTX 3090에서 직접 돌린 결과다(4096×4096 matmul 50회, gsplat 없이 순수 torch).

```python
import time, torch
a = torch.randn(4096, 4096, device="cuda"); b = torch.randn(4096, 4096, device="cuda")
def work(n=50):
    for _ in range(n):
        c = a @ b

work(5); torch.cuda.synchronize()                  # 워밍업

# ① 동기화 없이 (틀림)
t0 = time.perf_counter(); work(); t_nosync = (time.perf_counter() - t0) / 50 * 1e3

# ② 앞뒤 동기화 (맞음)
torch.cuda.synchronize(); t0 = time.perf_counter()
work()
torch.cuda.synchronize(); t_sync = (time.perf_counter() - t0) / 50 * 1e3

# ③ CUDA Event (맞음, 더 정밀)
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record(); work(); e.record(); torch.cuda.synchronize(); t_ev = s.elapsed_time(e) / 50
```

실제 출력:

```
NVIDIA GeForce RTX 3090
no-sync    0.014 ms/iter      ← 커널 하나 launch하는 데 걸린 CPU 시간
sync       5.556 ms/iter      ← 진짜 GPU 실행 시간
cudaEvent  5.564 ms/iter      ← GPU 타임스탬프 기준, sync와 일치
```

**0.014 ms vs 5.556 ms — 약 400배**. 동기화를 빼먹은 벤치마크는 "이 커널 엄청 빠르네요"라는
완전히 틀린 결론을 만들어 낸다. 더 나쁜 건, 틀린 쪽이 **그럴듯한 숫자**로 나온다는 점이다.
0.014 ms는 "말도 안 되게 빠름"이 아니라 "꽤 빠른 커널" 정도로 읽히기 때문에 의심 없이 지나가기 쉽다.

숫자를 보는 감각:
- **커널 launch 오버헤드**: 대략 5~20 µs (CUDA Graph를 쓰면 더 줄어든다)
- 그래서 동기화 없는 측정값은 **연산 크기와 거의 무관하게** 수십 µs 언저리에 머문다.
  입력을 10배로 키웠는데 측정 시간이 그대로라면 그건 최적화가 아니라 **동기화 누락 신호**다.

> 주의: 큐가 무한대는 아니다. 반복 횟수가 아주 크면 큐가 가득 차서 launch가 블로킹되고,
> 그때부터는 "우연히" 실제 시간에 가까워진다. 즉 동기화 없는 코드는 **경우에 따라 맞고 경우에 따라 틀린다** —
> 재현성 자체가 없다는 뜻이라 더 위험하다.

---

## 3. 암묵적 동기화 — 조용히 결과를 바꾸는 것들

`torch.cuda.synchronize()`만 동기화를 일으키는 게 아니다. **GPU 결과값을 CPU가 실제로 읽어야 하는 모든 연산**은
그 값을 만드는 커널이 끝날 때까지 기다린다.

| 연산 | 동기화 여부 | 비고 |
|---|---|---|
| `torch.cuda.synchronize()` | ✅ 전체 큐 | 명시적, 벤치마크의 정석 |
| `tensor.item()` | ✅ | 스칼라를 CPU로 가져옴. loss 로깅의 주범 |
| `tensor.cpu()` / `.numpy()` / `.tolist()` | ✅ | D2H 복사 (non_blocking=True + pinned memory여도 값을 읽는 순간 필요) |
| `print(tensor)` | ✅ | 값을 찍으려면 읽어야 하니까 |
| `float(t)`, `if t > 0:`, `t.max().item()` | ✅ | 파이썬 제어 흐름에 GPU 값이 들어가는 순간 |
| `torch.allclose(a, b)` (bool로 소비될 때) | ✅ | 노트북 검증 코드에 흔함 |
| `assert`, `tqdm`에 loss 표시 | ✅ | 학습 루프를 은근히 느리게 만든다 |
| `tensor.to("cuda")` (pinned + non_blocking) | ❌ | H2D는 비동기 가능 |
| `a @ b`, `conv2d`, 커스텀 CUDA op | ❌ | launch만 하고 return |

이게 벤치마크에 미치는 영향은 두 방향 모두다.

**(a) 우연히 맞게 만드는 경우** — 측정 구간 안에 `.item()`이나 `print()`가 들어 있으면 거기서 동기화가 걸려
숫자가 그럴듯해진다. 하지만 이건 "동기화가 필요 없다"는 증거가 아니라 **우연**이다. 그 줄을 지우는 순간 무너진다.

**(b) 결과를 왜곡하는 경우** — 재려는 대상이 아닌 D2H 복사·CPU 대기까지 측정에 포함된다.
특히 loop 안에서 매 반복 `.item()`을 부르면 **GPU가 파이프라인을 채울 기회를 잃어**
실제 처리량보다 느리게 측정된다(launch 오버헤드가 커널 실행에 가려지지 못한다).

---

## 4. 올바른 패턴

```python
def bench(fn, n=50, warmup=10):
    for _ in range(warmup):        # ① 워밍업
        fn()
    torch.cuda.synchronize()       # ② 이전 작업 전부 비우고 출발선 정렬
    t0 = time.perf_counter()       # ③ 시작 시각
    for _ in range(n):             # ④ N회 반복
        fn()
    torch.cuda.synchronize()       # ⑤ 마지막 커널 완료까지 대기
    return (time.perf_counter() - t0) / n * 1e3   # ⑥ 평균 (ms)
```

핵심 6단계: **워밍업 → sync → t0 → N회 반복 → sync → t1, 그리고 (t1−t0)/N**.

- **N회 반복 후 나누기**: 한 번만 재면 타이머 해상도·스케줄러 지터·클럭 부스트 변동에 묻힌다.
  반복하면 launch 오버헤드도 평균화된다.
- **중간에 sync를 넣지 않는다**: 반복마다 동기화하면 매 반복 GPU를 비워 실제 파이프라인 동작과 달라진다.
  (다만 반복 간 분산을 보고 싶다면 반복마다 CUDA Event 쌍으로 재는 편이 낫다.)
- **평균 대신 중앙값/최솟값**도 유용하다. 다른 프로세스가 GPU를 쓰는 환경이면 min이 노이즈에 강하다.

### 워밍업이 필요한 이유

첫 호출은 항상 느리다. 그것도 아주 많이. 원인:

1. **JIT / 커널 컴파일** — `torch.compile`, Triton, `torch.utils.cpp_extension` 로드, gsplat 같은 확장의
   지연 빌드. 첫 호출에서 컴파일이 일어나면 수 초~수십 분(!)이 그 측정에 들어간다.
2. **cuBLAS / cuDNN 초기화** — 핸들 생성, 워크스페이스 할당, **알고리즘 자동 튜닝**
   (`torch.backends.cudnn.benchmark=True`면 첫 호출에서 여러 알고리즘을 실측해 고른다).
3. **CUDA 컨텍스트 생성** — 프로세스당 최초 GPU 접근에서 수백 ms.
4. **캐싱 할당자(caching allocator)** — 첫 실행은 `cudaMalloc`(비싸고 암묵적 동기화 유발),
   이후에는 캐시에서 재사용해 거의 공짜. 워밍업 없이는 할당 비용이 커널 시간으로 둔갑한다.
5. **클럭/전력 상태** — 유휴 GPU는 저클럭이다. 몇 번 돌려야 부스트 클럭에 도달한다.
   (그래서 반대로 **너무 오래** 돌리면 thermal throttling으로 느려지기도 한다.)
6. **L2 캐시 / TLB warm** — 같은 텐서를 반복 접근하는 벤치마크는 실제보다 유리해질 수 있으니
   "캐시가 데워진 상태"가 재려는 시나리오와 맞는지 생각해야 한다.

---

## 5. 더 정밀한 대안 — `torch.cuda.Event`

```python
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

start.record()          # 스트림에 "타임스탬프 찍기" 마커를 큐잉 (이것도 비동기!)
for _ in range(n):
    fn()
end.record()
torch.cuda.synchronize()          # 또는 end.synchronize()
ms = start.elapsed_time(end) / n  # 밀리초, float
```

- `record()`는 **커널이 아니라 마커를 스트림에 넣는다**. GPU가 큐를 소화하다가 그 지점에 도달하는 순간의
  **GPU 하드웨어 타임스탬프**를 기록한다.
- `elapsed_time()`을 호출하기 전에는 반드시 이벤트가 완료되어야 한다 → `torch.cuda.synchronize()` 또는
  `end.synchronize()`. (덜 끝난 이벤트에 부르면 예외가 난다.) **동기화가 없어도 되는 게 아니라,
  동기화 지점이 "측정 창 밖"으로 밀려나는 것**이다.
- **장점**: CPU 스케줄링 지터·GIL·파이썬 오버헤드가 측정에서 빠진다. 순수한 GPU 구간 시간을 잰다.
  또 여러 이벤트 쌍으로 **파이프라인 내부 구간**(예: 투영만, 블렌딩만)을 CPU를 멈추지 않고 나눠 잴 수 있다.
- **주의**: 해상도는 약 0.5 µs. 멀티 스트림에서는 어느 스트림에 record했는지가 의미를 바꾼다.
  또 이벤트는 "GPU가 그 지점에 도달한 시각"이라 **CPU가 launch를 못 따라가 GPU가 굶는 상황(launch-bound)** 은
  이벤트 측정으로 잘 안 보인다 — 그런 경우는 벽시계+sync 측정과 비교해야 드러난다.

위 실측에서 `sync` 5.556 ms와 `cudaEvent` 5.564 ms가 거의 같았던 것은 이 워크로드가 충분히 커서
CPU 오버헤드가 무시할 수준(GPU-bound)이라는 뜻이다. 반대로 **커널이 작고 많은** 워크로드에서는
두 값이 벌어지고, 그 차이 자체가 "파이썬/런치 오버헤드가 병목"이라는 진단이 된다.

---

## 6. 직접 짜지 말고 쓰면 좋은 도구

### `torch.utils.benchmark.Timer`
```python
import torch.utils.benchmark as benchmark
t = benchmark.Timer(
    stmt="fn(x)",
    globals={"fn": fn, "x": x},
    label="rasterization", sub_label="fused", description="garden",
)
print(t.blocked_autorange(min_run_time=2.0))   # 워밍업·동기화·반복 횟수 자동 결정
```
- 워밍업, 동기화, 적응적 반복 횟수, 중앙값/IQR 통계를 알아서 처리한다.
- `Compare`로 여러 구현을 표로 나란히 비교, `Fuzzer`로 입력 shape 스윕까지.
- **손으로 짠 루프보다 거의 항상 낫다.** 특히 "동기화 빼먹기"를 구조적으로 못 하게 만든다.

### `torch.profiler`
```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    fn()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")   # chrome://tracing 또는 Perfetto에서 열기
```
- "전체 몇 ms"가 아니라 **어느 커널이 몇 ms**인지 준다. `Self CUDA time`으로 진짜 범인을 찾는다.
- 타임라인을 보면 **CPU 런치 갭**(GPU가 놀고 있는 빈틈)이 눈에 보인다 — 파이썬 오버헤드 진단에 결정적.
- 더 깊게는 **Nsight Systems**(`nsys profile`, 전체 타임라인)와 **Nsight Compute**(`ncu`, 커널 내부
  메모리/occupancy 분석)가 있다. gsplat 같은 커스텀 CUDA 커널을 실제로 튜닝하려면 이쪽이다.

---

## 7. 노트북의 벤치마크 코드 읽기

`.fm/assets/rasterization_walkthrough.py` 마지막 부분(§7 끝)에 있는 코드다.

```python
def bench(fn, n=10):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3


with torch.no_grad():
    t_fused = bench(lambda: rasterization(means, quats, scales, opacities, sh_coeffs, viewmats, Ks, W, H,
                                          sh_degree=SH_DEGREE, packed=False))
    t_step = bench(lambda: rasterize_stepwise(means, quats, scales, opacities, sh_coeffs, viewmats, Ks, W, H,
                                              sh_degree=SH_DEGREE))
print(f"forward 시간 (C={C}, N={N:,}, {W}x{H}):  rasterization() {t_fused:.2f} ms   stepwise {t_step:.2f} ms")
print("→ 계산 커널은 같고, C++ 오케스트레이터는 파이썬 오버헤드·중간 텐서 cat 몇 개를 절약할 뿐이다.")
```

체크포인트별로 뜯어보면:

- `torch.cuda.synchronize(); t0 = ...` — **출발선 정렬**. 바로 앞 셀의 backward 비교(`.backward()`,
  `grad.clone()`)에서 큐잉된 작업이 남아 있으면 그게 t_fused에 그대로 얹힌다. 이걸 막는다.
- `... synchronize(); return (perf_counter() - t0) / n * 1e3` — **결승선 정렬 + 평균 + ms 변환**.
  `n=10`으로 나누고 `1e3`을 곱해 반복당 밀리초를 얻는다.
- `with torch.no_grad()` — autograd 그래프 구축 비용과 저장 텐서 할당을 빼서 **순수 forward**만 잰다.
  이걸 빼면 fused/stepwise 양쪽에 backward용 저장 오버헤드가 섞인다.
- **워밍업은 어디에?** 이 `bench()` 안에는 명시적 워밍업 루프가 없다. 대신 노트북 흐름상
  §7 앞부분에서 `rasterization()`과 `rasterize_stepwise()`를 **이미 여러 번 호출**했기 때문에
  (forward 일치 검증, backward grad 비교) 커널 로드·cuBLAS 초기화·할당자 캐시가 모두 데워진 상태다.
  즉 **셀 순서가 워밍업 역할**을 한다. 이 셀만 따로 떼어 실행하면 첫 측정이 크게 부풀 수 있으니,
  독립 스크립트로 옮길 때는 `bench()` 안에 `for _ in range(3): fn()`을 넣어야 한다.
  `n=10`도 작은 편이라, 진지하게 비교하려면 `n`을 키우거나 `Timer.blocked_autorange`로 바꾸는 게 좋다.

### 결과 해석 — fused vs stepwise 차이가 왜 작은가

두 경로는 **완전히 같은 CUDA 커널**을 같은 순서로 호출한다.

- `rasterization(..., packed=False)` → C++ 오케스트레이터 `Rendering.cpp :: rasterization_3dgs()`가
  ①②③ 투영 → ⑤⑥ 타일 교차/오프셋 → ④ SH → ⑦ 블렌딩을 이어 붙인다.
- `rasterize_stepwise(...)` → **파이썬에서** `fully_fused_projection` → `isect_tiles` →
  `isect_offset_encode` → `spherical_harmonics` → `rasterize_to_pixels`를 같은 순서로 호출한다.

노트북이 앞선 셀에서 확인하듯 두 결과는 **비트 단위로 일치**한다(`torch.equal`이 True). 같은 커널이니까.
그러니 시간 차이로 나타날 수 있는 것은 **커널 바깥의 오케스트레이션 비용**뿐이다:

- 파이썬 인터프리터 오버헤드(함수 호출, 인자 검증, dispatch) — 커널 **launch당 수 µs**
- 중간 텐서 몇 개의 `cat`/`expand`/뷰 조작
- autograd Function 노드 생성 (여기선 `no_grad`라 거의 없음)

garden 씬 규모(수십만~수백만 Gaussian, 수백×수백 픽셀 이상)에서 커널 하나가 **밀리초 단위**로 도는 데 반해,
파이썬 오버헤드는 통틀어 **수십~수백 µs**다. 그래서 두 시간이 거의 같게 나오고, 노트북의 결론 문장이
바로 그것이다 — *"계산 커널은 같고, C++ 오케스트레이터는 파이썬 오버헤드·중간 텐서 cat 몇 개를 절약할 뿐이다."*

여기서 배울 점은 두 가지다.

1. **C++ 오케스트레이터의 이득은 커널이 작을 때 커진다.** 아주 작은 씬, 작은 해상도, 혹은 학습 루프에서
   커널 하나하나가 수십 µs 수준이면 파이썬 오버헤드가 상대적으로 커져 fused 쪽이 눈에 띄게 유리해진다.
   그런 영역이 CUDA Graph나 `torch.compile`이 노리는 지점이기도 하다.
2. **이 결론은 동기화가 있어야만 얻을 수 있다.** 만약 `synchronize()` 없이 쟀다면
   두 경로 모두 "launch 시간"만 재게 되고, 그건 **파이썬 호출 수에 정확히 비례**한다.
   즉 stepwise가 fused보다 몇 배 느리다는 **정반대의 그럴듯한 거짓 결론**이 나온다.
   동기화는 "정확한 숫자를 얻기 위한 형식"이 아니라 **무엇을 측정하고 있는지를 결정하는 스위치**다.

---

## 8. 자주 하는 실수 체크리스트

- [ ] **측정 뒤 `synchronize()` 누락** — 런치 오버헤드만 잰다. 가장 흔하고 가장 치명적.
- [ ] **측정 앞 `synchronize()` 누락** — 이전 작업의 잔여 시간이 내 측정에 얹힌다.
- [ ] **워밍업 없음** — 첫 호출의 JIT/cuBLAS 초기화/`cudaMalloc`이 커널 시간으로 둔갑.
- [ ] **반복 1회** — 타이머 지터·클럭 변동에 묻힌다. N회 후 평균(또는 중앙값/최솟값).
- [ ] **루프 안의 `.item()` / `print()` / `tqdm` 로깅** — 매 반복 강제 동기화로 파이프라인이 끊긴다.
- [ ] **`no_grad()` 깜빡** — forward만 재려 했는데 autograd 그래프 구축·저장 비용이 섞인다.
- [ ] **비교군 간 조건 불일치** — 한쪽만 워밍업, 한쪽만 캐시가 데워진 상태, dtype/TF32 설정 다름.
- [ ] **입력을 키웠는데 시간이 그대로** — 최적화가 아니라 동기화 누락 신호.
- [ ] **메모리 할당까지 측정에 포함** — 매 반복 새 출력 텐서를 잡으면 할당자 비용이 섞인다.
      필요하면 `torch.cuda.empty_cache()` + `reset_peak_memory_stats()`로 조건을 통제.
- [ ] **다른 프로세스와 GPU 공유** — `nvidia-smi`로 확인. MPS/다중 사용자 환경은 측정 자체가 무의미할 수 있다.
- [ ] **`elapsed_time()`을 동기화 전에 호출** — 이벤트가 아직 완료되지 않아 예외.
- [ ] **thermal throttling 무시** — 장시간 루프는 클럭이 떨어져 뒤로 갈수록 느려진다. `nvidia-smi -q -d CLOCK` 확인.

---

## 9. 함께 볼 카드

- `rasterization()` vs 단계별 호출의 **수치적 동일성** — 같은 커널을 쓰므로 비트 단위로 일치한다.
- **C++ 오케스트레이터(`Rendering.cpp :: rasterization_3dgs`)가 하는 일** — 커널 융합이 아니라
  단계 연결(launch 순서·중간 텐서 관리)이다. 그래서 절약되는 건 파이썬 오버헤드뿐.
- **타일 부하 불균형(`isect_offsets` 차분)** — 실제 래스터화 시간이 어디서 나오는지는
  전체 ms가 아니라 이 분포와 `torch.profiler`의 커널별 시간에서 읽어야 한다.
