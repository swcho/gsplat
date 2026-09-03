# `sh0`/`shN` Parameter 생성 시 `.contiguous()`가 필요한 이유

## 카드 내용

> **Q.** `sh0`/`shN` Parameter 생성 시 `.contiguous()`가 필요한 이유는?
> **A.** 슬라이스 뷰를 그대로 `Parameter`로 쓰면 fused Adam이 거부한다. `.contiguous()`로 연속 메모리 텐서를 만들어야 한다.

## 문제의 코드

워크스루 2단계(`training_walkthrough.py`, `init_splats_with_optimizers`)는 SH 계수를 **한 덩어리로 만든 뒤 둘로 쪼갠다**.

```python
C0 = 0.28209479177387814
colors = torch.zeros(N, (sh_degree + 1) ** 2, 3, device=device)  # [N,16,3]
colors[:, 0, :] = (rgbs - 0.5) / C0                              # DC(0차)만 SfM 색

splats = torch.nn.ParameterDict({
    ...
    # .contiguous()가 필요: 슬라이스 뷰를 그대로 Parameter로 쓰면 fused Adam이 거부한다
    "sh0": torch.nn.Parameter(colors[:, :1, :].contiguous()),   # [N,1,3]
    "shN": torch.nn.Parameter(colors[:, 1:, :].contiguous()),   # [N,15,3]
}).to(device)

optimizers = {name: torch.optim.Adam([...], eps=1e-15, fused=True) for ...}
```

`sh0`/`shN`을 애초에 나누는 이유는 **학습률이 다르기** 때문이다(`sh0`는 `2.5e-3`, 고차항 `shN`은 그 1/20). 렌더 직전에는 다시 `torch.cat([splats["sh0"], splats["shN"]], 1)`로 `[N,16,3]`을 복원해 `rasterization()`에 넘긴다.

## 왜 슬라이스 뷰가 문제인가

`colors[:, 1:, :]`는 **복사가 아니라 뷰(view)** 다. 같은 storage를 가리키면서 shape만 `[N,15,3]`으로 바꾼 것이므로 stride가 원본의 K=16 간격을 그대로 유지한다.

```
colors: [N,16,3], stride (48, 3, 1)      ← 한 Gaussian당 48개 float
sh0 = colors[:, :1, :]  → shape [N,1,3],  stride (48, 3, 1)   is_contiguous() == False
shN = colors[:, 1:, :]  → shape [N,15,3], stride (48, 3, 1)   is_contiguous() == False
```

즉 `shN`의 원소는 메모리에서 **45개 쓰고 3개 건너뛰기**를 반복하는 얼룩무늬 배치다. 두 뷰가 하나의 storage를 공유(`sh0.data_ptr() == colors.data_ptr()`, `shN`은 +12바이트 오프셋)하는 점도 주의할 점이다.

`.contiguous()`를 붙이면 새 버퍼로 복사(copy)되어 stride가 `(45, 3, 1)`로 촘촘해지고, `sh0`/`shN`은 서로 독립적인 메모리를 갖는다.

## fused Adam이 거부하는 정확한 메커니즘

`fused=True` Adam은 여러 텐서를 하나의 CUDA 커널(multi-tensor apply)로 한 번에 갱신한다. 이 커널은 텐서를 **평평한 연속 버퍼**로 보고 인덱싱하므로, `param`·`grad`·`exp_avg`·`exp_avg_sq`의 **layout(stride)이 전부 같아야** 한다.

여기서 어긋남이 발생한다.

- `param` = 뷰 → stride `(48, 3, 1)`
- `grad`(autograd가 만든 것)와 `exp_avg`/`exp_avg_sq`(`torch.zeros_like(p, memory_format=preserve_format)`) → 원본이 dense하지 않으므로 **연속 텐서로 생성**되어 stride `(45, 3, 1)`

torch 2.9.1 + CUDA에서 실제로 재현한 결과:

```
sh0 view stride (48, 3, 1) contig False
shN view stride (48, 3, 1) contig False
exp_avg stride (45, 3, 1)   param stride (48, 3, 1)
grad    stride (45, 3, 1)
→ RuntimeError: params, grads, exp_avgs, and exp_avg_sqs must have same
                dtype, device, and layout
```

`.contiguous()`를 붙인 파라미터는 같은 코드가 정상 동작한다(`stride (45, 3, 1)`로 전부 일치). 참고로 **`fused=False`(기본 Adam)는 비연속 텐서도 그냥 돌아간다** — 느린 elementwise 경로라 stride를 존중하기 때문이다. 즉 이 에러는 "fused를 켠 대가"다. 반대로 `grad`만 비연속이어도 같은 에러가 난다.

## 함정: 상류 `simple_trainer.py`는 `.contiguous()`가 없다

`examples/simple_trainer.py:346`은 `.contiguous()` 없이 슬라이스를 그대로 쓰는데도 `fused=True`로 잘 돌아간다.

```python
colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))   # ← device 인자 없음! CPU 텐서
params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
splats = torch.nn.ParameterDict({...}).to(device)     # ← CPU → CUDA 복사
```

차이는 `.to(device)`가 실제 복사를 하는지 여부다.

- `simple_trainer`: `colors`가 **CPU**에서 만들어지므로 `.to(device)`가 진짜 H2D 복사를 수행하고, `preserve_format`은 dense하지 않은 텐서의 stride를 보존하지 못해 결과가 **자동으로 contiguous**가 된다(`stride (45,3,1)`). 우연히 살아난 케이스.
- 워크스루: `colors`를 처음부터 `device=device`(CUDA)에 만들었으므로 `.to(device)`가 **no-op**이고, 비연속 뷰가 그대로 살아남는다 → 명시적 `.contiguous()`가 필수.

정리하면 `.to(device)`의 암묵적 정규화에 기대지 말고, **fused optimizer에 넘길 파라미터는 만들 때 명시적으로 연속화**하는 것이 안전하다.

## 같은 결의 규칙 — CUDA 커널 전반

이건 Adam만의 이야기가 아니라 gsplat의 CUDA 커널 전반에 적용되는 계약이다. 예: `gsplat/strategy/ops.py`의 MCMC 퍼터베이션 커널은

```python
# positions is modified in-place — cannot .contiguous()-copy;
# fall back to PyTorch if non-contiguous.
if not positions.is_contiguous():
    return False
...
torch.ops.gsplat.mcmc_perturb_positions(positions, quats.contiguous(),
                                        scales.contiguous(),
                                        opacities.flatten().contiguous(), ...)
```

- **읽기만 하는 입력**은 `.contiguous()`로 복사해 넘겨도 무해하다.
- **in-place로 갱신되는 텐서**는 복사하면 결과가 원본에 반영되지 않으므로 복사할 수 없다 → 비연속이면 PyTorch 경로로 폴백.

파라미터 텐서(`means`, `sh0`, `shN` …)는 optimizer가 in-place로 갱신하는 두 번째 부류이므로, **생성 시점에 연속으로 만드는 것**만이 유일한 해법이다. 밀도화(densification)로 파라미터가 교체될 때는 `torch.cat`/`index_select`가 새 연속 텐서를 만들므로 이 조건이 자연히 유지된다.

## 한 줄 요약

`colors[:, :1, :]` 같은 슬라이스는 stride `(48,3,1)`의 비연속 뷰이고, fused Adam의 multi-tensor 커널은 param/grad/exp_avg의 layout 일치를 요구한다 → `must have same dtype, device, and layout` RuntimeError. `.contiguous()`로 독립적인 연속 버퍼를 만들어야 한다.
