# %% [markdown]
# # `knn_mean_dist` — k-최근접 이웃 평균거리와 청크 처리
#
# 필요 패키지: torch, numpy, plotly, kaleido
#
# 3DGS 초기화에서 각 Gaussian의 `scales`는 **주변 점 밀도**에 맞춰 정해진다.
# 점 $p_i$에 대해 자기 자신을 뺀 $k$개의 최근접 이웃 거리를 $d_{i,1}\le\dots\le d_{i,k}$라 하면
#
# $$\bar d_i = \frac{1}{k}\sum_{j=1}^{k} d_{i,j},\qquad
#   \texttt{scales}_i = \log \bar d_i \;\;(\text{3축 동일})$$
#
# 즉 이웃이 멀면(희소) 큰 Gaussian, 가까우면(조밀) 작은 Gaussian이 된다.
#
# 문제는 계산량이 아니라 **메모리**다. `torch.cdist(points, points)`는 $[N,N]$ 행렬을
# 한 번에 만들므로 float32에서 $4N^2$ 바이트가 필요하다. $N=300{,}000$이면 335 GiB.
# 그래서 질의 점을 `chunk=8192`개씩 잘라 $[\text{chunk}, N]$ 행렬만 만들고
# 부분 결과를 이어붙인다: 최대 메모리가 $4\cdot\text{chunk}\cdot N$로 떨어진다.

# %%
import numpy as np
import torch
import plotly.graph_objects as go


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print("device:", DEV, "| torch:", torch.__version__)
# 출력: device: cuda | torch: 2.9.1+cu128

# %% [markdown]
# ## 1. 원본 구현과 순진한(naive) 구현
#
# 워크스루의 구현은 청크 루프 안에서 세 가지를 한다.
#
# 1. `torch.cdist(points[i:i+chunk], points)` → `[chunk, N]` 거리 행렬
# 2. `d.topk(k+1, largest=False).values[:, 1:]` → 가장 작은 $k+1$개 중 **첫 열(자기 자신, 거리 0)을 버림**
# 3. `.mean(dim=-1)` → 이웃 $k$개 거리의 평균
#
# 마지막에 `torch.cat(out)`으로 다시 `[N]` 텐서로 합친다.

# %%
def knn_mean_dist(points: torch.Tensor, k: int = 3, chunk: int = 8192) -> torch.Tensor:
    """각 점에서 k-최근접 이웃까지의 평균 거리 (GPU, 청크 처리)."""
    out = []
    for i in range(0, len(points), chunk):
        d = torch.cdist(points[i : i + chunk], points)  # [chunk, N]
        knn_d = d.topk(k + 1, largest=False).values[:, 1:]  # 자기 자신 제외
        out.append(knn_d.mean(dim=-1))
    return torch.cat(out)


def knn_mean_dist_naive(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """청크 없이 [N, N]을 한 번에 만드는 버전 — 결과는 같고 메모리만 폭발한다."""
    d = torch.cdist(points, points)  # [N, N]  <-- 여기가 문제
    return d.topk(k + 1, largest=False).values[:, 1:].mean(dim=-1)


pts = torch.rand(5000, 3, device=DEV) * 4.0
a = knn_mean_dist(pts, k=3, chunk=512)
b = knn_mean_dist_naive(pts, k=3)
print("shape:", tuple(a.shape), "| naive 대비 max abs diff:", (a - b).abs().max().item())
print("chunk=1 대비 max abs diff:", (a - knn_mean_dist(pts, 3, chunk=1)).abs().max().item())
# 출력: shape: (5000,) | naive 대비 max abs diff: 0.0
# 출력: chunk=1 대비 max abs diff: 2.3096799850463867e-05

# %% [markdown]
# **청크 크기는 결과를 (수치오차 범위 안에서) 바꾸지 않는다.** 각 행(질의 점)의 이웃 탐색이
# 서로 독립이고, 모든 청크가 항상 *전체* `points`를 상대로 거리를 재기 때문이다.
# (`chunk=1`에서 $10^{-5}$ 수준 차이가 보이는 건 행 수에 따라 `cdist`가 다른 커널 경로를
# 타면서 생기는 float32 반올림 차이일 뿐, 알고리즘 차이가 아니다.)
# `chunk`는 순수하게 메모리↔커널 호출 횟수의 트레이드오프 노브다.
#
# ## 2. 왜 청크인가 — $[N,N]$ 행렬의 크기

# %%
def gib(n_rows: int, n_cols: int) -> float:
    return n_rows * n_cols * 4 / 2**30  # float32


print(f"{'N':>10} {'[N,N] 전체':>14} {'[8192,N] 청크':>14} {'절감':>8}")
for N in (50_000, 100_000, 300_000, 1_000_000):
    full, ch = gib(N, N), gib(min(8192, N), N)
    print(f"{N:>10,} {full:>11.1f} GiB {ch:>11.1f} GiB {full / ch:>7.0f}x")
# 출력:          N     [N,N] 전체  [8192,N] 청크       절감
# 출력:     50,000         9.3 GiB         1.5 GiB       6x
# 출력:    100,000        37.3 GiB         3.1 GiB      12x
# 출력:    300,000       335.3 GiB         9.2 GiB      37x
# 출력:  1,000,000      3725.3 GiB        30.5 GiB     122x
#
# COLMAP SfM 포인트가 보통 10만~30만 개이므로 전체 행렬은 어떤 GPU에도 올라가지 않는다.
# 반대로 청크 버전은 N에 대해 **선형**으로만 커진다(위 표의 3번째 열).

# %%
# 실제 할당량 측정 (CUDA에서만 의미 있음)
if DEV == "cuda":
    P = torch.rand(40_000, 3, device=DEV) * 4.0
    for label, fn in (("chunk=8192", lambda: knn_mean_dist(P, 3, 8192)),
                      ("naive(N,N)", lambda: knn_mean_dist_naive(P, 3))):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        print(f"N=40,000 {label:>11}: peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
# 출력: N=40,000  chunk=8192: peak 2.45 GiB
# 출력: N=40,000  naive(N,N): peak 6.03 GiB

# %% [markdown]
# ## 3. 청크 크기에 따른 속도 / 메모리 곡선
#
# 청크를 너무 작게 잡으면 커널 호출·`topk` 오버헤드가 늘고, 너무 크게 잡으면 OOM이다.
# 8192는 "수 GiB 안에서 GPU를 충분히 채우는" 실용적 중간값이다.

# %%
import time

N = 40_000
P = torch.rand(N, 3, device=DEV) * 4.0
chunks, times, peaks = [256, 512, 1024, 2048, 4096, 8192, 16384], [], []
for c in chunks:
    if DEV == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    knn_mean_dist(P, 3, c)  # warm-up
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    knn_mean_dist(P, 3, c)
    if DEV == "cuda":
        torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1e3)
    peaks.append(torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else gib(c, N))
    print(f"chunk={c:>6}: {times[-1]:7.1f} ms, peak {peaks[-1]:.2f} GiB")
# 출력: chunk=   256:   108.0 ms, peak 0.09 GiB
# 출력: chunk=   512:    93.3 ms, peak 0.16 GiB
# 출력: chunk=  1024:    87.9 ms, peak 0.31 GiB
# 출력: chunk=  2048:    84.9 ms, peak 0.62 GiB
# 출력: chunk=  4096:    83.4 ms, peak 1.23 GiB
# 출력: chunk=  8192:    82.8 ms, peak 2.45 GiB
# 출력: chunk= 16384:    82.4 ms, peak 4.89 GiB
#
# 시간은 chunk≈2048부터 거의 평평해지고(83~85 ms) 메모리만 선형으로 증가한다.
# → 속도를 위해 chunk를 더 키울 이유가 없고, 8192는 안전한 상한 쪽 선택이다.

# %% [markdown]
# ## 4. 이 값이 실제로 무엇에 쓰이는가 — `scales` 초기화
#
# 조밀한 영역과 희소한 영역을 섞은 점군에서 $\bar d_i$가 밀도를 그대로 반영하는지 본다.

# %%
g = torch.Generator(device="cpu").manual_seed(1)
dense = torch.randn(3000, 3, generator=g) * 0.15 + torch.tensor([-1.5, 0.0, 0.0])
sparse = torch.randn(600, 3, generator=g) * 0.9 + torch.tensor([1.5, 0.0, 0.0])
mixed = torch.cat([dense, sparse]).to(DEV)

d_avg = knn_mean_dist(mixed, k=3, chunk=1024)
scales_log = torch.log(d_avg)  # simple_trainer 의 splats["scales"] 초기값
print(f"조밀 영역  mean d̄ = {d_avg[:3000].mean():.4f}, log = {scales_log[:3000].mean():.3f}")
print(f"희소 영역  mean d̄ = {d_avg[3000:].mean():.4f}, log = {scales_log[3000:].mean():.3f}")
print(f"희소/조밀 크기 비 = {d_avg[3000:].mean() / d_avg[:3000].mean():.1f}x")
# 출력: 조밀 영역  mean d̄ = 0.0337, log = -3.512
# 출력: 희소 영역  mean d̄ = 0.3428, log = -1.191
# 출력: 희소/조밀 크기 비 = 10.2x
#
# 희소 영역의 Gaussian이 약 10배 크게 초기화된다 → 빈틈 없이 덮되 과하게 겹치지 않는다.

# %% [markdown]
# ## 5. 함정: 중복 점 → 거리 0 → `log` 가 $-\infty$
#
# `[:, 1:]`은 "0번째 열이 자기 자신"이라는 가정이다. 완전히 겹친 점이 $k$개 이상 있으면
# 이웃 거리도 0이 되어 $\bar d_i = 0$, `torch.log(0) = -inf`가 `scales`에 들어간다.

# %%
dup = torch.cat([torch.zeros(4, 3), torch.rand(20, 3) + 1.0]).to(DEV)
d_dup = knn_mean_dist(dup, k=3, chunk=8)
print("중복점 d̄:", d_dup[:4].tolist(), "→ log:", torch.log(d_dup[:4]).tolist())
# 출력: 중복점 d̄: [0.0, 0.0, 0.0, 0.0] → log: [-inf, -inf, -inf, -inf]
print("clamp 후 log:", torch.log(d_dup[:4].clamp_min(1e-7)).tolist()[:2])
# 출력: clamp 후 log: [-16.11809539794922, -16.11809539794922]

# %% [markdown]
# ## 6. 상류 구현과의 차이
#
# `examples/utils.py:156`의 `knn()`은 sklearn `NearestNeighbors`(CPU, KD-tree)를 쓰고,
# `simple_trainer.py:321`은 거리의 **제곱평균제곱근(RMS)** 을 쓴다:
#
# $$\texttt{utils}: \sqrt{\tfrac{1}{k}\sum_j d_{i,j}^2}\quad\text{vs}\quad
#   \texttt{walkthrough}: \tfrac{1}{k}\sum_j d_{i,j}$$
#
# 젠센 부등식으로 RMS $\ge$ 산술평균이므로 워크스루 쪽이 살짝 작은 초기 스케일을 준다.

# %%
d = torch.cdist(mixed, mixed).topk(4, largest=False).values[:, 1:]
rms = (d**2).mean(dim=-1).sqrt()
am = d.mean(dim=-1)
print(f"RMS/AM 평균 비: {(rms / am).mean():.4f}  (항상 >= 1)")
print(f"log 차이 평균:  {(torch.log(rms) - torch.log(am)).mean():.4f}")
# 출력: RMS/AM 평균 비: 1.0279  (항상 >= 1)
# 출력: log 차이 평균:  0.0270

# %% [markdown]
# ## 7. 시각화
#
# 왼쪽: 조밀/희소 혼합 점군을 $\log_{10}\bar d_i$(= 초기 Gaussian 크기)로 색칠.
# 조밀 클러스터는 어둡고(작은 Gaussian), 희소 영역은 밝다(큰 Gaussian).
# 오른쪽: $[N,N]$ 전체($O(N^2)$) vs $[8192,N]$ 청크($O(N)$)의 메모리 스케일링과 24 GiB GPU 한계선.
#
# 주의: plotly의 log 축에서 `add_hline`/`add_vline`의 좌표는 **log10 값**이어야 한다
# (`y=24`로 쓰면 $10^{24}$ 위치에 선이 그려져 축 범위가 망가진다).

# %%
import math

from plotly.subplots import make_subplots

m = mixed.cpu().numpy()
dv = d_avg.cpu().numpy()
fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.19,
    subplot_titles=("점군 색 = k-NN 평균거리 d̄ (초기 scale)", "cdist 거리행렬 메모리: 전체 vs 청크"),
)
fig.add_trace(
    go.Scatter(
        x=m[:, 0], y=m[:, 1], mode="markers", customdata=dv,
        marker=dict(size=4, color=np.log10(dv), colorscale="Viridis", showscale=True,
                    colorbar=dict(title="log₁₀ d̄", x=0.435, thickness=12, len=0.85)),
        name="points", showlegend=False, hovertemplate="d̄=%{customdata:.4f}<extra></extra>",
    ), row=1, col=1,
)
Ns = np.array([1_000, 5_000, 20_000, 50_000, 100_000, 300_000, 1_000_000])
fig.add_trace(go.Scatter(x=Ns, y=Ns * Ns * 4 / 2**30, mode="lines+markers",
                         name="[N,N] 전체 (O(N²))", line=dict(color="#d62728")), row=1, col=2)
fig.add_trace(go.Scatter(x=Ns, y=np.minimum(8192, Ns) * Ns * 4 / 2**30, mode="lines+markers",
                         name="[8192,N] 청크 (O(N))", line=dict(color="#2ca02c")), row=1, col=2)
fig.add_hline(y=math.log10(24), line=dict(color="gray", dash="dash"),  # log 축 → log10 좌표
              annotation_text="24 GiB GPU", annotation_position="bottom left", row=1, col=2)
fig.update_xaxes(type="log", title_text="N (점 개수)", row=1, col=2)
fig.update_yaxes(type="log", title_text="거리 행렬 메모리 (GiB)", dtick=1, row=1, col=2)
fig.update_xaxes(title_text="x", row=1, col=1)
fig.update_yaxes(title_text="y", scaleanchor="x", row=1, col=1)
fig.update_layout(height=480, width=1150, title_text="knn_mean_dist: 무엇을 계산하고 왜 청크로 나누는가",
                  legend=dict(x=0.60, y=0.98, bgcolor="rgba(255,255,255,0.7)"))
fig.write_image("expy.png", scale=2)
_show(fig)
print("saved expy.png")
# 출력: saved expy.png
