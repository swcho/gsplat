A clean flat-design educational infographic, 16:9 landscape, titled at the top center in bold: "CUDA Kernel vs PyTorch rasterize_naive". Below the title a small subtitle in gray: "same math, different execution".

Layout: two tall vertical columns of equal width, left and right, separated by a narrow center gutter containing four horizontal double-headed arrows that link matching rows across the two columns. Reading flow is top-to-bottom within each column, and left-to-right across each arrow.

LEFT COLUMN, header bar in deep blue with white text: "CUDA kernel". Under it, four stacked cards, each with a small icon and a short label:
1. A grid of many small blue squares, one square highlighted with a bright outline. Label under it: "blockIdx = one tile". Small caption: "thousands run at once".
2. A single large square subdivided into a 16 x 16 checkerboard of tiny cells, with the text "16 x 16 = 256 threads" beside it. Label: "threadIdx = one pixel".
3. A horizontal strip of small circles labeled "Gaussians, front to back", with a bracket over the first 256 circles marked "batch of 256" and a small box beneath labeled "shared memory".
4. Four small rectangular chips drawn like hardware registers, labeled "T", "pix_out", "done", "cur_idx". Caption above them: "per-thread registers".

RIGHT COLUMN, header bar in warm orange with white text: "PyTorch rasterize_naive". Under it, four stacked cards mirroring the left ones:
1. A vertical stack of code-like lines with a looping circular arrow. Label: "for tile in range(th*tw)". Small caption: "one at a time".
2. A 2D tensor drawn as a bordered 16 x 16 grid of cells with a shape tag "[16, 16] tensor". Label: "elementwise = SIMT".
3. A horizontal strip of small circles with a bracket labeled "flatten_ids[start:end]". Caption: "plain indexing, no shared mem".
4. Four small rounded tensor blocks labeled "T", "out", "done", "cnt". Caption above them: "per-pixel tensors".

CENTER GUTTER: four double-headed arrows, one per row, each with a tiny label in dark gray: row 1 "block = tile", row 2 "thread = pixel", row 3 "serial walk", row 4 "register = tensor".

BOTTOM BAND spanning the full width, light gray background, split into two equal boxes side by side:
Left box titled "Branch becomes mask": show "if (!done) continue;" on the left and an arrow to "valid mask + torch.where" on the right.
Right box titled "Early exit": show "__syncthreads_count == 256" on the left and an arrow to "if done.all(): break" on the right.

Style: clean flat vector infographic, generous white space, off-white background, restrained palette of deep blue, warm orange, gray and white, thin rounded borders, monospace font for code snippets and sans-serif for labels. All text crisp and correctly spelled. No photographic elements, no gradients, no 3D shading. Aspect ratio 16:9.
