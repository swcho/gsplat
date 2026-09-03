Create a clean flat-design educational infographic, landscape 16:9, titled "rasterization(): The Whole Differentiable Renderer".

Overall layout: a top title band, a wide horizontal 4-stage pipeline occupying the middle two thirds, and a bottom feedback strip. Eye flow goes left to right through the pipeline, then a curved arrow loops back from right to left along the bottom.

TOP-LEFT INPUT BOX (before the pipeline), labeled "Gaussian params":
small stacked chips reading "means", "quats", "scales", "opacities", "SH coeffs".
A thick arrow leads right into the pipeline.

MIDDLE: four numbered panels in a row, connected by thick right-pointing arrows, each panel a rounded card with a number badge (1..4), a short English label, a kernel name in monospace, and a simple icon:

Panel 1 - badge "1", label "SH Evaluation" (한국어: SH 평가), monospace "spherical_harmonics".
Icon: a camera icon with a dashed view-direction ray hitting a sphere that shades from blue to orange, plus a tiny row of 16 small squares labeled "16 coeffs".
Small caption under panel: "view-dependent RGB".

Panel 2 - badge "2", label "Projection / EWA" (한국어: 투영), monospace "fully_fused_projection".
Icon: a 3D ellipsoid on the left, a camera frustum, and a flat 2D ellipse on an image plane on the right. Small formula chip reading "3D covariance to 2D conic". Tiny outputs list: "means2d", "conics", "depths", "radii".
Small caption: "radii = 0 means culled".

Panel 3 - badge "3", label "Tile Intersection" (한국어: 타일 교차), monospace "isect_tiles".
Icon: an image rectangle overlaid with a 16x16 grid, two ellipses each highlighting the few grid cells they overlap; beside it a small sorted bar strip labeled "sort by (tile, depth)".
Small caption: "one radix sort".

Panel 4 - badge "4", label "Pixel Rasterize" (한국어: 픽셀 래스터화), monospace "rasterize_to_pixels".
Icon: a single pixel column showing 4 stacked translucent discs front to back with decreasing opacity, and a small "early stop" tag with a stop marker on the last one. Formula chip: "front-to-back alpha blend".
Small caption: "C = sum c_i a_i T_i".

RIGHT OUTPUT COLUMN after panel 4: three small stacked output chips, each a distinct accent color:
"render_colors", "render_alphas", "info (meta)".

BOTTOM FEEDBACK STRIP: a wide curved dashed arrow starting at the "info (meta)" chip, sweeping left back to the "Gaussian params" input box. Along the curve place two small labels: "means2d.grad" and "radii", and a mid-curve rounded tag reading "Densify: split / prune". Under the curve a short line of text: "loss gradient flows all the way back".

Style: clean flat vector educational infographic, generous white space on an off-white background, one cool primary (deep blue) plus one warm accent (orange) plus soft grays, thin consistent line weights, rounded rectangles, subtle drop shadows, sans-serif labels, monospace only for kernel names. No photorealism, no 3D bevels, no clutter. All text must be crisp and correctly spelled. Aspect ratio 16:9.
