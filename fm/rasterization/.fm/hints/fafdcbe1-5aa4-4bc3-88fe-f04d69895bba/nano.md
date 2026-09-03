Create a clean, educational flat-design infographic, 16:9 landscape, titled "gsplat rasterization(): 7 Stages". White background, thin gray grid guides, a single left-to-right horizontal flow of 7 numbered stage cards connected by bold arrows. Use a cool color scheme: slate gray text, blue for geometry stages, amber for the color stage, green for the sorting stages, magenta for the final blending stage.

Top strip (input bar), a light gray rounded rectangle spanning the width, small monospace labels inside: "means [N,3]", "quats", "scales", "opacities", "SH colors", "viewmats", "Ks". Label the strip "INPUT".

Middle band: seven square cards in a row, each with a big circled number, a short bold English title, one tiny formula or icon, and a small monospace function name at the bottom.

1) "3D Covariance" — formula "S = R S S^T R^T", icon: a tilted 3D ellipsoid with 3 axis arrows — footer "quat_scale_to_covar_preci"
2) "Camera Transform" — formula "mu_c = R mu + t", icon: a camera frustum with a point inside — footer "(fused)"
3) "EWA Projection" — formula "J S_c J^T + eps", icon: 3D ellipsoid flattening into a 2D ellipse on an image plane — footer "fully_fused_projection"
4) "SH Color" — icon: a sphere with colored lobes and a view-direction arrow — footer "spherical_harmonics"
5) "Tile Intersection" — icon: a 4x3 tile grid with one ellipse overlapping several tiles, plus a tiny 64-bit key bar split into three segments labeled "image | tile | depth" — footer "isect_tiles"
6) "Tile Offsets" — icon: a sorted list with bracket markers pointing to tile start indices — footer "isect_offset_encode"
7) "Alpha Blending" — formula "C = sum c_i a_i T_i", icon: three overlapping translucent ellipses stacked front-to-back over a pixel grid — footer "rasterize_to_pixels"

Draw a prominent rounded dashed bracket enclosing ONLY cards 2 and 3, colored blue, with a bold callout label above it: "ONE FUSED CUDA KERNEL" and a smaller line under it: "fully_fused_projection". This bracket must be the visual highlight of the poster.

Under cards 5, 6, 7 draw a thin horizontal band labeled "GPU MAPPING" with three short labels aligned under them: "sort by tile+depth", "per-tile start index", "block = tile, thread = pixel".

Bottom strip (output bar), light gray rounded rectangle, monospace labels: "render_colors [C,H,W,D]", "render_alphas [C,H,W,1]", "meta". Label the strip "OUTPUT".

Group the seven cards visually into four phases with small caption text above the row: "SHAPE" over card 1, "3D to 2D" over cards 2-3, "COLOR" over card 4, "SORT" over cards 5-6, "PAINT" over card 7.

Style: modern educational poster, flat vector, crisp thin outlines, generous whitespace, sans-serif labels, all text short and correctly spelled English, no clutter, no photorealism.
