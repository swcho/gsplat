A clean flat-design educational infographic explaining the 64-bit tile-intersection sort key used in 3D Gaussian Splatting rasterization. Landscape 16:9 aspect ratio. Light off-white background (#F7F6F3), thin grey grid lines, modern geometric sans-serif labels, crisp vector style, no photorealism, no gradients except subtle flat tints, generous white space.

Title at top center in bold dark navy: "One 64-bit Key, Three-Level Sort" with a smaller grey subtitle underneath: "image_id | tile_id | depth bits".

Layout: three stacked horizontal panels, reading top to bottom, connected by a vertical flow.

PANEL 1 (top, largest): A single long horizontal bar spanning the panel width, representing 64 bits, divided into three colored segments with clear vertical dividers.
- Left segment, narrow, deep purple fill, label inside white text: "image_id".
- Middle segment, medium width, teal fill, label inside white text: "tile_id".
- Right segment, widest (half the bar), warm orange fill, label inside white text: "float32 depth bits".
Above the bar, small grey tick labels aligned to the dividers: "bit 63" at far left, "bit 32" at the boundary between teal and orange, "bit 0" at far right.
Below each segment, a small caption in dark grey: under purple "image_n_bits"; under teal "tile_n_bits (variable)"; under orange "32 bits (fixed)".
A thin arrow above the whole bar pointing left to right labeled "high bits to low bits".

PANEL 2 (middle): The sorting process, drawn left to right in three groups.
- Left group titled "Unsorted keys": five small horizontal three-color mini-bars stacked vertically in scrambled order, each mini-bar using the same purple/teal/orange segments.
- Center: a thick right-pointing arrow in dark navy with the bold white label "RADIX SORT" on it, and a small grey caption below the arrow: "CUB SortPairs, O(n k)".
- Right group titled "Sorted keys": the same five mini-bars now visibly grouped, with two brackets on the right side labeled "tile 0" and "tile 1", and a small vertical downward arrow beside them labeled "near to far".

PANEL 3 (bottom): Result and why it matters, two side-by-side boxes.
- Left box: a simple 3x3 grid of square tiles over a tiny camera icon; one tile highlighted teal, and from it a horizontal row of three orange circles of increasing size labeled beneath in small text "near", "mid", "far".
- Right box: three overlapping translucent ellipses stacked front to back with a small eye icon on the left and a left-pointing arrow, labeled "front-to-back alpha blend"; a short red caption below: "order matters".

Add small Korean glosses in parentheses next to three key labels only, in a smaller grey weight: next to "tile_id" write "(타일)", next to "float32 depth bits" write "(깊이)", next to "RADIX SORT" write "(기수 정렬)".

All labels must be short, at most four words, spelled exactly as given. Keep text minimal and perfectly legible. Consistent color coding across all panels: purple = image, teal = tile, orange = depth.
