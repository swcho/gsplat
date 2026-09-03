Create a clean flat-design educational infographic, 16:9 landscape, titled "EWA Splatting: Local Affine Approximation".

Style: modern minimal vector illustration on a light off-white background (#F7F6F3). Thin 2px outlines, muted palette of deep navy, teal, warm orange, and soft coral. Generous white space, crisp sans-serif labels, subtle drop shadows. No photorealism, no 3D rendering engine look. All text must be short, correctly spelled English.

Layout: one horizontal left-to-right pipeline across the top two thirds, and one comparison strip along the bottom third. Bold orange arrows connect panels left to right.

Panel 1 (far left), label above it "3D Gaussian in camera space": a teal wireframe ellipsoid tilted in space, with three thin axis lines through it showing its principal axes. Small caption below in dark navy: "Shape = covariance". Beside it, small formula text: "Sigma_c".

Arrow to the right.

Panel 2, label above it "Perspective projection is nonlinear": a simple flat camera icon (small dark navy box with a lens circle) with a light gray view frustum opening to the right. Inside the frustum draw a visibly CURVED grid surface, bending like a warped sheet, colored soft coral. Caption below: "x / z bends space". Small formula text: "pi(x,y,z) = (fx x/z, fy y/z)".

Panel 3, label above it "Linearize at the center": the same curved coral surface drawn faintly, with a flat teal TANGENT PLANE touching it at exactly one point. Mark that touch point with a bright orange dot labeled "mu_c". Caption below: "Tangent plane = Jacobian J". Show a small 2-by-3 matrix bracket beside it containing the entries "fx/z", "0", "-fx x/z^2" on the top row and "0", "fy/z", "-fy y/z^2" on the bottom row.

Arrow to the right.

Panel 4 (far right), label above it "2D ellipse on screen": a rectangular screen frame with a faint pixel grid, containing one tilted orange ellipse with two concentric dashed rings around it labeled "1 sigma" and "3 sigma". Caption below: "Now rasterizable".

Centered directly under the four panels, in a wide rounded box with a pale teal fill, display the key equation large and bold: "Sigma_2D = J Sigma_c J^T + eps I". Under it a short line of small text: "affine rule reused locally".

Bottom strip: two small side-by-side comparison cards separated by a vertical divider.
Left card, header "Accurate": a small camera far away and a tiny ellipsoid, with a green check mark, label "Small, far Gaussians".
Right card, header "Distorted": a small camera very close to a large ellipsoid, the resulting screen ellipse drawn stretched and skewed, with an orange warning triangle, label "Near or large Gaussians". Under the right card, a short note: "3DGUT uses Unscented Transform".

Add a small legend line at the very bottom left in gray: "Center is exact, only shape is approximated".

Keep every label under five words. Do not add any extra text, paragraphs, or watermarks.
