Create a clean flat-design educational infographic, 16:9 landscape, titled "5 Parameters of One 3D Gaussian".

Overall layout: a wide top hero band, a middle row of five equal vertical panels, and a thin bottom strip. Eye flow goes left to right across the five panels, then down to the bottom strip. Use a light neutral background (#F7F8FA), one soft accent color per panel, thin 1px gray panel borders with rounded corners, and a single clean sans-serif font.

TOP HERO BAND (left half): a large 3D ellipsoid blob drawn as a soft translucent gradient splat, tilted diagonally, with a small dot at its center. Annotate it with four thin leader lines and short labels: "center 위치", "3 axis radii 크기", "tilt 회전", "alpha 불투명도". TOP HERO BAND (right half): a formula card showing the text "Sigma = R S S^T R^T" in large type, with the caption "rotation x scale = shape" underneath.

MIDDLE ROW, five panels left to right. Each panel has a colored icon on top, then a bold parameter name, then a small gray shape tag, then one short label line:

Panel 1, blue: icon is a single dot with x-y-z axis arrows. Name "means". Tag "[N,3]". Label "Position, no activation".
Panel 2, teal: icon is an ellipse with three double-headed arrows along its axes. Name "scales". Tag "[N,3]". Label "Stored as log, use exp".
Panel 3, purple: icon is a curved rotation arrow circling an ellipse. Name "quats". Tag "[N,4]". Label "Quaternion, auto normalized".
Panel 4, amber: icon is three stacked translucent discs fading from solid to faint. Name "opacities". Tag "[N]". Label "Stored as logit, use sigmoid".
Panel 5, rose: icon is a sphere split into lobed spherical-harmonic patterns, with one big lobe labeled "sh0" and a grid of smaller lobes labeled "shN". Name "sh0 / shN". Tag "[N,1,3] + [N,15,3]". Label "SH coefficients, view color".

Draw a light bracket under panels 1 to 3 labeled "GEOMETRY 기하" and a light bracket under panels 4 to 5 labeled "APPEARANCE 외형".

BOTTOM STRIP: a horizontal arrow flow with four small nodes connected by arrows: "Unconstrained params" then "Activation exp / sigmoid" then "Differentiable rasterizer" then "Gradients back to all 5". Place a small right-aligned caption "59 numbers per Gaussian".

Style: modern educational infographic, flat vector illustration, generous whitespace, crisp geometric icons, no photorealism, no drop shadows, no clutter, all text short and perfectly legible.
