# Model Card

The candidate studies facial-expression recognition with a sparse pixel-level
graph. MediaPipe-derived priors determine selected face/context nodes and five
semantic anchors. Pixel, gradient, geometry, prior and local-detail channels
form 37-dimensional node evidence. Directed local/anchor edges carry eight
features.

This is a research candidate, not a production face-analysis system. FER2013
labels, imbalance, low resolution, annotation noise and demographic limitations
apply. Validation macro-F1 is the sole primary checkpoint selector. Test data
must not influence selection.

