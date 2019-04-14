import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

# using 800, 370
image_size = 300
image_mean = np.array([123, 117, 104])  # RGB layout
image_std = 1.0

iou_threshold = 0.4
iou_threshold2 = 0.6
center_variance = 0.1
size_variance = 0.2

# the input image size is changed to [800, 370] instead of [300, 300] used in voc.
# we need to redefine the SSDSpec, the shrinkage is still the same;
# specs = [
#     SSDSpec(35, 8, SSDBoxSizes(30, 60), [2]),
#     SSDSpec(17, 16, SSDBoxSizes(60, 111), [2, 3]),
#     SSDSpec(9, 32, SSDBoxSizes(111, 162), [2, 3]),
#     SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
#     SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
#     SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
# ]
# 20.48, 51.2, 133.12, 215.04, 296.96, 378.88 and 460.8
specs = [
    SSDSpec(100, 46, 8, SSDBoxSizes(20.48, 51.2), [2]),
    SSDSpec(50, 23, 16, SSDBoxSizes(51.2, 133.12), [2, 3]),
    SSDSpec(25, 12, 32, SSDBoxSizes(133.12, 215.04), [2, 3]),
    SSDSpec(13, 6, 64,  SSDBoxSizes(215.04, 296.96), [2, 3]),
    SSDSpec(7, 3, 114, SSDBoxSizes(296.96, 378.88), [2]),
    SSDSpec(2, 1, 370, SSDBoxSizes(378.88, 460.8), [2])
]
priors = generate_ssd_priors(specs, image_size)