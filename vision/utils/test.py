import numpy as np
import collections
import torch
import itertools
from typing import List
import math
import torch.nn as nn
import torch.nn.functional as F
from vision.ssd.inception3_ssd import create_inception3_ssd
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from vision.nn.inception import Inception3
from vision.ssd.inception3_ssd_network import Inception_SSD
from vision.ssd.config import inception3_ssd_config as config

image_size = 300
image_mean = np.array([123, 117, 104])  # RGB layout
image_std = 1.0

iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])

SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])

specs = [
    SSDSpec(35, 8, SSDBoxSizes(30, 60), [2]),
    SSDSpec(17, 16, SSDBoxSizes(60, 111), [2, 3]),
    SSDSpec(9, 32, SSDBoxSizes(111, 162), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
    SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
]

def generate_ssd_priors(specs: List[SSDSpec], image_size, clamp=True) -> torch.Tensor:
    """Generate SSD Prior Boxes.

    It returns the center, height and width of the priors. The values are relative to the image size
    Args:
        specs: SSDSpecs about the shapes of sizes of prior boxes. i.e.
            specs = [
                SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
                SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
                SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
                SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
                SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
            ]
        image_size: image size.
        clamp: if true, clamp the values to make fall between [0.0, 1.0]
    Returns:
        priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    priors = []
    for spec in specs:
        scale = image_size / spec.shrinkage
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale
            # small sized quare box
            size = spec.box_sizes.min
            h = w = size/image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])
            # big sized quare box
            size = math.sqrt(spec.box_sizes.min * spec.box_sizes.max)
            h = w = size/image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # change h/w ratio of the small sized box
            size = spec.box_sizes.min
            h = w = size / image_size
            for ratio in spec.aspect_ratios:
                ratio = math.sqrt(ratio)
                priors.append([
                    x_center,
                    y_center,
                    w * ratio,
                    h / ratio
                ])
                priors.append([
                    x_center,
                    y_center,
                    w / ratio,
                    h * ratio
                ])

    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False


if __name__ == '__main__':

     priors = generate_ssd_priors(specs, image_size)
     print("priors", priors)
     num_classes = 21

     input = torch.randn(3, 5, requires_grad=True)
     target = torch.randint(5, (3,), dtype=torch.int64)
     loss = F.cross_entropy(input, target)
     print("loss:", loss)
     print("Build network.")
     # let us go deep put the implementation of net archtecture here

     base_net = Inception3(1001).layers

     source_layer_indexes = [
         (10, BatchNorm2d(288)),
         len(base_net),
     ]

     extras = ModuleList([
         Sequential(
             Conv2d(in_channels=768, out_channels=256, kernel_size=1),
             ReLU(),
             Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
             ReLU()
         ),
         Sequential(
             Conv2d(in_channels=512, out_channels=128, kernel_size=1),
             ReLU(),
             Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
             ReLU()
         ),
         Sequential(
             Conv2d(in_channels=256, out_channels=128, kernel_size=1),
             ReLU(),
             Conv2d(in_channels=128, out_channels=256, kernel_size=3),
             ReLU()
         ),
         Sequential(
             Conv2d(in_channels=256, out_channels=128, kernel_size=1),
             ReLU(),
             Conv2d(in_channels=128, out_channels=256, kernel_size=3),
             ReLU()
         )
     ])

     regression_headers = ModuleList([
         Conv2d(in_channels=288, out_channels=4 * 4, kernel_size=3, padding=1),
         Conv2d(in_channels=768, out_channels=6 * 4, kernel_size=3, padding=1),
         Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
         Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
         Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
         Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
         # TODO: change to kernel_size=1, padding=0?
     ])

     classification_headers = ModuleList([
         Conv2d(in_channels=288, out_channels=4 * num_classes, kernel_size=3, padding=1),
         Conv2d(in_channels=768, out_channels=6 * num_classes, kernel_size=3, padding=1),
         Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
         Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
         Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
         Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
         # TODO: change to kernel_size=1, padding=0?
     ])

    #just build the network architecture
     net = Inception_SSD(num_classes, base_net, source_layer_indexes, extras, classification_headers,
                          regression_headers, is_test=False, config=config, transform_input=False)
     #net = create_inception3_ssd(21)
     print("Build network already")



