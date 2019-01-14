import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from ..nn.inception import Inception3
from .inception3_ssd_network import Inception_SSD
from .predictor import Predictor
from .config import inception3_ssd_config as config


def create_inception3_ssd(num_classes, is_test=False):

    base_net = Inception3(1001).layers

    source_layer_indexes = [
        (10, BatchNorm2d(288)),
        len(base_net),
    ]
    # source_layer_indexes = [
    #     (10, BatchNorm2d(288)), 15
    # ]

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
        Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=288, out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=768, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])

    return Inception_SSD(num_classes, base_net, source_layer_indexes, extras, classification_headers,
                          regression_headers, is_test=is_test, config=config, transform_input=False)


def create_inception_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
