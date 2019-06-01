from ..transforms.transforms import *


class TrainAugmentation:
    def __init__(self, width, height, mean=0, std=1.0):
        """
        Args:
            size: the size the of input image to the neural network.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.height = height
        self.width = width
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.width, self.height),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, width, height, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(width, height),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, width, height, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(width, height),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image