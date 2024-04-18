from monai.transforms import (Compose, LoadImage, EnsureChannelFirst,
                              Orientation, Resize, ToTensor,
                              RandGaussianNoise, RandGaussianSmooth,
                              RandRotate, RandAdjustContrast,
                              RandFlip, RandScaleCrop)
import numpy as np


def pci_transform_train(spatial_size=(128, 128, 128),
                        p_gaussianNoise=0.3, p_smooth=0.3,
                        p_rotate=0.9, p_contrast=0.9, p_flip=0.5):
    return Compose([
        # intensity normalization is done in the data pre-processing step
        LoadImage(),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Resize(spatial_size=spatial_size),
        RandGaussianNoise(prob=p_gaussianNoise, mean=0.2, std=0.03),
        RandGaussianSmooth(prob=p_smooth, sigma_x=(0.5, 1.5),
                            sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5), approx='erf'),
        RandFlip(spatial_axis=[0], prob=p_flip),
        RandFlip(spatial_axis=[1], prob=p_flip),
        RandFlip(spatial_axis=[2], prob=p_flip),
        RandScaleCrop(roi_scale=(0.8, 1.2), max_roi_scale=(1.2, 1.2), random_center=True, lazy=False),
        RandRotate(range_x=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    range_y=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    range_z=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    mode=("bilinear"),
                    align_corners=(True),
                    padding_mode=("zeros"),
                    prob=p_rotate),
        RandAdjustContrast(gamma=(0.7, 1.5), prob=p_contrast),
        ToTensor(),
    ])


def pci_transform_val(spatial_size=(128, 128, 128)):
    return Compose([
        LoadImage(),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Resize(spatial_size=spatial_size),
        ToTensor(),
    ])
