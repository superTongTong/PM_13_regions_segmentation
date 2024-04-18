from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
                              Orientationd, Resized, ToTensord,
                              RandGaussianNoised, RandGaussianSmoothd,
                              RandRotated, RandAdjustContrastd,
                              RandFlipd, RandScaleCropd)
import numpy as np


def pci_transform_train(all_key, cls123_key, spatial_size=(128, 128, 128),
                        p_gaussianNoise=0.3, p_smooth=0.3,
                        p_rotate=0.9, p_contrast=0.9, p_flip=0.5):
    train_transform = Compose([
        # intensity normalization is done in the data pre-processing step

        EnsureChannelFirstd(keys=all_key),
        Orientationd(keys=all_key, axcodes="RAS"),
        Resized(keys=all_key, spatial_size=spatial_size),
        RandGaussianNoised(keys=cls123_key, prob=p_gaussianNoise, mean=0.2, std=0.03),
        RandGaussianSmoothd(keys=cls123_key, prob=p_smooth, sigma_x=(0.5, 1.5),
                            sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5), approx='erf'),
        RandFlipd(keys=cls123_key, spatial_axis=[0], prob=p_flip),
        RandFlipd(keys=cls123_key, spatial_axis=[1], prob=p_flip),
        RandFlipd(keys=cls123_key, spatial_axis=[2], prob=p_flip),
        RandScaleCropd(keys=cls123_key, roi_scale=(0.8, 1.2), max_roi_scale=(1.2, 1.2), random_center=True, lazy=False),
        RandRotated(keys=cls123_key, range_x=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    range_y=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    range_z=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    mode=("bilinear"),
                    align_corners=(True),
                    padding_mode=("zeros"),
                    prob=p_rotate),
        RandAdjustContrastd(keys=all_key, gamma=(0.7, 1.5), prob=p_contrast),
        ToTensord(keys=all_key, ),
    ])
    return train_transform


def pci_transform_val(all_key, cls123_key, spatial_size=(128, 128, 128)):
    val_transform = Compose([
        EnsureChannelFirstd(keys=all_key),
        Orientationd(keys=all_key, axcodes="RAS"),
        Resized(keys=all_key, spatial_size=spatial_size),
        ToTensord(keys=all_key),
    ])
    return val_transform
