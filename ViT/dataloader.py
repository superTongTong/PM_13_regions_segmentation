from torchvision import transforms
import numpy as np
import glob
from PCI_dataset import PCI_Dataset
from base_data_loader import BaseDataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirst,
    CropForegroundd,
    Spacingd,
    Orientationd,
    SpatialPadd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandRotated,
    RandZoomd,
    CastToTyped,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandFlipd,
    ToTensord,
)


def pci_train_transform(image_key, spatial_size=(64,64,64)):
    train_transform = Compose([
        # AddChanneld(keys=all_keys),
        # EnsureChannelFirst(channel_dim=image_keys),
        # CropForegroundd(keys=all_keys, source_key=image_keys[0]),
        Orientationd(keys=image_key, axcodes="RAS"),
        SpatialPadd(keys=image_key, spatial_size=spatial_size),
        # RandZoomd(
        #     keys=image_key,
        #     min_zoom=0.7,
        #     max_zoom=1.5,
        #     mode=("bilinear",) * len(image_key) + ("nearest",),
        #     align_corners=(True,) * len(image_key) + (None,),
        #     prob=0.1,
        # ),
        # RandRotated(
        #     keys=all_keys,
        #     range_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
        #     range_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
        #     mode=("bilinear",) * len(image_keys) + ("nearest",),
        #     align_corners=(True,) * len(image_keys) + (None,),
        #     padding_mode=("border", ) * len(all_keys),
        #     prob=0.3,
        # ),
        # RandAdjustContrastd(  # same as Gamma in nnU-Net
        #     keys=image_keys,
        #     gamma=(0.7, 1.5),
        #     prob=0.3,
        # ),
        # RandFlipd(image_key, spatial_axis=[0], prob=0),  # Only right-left flip
        NormalizeIntensityd(keys=image_key, nonzero=True, channel_wise=True),
        # CastToTyped(keys=image_key, dtype=(np.float32,) * len(image_key) + (np.uint8,)),
        ToTensord(keys=image_key),
    ])
    return train_transform


def pci_validation_transform(image_key, spatial_size=(64,64,64)):
    val_transform = Compose([
        # AddChanneld(keys=all_keys),
        # EnsureChannelFirst(channel_dim=image_keys),
        # CropForegroundd(keys=all_keys, source_key=image_keys[0]),
        Orientationd(keys=image_key, axcodes="RAS"),
        SpatialPadd(keys=image_key, spatial_size=spatial_size),
        NormalizeIntensityd(keys=image_key, nonzero=True, channel_wise=True),
        # CastToTyped(keys=image_key, dtype=(np.float32,) * len(image_key) + (np.uint8,)),
        ToTensord(keys=image_key),
    ])
    return val_transform

class PCI_DataLoader(BaseDataLoader):
    """
    PCI score dataset Data Loader.
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, split='train',
                 prefetch_factor=2):

        train_transform = transforms.Compose([
            pci_train_transform(image_key='image')
        ])

        validation_transform = transforms.Compose([
            pci_validation_transform(image_key='image')
        ])

        # Store the train/test/validation dataset location
        self.data_dir = data_dir

        # train dataset
        if split == 'train':
            self.dataset = PCI_Dataset(self.data_dir, split=split, transform=train_transform)

        # test/validation dataset
        elif split == 'test' or split == 'validation':
            self.dataset = PCI_Dataset(self.data_dir, split=split, transform=validation_transform)
        else:
            raise ValueError('Invalid split name: {}'.format(split))

        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers, prefetch_factor=prefetch_factor)


def main():
    data_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/cropped_scan/'
    # test the data loader
    train_loader = PCI_DataLoader(data_dir, batch_size=1, shuffle=True, split='train')
    valid_loader = PCI_DataLoader(data_dir, batch_size=1, shuffle=False, split='validation')
    print('train_loader:', len(train_loader))
    print('valid_loader:', len(valid_loader))

if __name__ == '__main__':
    main()

