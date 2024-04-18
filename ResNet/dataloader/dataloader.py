import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.data import CacheDataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
from transforms import pci_transform_train, pci_transform_val
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
                              Resized, ToTensord, RandGaussianNoised,
                              RandGaussianSmoothd, RandRotated, RandAdjustContrastd,
                              RandFlipd, RandScaleCropd)

def pci_transform_train(data, spatial_size=(128, 128, 128),
                        p_gaussianNoise=0.3, p_smooth=0.3,
                        p_rotate=0.9, p_contrast=0.9, p_flip=0.5):
    if data['label'] == 0:
        return [
            # intensity normalization is done in the data pre-processing step
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Resized(keys=["image"], spatial_size=spatial_size),
            ToTensord(keys=["image"], ),
        ]
    elif data['label'] == 1:
        return [
            # intensity normalization is done in the data pre-processing step
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Resized(keys=["image"], spatial_size=spatial_size),
            RandGaussianNoised(keys=["image"], prob=p_gaussianNoise, mean=0.2, std=0.03),
            RandGaussianSmoothd(keys=["image"], prob=p_smooth, sigma_x=(0.5, 1.5),
                                sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5), approx='erf'),
            RandFlipd(keys=["image"], spatial_axis=[0], prob=p_flip),
            RandFlipd(keys=["image"], spatial_axis=[1], prob=p_flip),
            RandFlipd(keys=["image"], spatial_axis=[2], prob=p_flip),
            RandScaleCropd(keys=["image"], roi_scale=(0.8, 1.2), max_roi_scale=(1.2, 1.2), random_center=True, lazy=False),
            RandRotated(keys=["image"], range_x=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                        range_y=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                        range_z=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                        mode=("bilinear"),
                        align_corners=(True),
                        padding_mode=("zeros"),
                        prob=p_rotate),
            RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=p_contrast),
            ToTensord(keys=["image"], ),
        ]
    else:
        raise ValueError(f"Unexpected label value: {data['label']}")

def pci_transform_val(spatial_size=(128, 128, 128)):
    return [
        LoadImaged(keys=["image"], ),
        EnsureChannelFirstd(keys=["image"], ),
        Orientationd(keys=["image"], axcodes="RAS"),
        Resized(keys=["image"], spatial_size=spatial_size),
        ToTensord(keys=["image"], ),
    ]

def get_data_list(data_dir, split='train'):

    assert split in ['train', 'validation', 'test'], 'Only train, validation, test are supported options.'
    assert os.path.exists(data_dir), 'data_dir path does not exist: {}'.format(data_dir)

    print('Loading dataset from: {}'.format(data_dir + '/' + split + '/'))
    data_dir_img = os.path.join(data_dir, split)
    file_names = sorted(os.listdir(data_dir_img))
    n_files = len(file_names)
    print('Number of scans: {}'.format(n_files))
    images = [os.path.join(data_dir_img, file_name) for file_name in file_names]
    labels = [int(file_name[14:15]) for file_name in file_names]
    caseID = [file_name[0:14] for file_name in file_names]
    return images, labels, caseID


def PCI_DataLoader(data_dir, batch_size=1, shuffle=True, split='train', spatial_size=(128, 128, 128), num_workers=2,
                   p_gaussianNoise=0.1, p_Smooth=0.1, p_Rotate=0.5, p_Contrast=0.5, p_flip=0.5):
    imgs, labels, caseID = get_data_list(data_dir, split=split)
    class_counts = [count for num, count in sorted(Counter(labels).items())]
    num_samples = sum(class_counts)
    class_weights = [num_samples / class_count for class_count in class_counts]
    data_files = [{"image": i, "label": l, "CaseID": id} for i, l, id in zip(imgs, labels, caseID)]
    if split == 'train':
        train_trans = pci_transform_train(data_files, spatial_size=spatial_size, p_gaussianNoise=p_gaussianNoise,
                                          p_smooth=p_Smooth, p_rotate=p_Rotate, p_contrast=p_Contrast,
                                          p_flip=p_flip)

        weights = [class_weights[labels[i]] for i in range(int(num_samples))]
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples), replacement=True)
        ds = CacheDataset(data=data_files, transform=Compose(train_trans), progress=False)
        data_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler)
    elif split == 'validation':
        val_trans = pci_transform_val(spatial_size=spatial_size)
        ds = CacheDataset(data=data_files, transform=Compose(val_trans), progress=False)
        data_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader, class_weights

def plot_img(in_data_loder, save_path):
    for i, data in enumerate(in_data_loder):
        img = data['image']
        label = data['label']
        caseID = data['CaseID']
        # check if the csaeID already exists in the save_path

        img_array = img.numpy()
        squeezed = np.squeeze(img_array)
        plt.figure(figsize=(15, 5))

        # Plot the first image
        plt.subplot(1, 3, 1)
        plt.imshow(squeezed[64, :, :], cmap="gray")

        # Plot the second image
        plt.subplot(1, 3, 2)
        plt.imshow(squeezed[:, 64, :], cmap="gray")

        # Plot the third image
        plt.subplot(1, 3, 3)
        plt.imshow(squeezed[:, :, 64], cmap="gray")

        plt.title(f'{caseID}, label: {label.item()}')
        # Check if the file already exists and increment a counter until a unique filename is found
        counter = 0
        while os.path.exists(f'{save_path}{caseID}_{counter}.png'):
            counter += 1

        plt.savefig(f'{save_path}{caseID}_{counter}.png')
        print(f'processed {caseID}_{counter}.png')
        plt.close()
        # plt.show()


def main():
    data_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/cropped_scan_test/'
    train_img_save_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/val_images/'
    # data_dir = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/cropped_scan_v2/'
    # train_img_save_dir = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/train_images_v2/'
    # val_img_save_dir = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/val_images/'
    os.makedirs(train_img_save_dir, exist_ok=True)
    # os.makedirs(val_img_save_dir, exist_ok=True)

    train_loader, _ = PCI_DataLoader(data_dir, batch_size=1, shuffle=False, split='train',
                                     spatial_size=(128, 128, 128),
                                     p_gaussianNoise=0.3, p_Smooth=0.3, p_Rotate=0.9,
                                     p_Contrast=0.9, p_flip=0.5, num_workers=2)
    # val_loader, _ = PCI_DataLoader(data_dir, batch_size=1, shuffle=False, split='validation',
    #                                spatial_size=(128, 128, 128),
    #                                num_workers=2, use_sampler=False)
    plot_img(train_loader, train_img_save_dir)
    # plot_img(val_loader, val_img_save_dir)


if __name__ == '__main__':
    main()
