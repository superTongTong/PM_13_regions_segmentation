import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.data import CacheDataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Resized, ToTensord, RandGaussianNoised, RandGaussianSmoothd, RandZoomd, RandRotated, RandAdjustContrastd

def pci_transform_train(spatial_size=(128, 128, 128), p_gaussianNoise=0.3, p_Smooth=0.3, p_Rotate=0.9, p_Contrast=0.9, p_Zoom=0.5):
    return Compose([
        # intensity normalization is done in the data pre-processing step
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Resized(keys=["image"], spatial_size=spatial_size),
        # RandGaussianNoised(keys=["image"], prob=p_gaussianNoise, mean=0.0, std=0.1),
        # RandGaussianSmoothd(keys=["image"], prob=p_Smooth, sigma_x=(0.5, 1.5),
        #                     sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5), approx='erf'),
        RandZoomd(keys=["image"], min_zoom=0.9, max_zoom=1.5, mode=("area",), prob=p_Zoom),
        RandRotated(keys=["image"],
                    range_x=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    range_y=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    range_z=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    mode=("bilinear",) * len(["image"]),
                    align_corners=(True,) * len(["image"]),
                    padding_mode=("border",),
                    prob=p_Rotate),
        RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=p_Contrast),
        ToTensord(keys=["image"]),
    ])

def pci_transform_val(spatial_size=(128, 128, 128)):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Resized(keys=["image"], spatial_size=spatial_size),
        ToTensord(keys=["image"]),
    ])


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
                   p_gaussianNoise=0.1, p_Smooth=0.1, p_Rotate=0.5, p_Contrast=0.5, p_Zoom=0.5, use_sampler=True):
    imgs, labels, caseID = get_data_list(data_dir, split=split)
    data_files = [{"image": i, "label": l, "CaseID": id} for i, l, id in zip(imgs, labels, caseID)]
    transforms = pci_transform_train(spatial_size=spatial_size, p_gaussianNoise=p_gaussianNoise, p_Smooth=p_Smooth,
                                     p_Rotate=p_Rotate, p_Contrast=p_Contrast, p_Zoom=p_Zoom) if split == 'train' else pci_transform_val(spatial_size=spatial_size)
    class_counts = [count for num, count in sorted(Counter(labels).items())]
    num_samples = sum(class_counts)
    class_weights = [num_samples / class_count for class_count in class_counts]
    ds = CacheDataset(data=data_files, transform=transforms, progress=False)
    if use_sampler:
        weights = [class_weights[labels[i]] for i in range(int(num_samples))]
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples), replacement=True)
        data_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler)
    else:
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
    # data_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/cropped_scan_test/'
    # val_img_save_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/val_images/'
    data_dir = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/cropped_scan_v2/'
    train_img_save_dir = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/train_images_v2/'
    # val_img_save_dir = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/val_images/'
    os.makedirs(train_img_save_dir, exist_ok=True)
    # os.makedirs(val_img_save_dir, exist_ok=True)

    train_loader, _ = PCI_DataLoader(data_dir, batch_size=1, shuffle=False, split='train',
                                     spatial_size=(128, 128, 128),
                                     p_gaussianNoise=0.3, p_Smooth=0.3, p_Rotate=0.9,
                                     p_Contrast=0.9, p_Zoom=0.5, num_workers=2, use_sampler=True)
    # val_loader, _ = PCI_DataLoader(data_dir, batch_size=1, shuffle=False, split='validation',
    #                                spatial_size=(128, 128, 128),
    #                                num_workers=2, use_sampler=False)
    plot_img(train_loader, train_img_save_dir)
    # plot_img(val_loader, val_img_save_dir)


if __name__ == '__main__':
    main()
