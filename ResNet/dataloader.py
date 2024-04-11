import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torch
from monai.data import CacheDataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Resized,
    ToTensord,
    RandCropByPosNegLabeld,
    RandRotated,
    RandZoomd,
    CastToTyped,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandFlipd,
    ScaleIntensityRanged
)


def pci_transform(spatial_size=(128, 128, 128)):
    data_transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Resized(keys=["image"], spatial_size=spatial_size),
        # HU windowing for abdomen CT images: [-200, 300] to [0, 1], already done in the preprocessing
        # some DA
        RandGaussianNoised(keys=["image"], prob=0.3),
        RandGaussianSmoothd(keys=["image"], prob=0.3),
        RandZoomd(
            keys=["image"],
            min_zoom=0.9,
            max_zoom=1.5,
            mode=("bilinear",) * len(["image"]),
            align_corners=(True,) * len(["image"]),
            prob=0.3,
        ),
        RandRotated(
            keys=["image"],
            range_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            range_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            range_z=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            mode=("bilinear",) * len(["image"]),
            align_corners=(True,) * len(["image"]),
            padding_mode=("border",) * len(["image"]),
            prob=0.3,
        ),
        RandAdjustContrastd(
            keys=["image"],
            gamma=(0.7, 1.5),
            prob=0.3,
        ),
        # EnsureTyped(keys=["image"]),
        ToTensord(keys=["image"]),
    ])
    return data_transform


def get_data_list(data_dir, split='train'):

    assert split in ['train', 'validation', 'test'], 'Only train, validation, test are supported options.'
    assert os.path.exists(data_dir), 'data_dir path does not exist: {}'.format(data_dir)
    print('Loading dataset from: {}'.format(data_dir + '/' + split + '/'))

    data_dir_img = os.path.join(data_dir, split)
    # filenames
    file_names = sorted(os.listdir(data_dir_img))
    n_files = len(file_names)
    print('Number of scans: {}'.format(n_files))

    # get the list of all the cases (example=s0007_0001_R1_3) we only need 'sxxxx_xxxx_Rx'
    # cases = sorted([file_names[i][:13] for i in range(0, n_files)])
    images = []
    labels = []
    for idx in range(n_files):
        img_path = os.path.join(data_dir_img, file_names[idx])
        label = int(file_names[idx][14:15])
        # y_in = torch.tensor(label)
        # y_one_hot = F.one_hot(y_in, num_classes=4)
        images.append(img_path)
        labels.append(label)

    return images, labels


def PCI_DataLoader(data_dir, batch_size=1, shuffle=True, split='train', spatial_size=(128, 128, 128), num_workers=2, use_sampler=True):
    class_counts = []
    imgs, labels = get_data_list(data_dir, split=split)
    data_files = [{"image": i, "label": l} for i, l in zip(imgs, labels)]

    # Count the occurrences of each number
    number_counts = Counter(labels)
    for num, count in sorted(number_counts.items()):
        class_counts.append(count)
    num_samples = sum(class_counts)

    # prepare the weighted sampler
    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]

    if not use_sampler:
        ds = CacheDataset(data=data_files, transform=pci_transform(spatial_size=spatial_size), progress=True)
        data_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return data_loader, class_weights
    else:
        ###############
        weights = [class_weights[labels[i]] for i in range(int(num_samples))]
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
        #############
        ds = CacheDataset(data=data_files, transform=pci_transform(spatial_size=spatial_size), progress=True)
        data_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler)
        return data_loader, class_weights

def main():
    data_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/cropped_scan_test/'
    val_loader = PCI_DataLoader(data_dir, batch_size=1, shuffle=False, split='train',
                                spatial_size=(128, 128, 128), num_workers=2, use_sampler=True)

    for i, data in enumerate(val_loader):
        # img = data[0]
        img = data['image']
        # name = img[meta]['filename_or_obj']
        img_array = img.numpy()
        sqzzed = np.squeeze(img_array)
        plt.figure("visualize", (8, 4))
        # print("array:", sqzzed.shape)
        # plt.title(f"caseID: {name}")
        plt.imshow(sqzzed[:, :, 60], cmap="gray")
        plt.savefig(f'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/cropped_scan_test/train_noDA_{i}.png')
        # plt.show()

if __name__ == '__main__':
    start = time.time()
    main()
    print('Elapsed time: {}'.format(time.time() - start))

