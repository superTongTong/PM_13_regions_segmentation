from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.transforms import LoadImaged
from transforms import pci_transform_train, pci_transform_val

# Define keys for loading images
keys = ['cls0', 'cls1', 'cls2', 'cls3']

# Define transform for loading images
load_images = LoadImaged(keys)


class PCI_Dataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 transform=None,
                 ):

        assert split in ['train', 'validation', 'test'], 'Only train, validation, test are supported options.'
        assert os.path.exists(data_dir), 'data_dir path does not exist: {}'.format(data_dir)

        print('Loading dataset from: {}'.format(data_dir + '/' + split + '/'))

        self.data_dir = data_dir
        self.split = split
        self.data_dir_img = os.path.join(data_dir, split)
        self.transform = transform

        # filenames
        self.file_names = sorted(os.listdir(self.data_dir_img))
        self.n_files = len(self.file_names)
        print('Number of scans: {}'.format(self.n_files))

        # get the list of all the cases (example=s0007_0001_R1_3) we only need 'sxxxx_xxxx_Rx'
        self.cases = sorted([self.file_names[i][:13] for i in range(0, self.n_files)])

        # print('cases: {}'.format(self.cases))

    def __len__(self):
        return self.n_files

    def __getitem__(self, idx: int):
        img_dict = {}
        label = int(self.file_names[idx][14:15])
        img_path = os.path.join(self.data_dir_img, self.file_names[idx])

        # Update img_dict with image paths
        if label == 0:
            img_dict.update({'cls0': img_path})
        elif label == 1:
            img_dict.update({'cls1': img_path})
        elif label == 2:
            img_dict.update({'cls2': img_path})
        elif label == 3:
            img_dict.update({'cls3': img_path})
        else:
            print('label not found')

        # Load images as tensors
        img_dict = load_images(img_dict)

        if self.transform is not None:
            img_dict = self.transform(img_dict)

        # Check if all images are loaded and have the same shape
        for key in keys:
            if key not in img_dict or img_dict[key].shape != img_dict[keys[0]].shape:
                raise ValueError(f"Image {key} not loaded correctly")

        # Concatenate images along dimension 0
        data = torch.cat([img_dict[key] for key in keys], dim=0)

        return data, label


def test():
    data_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/cropped_scan_test/'
    # mask_dir = 'data/data_ViT/masks'
    train_transform = pci_transform_train(all_key=['cls0', 'cls1', 'cls2', 'cls3'], cls123_key=['cls1', 'cls2', 'cls3'],)
    # val_transform = pci_transform_val(all_key=['cls0', 'cls1', 'cls2', 'cls3'], cls123_key=['cls1', 'cls2', 'cls3'],)
    train_set = PCI_Dataset(data_dir, split='train', transform=train_transform)
    # val_set = PCI_Dataset(data_dir, split='validation', transform=val_transform)

    for data, label in enumerate(train_set):
        img = data[0]
        # img = img['image']
        img_array = img.numpy()
        sqzzed = np.squeeze(img_array)
        plt.figure("visualize", (8, 4))
        plt.title("image")
        plt.imshow(sqzzed[:, :, 30], cmap="gray")
        plt.show()
        plt.close()


if __name__ == '__main__':
    test()
