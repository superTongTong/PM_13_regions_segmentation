import torch
import SimpleITK as sitk
from torch.utils import data
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from monai.data import CacheDataset

class PCI_Dataset(data.Dataset):
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
        # self.n_cases = self.n_files // (155 * 5)
        print('Number of scans: {}'.format(self.n_files))

        # get the list of all the cases (example=s0007_0001_R1_3) we only need 'sxxxx_xxxx_Rx'
        self.cases = sorted([self.file_names[i][:13] for i in range(0, self.n_files)])

        # print('cases: {}'.format(self.cases))

    def __len__(self):
        return self.n_files

    def __getitem__(self,
                    idx: int):

        img_path = os.path.join(self.data_dir_img, self.file_names[idx])
        # img = self.transform(img_path[idx])

        label = int(self.file_names[idx][14:15])
        y_in = torch.tensor(label)
        y_one_hot = F.one_hot(y_in, num_classes=4)
        data_files = {"image": img_path, "label": y_one_hot}

        return data_files


def test():
    data_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/cropped_scan/'
    # mask_dir = 'data/data_ViT/masks'
    dataset = PCI_Dataset(data_dir, split='validation')

    for i, data in enumerate(dataset):
        img = data[0]
        # img = img['image']
        img_array = img.numpy()
        sqzzed = np.squeeze(img_array)
        plt.figure("visualize", (8, 4))
        plt.title("image")
        plt.imshow(sqzzed[:, :, 30], cmap="gray")
        plt.show()


if __name__ == '__main__':
    test()
