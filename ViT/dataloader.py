
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import SimpleITK as sitk
import torch
import glob

class PCI_Dataset_3D():

    def __init__(self, file_list, label_list, transform=None):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img)

        img2 = torch.tensor(np.array(img))
        img_transformed = img2.permute(3, 2, 0, 1)

        label = self.label_list[idx]
        y_one_hot = F.one_hot(label, num_classes=4).squeeze(0)
        y = torch.transpose(y_one_hot, 0, 2)

        return img_transformed, y


def main():
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    data_dir = '..data/data_ViT/images'
    mask_dir = '..data/data_ViT/masks'
    file_list = glob.glob(data_dir + '/*.nii.gz')

