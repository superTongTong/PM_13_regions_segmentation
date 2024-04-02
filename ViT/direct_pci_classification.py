from __future__ import print_function
import monai.networks.nets as nets
from torch import nn
import torch
import numpy as np
import os
import random
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from dataloader import PCI_DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import Compose, EnsureType, Activations, AsDiscrete
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
import time


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(epochs, val_interval, model, train_loader, val_loader, criterion, optimizer, post_label, post_pred, auc_metric):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    # metric_values = []
    # acc_values = []
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        writer = SummaryWriter()

        for batch_data in train_loader:
            step += 1
            data, label = batch_data["image"].to(device), batch_data["label"].to(device)
            model.to(device)

            output = model(data.float())
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_len = len(train_loader) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:

            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                # y_onehot = [post_label(i) for i in decollate_batch(y)]
                # y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                # # torch.tensor(y_pred_act, device="cpu")
                # # torch.tensor(y_onehot, device="cpu")
                # auc_metric(y_pred_act, y_onehot)
                # auc_result = auc_metric.aggregate()
                # auc_metric.reset()
                # del y_pred_act, y_onehot
                # metric_values.append(auc_result)

                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    # save_path = "C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/saved_model/best_metric_ResNet10.pth"
                    # os.makedirs(save_path, exist_ok=True)
                    # torch.save(model.state_dict(),
                    #            save_path)
                    # print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric,  best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        writer.close()




def mian():

    batch_size = 2  # 64
    epochs = 10  # 10   #20 #10 #50 #20
    val_interval = 2
    lr = 3e-5
    gamma = 0.7
    seed = 42  # 42
    seed_everything(seed)
    # model = nets.ViT(
        # in_channels=1,
        # img_size=(96,96,96),
        # patch_size=(16,16,16),
        # pos_embed='conv',
        # classification=True
        # num_classes=4
    # )
    model = nets.resnet10(
        pretrained=False,
        n_input_channels=1,
        widen_factor=2,
        conv1_t_stride=2,
        num_classes=4,
    )
    data_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/cropped_scan/'
    train_loader = PCI_DataLoader(data_dir, batch_size=batch_size, shuffle=True,
                                  split='train', spatial_size=(64, 64, 64), num_workers=2)
    val_loader = PCI_DataLoader(data_dir, batch_size=1, shuffle=False,
                                split='validation', spatial_size=(64, 64, 64), num_workers=2)

    post_pred = Compose([EnsureType(), Activations(softmax=True)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=4, n_classes=4)])
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    # metric
    auc_metric = ROCAUCMetric()
    train(epochs, val_interval, model, train_loader, val_loader, criterion, optimizer, post_label, post_pred, auc_metric)


if __name__ == '__main__':
    start = time.time()
    mian()
    print('Elapsed time: {}'.format(time.time() - start))
