from __future__ import print_function
import monai.networks.nets as nets
from torch import nn
import torch
import numpy as np
import os
import random
from torch.optim.lr_scheduler import StepLR, PolynomialLR
import torch.optim as optim
from dataloader import PCI_DataLoader
from monai.transforms import Compose, EnsureType, Activations, AsDiscrete
from monai.metrics import ROCAUCMetric
import time
from plot_results import plot_metrics
from monai.data import decollate_batch
import wandb
from Loss_function import CombinedLoss


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def ResNet_train(epochs, val_interval, model, train_loader, val_loader, criterion, optimizer, scheduler, post_label, post_pred, auc_metric, device):
    # Log gradients and model parameters
    wandb.watch(model)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_metric = -1
    best_metric_epoch = -1
    train_loss_list = []
    val_loss_list = []
    metric_values = []
    acc_values = []
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        print(f"Current learning rate: {np.round(optimizer.param_groups[0]['lr'], decimals=10)}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            data, label = batch_data["image"].to(device), batch_data["label"].to(device)
            print('sampled data label:', label)
            # model.to(device)

            output = model(data.float())
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_len = len(train_loader) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

        scheduler.step()
        epoch_loss /= step
        train_loss_list.append(epoch_loss)
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
                val_l = criterion(y_pred, y)
                val_loss = val_l.item()
                val_loss_list.append(val_loss)
                print(f"epoch {epoch + 1} validation loss: {val_loss:.4f}")
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                acc_values.append(acc_metric)
                y_onehot = [post_label(i) for i in decollate_batch(y)]
                y_onehot = torch.stack(y_onehot, dim=0).to(device)
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(auc_result)
                print(f"epoch {epoch + 1} AUC: {auc_result:.4f}")

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
                # Log metrics to wandb
                wandb.log(
                    {"Learning Rate": optimizer.param_groups[0]['lr'], "Train Loss": epoch_loss,
                     "Validation Loss": val_loss, "AUC": auc_result, "Accuracy": acc_metric})
        # # plot and save train and val loss curve, accuracy curve
        # os.makedirs(save_dir, exist_ok=True)
        # plot_metrics(train_loss_list, val_loss_list, acc_values, save_path=save_dir)
        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


def mian():
    # Log in to wandb
    wandb.login(key='f20a2a6646a45224f8e867aa0c94a51efb8eed99')
    # Initialize wandb
    run = wandb.init(project="my-project", name="mficb_DA4_lr5e-4_batch8_datasetv4")
    # specify all the directories
    # data_dir = 'C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/cropped_scan_test/'
    # save_plot_dir = "C:/Users/20202119/PycharmProjects/segmentation_PM/data/data_ViT/plot/"
    #
    # # pretrain = torch.load(
    # #     "C:/Users/20202119/PycharmProjects/segmentation_PM/data/MedicalNet_pretrained_weights/resnet_50_23dataset.pth")
    #
    # pretrain = torch.load(
    #     "C:/Users/20202119/PycharmProjects/segmentation_PM/data/MedicalNet_pretrained_weights/model_weights.torch")
    #
    data_dir = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/cropped_scan_v4/'
    # save_plot_dir = "/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/loss_acc_plot_mfcib/"
    # pretrain = torch.load(
    #     "/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/MedicalNet_pretrained_weights/resnet_50_23dataset.pth")
    pretrain = torch.load(
        "/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/MedicalNet_pretrained_weights/model_weights.torch")

    # set hyperparameters
    batch_size = 8  #64 out of memory
    epochs = 50
    val_interval = 1
    lr = 5e-4 # 3e-5
    gamma = 1.3
    seed = 42
    num_classes = 2
    seed_everything(seed)

    #set model
    model = nets.resnet50(
        pretrained=False,
        n_input_channels=1,
        widen_factor=1,
        # conv1_t_stride=2,
        num_classes=num_classes
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(model)
    # load pretrain model
    # pretrain['state_dict'] = {k.replace("module.", ""): v for k, v in pretrain['state_dict'].items()}
    model.to(device)
    model.load_state_dict(pretrain, strict=False)
    print("load pretrain model")

    # prepare dataloader

    train_loader, _ = PCI_DataLoader(data_dir, batch_size=batch_size, shuffle=False,
                                     split='train', spatial_size=(128, 128, 128), num_workers=2, use_sampler=True)
    val_loader, _ = PCI_DataLoader(data_dir, batch_size=1, shuffle=False,
                                   split='validation', spatial_size=(128, 128, 128), num_workers=2, use_sampler=False)

    # convert class weights to tensor

    # class_weights_train = torch.tensor(class_weights_train, device=device)
    # class_weights_val = torch.tensor(class_weights_val, device=device)
    post_pred = Compose([EnsureType(), Activations(softmax=True)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=num_classes, n_classes=num_classes)])

    criterion = nn.CrossEntropyLoss()
    # combine cross entropy loss with focal loss
    # criterion = CombinedLoss(alpha=1, gamma=2, weight=None)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # scheduler
    scheduler = PolynomialLR(optimizer, total_iters=epochs, power=gamma)
    # metric
    auc_metric = ROCAUCMetric()
    ResNet_train(epochs, val_interval, model, train_loader, val_loader, criterion,
                 optimizer, scheduler, post_label, post_pred, auc_metric, device)


if __name__ == '__main__':
    start = time.time()
    mian()
    # Finish the wandb run
    wandb.finish()
    print('Elapsed time: {}'.format(time.time() - start))
