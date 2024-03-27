from __future__ import print_function
import monai.networks.nets as nets
from tqdm import tqdm
from torch import nn
import torch
import numpy as np
import os
import random
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import io


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train(epochs, model, train_loader, valid_loader, criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)
            # print('data,label =', data.shape, len(label))

            output = model(data.float())
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            # save a checkpoint

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data.float())
                # print('val-data, max=',val_output.data[0], torch.max(val_output.data[0],0))
                val_loss = criterion(val_output, label)
                # print the resulrs
                cls = torch.max(val_output[0], 0)
                cls = cls.indices
                # print('val-output',cls)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)
        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
        torch.save(model.state_dict(), 'xg_vit_model_covid_3D.pt')  # xg

        # saving the data
    with io.open('3d16-val-output.txt', 'w') as f:
        for i in range(len(label)):
            if cls > 0:
                f.write("%s %s\n" % ('1', int(label[i].data)))
            else:
                f.write("%s %s\n" % ('0', int(label[i].data)))


def mian():

    batch_size = 4  # 64
    epochs = 5  # 10   #20 #10 #50 #20
    lr = 3e-5
    gamma = 0.7
    seed = 42  # 42
    seed_everything(seed)

    model = nets.resnet50(
        pretrained=False,
        n_input_channels=1,
        widen_factor=2,
        conv1_t_stride=2,
    )

    dataset = ...  # outputs crops per region

    train_loader, valid_loader = Dataloader(dataset)

    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    train(epochs, model, train_loader, valid_loader, criterion, optimizer)

