from torch import nn
import torch
from .resnet_base import resnet50


class MedicalNet(nn.Module):
    def __init__(self, path_to_weights, device, sample_input_D=128, sample_input_H=128, sample_input_W=128, num_classes=4):
        super(MedicalNet, self).__init__()
        self.num_classes = num_classes
        self.model = resnet50(sample_input_D=sample_input_D, sample_input_H=sample_input_H,
                              sample_input_W=sample_input_W, num_seg_classes=num_classes)
        self.model.conv_seg = nn.Sequential(
            nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),
            nn.Flatten(start_dim=1)
        )
        net_dict = self.model.state_dict()
        pretrained_weights = torch.load(path_to_weights, map_location=torch.device(device))
        pretrain_dict = {
            k.replace("module.", ""): v for k, v in pretrained_weights['state_dict'].items() if
            k.replace("module.", "") in net_dict.keys()
        }
        net_dict.update(pretrain_dict)
        self.model.load_state_dict(net_dict)
        self.fc = nn.Linear(2048, out_features=num_classes)

    def forward(self, x):
        features = self.model(x)
        # return torch.sigmoid_(self.fc(features))
        if self.num_classes == 2:
            return torch.sigmoid_(self.fc(features))
        elif self.num_classes == 4:
            return torch.softmax_(self.fc(features))
        else:
            print("Number of classes must be 2 or 4. Current number of classes is ", self.num_classes)
