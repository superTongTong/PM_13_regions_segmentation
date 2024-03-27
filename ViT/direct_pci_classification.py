import monai.networks.nets as nets
from torch.utils import Dataloader
from tqdm import tqdm


model = nets.resnet50(
    pretrained=False,
    n_input_channels=1,
    widen_factor=2,
    conv1_t_stride=2,
)

dataset = ...  # outputs crops per region

dataloader = Dataloader(
    dataset
)

criterion = nn.CrossEntropyLoss

for batch in tqdm(dataloader):

    output = model(batch[0])
    loss = criterion(output, batch[1])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


