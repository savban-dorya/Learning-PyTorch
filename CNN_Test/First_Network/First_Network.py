import torch                                    # main library
from torch import nn                            # neural network
from torch.utils.data import DataLoader         # dataloader
from torchvision import datasets                # datasets
from torchvision.transforms import ToTensor     # convert to tensor


# get our training data
training_data = datasets.MNIST(
    root="data",                    # root directory of the data
    train=True,                     # it is training data
    transform=ToTensor(),           # transforms the images to tensors
    download=True                   # download it if not found
)

# get our evaluation data
testing_data = datasets.MNIST(
    root="data",                    # root directory of data
    train=False,                    # it is not training data (evaluation data)
    transform=ToTensor(),           # transforms images to tensor
    download=True                   # downloads if not available
)
