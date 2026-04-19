print(f"----------------------- Imports -----------------------\n\n")



# **** Imports ****
# =================
import torch                                    # main library
from torch import nn                            # neural network
from torch.utils.data import DataLoader         # dataloader
from torchvision import datasets                # datasets
from torchvision.transforms import ToTensor     # convert to tensor
# -----------------------------------------------------------------

print(f"----------------------- HyperParams -----------------------\n\n")

# **** HyperParameters ****
# =========================
learning_rate: float = 1e-3    # how much the optimizer changes the paramaters every cycle
batch_size: int = 64         # how many images to process at once
epochs: int = 10             # how many times we should go through the whole dataset
# ---------------------------------------------------------------------------------

print(f"Learning Rate: {learning_rate}\nBatch Size: {batch_size}\nEpochs: {epochs}\n")
print(f"----------------------- Download Data -----------------------\n\n")

# **** Downloading Data ****
# ==========================

# get our training data
training_data: datasets = datasets.MNIST(
    root="data",                    # root directory of the data
    train=True,                     # it is training data
    transform=ToTensor(),           # transforms the images to tensors
    download=True                   # download it if not found
)

# get our evaluation data
testing_data: datasets = datasets.MNIST(
    root="data",                    # root directory of data
    train=False,                    # it is not training data (evaluation data)
    transform=ToTensor(),           # transforms images to tensor
    download=True                   # downloads if not available
)
# -----------------------------------------------------------------------------

print(f"Training Data: {training_data}\nTesting Data: {testing_data}\n")
print(f"----------------------- DataLoaders -----------------------\n\n")

# **** DataLoaders ****
# =====================

# dataloader for the training data
training_dataloader: DataLoader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# dataloader for the evaluation data
testing_dataloader: DataLoader = DataLoader(testing_data, batch_size=batch_size)
# ------------------------------------------------------------

print(f"Training DataLoader: {training_dataloader}\n\nTesting DataLoader: {testing_dataloader}")
print(f"----------------------- Define Network -----------------------\n\n")


# **** Define Network ****
# ========================

# neural network class inherits from `nn.Module`
class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()              # run the `nn.module` initialize function
        self.flatten_function = nn.Flatten()     # define the flatten function

        # a simple sequential linear neural network
        # where every layer connects with everything
        # in the last
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # takes in all 28*28 pixels and passes them to 512 neurons
            nn.ReLU(),              # applies ReLU fucntion
            nn.Linear(512,512),     # a hidden layer of 512 neurons
            nn.ReLU(),              # ReLU again
            nn.Linear(512,10)       # final layer to outputs the logits
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten_function(x)
        logits: torch.Tensor = self.linear_relu_stack(x)
        return logits


# instantiate the model
model = NeuralNetwork()

# move to gpu
device = "cuda"
model.to(device)
#----------------------------------------------------------------------------------------------

print(f"\nModel:{model}\n")
print(f"----------------------- Define Network -----------------------\n\n")


# **** Loss Function & Optimizer ****
# ===================================

# loss function
loss_function: nn.CrossEntropyLoss = nn.CrossEntropyLoss()   # usually used for classification

# optimizer
optimizer: torch.optim.SGD = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------

print(f"\nLoss: {loss_function}\n\nOptimizer: {optimizer}\n")
print(f"\n\n\n----------------------- Define Training and Testing Loops -----------------------\n\n")


# **** Training and Testing Loops ****
# ====================================

# ** train loop **
def train_loop(model: NeuralNetwork, dataloader: DataLoader, loss_func: nn.CrossEntropyLoss, optimizer: torch.optim.SGD) -> None:

    data_size = len(dataloader.dataset) # for printing later

    model.train() # turns the model into training mode

    # actual loop
    for batch_number, (images, labels) in enumerate(dataloader):

        # move images and labels to gpu
        images, labels = images.to(device), labels.to(device)

        # get the predicion and the loss
        output_logits: torch.Tensor = model(images)
        loss: torch.Tensor = loss_func(output_logits, labels)

        # backpropagation
        loss.backward()     # calculate the gradients
        optimizer.step()    # adjusts the weights a step towards the gradient
        optimizer.zero_grad()

        # print every hundred batches
        if batch_number % 100  == 0:
            loss: int = loss.item()                                         # get the actual computed loss
            current_index: int = batch_number * batch_size + len(images)    # compute the current index

            # actually print
            print(f"Loss: {loss:>7f} [{current_index:>5d}/{data_size:>5d}]")

# ** test loop **
def test_loop(model: NeuralNetwork, dataloader: DataLoader, loss_func: nn.CrossEntropyLoss) -> None:

    model.eval() # set to evaluate mode

    data_size: int = len(dataloader.dataset) # gets the size of the dataset

    num_batches: int = len(dataloader)  # number of batches
    sum_loss: float = 0                 # the average loss to be computed
    num_correct:int = 0                 # the number that are correct

    # dont compute gradients do to ineffciencies
    with torch.no_grad():
        for batch_number, (images, labels) in enumerate(dataloader):

            # move images and labels to gpu
            images, labels = images.to(device), labels.to(device)

            prediction: torch.Tensor = model(images) # images
            
            #  just keep summing the losses so we can divide them after
            sum_loss += loss_func(prediction, labels).item()

            # add all the correct answers from the batch
            num_correct += (prediction.argmax(1) == labels).type(torch.float).sum().item()

    avg_loss = sum_loss / num_batches
    correct_percentage = num_correct / data_size

    print(f"==============================================")
    print(f"Test Error:\n")
    print(f"    Correct: {(100*correct_percentage):0.1f}%")
    print(f"    Average Loss: {avg_loss:>8f}")
# --------------------------------------------------------------------------------------------------



# **** Final Logic ****
# =====================
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------")
    train_loop(model, training_dataloader, loss_function, optimizer)
    test_loop(model, testing_dataloader, loss_function)

torch.save(model.state_dict(), 'model_weights.pth')
print(f"Done")
# ---------------------



