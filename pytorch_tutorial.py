import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size) #wrap into iterables.
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader: #print the shape 
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = "cuda" if torch.cuda.is_available() else "cpu" #I'll use cpu
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #
        self.linear_relu_stack = nn.Sequential( #define layers and activation function
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ) #linear_relu_stack is a function that takes data.

    def forward(self, x): #define how data gets taken in
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)
'''
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )'''



loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    '''
    enumerate data, for each, make guess, compute loss, backpropogate.
    '''
    size = len(dataloader.dataset)
    model.train() # put model in training mode.
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X) # y_pred vector
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward() #what type is this?
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0: #every 100 batches,     
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    '''
    run the current model on the input dataloader 
    '''
    size = len(dataloader.dataset) # all datapoints?
    num_batches = len(dataloader) # all batches (fewer than total datapoints?)
    model.eval() #I'm guessing this is changing the mode the model is in.
    test_loss, correct = 0, 0 
    with torch.no_grad(): #what is grad? gradient descent?
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() #?
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!") 

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")