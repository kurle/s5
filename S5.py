import torch
import torch.optim as optim
from tqdm import tqdm
from model import Net
from utils import get_data_loaders, get_device, visualize_batch, plot_loss_and_accuracy
!pip install torchsummary
from torchsummary import summary

def GetCorrectPredCount(pPrediction, pLabels):
    """
      Calculates the number of correct predictions in a batch.

      Args:
          pPrediction (torch.Tensor): Tensor of predicted class probabilities.
          pLabels (torch.Tensor):  Tensor of true target labels.

      Returns:
          int: The number of correct predictions in the batch.
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
    """
      Trains the model for a single epoch.

      Args:
          model (torch.nn.Module): The PyTorch model to train.
          device (torch.device): Device (CPU or GPU) to use for training.
          train_loader (torch.utils.data.DataLoader): DataLoader providing training data.
          optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
          criterion (torch.nn): Loss function used for training.
          progress_bar (bool, optional): Whether to display a progress bar using tqdm.
                                       Defaults to True.
    """
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0
    device = get_device()

    for batch_idx, (data, target) in enumerate(pbar):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()

      # Predict
      pred = model(data)

      # Calculate loss
      loss = criterion(pred, target)
      train_loss+=loss.item()

      # Backpropagation
      loss.backward()
      optimizer.step()

      correct += GetCorrectPredCount(pred, target)
      processed += len(data) #increments the number representing how many data samples have been seen.

      pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed) #average loss
    train_losses.append(train_loss/len(train_loader)) #average process

def test(model, device, test_loader, criterion):
    """
      Evaluates the model on the test dataset.

      Args:
          model (torch.nn.Module): The PyTorch model to evaluate.
          device (torch.device): Device (CPU or GPU) to use for evaluation.
          test_loader (torch.utils.data.DataLoader): DataLoader providing test data.
          criterion (torch.nn): Loss function used for evaluation.
    """
    model.eval() #evaluation mode

    test_loss = 0
    correct = 0

    with torch.no_grad(): #disable gradient propogation
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():

      train_losses = []
      test_losses = []
      train_acc = []
      test_acc = []

      train_loader, test_loader = get_data_loaders()
      model = Net().to(device)
      optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
      scheduler = StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
      criterion = F.nll_loss

      for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        train(model, device, train_loader, optimizer, criterion)
        test(model, device, test_loader, criterion)
        scheduler.step()

      plot_loss_and_accuracy(plot_loss_and_accuracy, test_losses, train_acc, test_acc)
      summary(model, input_size=(1, 28, 28))
