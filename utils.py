from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt

def get_data_loaders(train_batch_size, val_batch_size):
    """
    Prepares and returns data loaders for MNIST training and testing.

    Returns:
        tuple(torch.utils.data.DataLoader, torch.utils.data.DataLoader):
            A tuple containing the training data loader and testing data loader.
    """
    # Train transformations
    train_transforms = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Test transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=train_transforms),
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=test_transforms),
        shuffle=False)


    return train_loader, test_loader

def get_device():
    """
      Determines the most appropriate device (CUDA or CPU) for PyTorch computations.

      Returns:
          torch.device: The selected device.
    """
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    return device

def visualize_batch(data_loader):
    """
    Displays a batch of images from a data loader.

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader from which to fetch a batch.
    """
    batch_data, batch_label = next(iter(data_loader))

    fig = plt.figure()
    for i in range(12):  # Adjust if you want a different number of images displayed
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])
    plt.show()  # Display the plot



def plot_loss_and_accuracy(train_losses, test_losses, train_acc, test_acc):
    """
    Plots training and testing loss and accuracy curves.

    Args:
        train_losses (list): List of training losses per epoch.
        test_losses (list): List of test losses per epoch.
        train_acc (list): List of training accuracies per epoch.
        test_acc (list):  List of test accuracies per epoch.
    """

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()  # Display the plot
