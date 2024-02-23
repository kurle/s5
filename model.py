  import torch
  import torch.nn as nn
  import torch.nn.functional as F

  class Net(nn.Module):
      """
          Defines a convolutional neural network (CNN) for image classification.
              """

      def __init__(self):
          """
                  Initializes the CNN layers.
                          """
          super(Net, self).__init__()
          # Convolutional layers
          self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
          self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
          self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
          self.conv4 = nn.Conv2d(128, 256, kernel_size=3)

          # Fully connected layers
                  self.fc1 = nn.Linear(4096, 50)
          self.fc2 = nn.Linear(50, 10)  # 10 output classes for MNIST

      def forward(self, x):
          """
                  Defines the forward pass (data flow) through the network.

                          Args:
                                      x (torch.Tensor): Input image tensor.

                                              Returns:
                                                          torch.Tensor: Log-softmax probabilities for each class.
                                                                  """
          # Convolutional layers with ReLU and max pooling
          x = F.relu(self.conv1(x))
          x = F.relu(F.max_pool2d(self.conv2(x), 2))
          x = F.relu(self.conv3(x))
          x = F.relu(F.max_pool2d(self.conv4(x), 2))

          # Flatten for fully connected layers
                  x = x.view(-1, 4096)

          # Fully connected layers
                  x = F.relu(self.fc1(x))
          x = self.fc2(x)

          # Output with log-softmax for probabilities
                  return F.log_softmax(x, dim=1)
