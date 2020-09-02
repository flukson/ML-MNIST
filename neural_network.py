import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):

  def __init__(self):
    '''
    This function contains definitions of layers in the neural network
    '''

    super(NeuralNetwork, self).__init__()

    # Convolutional layers are used for extraction of features.
    # Nice movie about CNN networks: https://www.youtube.com/watch?v=RANnxwUGAks

    # First 2D convolutional layer, taking in 1 input channel (image),
    # outputting 10 convolutional features, with a square kernel size of 5
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

    # Second 2D convolutional layer, taking in 10 input channels,
    # outputting 20 convolutional features, with a square kernel size of 5
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

    # Dropout layer to prevent overfitting
    self.conv2_drop = nn.Dropout2d()

    # First fully connected layer
    self.fc1 = nn.Linear(320, 50)

    # Second fully connected layer that outputs 10 labels
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    '''
    This function defines how data pass through subsequent layers of the neural network
    Args:
        x: data
    '''

    # Maximum pooling, or max pooling, is a pooling operation that calculates
    # the maximum, or largest, value in each patch of each feature map.
    # Here size of pool of square window is 2.
    # Max pooling is used to reduce the amount of information.

    # relu - use the rectified-linear (ReLU) activation function (a piecewise linear function
    # that will output the input directly if it is positive, otherwise, it will output zero)
    x = F.relu(F.max_pool2d(self.conv1(x), 2))

    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

    # https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch
    # Reshape output from convolutional layer to have 320 columns
    # (number of inputs of the 1st fully connected layer). -1 means that the number of rows
    # will be adjusted automatically.
    # Output from the CNN network must be flattened before sending to classifying neural network.
    x = x.view(-1, 320)

    # Apply dropout
    x = F.dropout(x, training=self.training)

    # Use the rectified-linear activation function over x
    x = F.relu(self.fc1(x))

    # Apply softmax to x
    # https://en.wikipedia.org/wiki/Softmax_function
    # Softmax function - generalization of the logistic regression function to multiple dimensions
    return F.log_softmax(self.fc2(x))
