import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeLayerCNN(nn.Module):

    """

    A PyTorch implementation of a three-layer convolutional network

    with the following architecture:



    conv - relu - 2x2 max pool - fc - dropout - relu - fc



    The network operates on minibatches of data that have shape (N, C, H, W)

    consisting of N images, each with height H and width W and with C input

    channels.

    """



    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,

                 stride=1, weight_scale=0.001, pool=2, hidden_dim=100,

                 num_classes=10, dropout=0.0):

        """

        Initialize a new network.



        Inputs:

        - input_dim: Tuple (C, H, W) giving size of input data.

        - num_filters: Number of filters to use in the convolutional layer.

        - filter_size: Size of filters to use in the convolutional layer.

        - hidden_dim: Number of units to use in the fully-connected hidden layer-

        - num_classes: Number of scores to produce from the final affine layer.

        - stride: The size of the jump of convolution window for conv layer.

        - weight_scale: Scale for the convolution weights initialization-

        - pool: The size of the max pooling window.

        - dropout: Probability of an element to be zeroed.

        """

        super(ThreeLayerCNN, self).__init__()

        channels, height, width = input_dim



        ############################################################################

        # TODO: Initialize the necessary layers to resemble the ThreeLayerCNN      #

        # architecture  from the class docstring. In- and output features should   #

        # not be hard coded which demands some calculations especially for the     #

        # input of the first fully convolutional layer. The convolution should use #

        # "same" padding which can be derived from the kernel size and its weights #

        # should be scaled. Layers should have a bias if possible.                 #

        ############################################################################
        
        # conv - relu - 2x2 max pool - fc - dropout - relu - fc
        
        C,H,W = input_dim
        conv_pad = kernel_size // 2
        size_change = lambda x: ((x + 2 * conv_pad - kernel_size) // stride + 1 - pool) // pool + 1
        fc_in = num_filters * size_change(H) * size_change(W)
        
        self.conv = nn.Conv2d(C, num_filters, kernel_size=kernel_size, stride=stride, padding=conv_pad, bias=True)
        self.conv.weight.data *= weight_scale
        self.relu_conv = nn.ReLU()
        self.pool = nn.MaxPool2d(pool)
        self.fc1 = nn.Linear(fc_in, hidden_dim, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=True)


        ############################################################################

        #                             END OF YOUR CODE                             #

        ############################################################################



    def forward(self, x):

        """

        Forward pass of the convolutional neural network. Should not be called

        manually but by calling a model instance directly.



        Inputs:

        - x: PyTorch input Variable

        """



        ############################################################################

        # TODO: Chain our previously initialized convolutional neural network      #

        # layers to resemble the architecture drafted in the class docstring.      #

        # Have a look at the Variable.view function to make the transition from    #

        # convolutional to fully connected layers.                                 #

        ############################################################################
        
        # conv - relu - 2x2 max pool - fc - dropout - relu - fc
        n,_,_,_ = x.size()
        
        x = self.conv(x)
        x = self.relu_conv(x)
        x = self.pool(x)
        x = x.view(n,-1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu_fc(x)
        x = self.fc2(x)

        ############################################################################

        #                             END OF YOUR CODE                             #

        ############################################################################



        return x



    def save(self, path):

        """

        Save model with its parameters to the given path. Conventionally the

        path should end with "*.model".



        Inputs:

        - path: path string

        """

        print 'Saving model... %s' % path

        torch.save(self, path)

