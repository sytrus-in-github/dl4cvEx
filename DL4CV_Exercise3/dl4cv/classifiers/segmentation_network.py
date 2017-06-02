import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class SegmentationNetwork(nn.Module):

    def __init__(self):
        super(SegmentationNetwork, self).__init__()

        ############################################################################
        #                             YOUR CODE                                    #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ############################################################################
        #                             YOUR CODE                                    #
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
