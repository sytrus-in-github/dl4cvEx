import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
  
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
  
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
  
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    D, C = W.shape
    N, _ = X.shape
    for i in range(N):
        z_i = X[i,:].dot(W)
        z_i -= np.max(z_i)
        exp_i = np.exp(z_i)
        sum_i = np.sum(exp_i)
        loss += -z_i[y[i]] + np.log(sum_i)
        p_i = exp_i / sum_i
        p_i[y[i]] -= 1.
        dW += np.reshape(X[i,:],(-1,1)) * (np.reshape(p_i, (1,-1)))
    loss /= N
    dW /= N
    loss += 0.5 * reg * np.sum(W**2)
    dW += reg * W
            
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
  
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    n = len(y)
    Z = X.dot(W)
    Z -= np.expand_dims(np.max(Z, 1),axis=1)
    Exp = np.exp(Z)
    Sum = np.sum(Exp, 1)
    loss = np.sum(np.log(Sum)-np.choose(y, Z.T))/n + 0.5 * reg * np.sum(W**2) 
    P = Exp/np.expand_dims(Sum,axis=1)
    P[np.arange(n),y] -= 1
    dW = X.transpose().dot(P)/n + reg * W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

