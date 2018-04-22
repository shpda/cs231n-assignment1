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
  # X: (N, D)
  # W: (D, C)
  # y: (N,)
  # score: (N, C)
  score = np.dot(X, W)
  score -= np.max(score)
  exp_score = np.exp(score)
  for i in range(y.shape[0]):
      prob = exp_score[i, :]/ np.sum(exp_score[i, :])
      loss -= np.log(prob[y[i]])
      tmp = prob
      tmp[y[i]] -= 1 
      dW += np.outer(X[i, :], tmp)

  loss /= y.shape[0]
  loss += reg * np.sum(W * W)
  dW /= y.shape[0]
  dW += 2 * reg * W
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
  # score: (N, C)
  n_example = X.shape[0]
  score = np.dot(X, W)
  score -= np.max(score)
  exp_score = np.exp(score)
  prob = (exp_score / np.expand_dims(np.sum(exp_score, axis=1), axis=1))
  mask = np.zeros(score.shape)
  mask[range(score.shape[0]), y] = 1
  loss = (-np.sum(np.log(prob) * mask)) / n_example + reg * np.sum(W * W)
  # dW: (D, C) -- (D, N) * (N, C) 
  dW = np.dot(np.transpose(X), prob - mask) / n_example + 2* reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

