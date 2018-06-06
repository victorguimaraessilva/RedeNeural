import numpy as np

class Perceptron(object):

  '''
    Class Perceptron: 

    Parameters: eta -> float (Learning Rate [between 0.0 and 1.0])
                n_iter -> int (number of training set repetitions)
    
    Attributes: w_ : array (Weights)
                errors_ : List (Error list)
                
    Coded by VictorGUimas
  '''

  def __init__(self, eta=0.01, n_iter=10):

    self.eta = eta
    self.n_iter = n_iter

  def fit(self, X, y):

    '''
      Method for training data

      Parameters: X: array (shape = [n_samples, n_features])
                  Y: array (shape = [n_samples]) - Target Values

      Return: self: object
    '''

    self.w_ = np.zeros(1 + X.shape[1])
    self.errors_ = []

    for _ in range(self.n_iter):
      errors = 0
      for xi, target in zip(X, y):
        update = self.eta * (target - self.predict(xi))
        self.w_[1:] += update * xi
        self.w_[0] += update
        errors += int(update != 0.0)
      
      self.errors_.append(errors)

    return self

  def net_input(self, X):
    '''
      Method for calculate input

    '''
    return np.dot(X, self.w_[1:] + self.w_[0])

  def predict(self, X):
    '''
      Returns the predict about X
    '''
    return np.where(self.net_input(X) >= 0.0, 1, -1)