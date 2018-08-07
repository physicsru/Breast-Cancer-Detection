import numpy as np
from sklearn.preprocessing import MinMaxScaler, Normalizer

class RFScaler(object):
    
    def __init__(self, feature_range=(0, 1), epsilon=10e-10):
        self.feature_range = feature_range
        self.epsilon = epsilon
        return 
    
    def preprocess(self, X):
        return np.log(self.epsilon + X)
    
    def inverse_preprocess(self, X):
        return np.exp(X) - self.epsilon
        
    def fit(self, X):
        X = self.preprocess(X)
        self.data_min = X.min()
        self.data_max = X.max()
        self.data_range = self.data_max - self.data_min
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / self.data_range
        self.min_ = self.feature_range[0] - self.data_min * self.scale_
        return self
    
    def transform(self, X):
        X = self.preprocess(X)
        X *= self.scale_
        X += self.min_
        return X
    
    def inverse_transform(self, X):
        X -= self.min_
        X /= self.scale_
        X = self.inverse_preprocess(X)
        return X
    