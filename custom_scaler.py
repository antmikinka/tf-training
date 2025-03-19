import numpy as np

class CustomMinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        
    def fit_transform(self, X):
        self.min_ = X.min(axis=0)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (X.max(axis=0) - self.min_ + 1e-8)
        return self.transform(X)
    
    def transform(self, X):
        return (X - self.min_) * self.scale_ + self.feature_range[0]
    
    def inverse_transform(self, X):
        return (X - self.feature_range[0]) / self.scale_ + self.min_