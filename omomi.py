import sys, os
sys.path.append(os.pardir)
import numpy as np
from ainofunction import softmax, cross_entropy
class omomi:
    def __init__(self):  #これはクラスを指定された時点で発動する
        self.W = np.random.rand(2 ,3)   #こいつがとりあえずの重みとなるのか？
    
    def predict (self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy(y, t)