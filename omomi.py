import sys, os
sys.path.append(os.pardir)
import numpy as np
from ainofunction import softmax, cross_entropy
class omomi:
    def __init__(self):  #これはクラスを指定された時点で発動する
        self.W = np.random.rand(2 ,3)   #こいつがとりあえずの重みとなるのか？
    
    def predict (self, x):
        return np.dot(x, self.W)       #名前.predict(x)で簡単に予測　　predictは次の層にデータを渡す行為

    def loss(self, x, t):
        z = self.predict(x)           #名前.loss(x, t)で簡単に誤差
        y = softmax(z)
        return cross_entropy(y, t)