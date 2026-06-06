import sys, os
sys.path.append(os.pardir)
import numpy as np
from ainolayer import *
from ainogradient import *
from collections import OrderedDict

class nisou_gyaku:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std):
        self.params = {}
        self.params['w1'] = weight_init_std *  np.random.randn(input_size, hidden_size)  #input_sizeが入力値のデータの大きさ　
        self.params['b1'] = np.zeros(hidden_size)
        self.params["w2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(hidden_size)

        self.layers = OrderedDict()
        self.layers['gyouretuseki_layer1'] = gyouretuseki_layer(self.params['w1'], self.params['b1'])
        self.layers['relu_layer1'] = relu_layer()
        self.layers['gyouretuseki_layer2'] = gyouretuseki_layer(self.params['w2'], self.params['b2'])
        self.lastlayer = softmax_with_loss_layer()

    def predict(self, x):#xは入力から運ばれてくる値。self.layers.values()は順伝播の処理をするレイヤがなにかを順に指定している
        for layer in self.layers.values():
            x = layer.forward(x)
        return x#結果としてpredictしてる
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer(y, t)
