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

        self.layers = OrderedDict()#kononyu-ronsonomono
        self.layers['gyouretuseki_layer1'] = gyouretuseki_layer(self.params['w1'], self.params['b1'])
        self.layers['relu_layer1'] = relu_layer()
        self.layers['gyouretuseki_layer2'] = gyouretuseki_layer(self.params['w2'], self.params['b2'])
        self.lastlayer = softmax_with_loss_layer()#gosadasiteru

    def predict(self, x):#xは入力から運ばれてくる値。self.layers.values()は順伝播の処理をするレイヤがなにかを順に指定している
        for layer in self.layers.values(): #.values(辞書)は辞書の添え字でないものをすべて出す
            x = layer.forward(x)
        return x#結果としてpredictしてる
    
    def loss(self, x, t):#xは入力値
        y = self.predict(x)
        return self.lastlayer.forward(y, t)
    
    def accuracy(self, x, t):#xは入力値
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1:
            t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        grads['w1'] = koubai_tazigenn(loss_w, self.params['w1'])
        grads['b1'] = koubai_tazigenn(loss_w, self.params['b1'])
        grads['w2'] = koubai_tazigenn(loss_w, self.params['w2'])
        grads['b2'] = koubai_tazigenn(loss_w, self.params['b2'])
        return grads
    
    def gradient(self, x, t):
        #forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())#.valuesのせいで出てきたdict_values型を配列に変えてる
        layers.reverse()#layersの順番を逆にしている？
        for layer in layers:
            dout = layer.backward(dout)#layerには処理の種類が入っている
        
        #設定
        grads = {}
        grads['w1'] = self.layers['gyouretuseki_layer1'].dw
        grads['b1'] = self.layers['gyouretuseki_layer1'].db
        grads['w1'] = self.layers['gyouretuseki_layer2'].dw
        grads['w1'] = self.layers['gyouretuseki_layer2'].db
        return grads#逆伝播の勾配？