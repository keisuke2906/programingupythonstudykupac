import sys, os
sys.path.append(os.pardir)
import numpy as np
from ainofunction import softmax, cross_entropy, sigmoid
from ainogradient import koubai, mugen_koubai
class omomi_nisou:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):  #これはクラスを指定された時点で発動する　　これらの因数は一度ですべてのデータを読み込ませ計算する
        #omominoseisei
        self.params = {}
        self.params['w1'] = weight_init_std *  np.random.randn(input_size, hidden_size)  #input_sizeが入力値のデータの大きさ　
                                                         #output_sizeが出力値のデータの大きさ。かけるとｗになる
        self.params['b1'] = np.zeros(hidden_size)
        self.params["w2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(hidden_size)

    def predict(self, x):
        w1 , w2 = self.params['w1'] , self.params['w2']
        b1 , b2 = self.params["b1"] , self.params["b2"]

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1) 
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)

        return y
    
    def loss(self, x, t): #t ha kyousidata
        y = self.predict(x)           #名前.loss(x, t)で簡単に誤差

        return cross_entropy(y, t)
    
    def accuracy(self , x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)  #np.argmaxは行列の行方向で一番大きい値のインデックスを配列にして出力する
        accuracy = np.sum(y == t) / float(x.shape[0]) #float(x.shape[0])はこの時一気に読み込ませたデータの総数である。
    
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
    def f(self,):
        return mugen_koubai(self.loss)