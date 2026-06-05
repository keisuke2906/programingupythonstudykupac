import sys, os
sys.path.append(os.pardir)
import numpy as np
from ainofunction import softmax, cross_entropy, sigmoid
from ainogradient import koubai, mugen_koubai, koubai_tazigenn
class omomi_suika:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):  #これはクラスを指定された時点で発動する　　これらの因数は一度ですべてのデータを読み込ませ計算する
        #omominoseisei
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.params = {}
        self.params['w1'] = weight_init_std *  np.random.randn(input_size, hidden_size)  #input_sizeが入力値のデータの大きさ　
                                                         #output_sizeが出力値のデータの大きさ。かけるとｗになる
        self.params['b1'] = np.zeros(hidden_size)
        self.params["w2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(hidden_size)

    def predict(self, a):
        w1 , w2 = self.params['w1'] , self.params['w2']
        b1 , b2 = self.params["b1"] , self.params["b2"]

        a1 = np.dot(a, w1) + b1
        z1 = sigmoid(a1) 
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)

        return y
    
    def loss(self, w, t, a): #t ha kyousidata
        y = self.predict(a)           #名前.loss(x, t)で簡単に誤差

        return cross_entropy(y, t)
    
    def accuracy(self , x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)  #np.argmaxは行列の行方向で一番大きい値のインデックスを配列にして出力する
        accuracy = np.sum(y == t) / float(x.shape[0]) #float(x.shape[0])はこの時一気に読み込ませたデータの総数である。
    
        return accuracy
    
    def numerical_gradient(self, w, t, a):#aは入力値
        loss_W = lambda W: self.loss(w, t, a)
        
        grads = {}
        grads['w1'] = koubai_tazigenn(loss_W, self.params['w1'])
        grads['b1'] = koubai_tazigenn(loss_W, self.params['b1'])
        grads['w2'] = koubai_tazigenn(loss_W, self.params['w2'])
        grads['b2'] = koubai_tazigenn(loss_W, self.params['b2'])
        
        return grads
    def gakusyuu(self, a, t):
        h = 0.01
        grads = self.numerical_gradient(None, t, a)
        
        # 【修正】勾配（傾き）の方向に、h（学習率）を掛けて「引き算」する（勾配降下法）
        self.params['w1'] -= h * grads['w1']
        self.params['b1'] -= h * grads['b1']
        self.params['w2'] -= h * grads['w2']
        self.params['b2'] -= h * grads['b2']
    
    def takusann_gakusyuu(self, t):
        for i in range(100):
            a = np.random.randn(self.input_size, 1)
            y = self.gakusyuu(a, t)

unnko = omomi_suika(input_size = 3, hidden_size = 10, output_size = 3, weight_init_std = 0.01)

for i in range(100):
    a = np.random.randn(unnko.input_size, 1)
    t = np.random.randn(unnko.input_size, 1)
    unnko.params['w1'],unnko.params['w2'] = unnko.gakusyuu(a, t)['w1'],unnko.gakusyuu(a, t)['w2']
    unnko.params['b1'],unnko.params['b2'] = unnko.gakusyuu(a, t)['b1'],unnko.gakusyuu(a, t)['b2']
print (unnko.params)