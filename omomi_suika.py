import sys, os
sys.path.append(os.pardir)
import numpy as np
from ainofunction import softmax, cross_entropy, sigmoid
from ainogradient import koubai, mugen_koubai, koubai_tazigenn
class omomi_suika:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std):  #これはクラスを指定された時点で発動する　　これらの因数は一度ですべてのデータを読み込ませ計算する
        #omominoseisei
        self.params = {}
        self.params['w1'] = weight_init_std *  np.random.randn(input_size, hidden_size)  #input_sizeが入力値のデータの大きさ　
                                                         #output_sizeが出力値のデータの大きさ。かけるとｗになる
        self.params['b1'] = np.zeros(hidden_size)
        self.params["w2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(hidden_size)
        self.input_size = input_size

    def predict(self, x):
        w1 , w2 = self.params['w1'] , self.params['w2']
        b1 , b2 = self.params["b1"] , self.params["b2"]

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1) 
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)

        return y
    
    def predict2(self, x):
        w1 = self.gakusyuu['w1'] 
        b1 = self.gakusyuu['b1'] 
        w2 = self.gakusyuu['w2'] 
        b2 = self.gakusyuu['b2'] 

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1) 
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)

        return y
    
    def loss(self, a, t): #t ha kyousidata
        y = self.predict2(a)           #名前.loss(x, t)で簡単に誤差

        return cross_entropy(y, t)
    
    def loss2(self, a, t): #t ha kyousidata
        y = self.predict(a)           #名前.loss(x, t)で簡単に誤差

        return cross_entropy(y, t)
    
    def accuracy(self , x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)  #np.argmaxは行列の行方向で一番大きい値のインデックスを配列にして出力する
        accuracy = np.sum(y == t) / float(x.shape[0]) #float(x.shape[0])はこの時一気に読み込ませたデータの総数である。
    
        return accuracy
    
    def numerical_gradient(self, x, t, a):
        loss_W = lambda W: self.loss(a, t)
        
        grads = {}
        grads['W1'] = koubai_tazigenn(loss_W, self.params['W1'])
        grads['b1'] = koubai_tazigenn(loss_W, self.params['b1'])
        grads['W2'] = koubai_tazigenn(loss_W, self.params['W2'])
        grads['b2'] = koubai_tazigenn(loss_W, self.params['b2'])
        
        return grads
    
    def numerical_gradient2(self, x, t, a):
        loss_W = lambda W: self.loss2(a, t)
        
        grads = {}
        grads['W1'] = koubai_tazigenn(loss_W, self.params['W1'])
        grads['b1'] = koubai_tazigenn(loss_W, self.params['b1'])
        grads['W2'] = koubai_tazigenn(loss_W, self.params['W2'])
        grads['b2'] = koubai_tazigenn(loss_W, self.params['b2'])
        
        return grads
    
    def f(self):
        return 
    
    def gakusyuu(self, x, t):
        h = 0.01
        w1 , w2 = self.params['w1'] , self.params['w2']
        b1 , b2 = self.params["b1"] , self.params["b2"]
        w1 += self.numerical_gradient(w1, t, x)['w1']
        b1 += self.numerical_gradient(b1, t, x)['b1']
        w2 += self.numerical_gradient(w2, t, x)['w2']
        b2 += self.numerical_gradient(b2, t, x)['b2']
        gakusyuu_zumi = {}
        gakusyuu_zumi['w1'] = w1
        gakusyuu_zumi['b1'] = b1
        gakusyuu_zumi['w2'] = w2
        gakusyuu_zumi['b2'] = b2


    def gakusyuu2(self, x, t):
        h = 0.01
        w1 , w2 = 
        b1 , b2 = 
        w1 += self.numerical_gradient2(w1, t, x)['w1']
        b1 += self.numerical_gradient2(b1, t, x)['b1']
        w2 += self.numerical_gradient2(w2, t, x)['w2']
        b2 += self.numerical_gradient2(b2, t, x)['b2']
        gakusyuu_zumi = {}
        gakusyuu_zumi['w1'] = w1
        gakusyuu_zumi['b1'] = b1
        gakusyuu_zumi['w2'] = w2
        gakusyuu_zumi['b2'] = b2
        return gakusyuu_zumi
    def takusann_gakusyuu(self, t):
        x = np.random.randn(self.input_size, 1)
        y = self.gakusyuu(x, t)
        for i in range(99):
            x = np.random.randn(self.input_size, 1)
            y = self.gakusyuu(x, t)

unnko = omomi_suika(input_size = 3, hidden_size = 10, output_size = 3, weight_init_std = 0.01)
