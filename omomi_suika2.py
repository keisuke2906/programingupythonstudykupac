import sys, os
sys.path.append(os.pardir)
import numpy as np
from ainofunction import softmax, cross_entropy, sigmoid, nizyou_heikinn
from ainogradient import koubai, mugen_koubai, koubai_tazigenn
class omomi_suika:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.5):  #これはクラスを指定された時点で発動する　　これらの因数は一度ですべてのデータを読み込ませ計算する
        #omominoseisei
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)  #input_sizeが入力値のデータの大きさ　
                                                         #output_sizeが出力値のデータの大きさ。かけるとｗになる
        self.params['b1'] = np.zeros(hidden_size)
        self.params["w2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, a):
        w1 , w2 = self.params['w1'] , self.params['w2']
        b1 , b2 = self.params["b1"] , self.params["b2"]
        a1 = np.dot(a, w1) + b1
        z1 = 4 * sigmoid(a1) 
        a2 = np.dot(z1, w2) + b2
        y = abs(softmax(a2))

        return a2
    
    def loss(self, t, a): #t ha kyousidata
        y = self.predict(a)           #名前.loss(x, t)で簡単に誤差
        return nizyou_heikinn(y, t)
    
    
    def numerical_gradient(self, t, a):#aは入力値
        loss_W = lambda W: self.loss(t, a)
        
        grads = {}
        grads['w1'] = koubai_tazigenn(loss_W, self.params['w1'])
        grads['b1'] = koubai_tazigenn(loss_W, self.params['b1'])
        grads['w2'] = koubai_tazigenn(loss_W, self.params['w2'])
        grads['b2'] = koubai_tazigenn(loss_W, self.params['b2'])
        
        return grads
    def gakusyuu(self, a, t):
        h = 0.0005
        grads = self.numerical_gradient(t, a)

        self.params['w1'] -= h * grads['w1']
        self.params['b1'] -= h * grads['b1']
        self.params['w2'] -= h * grads['w2']
        self.params['b2'] -= h * grads['b2']
    

unnko = omomi_suika(input_size = 2, hidden_size = 3, output_size = 2, weight_init_std = 0.5)

for i in range(200):
    a = np.random.randn(1, unnko.input_size)
    for k in range(50):
        unnko.gakusyuu(a, a)
    print(f"{i+1}回目の学習完了")
print (unnko.params)
k = 0
for i in range(100):
    a = np.random.randn(1, unnko.input_size)
    k = k + unnko.loss(a, a)
    print(f"{i+1}回目の学習完了")
print("heikinnha")
print (k / 100)
x = input("suikanoitiha?x")
y = input("suikanoitiha?y")
#z = input("suikanoitiha?z")
aa = np.array([[float(x), float(y)]])
print (unnko.predict(aa))


