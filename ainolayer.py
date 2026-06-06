#d~は大体その変数の偏微分を表します

class zyousann_layer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):#doutは上から降ってくる微分
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
    

class kasann_layer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out


class relu_layer:
    def __init__(self):
        self.mask = None

    def forward(self, x,):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    

class sigmoid_layer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        import numpy as np
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    

class gyouretuseki_layer:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
    
    def forward(self, x):
        import numpy as np
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out
    
    def backward(self, dout):
        import numpy as np
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    

class softmax_with_loss_layer:
    def __init__(self):
        self.loss = None
        self.y = None #ソフトマックス関数の出力
        self.t = None

    def forward(self, x, t):
        from ainofunction import softmax, cross_entropy
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y,self.t)
        return self.loss
    
    def backward(self, dout=1):#一番最後だからdoutは１
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx