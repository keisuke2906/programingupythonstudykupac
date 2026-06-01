def softmax(x):   #指数関数的で人間的な確率のとらえ方
    import numpy as np
    c = np.max(x)
    exp_a = np.exp(x - c)
    exp_sum = np.sum(exp_a)
    y = exp_a / exp_sum
    print (exp_a)
    return y


def sigmoid(x):
    import numpy as np
    y = 1 / (1 + np.exp(- x))
    return y

def cross_entropy(x , t):
    import numpy as np
    y = - np.sum(t * np.log(x + 1e-7)) 
    return y