def softmax(x):
    import numpy as np
    c = np.max(x)
    exp_a = np.exp(x - c)
    exp_sum = np.sum(exp_a)
    y = exp_a / exp_sum
    print (exp_a)
    return (y)