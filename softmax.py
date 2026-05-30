def softmax(x):
    import numpy as np
    exp_a = np.exp(x)
    exp_sum = np.sum(exp_a)
    y = exp_a / exp_sum
    print (exp_a)
    return (y)