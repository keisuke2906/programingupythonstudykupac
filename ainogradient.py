def koubai (f , x):  #一列の勾配
    import numpy as np
    h = 0.0001
    koubai_atai = np.zeros_like(x)
    for i in range (x.size):
        a = x[i]
        x[i] = x[i] + h
        fxh1 = f(x)

        x[i] = a - h
        fxh2 = f(x)
        koubai_atai[i] = (fxh1 - fxh2) / 2*h
        x[i] = a
    return koubai_atai

def mugen_koubai (f , x): #勾配の向きに動く
    import numpy as np
    for k in range(100):
        dfx = koubai(f , x)
        x = x - dfx * 0.01
    return x

def koubai_tazigenn(f, x): #多数列の勾配 xは元の入力値ではなく重みやバイアス
    import numpy as np
    if x.ndim == 1: #ｘが一列だった時
        return koubai(f, x)
    else:
        grad = np.zeros_like(x)
        for i, X in enumerate(x):#大文字xにその列のｘの値を入れた一列行列にしている。
            grad[i] = koubai(f, X)
        
        return grad

            