def koubai (f , x):
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
def mugen_koubai (f , x):
    import numpy as np
    for k in range(100):
        dfx = koubai(f , x)
        x = x + dfx * 0.01
    return x