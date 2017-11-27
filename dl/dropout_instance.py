import numpy as np

x = np.random.random((10, 100)) #模拟一个batch_size=10、维度为100的输入
print(x.shape)


def Dropout(x, drop_proba):
    return x*np.random.choice(
                              [0, 1],
                              x.shape,
                              p=[drop_proba, 1-drop_proba]
                             )/(1.-drop_proba)


def Dropout2(x, drop_proba, noise_shape):
    return x*np.random.choice(
                              [0, 1],
                              noise_shape,
                              p=[drop_proba, 1-drop_proba]
                             )/(1.-drop_proba)

# print(Dropout(x, 0.6))
print(Dropout2(x, 0.5, [10, 1]))
print(Dropout2(x, 0.5, [1, 100]))
