import numpy as np
import torch as th
import torchlib as tl
import matplotlib.pyplot as plt

ftshift = False
# ftshift = True

X_th = tl.imread('../../data/images/Einstein256.png')
X_th = X_th + 1j * X_th

X_np = X_th.numpy()

device = th.device('cuda:0')
X_th = X_th.to(device)
print(X_th.shape, type(X_th))
# print(x_th)

Y1 = np.fft.fft(X_np, axis=0)
Y1 = np.fft.fft(Y1, axis=1)
# Y1 = np.fft.fft(np.fft.fft(X_np, axis=0), axis=1)
Y1 = np.abs(Y1)

Y2 = tl.fft(X_th, axis=0, shift=ftshift)
Y2 = tl.fft(Y2, axis=1, shift=ftshift)
Y2 = th.abs(Y2).cpu()

print(np.sum(Y1 - Y2.numpy()))
Y1 = np.log10(Y1)
Y2 = th.log10(Y2)

plt.figure()
plt.subplot(131)
plt.imshow(np.abs(X_np))
plt.subplot(132)
plt.imshow(Y1)
plt.subplot(133)
plt.imshow(Y2)
plt.show()
