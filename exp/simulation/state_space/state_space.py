#%%
import matplotlib.pyplot as plt
import numpy as np

f,ax=plt.subplots(figsize=(5,5))
x = np.arange(0, 5, 1)
y = np.arange(0, 5, 1)

X, Y = np.meshgrid(x, y)

plt.plot(X, Y, marker='s', color='k', linestyle='')
plt.xticks([0,1,2,3,4],[1,2,3,4,5])
plt.yticks([0,1,2,3,4],[1,2,3,4,5])
plt.show()