from gym import spaces
import numpy as np
from scipy.special import softmax


x = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(20,), dtype=np.float32)
y = spaces.Discrete(5)
z = [1,2,3,3]
obs = {f'{i}': z for i in range(4)}
n = np.ones((1,4))
print(n.shape)
m = np.reshape(n, (-1))
print(m.shape)


