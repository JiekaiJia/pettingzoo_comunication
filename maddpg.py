import tensorflow as tf
from ray.rllib.utils.tf_ops import one_hot
from gym.spaces import Discrete
a = tf.ones([], dtype=tf.int32)
b = a
print(a)
b = one_hot(b, Discrete(3))
print(a)
print(b)