import tensorflow as tf
import numpy as np
x=tf.constant(np.array([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03]))
y=tf.constant(np.array([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84]))
wx=10*tf.reduce_sum(x*y)-tf.reduce_sum(x)*tf.reduce_sum(y)
wy=10*tf.reduce_sum(tf.square(x))-tf.square(tf.reduce_sum(x))
w=wx/wy
bx=tf.reduce_sum(y)-w*tf.reduce_sum(x)
b=bx/10
print("w=%f"%w)
print("b=%f"%b)