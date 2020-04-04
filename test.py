import matplotlib.pyplot as plt
import tensorflow as tf
mnist=tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y)=mnist.load_data()
print("train_set",len(train_x))
print('test_set',len(test_x))