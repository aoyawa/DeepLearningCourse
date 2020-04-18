import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
from sklearn.utils import shuffle

df=pd.read_csv('d:\\data\\boston.csv',header=0)
df=df.values
x_data=df[:,:12]
for i in range(12):
    x_data[:,i]=(x_data[:,i]-x_data[:,i].min())/(x_data[:,i].max()-x_data[:,i].min())
y_data=df[:,12]

train_num=300
valid_num=100
test_num=len(x_data)-train_num-valid_num

x_train=x_data[:train_num]
y_train=y_data[:train_num]

x_valid=x_data[train_num:train_num+valid_num]
y_valid=y_data[train_num:train_num+valid_num]

x_test=x_data[train_num+valid_num:train_num+valid_num+test_num]
y_test=y_data[train_num+valid_num:train_num+valid_num+test_num]

x_train=tf.cast(x_train,dtype=tf.float32)
x_valid=tf.cast(x_valid,dtype=tf.float32)
x_test=tf.cast(x_test,dtype=tf.float32)

def model(x,w,b):
    return tf.matmul(x,w)+b

W=tf.Variable(tf.random.normal([12,1],mean=0.0,stddev=1.0,dtype=tf.float32))
B=tf.Variable(tf.zeros(1),dtype=tf.float32)
def loss(x,y,w,b):
    err=model(x,w,b)-y
    squared_err=tf.square(err)
    return tf.reduce_mean(squared_err)
def grad(x,y,w,b):
    with tf.GradientTape()  as tape:
        loss_=loss(x,y,w,b)
    return tape.gradient(loss_,[w,b])
training_epochs=100
learning_rate=0.02
batch_size=10
optimizer=tf.keras.optimizers.SGD(learning_rate)

loss_list_train=[]
loss_list_valid=[]
total_step=int(train_num/batch_size)

for epoch in range(training_epochs):
    for step in range(total_step):
        xs=x_train[step*batch_size:(step+1)*batch_size]
        ys=y_train[step*batch_size:(step+1)*batch_size]
        grads=grad(xs,ys,W,B)
        optimizer.apply_gradients(zip(grads,[W,B]))
    loss_train=loss(x_train,y_train,W,B).numpy()
    loss_valid=loss(x_valid,y_valid,W,B).numpy()
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    print("epoch={:3d}  train_loss={:.4f}  valid_loss={:.4f}".format(epoch+1,loss_train,loss_valid))
print("b=",B.numpy())
print("w=",W.numpy())
