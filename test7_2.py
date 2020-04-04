import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist=tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y)=mnist.load_data()
plt.rcParams["font.sans-serif"]="SimHei"
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(3,3))
plt.suptitle("MNIST测试集样本",fontsize=20,color="red")

for i in range(16):
    plt.subplot(4,4,i+1)
    num=np.random.randint(10000)
    plt.imshow(test_x[num],cmap="gray")
    strtitle="标签值："+str(test_y[num])
    plt.title(strtitle,fontsize=14)
    plt.axis("off")
plt.show()