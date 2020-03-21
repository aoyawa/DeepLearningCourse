import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
boston_housing=tf.keras.datasets.boston_housing
(train_x,train_y),(_,_)=boston_housing.load_data(test_split=0)
plt.rcParams["font.sans-serif"]="SimHei"
plt.rcParams['axes.unicode_minus']=False
titles=np.array(['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'])
plt.figure(figsize=(8,8))
for i in range(13):
    plt.subplot(4,4,(i+1))
    plt.scatter(train_x[:,i],train_y,24)
    plt.xlabel(titles[i])
    plt.ylabel("Prices($1000's)")
    plt.title(str(i+1)+ "."+titles[i]+" 一Price")
plt.tight_layout(rect=[0,0,1,0.9])
plt.suptitle("各个属性与房价的关系",fontsize=20)
plt.show()
for j in range(13):
    if (j>=9):
        print("%d - %s"%(j+1,titles[j]))
    else:
        print("%d -- %s"%(j+1,titles[j]))
    
x=int(input("请选择属性："))
plt.scatter(train_x[:,x-1],train_y,14)
plt.xlabel(titles[x-1])
plt.ylabel("Prices($1000's)")
plt.title(str(x)+ "."+titles[x-1]+" 一Price")
plt.show()