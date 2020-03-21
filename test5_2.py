import numpy as np
x=np.array([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03])
y=np.array([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84])
x1=x.mean()
y1=y.mean()
x2=x-x1
x3=x2**2
y2=y-y1
f=x2*y2
f1=np.sum(f)
m=np.sum(x3)
w=f1/m
b=y1-w*x1
print("w=%f"%w)
print("b=%f"%b)