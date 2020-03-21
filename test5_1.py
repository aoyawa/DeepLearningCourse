import numpy as np
a=int(input("Enter a number between 1 and 100:"))
i=0
j=1
np.random.seed(612)
while i<1000:
  x=np.random.uniform(0,1)
  if i%a==0:
    print("%d\t%d\t%f"%(j,i,x))
    j+=1
  i+=1