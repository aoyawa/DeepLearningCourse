import matplotlib.pyplot as plt
from PIL import Image
img=Image.open("lena.tiff")
img_r,img_g,img_b=img.split()
plt.rcParams["font.sans-serif"]="SimHei"
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(5,5))
plt.suptitle("图像基本操作",fontsize=20,color="blue")
plt.tight_layout(rect=[0,0,1,0.9])

plt.subplot(221)
plt.title("R-缩放",fontsize=14)
img1=img_r.resize((50,50))
plt.imshow(img1,cmap="gray")
plt.axis("off")

plt.subplot(222)
plt.title("G-镜像+旋转",fontsize=14)
img_r90=img_g.transpose(Image.ROTATE_90)
img_flr=img_r90.transpose(Image.FLIP_LEFT_RIGHT)
plt.imshow(img_flr,cmap="gray")

plt.subplot(223)
plt.title("B-裁剪",fontsize=14)
img_crop=img_b.crop((0,0,150,150))
plt.imshow(img_crop,cmap="gray")
plt.axis("off")

plt.subplot(224)
plt.title("RGB",fontsize=14)
img2=Image.merge("RGB",[img_r,img_g,img_b])
plt.imshow(img2)
plt.axis("off")
img2.save("test.png")

plt.show()