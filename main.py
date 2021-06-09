import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("C:\\Users\\K\\Desktop\\123.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# aaa = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
'''获取图片大小'''
rows, cols, _ = img.shape
print(rows, cols)

'''初始化一个大小一样的图像，元素初始化为0 '''
created1 = np.zeros((rows, cols))

''' for in 语句，遍历数组，但不能修改数组'''
for i in img:
    for j in i:
        # print(j)
        pass

'''给数组赋值 '''
for i in range(0, rows):
    for j in range(0, cols):
        created1[i, j] = img[i, j, 2]  # 第三个参数，可以是0,1,2分别表示3个通道

'''必须加这一条语句,否则无法正确显示图像 '''
created1 = created1.astype(np.uint8)  # 转换类型，才能正确的显示图像
cv.imshow('Red Channel', created1)

cv.imshow("Image", img)

'''获取原图（100，125）的三通道像素值'''
test = img[100, 125]
print("原图三通道像素值", test)

test = created1[100, 125]
print("通道2 红色", test)

# cv.imshow("girl", img)

# girl.ravel()函数是将图像的三位数组降到一维上去，256为bins的数目,[0, 256]为范围

plt.hist(img.ravel(), 256, [0, 256])
'''直方图计算'''
color = ("b", "g", "r")
for i, color in enumerate(color):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.title("img of histogram")
    plt.xlabel("Bins")
    plt.ylabel("num of perlex")
    plt.plot(hist, color=color)
    plt.xlim([0, 260])


plt.show()

# cv2.imshow("Iaa", gray)
# cv2.imshow("aaa", aaa)

cv.waitKey()              # 按下任意键退出
cv.destroyAllWindows()
