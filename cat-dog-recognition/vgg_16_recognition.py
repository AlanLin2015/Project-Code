#coding:utf-8  
import os   #处理字符串路径  
import glob   #查找文件  
  
from keras.models import Sequential   #导入Sequential模型  
  
from keras.layers.core import Flatten, Dense, Dropout  
  
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D  
  
from keras.optimizers import SGD  
  
import cv2, numpy as np  
  
from pandas import Series,DataFrame  
  
def VGG_16(weights_path=None):    #根据keras官方文档建立VGG_16模型  
  
    model = Sequential()  
  
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))  
    model.add(Convolution2D(64, 3, 3, activation='relu'))  
    model.add(ZeroPadding2D((1,1)))  
    model.add(Convolution2D(64, 3, 3, activation='relu'))  
    model.add(MaxPooling2D((2,2), strides=(2,2)))  
  
  
    model.add(ZeroPadding2D((1,1)))  
    model.add(Convolution2D(128, 3, 3, activation='relu'))  
    model.add(ZeroPadding2D((1,1)))  
    model.add(Convolution2D(128, 3, 3, activation='relu'))  
    model.add(MaxPooling2D((2,2), strides=(2,2)))  
  
  
    model.add(ZeroPadding2D((1,1)))  
    model.add(Convolution2D(256, 3, 3, activation='relu'))  
    model.add(ZeroPadding2D((1,1)))  
    model.add(Convolution2D(256, 3, 3, activation='relu'))  
    model.add(ZeroPadding2D((1,1)))  
    model.add(Convolution2D(256, 3, 3, activation='relu'))  
    model.add(MaxPooling2D((2,2), strides=(2,2)))  
  
    model.add(ZeroPadding2D((1,1)))  
    model.add(Convolution2D(512, 3, 3, activation='relu'))  
    model.add(ZeroPadding2D((1,1)))  
    model.add(Convolution2D(512, 3, 3, activation='relu'))  
    model.add(ZeroPadding2D((1,1)))  
    model.add(Convolution2D(512, 3, 3, activation='relu'))  
    model.add(MaxPooling2D((2,2), strides=(2,2)))  
  
  
    model.add(ZeroPadding2D((1,1)))  
    model.add(Convolution2D(512, 3, 3, activation='relu'))  
    model.add(ZeroPadding2D((1,1)))  
    model.add(Convolution2D(512, 3, 3, activation='relu'))  
    model.add(ZeroPadding2D((1,1)))  
    model.add(Convolution2D(512, 3, 3, activation='relu'))  
    model.add(MaxPooling2D((2,2), strides=(2,2)))  
  
  
    model.add(Flatten())  
    model.add(Dense(4096, activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(4096, activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(1000, activation='softmax'))  
  
    if weights_path:  
        model.load_weights(weights_path)  
  
    return model  
  
model = VGG_16('vgg16_weights.h5')   #用训练好的vgg16_weights权重做预测  
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)    #采用随机梯度下降法，学习率初始值0.1,动量参数为0.9,学习率衰减值为1e-6,确定使用Nesterov动量  
model.compile(optimizer=sgd, loss='categorical_crossentropy')    #配置模型学习过程，目标函数为categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列  
  
dogs=[251, 268, 256, 253, 255, 254, 257, 159, 211, 210, 212, 214, 213, 216, 215, 219, 220, 221, 217, 218, 207, 209, 206, 205, 208, 193, 202, 194, 191, 204, 187, 203, 185, 192, 183, 199, 195, 181, 184, 201, 186, 200, 182, 188, 189, 190, 197, 196, 198, 179, 180, 177, 178, 175, 163, 174, 176, 160, 162, 161, 164, 168, 173, 170, 169, 165, 166, 167, 172, 171, 264, 263, 266, 265, 267, 262, 246, 242, 243, 248, 247, 229, 233, 234, 228, 231, 232, 230, 227, 226, 235, 225, 224, 223, 222, 236, 252, 237, 250, 249, 241, 239, 238, 240, 244, 245, 259, 261, 260, 258, 154, 153, 158, 152, 155, 151, 157, 156]   #训练好的权重文件里，属于狗的位置  
  
cats=[281,282,283,284,285,286,287]   #权重文件里，猫的位置  
  
path = os.path.join('imgs', 'test', '*.jpg')   #拼接路径  
files = glob.glob(path)    #打开这个路径  
  
result=[]  
flbase=0  
p=0  
a=0  
  
  
for fl in files:  
    a=cv2.imread(fl)   #读入图像  
    if  a ==None:     #异常检测，检测空图  
        pass  
    else:  
        im = cv2.resize(a,(224,224)).astype(np.float32)   #使图片变为224*224大小并转化为float32格式  
        im[:,:,0] -= 103.939  
        im[:,:,1] -= 116.779  
        im[:,:,2] -= 123.68  
        im = im.transpose((2,0,1))  
        im = np.expand_dims(im,axis=0)      
        out = model.predict(im)   #输出的预测列表是二维列表，即[[x]]的形式，这个列表里储存了该图像与权重文件中每个物品的相似度  
        flbase = os.path.basename(fl)  
        p = np.sum(out[0,dogs]) / (np.sum(out[0,dogs]) + np.sum(out[0,cats]))   #计算该图像与狗的相似度/（该图像与狗的相似度+该图像与猫的相似度）  
        result.append((flbase,p))   #将（文件名，相似概率）加到result列表里  
  
result=sorted(result, key=lambda x:x[1], reverse=True)   #按概率逆序排序  
  
for x in result:  
    print x[0],x[1]  
  
for x in result:  
    print x[0]  
  
dataframe=DataFrame(result)  
dataframe.to_csv("~/dogs-recognition.csv")  