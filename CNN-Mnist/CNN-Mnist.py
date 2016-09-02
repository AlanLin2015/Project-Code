#coding:utf-8  
  
import os  
from PIL import Image  
import numpy as np  
  
#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，  
#如果是将彩色图作为输入,则将1替换为3，并且data[i,:,:,:] = arr改为data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]  
def load_data():  
    data = np.empty((42000,1,28,28),dtype="float32")   #empty与ones差不多原理，但是数值随机，类型随后面设定  
    label = np.empty((42000,),dtype="uint8")  
  
    imgs = os.listdir("./mnist")   #打开文件夹里的目录  
    num = len(imgs)  
    for i in range(num):  
        img = Image.open("./mnist/"+imgs[i])   #用图像处理函数Image处理图像  
        arr = np.asarray(img,dtype="float32")  #将img数据转化为数组形式  
        data[i,:,:,:] = arr   #将每个三维数组赋给data  
        label[i] = int(imgs[i].split('.')[0])   #取该图像的数值属性作为标签  
    return data,label 
    
    
from __future__ import absolute_import  
from __future__ import print_function  
from keras.preprocessing.image import ImageDataGenerator  
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation, Flatten  
from keras.layers.advanced_activations import PReLU  
from keras.layers.convolutional import Convolution2D, MaxPooling2D  
from keras.optimizers import SGD, Adadelta, Adagrad  
from keras.utils import np_utils, generic_utils  
from six.moves import range  
from data import load_data  
  
  
data, label = load_data()   #导入数据  
print(data.shape[0], ' samples')   #输出数据数量  
  
  
label = np_utils.to_categorical(label, 10)   #label为0~9共10个类别，keras要求格式为binary class matrices,调用keras这个函数转化  
  
#建立CNN模型
model = Sequential()   #建立一个Sequential模型  
  
model.add(Convolution2D(4, 5, 5, border_mode='valid',input_shape=(1,28,28)))    #第一个卷积层，4个卷积核，每个卷积核5*5,卷积后24*24，第一个卷积核要申明input_shape(通道，大小)  
model.add(Activation('tanh'))   #激活函数采用“tanh”  
  
model.add(Convolution2D(8, 3, 3, subsample=(2,2), border_mode='valid'))   #第二个卷积层，8个卷积核，不需要申明上一个卷积留下来的特征map，会自动识别，下采样层为2*2,卷完且采样后是11*11  
model.add(Activation('tanh'))  
  
model.add(Convolution2D(16, 3, 3, subsample=(2,2), border_mode='valid'))   #第三个卷积层，16个卷积核，下采样层为2*2,卷完采样后是4*4  
model.add(Activation('tanh'))  
  
model.add(Flatten())   #把多维的模型压平为一维的，用在卷积层到全连接层的过度  
model.add(Dense(128, input_dim=(16*4*4), init='normal'))   #全连接层，首层的需要指定输入维度16*4*4,128是输出维度，默认放第一位  
model.add(Activation('tanh'))  
  
model.add(Dense(10, input_dim= 128, init='normal'))   #第二层全连接层，其实不需要指定输入维度，输出为10维，因为是10类  
model.add(Activation('softmax'))   #激活函数“softmax”，用于分类  
  
#训练CNN模型   
  
sgd = SGD(lr=0.05, momentum=0.9, decay=1e-6, nesterov=True)   #采用随机梯度下降法，学习率初始值0.05,动量参数为0.9,学习率衰减值为1e-6,确定使用Nesterov动量  
model.compile(loss='categorical_crossentropy', optimizer=sgd)   #配置模型学习过程，目标函数为categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列，第18行已转化，优化器为sgd  
  
model.fit(data, label, batch_size=100,nb_epoch=10,shuffle=True,verbose=1,validation_split=0.2)   #训练模型，训练nb_epoch次，bctch_size为梯度下降时每个batch包含的样本数，验证集比例0.2,verbose为显示日志，shuffle是否打乱输入样本的顺序  