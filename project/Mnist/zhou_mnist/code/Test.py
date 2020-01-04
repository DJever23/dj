﻿from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
import time

def imageprepare():
    """
        功能：图片预处理
        输入：读取灰度png图片
        输出：灰度图像像素值
    """
    file_name='C:/Users/mechrevo/Desktop/9.png'#导入自己的图片地址
    #图片格式转换
    im = Image.open(file_name)
    plt.imshow(im)  #显示需要识别的图片
    plt.show()
    im = im.convert('L')#'L'模式图片
    im.save("C:/Users/mechrevo/Desktop/sample.png")#保存经过格式转换后的图片
    tv = list(im.getdata()) #获取图片像素值
    #格式化像素：0-1为纯白色, 1 是纯黑.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    print(tva)
    return tva

    """
        输入：图片预处理返回像素值
        输出：测试集
    """
    #定义模型
result=imageprepare()#调用图片预处理函数
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

"""
输入：一维的张量，即输出张量
输出：返回随机值
shape：[batch, height, width, channels]形式
"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1) #从截断的正态分布中输出随机值，stddev为正太分布标准差
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape) #创建常量
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME') #进行二维卷积


"""
功能：最大值池化操作，紧接卷积操作之后，返回一个Tensor
函数输入参数：
    第一个参数：需要池化的输入
    第二个参数：池化窗口的大小
    第三个参数：和卷积类似，窗口在每一个维度上滑动的步长
    第四个参数：和卷积类似，可以取'VALID' 或者'SAME
"""
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') #CNN当中的最大值池化操作，和卷积类似


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 这里使用了之前保存的模型参数
    saver.restore(sess, "H:/CSDN/SAVE/model.ckpt")
    #print ("Model restored.")

    prediction=tf.argmax(y_conv,1)
    predint=prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)
    print(h_conv2)

    print('识别结果:')
    print(predint[0])