#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import matplotlib.pyplot as plt
import glob
import re
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from PIL import Image
import time


# In[14]:


#list1 = os.listdir('class10_high')
kd = 'dataset_split'
list2 = os.listdir(kd)
imgbox = []
labels = []
#for i in range(1):
for i in tqdm(range(len(list2))):
    c = glob.glob(kd+list2[i]+'/*.jpg')
    for j in range(len(c)):
        a1 = c[j]
        a1 = a1.replace('\\','/')
        image = Image.open(a1)
        a11 = np.array(image.resize((50, 50))).reshape(50,50,3)
        imgbox.append(a11)
        labels.append(i)


# In[15]:



        
generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 30, #각도
    width_shift_range = 0.1,  #확대 축소
    height_shift_range = 0.1)  #확대 축소

gen2 = []
gen3 = []
# fld = '9240202000/' #이미지가 있는 폴더
make_num = 4  #생산할 이미지 갯수
# c1 = os.listdir(fld)

for i in tqdm(range(len(imgbox))):
    aa = imgbox[i]
    cc = labels[i]
    aa = aa.reshape((1,)+aa.shape)
    a1 = generator.flow(aa)
    for j in range(make_num):
        a3 = a1.next()[0]
        a3 = np.int32(a3)
        gen2.append(a3)
        gen3.append(cc)
#gen2 에 이미지가 있습니다 순서대로..
#gen3 는 라벨 이름입니다. (필요시사용)
print(len(imgbox),len(gen2),len(gen3))
#a2 = plt.imread(imgbox[0])
plt.imshow(gen2[1])
x_data = []
y_data = [] 
for i in range(len(gen3)):
    x_data.append(gen2[i])
    y_data.append(gen3[i])
import keras
y_data = keras.utils.to_categorical(y_data, num_classes=None, dtype='float32')
x_data = np.array(x_data)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2,random_state=42)
print(x_train.shape, y_train.shape,x_test.shape,y_test.shape)


# In[ ]:





# In[ ]:





# In[16]:


ranstate = 3,6,42,21,27

learning_rate = 0.005
training_epochs = 120
batch_size = 100
#94
acc_list = []
from sklearn.model_selection import train_test_split
for z in tqdm(ranstate):
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2,random_state=z)
    print(x_train.shape, y_train.shape,x_test.shape,y_test.shape)

    fil = 5
    tf.reset_default_graph()

    #tf.set_random_seed(777)
    x= tf.placeholder(tf.float32, shape=[None,50,50,3], name="x_data")
    y= tf.placeholder(tf.float32, shape=[None,10], name="y_data")  
    #num = tf.placeholder(tf.float32, name="drop")
    batch_prob = tf.placeholder(tf.bool)
    f1 = tf.Variable(tf.random_normal([fil,fil,3,32], stddev=0.01))
    c1 = tf.nn.conv2d(x, f1, strides=[1,1,1,1], padding='SAME')
    n1 = tf.layers.batch_normalization(c1,center=True,  scale=True, training=batch_prob)
    r1 = tf.nn.relu(n1)
    m1 = tf.nn.max_pool(r1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #m1 = tf.nn.dropout(m1, num)

    f2 = tf.Variable(tf.random_normal([fil,fil,32,64], stddev=0.01))
    c2 = tf.nn.conv2d(m1, f2, strides=[1,1,1,1], padding='SAME')
    n2 = tf.layers.batch_normalization(c2, center=True, scale=True, training=batch_prob)
    r2 = tf.nn.relu(n2)
    m2 = tf.nn.max_pool(r2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #m2 = tf.nn.dropout(m2, num)

    f3 = tf.Variable(tf.random_normal([fil,fil,64,128], stddev=0.01))
    c3 = tf.nn.conv2d(m2, f3, strides=[1,1,1,1], padding='SAME')
    n3 = tf.layers.batch_normalization(c3,center=True,  scale=True, training=batch_prob)
    r3 = tf.nn.relu(n3)
    m3 = tf.nn.max_pool(r3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #m3 = tf.nn.dropout(m3, num)

    f4 = tf.Variable(tf.random_normal([fil,fil,128,256], stddev=0.01))
    c4 = tf.nn.conv2d(m3, f4, strides=[1,1,1,1], padding='SAME')
    n4 = tf.layers.batch_normalization(c4, center=True,scale=True, training=batch_prob)
    r4 = tf.nn.relu(n4)
    m4 = tf.nn.max_pool(r4, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
    #m4 = tf.nn.dropout(m4, num)

    f5 = tf.Variable(tf.random_normal([fil,fil,256,512], stddev=0.01))
    c5 = tf.nn.conv2d(m4, f5, strides=[1,1,1,1], padding='SAME')
    n5 = tf.layers.batch_normalization(c5, center=True,scale=True, training=batch_prob)
    r5 = tf.nn.relu(n5)
    m5 = tf.nn.max_pool(r5, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
    #m5 = tf.nn.dropout(m5, num)

    f6 = tf.Variable(tf.random_normal([fil,fil,512,256], stddev=0.01))
    c6 = tf.nn.conv2d(m5, f6, strides=[1,1,1,1], padding='SAME')
    n6 = tf.layers.batch_normalization(c6, center=True,scale=True, training=batch_prob)
    r6 = tf.nn.relu(n6)
    m6 = tf.nn.max_pool(r6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #m6 = tf.nn.dropout(m6, num)

    a4f = tf.reshape(m6,[-1,4*4*256])

    w1 = tf.Variable(tf.random_normal([4*4*256,10], stddev=0.1))
    b1 = tf.Variable(tf.random_normal([10]))
    a4f = tf.layers.batch_normalization(a4f,center=True, scale=True, training=batch_prob)
    #hl1 = tf.matmul(a4f,w1)+b1
    # hl1 = tf.nn.dropout(hl1, num)

    # w2 = tf.Variable(tf.random_normal([64,32], stddev=0.1))
    # b2 = tf.Variable(tf.random_normal([32]))
    # hl2 = tf.matmul(hl1,w2)+b2
    # hl2 = tf.nn.dropout(hl2, num)

    # w3 = tf.Variable(tf.random_normal([32,4], stddev=0.1))
    # b3 = tf.Variable(tf.random_normal([4]))


    # logits = tf.matmul(hl2,w3)+b3
    logits = tf.matmul(a4f,w1)+b1
    logits = tf.identity(logits, "logits")




    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels =y))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train = tf.train.AdamOptimizer(learning_rate).minimize(cost)



    target = tf.nn.softmax(logits)
    target = tf.identity(target, "target")
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(target,1), tf.argmax(y,1)),dtype=tf.float32))

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(batch_size) #배치사이즈
    iterator = dataset.make_initializable_iterator() #함수 호출
    x_epoch, y_epoch = iterator.get_next() #next batch 
    step = 0 

    saver = tf.train.Saver() #save
    st_time = time.time()
    with tf.Session() as sess:
        #haha = tf.summary.FileWriter('/tmp/1', sess.graph) 
        sess.run(tf.global_variables_initializer()) 
 
        for epoch in range(140):
            sess.run(iterator.initializer) #iterator 상태를 처음 초기화하
 
            while True:
                try:
                    batch_x_data, batch_t_data = sess.run([x_epoch, y_epoch])
                    loss_v, _ = sess.run([cost,train], feed_dict={x:batch_x_data,y:batch_t_data,batch_prob:True})
                    step = step + 20
                    
                except tf.errors.OutOfRangeError:
                    step = 0 
                    break

            ex_time = time.time() - st_time
            print(epoch, '|', step, '|', loss_v)
            print('Epoch :', '%04d'%(epoch+1),'loss = ','{:9f}'.format(loss_v),
                  'Train Accuracy = ',sess.run(acc, feed_dict={x:batch_x_data,y:batch_t_data,batch_prob:False}),
                  '\nVal loss = ',sess.run(cost, feed_dict={x:x_test,y:y_test,batch_prob:True}),
                  'Val Accuracy = ',sess.run(acc,feed_dict={x:x_test,y:y_test, batch_prob:False}),
                  'Time = ',ex_time,'sec')
        saver.save(sess, '1024_model') #save
        acc_v = sess.run(acc, feed_dict={x:x_test,y:y_test,batch_prob:False})
        acc_list.append(acc_v)
        print(acc_v)

    


# In[17]:


print(acc_list)
print('10','평균 acc:',sum(acc_list)/5)

