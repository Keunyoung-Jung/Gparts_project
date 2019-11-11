import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
import random
import cv2
import time
from tqdm import tqdm

def mkdir_parts(path):
    parts_num = ''
    parts_idx = -1
    count = 0
    for f in os.listdir(path) :
        count += 1
        #if parts_idx == 70 :
        #    print(' ',count-1)
        #    break
        imgPath = os.path.join(path,f)
        if parts_num != f.split('_')[0] :
            print(' ',count-1)
            count = 1
            parts_num = f.split('_')[0]
            parts_idx += 1
            print(parts_num, parts_idx, end='')
            try:
                if not os.path.exists(os.path.join('./dataset2/',parts_num)) :
                    os.makedirs('./dataset2/'+parts_num)
            except :
                pass
        try :
            parts_img = cv2.imread(imgPath)
            cv2.imwrite('./dataset2/'+parts_num+'/'+f, parts_img)
        except :
            pass
    print(' ',count-1)
def onehot(test_DIR):
    parts_class = []
    test_folder_list = os.listdir(test_DIR)

    X = []
    Y = []
    for index in tqdm(range(len(test_folder_list))):
        path = os.path.join(test_DIR, test_folder_list[index])
        path += '/'
        img_list = os.listdir(path)
        parts_class.append(test_folder_list[index])
        
        for img in img_list:
            img_path = os.path.join(path, img)
            #print(img)
            try:
                #print('#')
                img_read = cv2.imread(img_path)
                img_read = cv2.resize(img_read , (50,50))
                #cv2.imshow(img,img_read)
                #cv2.waitKey()
                X.append(img_read)
                Y.append(index)
            except:
                pass
        
    tmpx = np.array(X)
    
    Y = np.array([[i] for i in Y])
    enc = OneHotEncoder()
    enc.fit(Y)
    tmpy = enc.transform(Y).toarray()
    print('')
    print(tmpx.shape, tmpy.shape, len(parts_class))
    return tmpx , tmpy , parts_class

st_time = time.time()
#dirpath = './dataset_split'
train_dirpath = './class30_train/'
test_dirpath = './class30_test'
#mkdir_parts(dirpath)
X_train, Y_train , parts_class = onehot(train_dirpath)
X_test, Y_test , parts_class = onehot(test_dirpath)
#X_train, X_test, Y_train, Y_test = train_test_split(tmpx,tmpy,random_state = 0)
print(int(st_time))
print(X_train.shape,Y_train.shape, X_test.shape, Y_test.shape)

#이미지 shape 640,480,3
learning_rate = 0.0001
training_epochs = 30
batch_size = 64

X = tf.placeholder(tf.float32, [None, 50, 50, 3], name='input')
Y = tf.placeholder(tf.float32, [None, 30], name='output')
keep_prob = tf.placeholder(tf.float32, name = 'dropout')

W0 = tf.Variable(tf.random_normal([5, 5, 3, 16], stddev=0.01),name='w0')
L0 = tf.nn.conv2d(X, W0, strides=[1, 1, 1, 1], padding='SAME')
#640x480x32 , 320x240x32 , 160x120x32 ,50x50x16
L0 = tf.nn.relu(L0)
L0 = tf.nn.max_pool(L0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#320x240x32 , 160x120x32 , 80x60x32 , 25x25x16


W1 = tf.Variable(tf.random_normal([5, 5, 16, 32], stddev=0.01),name='w1')
L1 = tf.nn.conv2d(L0, W1, strides=[1, 1, 1, 1], padding='SAME')
#640x480x32 , 320x240x32 , 160x120x32 ,25x25x32 
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#320x240x32 , 160x120x32 , 80x60x32 , 13x13x32 

W2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01),name='w2')
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
#320x240x64 , 160x120x64 , 80x60x64 , 13x13x64
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#160x120x64 , 80x60x64 , 40x30x64 , 7x7x64

W3 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev =0.01),name='w3')
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
#160x120x128 , 80x60x128 , 40x30x128 , 7x7x128
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#80x60x128 , 40x30x128 , 20x15x128 , 4x4x128
L3 = tf.nn.dropout(L3, keep_prob)

L3_flat = tf.reshape(L3,[-1, 4 * 4 * 128])
# 최종 출력값 L3 에서의 출력 256개를 입력값으로 받아서 0~29 레이블인 30개의 출력값을 만듭니다.
W4 = tf.Variable(tf.random_normal([4 * 4 * 128, 30], stddev=0.01),name='w4')
hypothesis = tf.matmul(L3_flat, W4)
hypothesis = tf.identity(hypothesis,'hypothesis')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, 
                                                              labels = Y))    #costfunction

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

target1 = tf.nn.softmax(hypothesis)

saver = tf.train.Saver()
ing_time = st_time
chk_time = 0
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs) :
        avg_cost = 0
        total_batch = int(len(X_train)/ batch_size)
        
        for  i in range(total_batch) :
            #batch_xs, batch_ys = 
            #batch_xs = batch_xs.reshape(-1,28,28,1)
            feed_dict = {X:X_train, Y:Y_train , keep_prob: 1}
            cost_val, _ = sess.run([cost,optimizer],feed_dict=feed_dict)
            avg_cost += cost_val / total_batch
            
        correct_prediction2 = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
        accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2,tf.float32))
        ex_time2 = time.time() - st_time - chk_time
        print('Epoch :', '%04d'%(epoch+1),'cost = ','{:9f}'.format(avg_cost),
              'Accuracy = ',sess.run(accuracy2,feed_dict={X:X_test,Y:Y_test, keep_prob:1}),
              'Time = ',ex_time2,'sec')
        
        #chk_time = ex_time2
    print('Learning Finished!!')
    
    saver.save(sess,'savedmodel/class30_test_cnn_model.ckpt', global_step = 1000)
    print('model saved!!')
    correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    accuracy2 = tf.cast(correct_prediction,tf.float32)
    
    ex_time = time.time() - st_time
    
    print('Excute Time : ', int(ex_time) ,'sec')
    print('Total Accuracy : ', sess.run(accuracy, feed_dict={X:X_test,
                                                       Y:Y_test, keep_prob:1}))
    
    r = random.randint(0,100)
    parts_idx = sess.run(tf.argmax(Y_test[r:r+1],1))[0]
    parts_pre_idx = sess.run(tf.argmax(hypothesis,1),
                             feed_dict={X:X_test[r:r+1],
                                        keep_prob: 1 })[0]
    print('Label : ',sess.run(tf.argmax(Y_test[r:r+1],1)))
    print('Parts_Label :',parts_class[parts_idx])
    #print('Parts_class : ', parts_class)
    print('Prediction : ',sess.run(tf.argmax(hypothesis,1),
                                   feed_dict={X:X_test[r:r+1],
                                              keep_prob: 1}))
    print('Parts_prediction : ',parts_class[parts_pre_idx])
    
    #print('Prediction : ',sess.run(target1,
    #                               feed_dict={X:X_test[r:r+1],
    #                                          keep_prob: 1}))

plt.imshow(X_test[r:r+1].reshape(50,50,3),cmap = 'Greys',
           interpolation='nearest')
plt.show()
