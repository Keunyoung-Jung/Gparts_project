import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import shutil
import pandas as pd

def gen_image(path) :
    generator = ImageDataGenerator(
        rotation_range = 40, #각도
        width_shift_range = 0.1,  #확대 축소
        height_shift_range = 0.1)  #확대 축소
    
    make_num = 4  #생산할 이미지 갯수
    for f in os.listdir(path) :
        for i in os.listdir(os.path.join(path,f)):
            gen2 = []
            gen3 = []
            print(i)
            aa = cv2.imread(os.path.join(path,f,i))
            aa = aa.reshape((1,)+aa.shape)
            a1 = generator.flow(aa)
            for k in range(make_num):
                a3 = a1.next()[0]
                a3 = np.int32(a3)
                gen2.append(a3)
                gen3.append(k)
            for j in range(len(gen2)) :
                filename = i.split('.')[0]
                cv2.imwrite(os.path.join(path,f,filename+'_'+str(j)+'.jpg'),gen2[j])
                print(os.path.join(path,f,filename+'_'+str(j)+'.jpg'))
    
    print('Generating is Finished!!')
    #gen2 에 이미지가 있습니다 순서대로..
    #gen3 는 라벨 이름입니다. (필요시사용)
def img_resize(path):
    count = 0
    for flist in os.listdir(path) :
        count += 1
        for f in os.listdir(os.path.join(path,flist)) :
            imgPath = os.path.join(path,flist,f)
            try :
                print(f)
                parts_img = cv2.imread(imgPath)
                parts_img = cv2.resize(parts_img,(50,50))
                cv2.imwrite(imgPath, parts_img)
            except :
                pass
        print(' ',count-1)
    
def mkdir_parts(path):
    parts_num = ''
    parts_idx = -1
    count = 0
    for f in os.listdir(path) :
        count += 1
        imgPath = os.path.join(path,f)
        if parts_num != f.split('_')[0] :
            print(' ',count-1)
            count = 1
            parts_num = f.split('_')[0]
            parts_idx += 1
            print(parts_num, parts_idx, end='')
            try:
                if not os.path.exists(os.path.join('./dataset_split/',parts_num)) :
                    os.makedirs('./dataset_split/'+parts_num)
            except :
                pass
        try :
            shutil.copy(imgPath,'./dataset_split/'+parts_num+'/'+f)
        except :
            print('----------',parts_num)
            pass
    print(' ',count-1)
    
def move_testset(path):
    #lh = 'low'
    lh = 'high'
    for flist in os.listdir(path) :
        count = 0
        for f in os.listdir(os.path.join(path,flist)) :
            count += 1
            imgPath = os.path.join(path,flist,f)
            if not os.path.exists(os.path.join('./class100_test_'+lh+'/',flist)) :
                    os.makedirs('./class100_test_'+lh+'/'+flist)
            testsetPath = os.path.join('class100_test_'+lh+'/',flist,f)
            shutil.move(imgPath,testsetPath)
            #print(len(f))
            print('moved '+testsetPath)
            if count == 15 :
                break
        print(' ',count-1)
        
def split_high_low(path):
    df = pd.read_csv('품번카운트.csv')
    print(df.head())
    print(df['count'])
    imgdir = 'dataset_split/'
    for i in range(len(df['품번'])) :
        if df['count'][i] >= 50 :
            print(df['품번'][i],'= high')
            shutil.move(imgdir+df['품번'][i],'class100_train_high/'+df['품번'][i])
        
        else :
            print(df['품번'][i],'= low')
            shutil.move(imgdir+df['품번'][i],'class100_train_low/'+df['품번'][i])
        
        
        
path = 'dataset_split/'
#path = './'
#path = 'class100_train_high'
#path = 'class100_train_low'
#img_resize(path)
gen_image(path)
#mkdir_parts(path)
#split_high_low(path)
#move_testset(path)