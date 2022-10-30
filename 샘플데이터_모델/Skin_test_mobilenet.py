#!/usr/bin/env python
# coding: utf-8

# In[2]:

#각종 라이브러리 호출 부분#
from PIL import Image #이미지 불러오는 등 이미지 처리에 필요한 라이브러리
from tensorflow.keras.utils import *
from tensorflow.keras.layers import * #딥러닝 모델의 layer를 사용하기 위한 라이브러리
from tensorflow.keras.models import * #딥러닝 모델 구성을 위한 라이브러리
from tensorflow.keras.callbacks import * #체크포인트 저장 등에 필요한 라이브러리
from tensorflow.keras.optimizers import * #딥러닝 모델의 optimizer 함수 호출을 위한 라이브러리
from tensorflow.keras.applications import * #SOTA 딥러닝 모델 호출을 위한 라이브러리
from sklearn.model_selection import train_test_split #trian 데이터셋, test 데이터셋 구분시 사용하는 라이브러리
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix #딥러닝 모델의 검증에 필요한 라이브러리

import datetime
import numpy as np 
import pandas as pd
import tensorflow as tf


# In[129]:


df = pd.read_excel('C:/Users/swer5/PycharmProjects/Skin_study/Skin_test_class.xlsx', engine='openpyxl') #데이터의 경로, 클래스 등이 저장되어있는 엑셀 파일 호출


# In[188]:


df = df.sample(frac=1).reset_index(drop=True) #데이터 전체 셔플링

print(df)

# In[189]:


x = [] #x라는 빈 리스트를 생성

for i in range(len(df)): #앞서 호출한 df 전체 개수 만큼 반복을 하면서
    img = Image.open(df['dir'][i]).convert('RGB') #df['dir'] 이라는 컬럼에 있는 행을 순서대로 읽으면서 이미지를 RGB 값으로 연다.
    img = img.resize((224,224)) # 불러온 이미지를 가로 224, 세로 224 크기로 조정한다.
    img = np.asarray(img).astype(np.float32) / 255.0 # 이미지의 각 픽셀값을 0~1 사이의 숫자로 변환
    x.append(img) # x라는 빈 리스트에 이미지를 담는다.

x = np.array(x) #리스트 x를 array 값으로 변환


# In[190]:


y = df['class'].to_numpy() #df['class']의 행을 array값으로 변환


# In[191]:


y = np.asarray(y).astype(np.float32) #float32의 형태로 변환  / <lee> : as array는 입력된 데이터를 ndarray(n차원 배열객체)형식으로 만듬


# In[192]:


y = to_categorical(y) #categorical 형태로 변형 (categorical의 경우 분류하고자 하는 클래스가 멀티 클래스인 것을 의미함)

##딥러닝에서 x는 인풋 y는 예측하고자 하는 분류 쿨래스를 주로 의미함#
# In[193]:


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1) #학습용 이미지, 검증용 이미지, 학습용 라벨값, 검증용 라벨값을 전체 데이터셋에서 9:1 비율로 나눔)
# <lee> : 오버피팅문제 방지를 위함// test_size = 테스트 데이터셋의 비율 10%

# In[194]:


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1) #학습용 이미지, 테스트용 이미지, 학습용 라벨값, 태스트용 라벨값을 앞서 나눈 학습용 데이터셋에서 9:1 비율로 나눔)


# In[195]:


print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

#각 데이터의 형태(ex. 아마 (데이터 개수, 가로길이, 세로길이, 3) 이 형태로 나올 것임
# In[138]:


base_model = VGG16(input_shape=x_train.shape[1:],
                      include_top=False,
                      weights='imagenet')

# (베이스가 되는 백본 모델은 MobileNet 사용하고, 해당 모델의 인풋 형태는 x_train의 형태의 2번째꺼부터 끝까지 사용하겠다. 파이썬의 경우 인덱싱을 0부터 시작함 따라서 1은 실제 데이터에서 2번째에 위치한 것)
#x_train.shape을 찍어보았을 때 (데이터 개수, 가로길이, 세로길이, 3)이 나올텐데 [1:]을 붙일경우 (가로길이, 세로길이, 3) 이렇게 인풋이 되어 들어감
# include_top = True를 했을 경우에는 fully connected layer의 구조를 변경할 수 가 없음. 일반적인 분류 SOTA 모델의 경우 이미지넷 데이터를 활용했으므로 1,000개의 예측 클래스를 가지고 있음
# 따라서 우리는 우리 데이터에 맞는 클래스를 예측해야하므로 Fully connected layer의 구조를 변경해야 하므로 include_top=False를 사용
# weights의 경우 본 BackBone 모델이 사용한 Imagenet 데이터를 통해 학습된 이미지 가중치를 사용하여 학습

global_average_layer = GlobalAveragePooling2D() #Flatten Layer를 사용할 경우 학습 파라미터의 수가 많이 늘어나므로 Global Average Pooling을 활용하여 Fully connected layer 구성 / 해당 부분은 별도로 딥러닝 모델의 구조에 대해서 따로 공부하셔야함

dropout_layer = Dropout(0.3) #Dropout의 경우 overfitting을 방지하기 위해 사용함. 필수값은 아님

pred_layer = Dense(8, activation='softmax') #최종 fully connected layer에 해당하며 예측하고자 하는 클래스 개수와 활성화 함수 softmax를 사용 / 활성화 함수에 대해서도 별도 공부필요


# In[139]:


model = Sequential([
    base_model,
    global_average_layer,
    dropout_layer,
    pred_layer
])

# 앞서 정의한 전체 모델의 레이어를 Sequential Layer를 사용하여 한층한층 쌓음

# In[140]:


cp_path = 'C:/Users/swer5/PycharmProjects/Skin_study/body/weights/body.h5' #학습한 모델의 가중치를 저장할 경로와 파일명


# In[141]:


cp = ModelCheckpoint(cp_path, monitor='val_acc', mode='max', verbose=1, save_best_only=True, save_weight_only=False)
# 중간중간 가중치를 저장하는 콜백 함수 / monitor의 값을 mode에 해당할 때 저장 / 우리는 추후에 tflite로 변경을 해야하기때문에 모델의 전체 구조도 함께 저장해야하므로 save _weights_only를 False
es = EarlyStopping(patience=15, monitor='val_acc', mode='max')
# 학습이 진행되는 과정에서 monitor의 값이 patience 수치만큼 변동이 없을 경우 학습을 미리 종료시키는 콜백 함수


# In[142]:


model.compile(loss='categorical_crossentropy',
             optimizer=Adam(learning_rate=0.0001, decay=0.01),
             metrics=['acc'])
# 학습 전 모델의 loss 함수, optimizer, metrics를 정의해주는 부분
# loss 함수, optimizer의 경우 다양한 종류가 있고 상황에 따라 사용하는 함수가 다르므로 별도 공부 필요

# In[143]:


model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=24, callbacks=[cp, es])
# 모델의 학습을 진행하는 부분 (x_train, y_train) 이 학습용 데이터를 batch_size 만큼 묶어서 총 50번간 학습을 하고 , 검증의 경우 (x_val, y_val)을 이용해 검증함. callbacks의 경우 앞서 정의한 각 콜백함수

# In[196]:


model.load_weights(cp_path)
# 학습이 끝난 후 가중치가 저장된 경로에서 가중치를 불러옴

# In[197]:


y_pred = model.predict(x_test)
#불러온 가중치를 토대로 x_test를 예측

# In[198]:


y_pred = np.argmax(y_pred, axis=-1)
# 처음 y_pred는 총 8개의 클래스별로 개별 확률 값을 가지고 나옴. np.argmax를 통해 가장 높은 확률 값을 가진 인덱스를 반환 (결국 가장 높은 확률은 가진 인덱스가 예측한 클래스가 됨)

# In[199]:


y_test = np.argmax(y_test, axis=-1)

# In[200]:


with open('C:/Users/swer5/PycharmProjects/Skin_study/result.txt', 'w') as f:
    f.write('F1_score : {}'.format(np.round(f1_score(y_test, y_pred, average='micro')*100, 2)))
    f.write('\n')
    f.write('체크 일자 : {}'.format(datetime.datetime.now()))


# In[201]:


model = load_model(cp_path)
# 모델 구조를 불러오기

# In[203]:


# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
#
# with open('C:/Users/swer5/PycharmProjects/Skin_study/body/tflite/mobilenet/mobilenet.tflite', 'wb') as f:
#     f.write(tflite_model)
# #어플에 임베딩 하기 위해 tflite로 변환하는 과정
#
# # In[ ]:




