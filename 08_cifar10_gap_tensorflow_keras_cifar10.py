

from tensorflow.keras.datasets.cifar10 import load_data
(x_train, y_train), (x_test, y_test) = load_data()

# 利用陣列的 shape 屬性可查詢文字圖片的外型屬性, 包含筆數, 解析度, 以及色版數目
  #圖片資料為陣列
  #第 1 列畫素開始 (0) 
  #第 1 列畫素結束 (31)......
  #第 32 列畫素開始 (0)
  #第 32 列畫素結束 (0)

print(x_train.shape) #  #訓練集為 5 萬筆之 32*32 RGB彩色圖片(色版=3)，為3維陣列
print(x_test.shape) # 訓練集標籤為 5 萬筆 0~9 數字
print(y_train.shape) 
print(y_test.shape)

# !!! (50000, 1)：2維/表格 =/= (50000, )：1維/1列

y_train

#第 1 列畫素開始 (0)# 轉換名稱：每個array第一個數字即第[0]值，代表物種種類
trans = [
    "airplane",										
    "automobile",										
    "bird",										
    "cat",										
    "deer",										
    "dog",										
    "frog",										
    "horse",										
    "ship",										
    "truck",
]

import matplotlib.pyplot as plt
idx = 10000            #查看第幾筆資料
print(trans[y_train[idx][0]])    #查看第idx筆資料的答案(物種名稱)
plt.imshow(x_train[idx])       #第idx筆的xtrain資料以視覺化方式呈現

"""**設定模型**
* 強調前重後輕，重視特徵萃取，改用GlobalAveragePooling2D
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D    # 長x寬+通道，選用2D
from tensorflow.keras.layers import Dense, Flatten, Dropout

#設定模型階層內容 #可依model.evaluate結果調整CNN階層層數
    
''' 
    捨去中間過程  #改用GlobalAveragePooling2D
    # MaxPooling2D(),
    # MLP
    # Flatten(),   # 將3維轉為1維 #將input_shape轉為input_dim
    # Dense(256, activation="relu"),
    # Dropout(0.25), # Dense 之間設定Dropout(0.25)，表示每次隨機丟棄(1/4 * 256)的訓練資料，只用剩下3/4去訓練
'''


layers = [
    # CNN
    # 一個過濾器(九宮格): 3 * 3 * 3(前一層通道數) * 64(過濾器數目) + 64(bias) = 1792
    Conv2D(64, 3, padding="same", activation="relu", input_shape=(32, 32, 3)),   # 卷積：萃取圖片特徵，改變通道數(變大)
    MaxPooling2D(),   # 可刪掉，但會增加計算量及計算時間   # 池化：調整圖片尺寸(變小)_長/寬各縮減一半
    # 一個過濾器: 3 * 3 * 64(前一層通道數) * 128(過濾器數目) + 128(bias) = 73856
    Conv2D(128, 3, padding="same", activation="relu"),  # activation：啟動函數
    MaxPooling2D(),   # 縮小圖像尺寸，減低增加計算量及計算時間
    Conv2D(256, 3, padding="same", activation="relu"),
    MaxPooling2D(),
    Conv2D(512, 3, padding="same", activation="relu"),
    GlobalAveragePooling2D(),
    Dense(10, activation="softmax")
]

# 原本資料量為 524544+2570，調整後剩5130

model = Sequential(layers)  #建立模型
model.summary(line_length=100) #模型摘要

"""**學習過程配置**
SparseCategoricalCrossentropy(請參考[說明](https://ithelp.ithome.com.tw/articles/10271081?sc=iThelpR))
"""

# 一個輸出(二元分類): BinaryCrossEntropy p log 1/q + (1 - p) log 1/1-q
# 多個輸出(多元分類): CategoricalCrossEntropy pi log1/qi
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(loss=SparseCategoricalCrossentropy(),
              # "adam"也可以
              optimizer="adam",
              metrics=["accuracy"])

x_train_norm = x_train / 255
x_test_norm = x_test / 255

# batch_size: 看多少筆, 做一次梯度下降(幾10~幾100)
# epochs: 所有資料看幾輪(負責結束訓練)
# batch_size=200
# 一epochs: 54000 / 200 = 270(次梯度下降)
# verbose=0(quiet) 1(default) 2(no bar)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("cnn.h5", save_best_only=True)
]
model.fit(x_train_norm,
          y_train,
          batch_size=200,
          epochs=100,
          validation_split=0.1,
          verbose=2,
          callbacks=callbacks)

model.evaluate(x_test_norm, y_test)

pre = model.predict(x_test_norm).argmax(axis=1)
pre

# (10000, 1)和(10000,)是兩回事
y_test_1d = y_test.reshape(-1)

# keras: y_test_cat sklearn: y_test
import pandas as pd
from sklearn.metrics import confusion_matrix
pre = model.predict(x_test_norm).argmax(axis=1)
mat = confusion_matrix(y_test_1d, pre)
pd.DataFrame(mat,
             index=["{}(正確)".format(trans[i]) for i in range(10)],
             columns=["{}(預測)".format(trans[i]) for i in range(10)])

import numpy as np
# 找出True(預測錯誤)的位置
idx = np.nonzero(pre != y_test_1d)[0]
idx = idx[:200]
pre_false_label = y_test_1d[idx]
pre_false_pre = pre[idx]
pre_false_img = x_test[idx]

plt.figure(figsize=(15, 45))
width = 10
height = len(idx) // width + 1
for i in range(len(idx)):
    plt.subplot(height, width, i+1)
    t = "[O]:{}\n[P]:{}".format(trans[pre_false_label[i]], trans[pre_false_pre[i]])
    plt.title(t)
    plt.axis("off")
    plt.imshow(pre_false_img[i])

import requests
# pillow
from PIL import Image
url = input("url:")
response = requests.get(url, stream=True)
img = Image.open(response.raw).convert("RGB").resize((32, 32))
img_np = np.array(img) / 255
# (32, 32, 3) -> (1, 32, 32, 3)
img_np = img_np.reshape(1, 32, 32, 3)
proba = model.predict(img_np)[0]
for p, n in zip(proba, trans):
    print(n, "的機率:", round(p, 3))
plt.imshow(img)

from google.colab import drive
drive.mount('/content/drive')