
import tensorflow as tf

# tf.keras.utils.get_file：從某個URL下載資料, fname：文件名稱，origin：文件的URL，extract=True提取壓縮檔內容(執行解壓縮)
dataset = tf.keras.utils.get_file(
    fname="aclImdb.tar.gz", 
    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
    extract=True,
)

# 查看資料路徑
dataset

import os
import glob
import pandas as pd

# 定義資料來源
def getdata(mid):
    dn = os.path.dirname(dataset)
    posfn = glob.glob(os.path.join(dn, "aclImdb", mid, "pos", "*")) # 資料夾路徑,dn/root/.keras，之後每個逗號後面就是下層資料夾
    negfn = glob.glob(os.path.join(dn, "aclImdb", mid, "neg", "*"))
    contents = []
    for fn in posfn + negfn:
        with open(fn, encoding="utf-8") as f:
            contents.append(f.read())
    df = pd.DataFrame({
        "content":contents,
        "sentiment":[1] * len(posfn) + [0] * len(negfn)
    })
    return df

train_df = getdata("train")
test_df = getdata("test")

# test_df
train_df # 有25000篇文章 #sentiment標註正/負評

# Tokenize: 把你的詞變成數字
from tensorflow.keras.preprocessing.text import Tokenizer
# tok = Tokenizer()
tok = Tokenizer(num_words=3000) # 針對出現頻率最高的3000個字
tok.fit_on_texts(train_df["content"])

# 每個字對應的編號
# tok.word_index
# tok.index_word

# Sequence: 化成數字的序列
x_train_seq = tok.texts_to_sequences(train_df["content"])
x_test_seq = tok.texts_to_sequences(test_df["content"])

#每篇評論的字數
x_train_seq
pd.DataFrame(x_train_seq)

# 資料處理：讓每篇評論字數一致_直接寫死字數長度
# Padding: 截長補短變成一樣長，pad_sequences函式說明https://keras.io/zh/preprocessing/sequence
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# x_train_pad = pad_sequences(x_train_seq, maxlen=512) # maxlen 自訂每篇文章要留幾個數字，預設從前面(文章起始)截長補短，可改
# x_test_pad = pad_sequences(x_test_seq, maxlen=512)
# pd.DataFrame(x_train_pad)

INPUT_LENGTH = 512
INPUT_DIM = 3000 #1~3000(未包含0)
OUTPUT_DIM = 128

# 資料處理：讓每篇評論字數一致_利用變數動態調整字數長度(配合模型調整)
# pd.DataFrame(x_train_seq)
# Padding: 截長補短變成一樣長
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train_pad = pad_sequences(x_train_seq, maxlen=INPUT_LENGTH)
x_test_pad = pad_sequences(x_test_seq, maxlen=INPUT_LENGTH)
pd.DataFrame(x_train_pad)

"""建立模型"""

# 建立模型--舊方法，缺點：係數過大，容易過擬合
# 係數 16777472 = (65536 + 1)*256
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout


layers = [
    # 3001(種詞) * 128(個情緒/語意/係數)   # INPUT_DIM+1，因為0也是一種 ，mask_zero=True：遇到0也要做情緒判斷
    Embedding(INPUT_DIM+1, OUTPUT_DIM, mask_zero=True, input_length=INPUT_LENGTH),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.25),
    Dense(2, activation="softmax")
]
model = Sequential(layers)
model.summary()

# 建立模型--現行常用方法
# Embedding把每個詞作情緒轉換(做多少情緒係數轉換)：https://keras.io/zh/layers/embeddings/
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
layers = [
    # 3001(種詞) * 128(個情緒/語意/係數)
    Embedding(INPUT_DIM+1, OUTPUT_DIM, mask_zero=True, input_length=INPUT_LENGTH),
    GlobalAveragePooling1D(),
    Dense(2, activation="softmax")
]
model = Sequential(layers)
model.summary()

"""模型訓練"""

# 一個輸出(二元分類): BinaryCrossEntropy p log 1/q + (1 - p) log 1/1-q
# 多個輸出(多元分類): CategoricalCrossEntropy pi log1/qi
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(loss=SparseCategoricalCrossentropy(),
              # "adam"也可以
              optimizer="adam",
              metrics=["accuracy"])

import numpy as np
y_train = np.array(train_df["sentiment"])
y_test = np.array(test_df["sentiment"])

# batch_size: 看多少筆, 做一次梯度下降(幾10~幾100)
# epochs: 所有資料看幾輪(負責結束訓練)
# batch_size=200
# 一epochs: 54000 / 200 = 270(次梯度下降)
# verbose=0(quiet) 1(default) 2(no bar)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("imdb.h5", save_best_only=True)
]
model.fit(x_train_pad,
          y_train,
          batch_size=200,
          epochs=100,
          validation_split=0.1,
          verbose=2,
          callbacks=callbacks)

model.evaluate(x_test_pad, y_test)

# model.layers[1:]
# layers[0].get_weights()

''' https://keras.io/zh/layers/about-keras-layers/
•	layer.get_weights(): 以含有Numpy矩陣的列表形式返回層的權重。
•	layer.set_weights(weights): 從含有Numpy矩陣的列表中設置層的權重（與get_weights的輸出形狀相同）。
'''

# 讓INPUT變成不限資料長度
l = [
    Embedding(INPUT_DIM+1, OUTPUT_DIM, mask_zero=True) #新設第一層：不限字數
]
remain = model.layers[1:] #原模型的第2層以後
model_use = Sequential(l+remain) #建立模型：第一層用l，第二層以後用remain

model_use.layers[0].set_weights(model.layers[0].get_weights()) # 使用新模型第一層[0]重設權重為原模型的第一層(直接套用權重就不用訓練了)
model_use.summary()

review = input("影評:")
review_seq = tok.texts_to_sequences([review])
proba = model_use.predict(review_seq)[0]
trans = ["neg", "pos"]
for p, sentiment in zip(proba, trans):
    print(sentiment, ":", p)
