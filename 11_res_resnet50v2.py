
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.utils import plot_model
model = ResNet50V2(include_top=False, input_shape=(224, 224, 3))
plot_model(model, to_file="res.png", show_shapes=True)

"""解析模組某區塊(兩循環圈)"""

# 通道數有增加的圈
# 短路線: 1 x 1卷積(256)
w = model.get_layer("conv2_block1_0_conv").get_weights()
print(w[0].shape)
# 殘差線: 1 x 1卷積(64) -> 3 x 3卷積(64) -> 1 x 1卷積(256)
w = model.get_layer("conv2_block1_1_conv").get_weights()
print(w[0].shape)
w = model.get_layer("conv2_block1_2_conv").get_weights()
print(w[0].shape)
w = model.get_layer("conv2_block1_3_conv").get_weights()
print(w[0].shape)

# 沒有改變通道數的圈圈
# 殘差線: 1x1卷積(64) -> 3x3卷積(64) -> 1x1卷積(256)
w = model.get_layer("conv2_block2_1_conv").get_weights()
print(w[0].shape)
w = model.get_layer("conv2_block2_2_conv").get_weights()
print(w[0].shape)
w = model.get_layer("conv2_block2_3_conv").get_weights()
print(w[0].shape)

"""模仿"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, Add

i = Input(shape=(56, 56, 64))
# 短路
x1 = Conv2D(256, 1, padding="same")(i)
# 殘差
x2 = Conv2D(64, 1, padding="same")(i)
x2 = BatchNormalization()(x2)
x2 = Activation("relu")(x2)
x2 = Conv2D(64, 3, padding="same")(x2)
x2 = BatchNormalization()(x2)
x2 = Activation("relu")(x2)
x2 = Conv2D(256, 1, padding="same")(x2)
# ++
o = Add()([x1, x2])
p = Model(inputs=i, outputs=o)
plot_model(p, show_shapes=True)

i = Input(shape=(56, 56, 256))

# 殘差
x2 = Conv2D(64, 1, padding="same")(i)
x2 = BatchNormalization()(x2)
x2 = Activation("relu")(x2)
x2 = Conv2D(64, 3, padding="same")(x2)
x2 = BatchNormalization()(x2)
x2 = Activation("relu")(x2)
x2 = Conv2D(256, 1, padding="same")(x2)
# ++
o = Add()([i, x2])
p = Model(inputs=i, outputs=o)
plot_model(p, show_shapes=True)