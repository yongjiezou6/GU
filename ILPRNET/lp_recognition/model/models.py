import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
from lp_recognition.parameter import *
from keras import backend as K
from keras.layers import Conv2D, Conv2DTranspose, Add, Concatenate, MaxPooling2D, Dropout, Input, GlobalMaxPooling2D, Reshape, Multiply, TimeDistributed, Lambda,Softmax
from keras.layers import Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model

def conv_block(input_tensor, filters, kernel_size, strides=1, padding='same', pooling=False, dropout=False):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    if pooling:
        x = MaxPooling2D()(x)
    if dropout:
        x = Dropout(0.2)(x)
    return x

def deconv_block(input_tensor, filters, kernel_size, padding, strides=1, activation='relu'):
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    if activation == 'relu':
        x = LeakyReLU()(x)
    elif activation == 'sigmoid':
        x = Activation('sigmoid')(x)
    return x


def union_location_block(input_feature, input_shape):
    in_W, in_H, in_C = input_shape  # 38, 14, 128
    # Merge
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(input_feature)
    add = Add()([avg_pool, max_pool]) ######
    # Encode & Decode
    x_Concat = Concatenate(axis=-1)([avg_pool, max_pool,add])  # 38, 14, 2#######
    x = conv_block(x_Concat, 8, (2, 2), strides=2, padding='valid')  # 19, 7, 8 #8
    x_a = conv_block(x, 16, (5, 5), padding='valid')  # 15, 3, 12  #16
    x_b = conv_block(x_a, 24, (5, 3), padding='valid')  # 11, 1, 24 #24
    x = conv_block(x_b, 256, (11, 1), padding='valid')  # 1, 1, 256 #256
    x = deconv_block(x, 24, (11, 1), padding='valid')  # 11, 1, 24#24
    x = Concatenate(axis=-1)([x, x_b])
    x = deconv_block(x, 16, (5, 3), padding='valid')  # 15, 3, 24 #16
    x = Concatenate(axis=-1)([x, x_a])
    x = deconv_block(x, 8, (5, 5), padding='valid')  # 19, 7, 6#8
    x = deconv_block(x, 7, (2, 2), strides=2, padding='valid', activation='sigmoid')  # 38, 14, 7


    def routing(x, position):
        x = Lambda(lambda x: x[:, :, :, position])(x)
        x = Reshape((in_W, in_H, 1))(x)
        return x

    x1 = Multiply()([input_feature, routing(x, 0)])
    x2 = Multiply()([input_feature, routing(x, 1)])
    x3 = Multiply()([input_feature, routing(x, 2)])
    x4 = Multiply()([input_feature, routing(x, 3)])
    x5 = Multiply()([input_feature, routing(x, 4)])
    x6 = Multiply()([input_feature, routing(x, 5)])
    x7 = Multiply()([input_feature, routing(x, 6)])

    return [x1, x2, x3, x4, x5, x6, x7, x]


def prov_classifier(x, name):
    x = GlobalMaxPooling2D()(x)
    x = Softmax(axis=-1, name=name)(x)
    return x

def lett_classifier(x, position, name):
    x = Lambda(lambda x: x[:, position, :, :, :])(x)
    x = GlobalMaxPooling2D()(x)
    x = Softmax(axis=-1, name=name)(x)
    return x


def ILPRnet(input_shape, debug=False):
    W, H ,C = input_shape
    input_img = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 256, 64, 1)

    x = conv_block(input_img, 32, (5, 5), pooling=True, dropout=True)  # 32
    x = conv_block(x, 64, (3, 3), pooling=True, dropout=True)
    x2 = conv_block(x, 128, (3, 3), pooling=False, dropout=True)  # 128
    x3 = conv_block(x2, 128, (3, 3), pooling=False, dropout=True)  # 128
    x = Add()([x2, x3])  # 18 8
    x = conv_block(x, 64, (3, 3), pooling=False, dropout=True)  # 18 8
    x = union_location_block(x, input_shape=(W // 4, H // 4, 64))  # 38,14#128

    x1 = x[0]
    x2 = Lambda(lambda x: K.expand_dims(x, axis=1))(x[1])
    x3 = Lambda(lambda x: K.expand_dims(x, axis=1))(x[2])
    x4 = Lambda(lambda x: K.expand_dims(x, axis=1))(x[3])
    x5 = Lambda(lambda x: K.expand_dims(x, axis=1))(x[4])
    x6 = Lambda(lambda x: K.expand_dims(x, axis=1))(x[5])
    x7 = Lambda(lambda x: K.expand_dims(x, axis=1))(x[6])
    locations = x[7]

    x_prov = Conv2D(128, (3, 3), padding='same', use_bias=False)(x1)
    x_prov = BatchNormalization()(x_prov)
    x_prov = LeakyReLU()(x_prov)
    x_prov = Conv2D(province_num, (1, 1), padding='same')(x_prov)

    x_lett = Concatenate(axis=1)([x2, x3, x4, x5, x6, x7])
    x_lett = TimeDistributed(Conv2D(128, (3, 3), padding='same', use_bias=False))(x_lett)
    x_lett = BatchNormalization()(x_lett)
    x_lett = LeakyReLU()(x_lett)
    x_lett = TimeDistributed(Conv2D(letter_num, (1, 1), padding='same'))(x_lett)

    x1 = prov_classifier(x_prov, name='x1')
    x2 = lett_classifier(x_lett, 0, name='x2')
    x3 = lett_classifier(x_lett, 1, name='x3')
    x4 = lett_classifier(x_lett, 2, name='x4')
    x5 = lett_classifier(x_lett, 3, name='x5')
    x6 = lett_classifier(x_lett, 4, name='x6')
    x7 = lett_classifier(x_lett, 5, name='x7')

    if debug:
            outputs = [x1, x2, x3, x4, x5, x6, x7, locations]
    else:
        outputs = [x1, x2, x3, x4, x5, x6, x7]

    return Model(input_img, outputs)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu
    model = ILPnet((152, 56, 1))