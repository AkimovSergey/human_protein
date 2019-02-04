
from keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization, GRU
from keras.layers import Conv2D, LSTM, Input, Lambda, Concatenate, TimeDistributed, Bidirectional
from keras.layers import MaxPooling2D, UpSampling2D, DepthwiseConv2D, LeakyReLU, Softmax, AveragePooling2D
from keras.optimizers import *

from utils import *
from keras.models import Model
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy

input_shape = (KERNEL_SZ, KERNEL_SZ, 64)

def lstm_block(img_2_layers):
    inp_1 = Lambda(lambda x: x[:, :, :, 0])(img_2_layers)
    out_1 = LSTM(SZ, activation='softsign')(inp_1)
    inp_2 = Lambda(lambda x: x[:, :, :, 1])(img_2_layers)
    out_2 = LSTM(SZ, activation='softsign')(inp_2)
    return out_1, out_2

def conv_block_small(input, layers, pooling=True):
    conv1 = Conv2D(layers, (1, 1), activation='lrelu', padding='same')(input)
    conv1 = Conv2D(layers, (1, 1), activation='lrelu', padding='same')(conv1)
    conv1 = Conv2D(layers, (1, 1), activation='lrelu', padding='same')(conv1)
    '''conv2 = Conv2D(layers, (2, 2), activation='relu', padding='same')(input)
    conv2 = Conv2D(layers, (2, 2), activation='lrelu', padding='same')(conv2)'''
    conv3 = Conv2D(layers, (3, 3), activation='lrelu', padding='same')(input)
    #conv3 = Conv2D(layers, (5, 5), activation='tanh', padding='same')(input)
    conv = Concatenate()([conv1, conv3])
    if pooling:
        mx = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)
        return BatchNormalization()(mx)
    else:
        return conv

def conv_block(img):
    #lstm_1, lstm_2 = lstm_block(img)
    conv1 = Conv2D(16, (1, 1), activation='tanh', padding='same')(img)
    conv2 = Conv2D(16, (3, 3), activation='tanh', padding='same')(img)
    conv3 = Conv2D(16, (5, 5), activation='tanh', padding='same')(img)
    conv = Concatenate()([conv1, conv2, conv3])
    mx = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)
    bn = BatchNormalization()(mx)
    conv1 = Conv2D(32, (1, 1), activation='tanh', padding='same')(bn)
    conv2 = Conv2D(32, (3, 3), activation='tanh', padding='same')(bn)
    conv3 = Conv2D(32, (5, 5), activation='tanh', padding='same')(bn)
    conv = Concatenate()([conv1, conv2, conv3])
    mx = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)
    bn = BatchNormalization()(mx)
    conv = Conv2D(64, (3, 3), activation='tanh', padding='same')(bn)
    conv = Conv2D(64, (3, 3), activation='tanh', padding='same')(conv)
    mx = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)
    bn = BatchNormalization()(mx)
    conv = Conv2D(128, (3, 3), activation='tanh', padding='same')(bn)
    conv = Conv2D(128, (3, 3), activation='tanh', padding='same')(conv)
    mx = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)
    bn = BatchNormalization()(mx)
    conv = Conv2D(256, (3, 3), activation='tanh', padding='same')(bn)
    conv = Conv2D(256, (3, 3), activation='tanh', padding='same')(conv)


    ft = Flatten()(conv)
    dp = Dropout(0.3)(ft)
    #ctk = Concatenate()([dp, lstm_1, lstm_2])
    dense = Dense(2048, activation='tanh')(dp)
    dense = Dense(2048, activation='tanh')(dense)
    return dense

def model_conv_4_colors():
    inp = Input(shape=input_shape)
    cnv_branch1 = conv_block(Lambda(lambda x: x[:, :, :, 0:1])(inp))
    cnv_branch2 = conv_block(Lambda(lambda x: x[:, :, :, 1:2])(inp))
    cnv_branch3 = conv_block(Lambda(lambda x: x[:, :, :, 2:3])(inp))
    out = Concatenate()([cnv_branch1, cnv_branch2, cnv_branch3])
    dp = Dense(4096, activation='relu')(out)
    dp = Dense(4096, activation='relu')(dp)
    dp = Dense(28, activation=LeakyReLU(alpha=0.1))(dp)
    model = Model(inputs=inp, outputs=dp)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
                  metrics=[categorical_crossentropy, categorical_accuracy, f1])
    model.summary()
    return model


def conv_net():
    inp = Input(shape=input_shape)
    out = conv_block_small(inp, 256)
    out = conv_block_small(out, 512)
    out = conv_block_small(out, 1024)
    '''out = conv_block_small(out, 256)
    out = conv_block_small(out, 512)'''
    '''cnv_branch1 = conv_block(Lambda(lambda x: x[:, :, :, 0:2])(inp))
    cnv_branch2 = conv_block(Lambda(lambda x: x[:, :, :, 2:4])(inp))
    out = Concatenate()([cnv_branch1, cnv_branch2])'''
    ft = Flatten()(out)
    dp = Dropout(0.3)(ft)
    dense = Dense(2048, activation='relu')(dp)
    dense = Dense(2048, activation='relu')(dense)
    dense = Dense(2048, activation='relu')(dense)
    dense = Dense(28, activation='sigmoid')(dense)

    model = Model(inputs=inp, outputs=dense)
    #loss = [focal_loss(alpha=.25, gamma=2)]
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def unet(pretrained_weights = None,input_size=(128, 64, 4)):
    concat_axis = 3
    inputs = Input(shape=input_size)

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    bn = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn)


    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    bn = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn)

    conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)
    bn = BatchNormalization()(conv2)
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    up_conv8 = UpSampling2D(size=(2, 2))(bn)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(up_conv8)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)


    up_conv8 = UpSampling2D(size=(2, 2))(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up_conv8)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(4, (1, 1))(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['accuracy'])

    model.summary()

    return model

def update_model(model):
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()


    model.layers[1].trainable = False
    model.layers[2].trainable = False
    model.layers[5].trainable = False
    model.layers[6].trainable = False
    model.layers[9].trainable = False
    model.layers[10].trainable = False

    conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(model.layers[-1].output)
    conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #lb = Lambda(lambda x: K.stop_gradient(x))(pool1)
    ft = Flatten()(conv1)
    dp = Dropout(0.5)(ft)
    dp = Dense(4096, activation='relu')(dp)
    dp = Dense(4096, activation='relu')(dp)
    dp = Dense(2048, activation='relu')(dp)
    dp = Dense(28, activation='sigmoid')(dp)
    model = Model(inputs=model.layers[0].input, outputs=dp)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model



