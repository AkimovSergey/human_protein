
from os import path
import pandas as pd
import numpy as np
from data_loader import DataGenerator
from model import unet, update_model, model_conv_4_colors, conv_net
from utils import *
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tqdm
import os


saved_encoder_net = 'vgg_cells_last.hdf5'
saved_detector_net = 'detector_net.hdf5'

train_dataset = load_dataset()
train_datagen = DataGenerator.create_train(train_dataset, 10)

model = conv_net()

image = DataGenerator.load_set_from_image(os.path.join(train_path, '000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0'))

#model.load_weights(os.path.join(data_path, saved_detector_net))
'''callbacks = [
    ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5,
                      min_delta=0.005, mode='max', cooldown=3, verbose=1),
             ModelCheckpoint(filepath=os.path.join(data_path, saved_detector_net),
                             monitor='loss', save_best_only=True, verbose=1)
]'''
callbacks = [EarlyStopping(monitor='loss', patience=2),
             ModelCheckpoint(filepath=os.path.join(data_path, saved_detector_net),
                             monitor='loss', save_best_only=True, verbose=1)]

DataGenerator.is_encoder_train = False
for i in range(100):
    model.fit_generator(train_datagen, steps_per_epoch=100, epochs=1, callbacks=callbacks)
    image = DataGenerator.load_set_from_image(os.path.join(train_path, '000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0'))
    # visualize_layer('concatenate_4', model, image)

    score_predict = model.predict(image)
    scp = np.transpose(score_predict)
    mn = [np.mean(x) for x in scp]
    mx = [np.max(x) for x in scp]
    pretty_print('mean', mn)
    pretty_print('max', mx)

'''image = DataGenerator.load_set_from_image(os.path.join(train_path, '008761b4-bbad-11e8-b2ba-ac1f6b6435d0'))
#visualize_layer('concatenate_4', model, image)

score_predict = model.predict(image)
scp = np.transpose(score_predict)

sm = [sum(x) for x in scp]
mn = [np.mean(x) for x in scp]
mx = [np.max(x) for x in scp]
pretty_print('sum', sm)
pretty_print('mean', mn)
pretty_print('max', mx)'''

#res = model.evaluate_generator(train_datagen, steps=100)
#print(res)

#model_checkpoint = ModelCheckpoint('vgg_cells.hdf5', monitor='loss', verbose=1, save_best_only=True)
#model.fit_generator(train_datagen, steps_per_epoch=50, epochs=400, callbacks=[model_checkpoint])


submit = pd.read_csv(os.path.join(data_path, 'train_check.csv'))

predicted = []
error = 0
for name in tqdm.tqdm(submit['Id']):
    path = os.path.join(os.path.join(data_path, 'train/'), name)
    image = DataGenerator.load_set_from_image(path)
    #image = DataGenerator.load_image(path)
    if image is None or len(image) == 0:
        error = error + 1
        out = '0'
    else:
        score_predict = model.predict(image)
        scp = np.transpose(score_predict)
        mx = np.array([np.max(x) for x in scp])
        label_predict = np.arange(28)[mx >= 0]
        out = ' '.join(str(l) for l in label_predict)
    str_predict_label = out
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
submit.to_csv('train_check_v.csv', index=False)

print('error = {0}'.format(error))
