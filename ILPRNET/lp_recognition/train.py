import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras import optimizers
from lp_recognition.parameter import *
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from lp_recognition.Image_Generator import TextImageGenerator

from keras.utils import multi_gpu_model, plot_model

from lp_recognition.model.models import ILPRnet



# 自定义回调函数，保存模型
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss',
                 save_best_only=True, mode='min'):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, save_best_only, mode)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu
    plt_model_path = plt_model_path
    if not os.path.exists(plt_model_path):
        os.makedirs(plt_model_path)

    original_model = ILPnet(input_shape=(img_w, img_h, 1),debug=True)

    original_model.summary()
    plot_model(original_model, to_file=plt_model_path+'/ILPNET.png')
    train_model = multi_gpu_model(original_model, gpus=gpu_num) if (gpu_num > 1) else original_model

    # data generator
    tiger_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size)
    tiger_train.build_data()
    tiger_val = TextImageGenerator(valid_file_path, img_w, img_h, val_batch_size)
    tiger_val.build_data()

    optimizer = optimizers.Adam()

    train_model.compile(optimizer=optimizer,loss={'x1':'categorical_crossentropy',
                                                  'x2':'categorical_crossentropy',
                                                  'x3':'categorical_crossentropy',
                                                  'x4':'categorical_crossentropy',
                                                  'x5':'categorical_crossentropy',
                                                  'x6':'categorical_crossentropy',
                                                  'x7':'categorical_crossentropy'}, metrics=['accuracy'])

    reduce_lr_train= ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-12, verbose=1)  #
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
    backup_path = plt_model_path+'/ILPRNET-tinue-{epoch:02d}-{val_loss:.4f}.hdf5'
    checkpoint = ParallelModelCheckpoint(original_model, backup_path, monitor='val_loss', save_best_only=True, mode='min')

    H = train_model.fit_generator(generator=tiger_train.next_batch(),
                                  steps_per_epoch=min(200, int(tiger_train.n / batch_size)),
                                  epochs=epochs,
                                  callbacks=[reduce_lr_train,checkpoint],
                                  validation_data=tiger_val.next_batch(),
                                  validation_steps=int(tiger_val.n / val_batch_size))
    original_model.save(plt_model_path+'/ILPNET.h5')
    original_model.save_weights(plt_model_path+'/ILPNET_weight.h5')

    plt.style.use("ggplot")
    plt.figure()
    N = len(H.history['loss'])
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(plt_model_path + '/plot.png')

