import cv2
import random
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import numpy as np
from tqdm import tqdm
from lp_recognition.parameter import *


# # Input data generator
def labels_to_text(labels):     # letters의 index -> text (string)
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))


class TextImageGenerator:
    def __init__(self, img_dirpath, img_w, img_h,
                 batch_size, provinces=provinces, letters=letters, max_textlen=lp_text_len):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_textlen
        self.province_list = provinces
        self.province_num = len(provinces)
        self.letter_list = letters
        self.letter_num = len(letters)
        self.img_dirpath = img_dirpath                  # image dir path
        self.img_dir = os.listdir(self.img_dirpath)     # images list
        self.n = len(self.img_dir)                      # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []

    ##
    def build_data(self):
        print(self.n, " Image Loading start...")
        for i, img_file in tqdm(enumerate(self.img_dir)):
            img = cv2.imread(os.path.join(self.img_dirpath , img_file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = (img / 255.0) * 2.0 - 1.0

            self.imgs[i, :, :] = img
            self.texts.append(img_file.split('.')[1])
        #print(self.texts[0:10], '...')
        assert (len(self.texts) == self.n) and (self.imgs.shape[0] == self.n)
        print(self.n, " Image Loading finish...")

    def next_sample(self):      ## index max -> 0
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):       ## batch size
        while True:
            X_data = np.zeros((self.batch_size, self.img_w, self.img_h, 1))     # (bs, 128, 64, 1)
            Y_data = [np.zeros((self.batch_size, self.province_num))]     # (bs, 9)
            for i in range(self.max_text_len-1):
                Y_data.append(np.zeros((self.batch_size, self.letter_num)))

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                #print(text)
                for j, ch in enumerate(text):
                    #print(j)
                    if j == 0:
                        Y_data[j][i, self.province_list.index(ch)] = 1
                    else:
                        Y_data[j][i, self.letter_list.index(ch)] = 1

            yield (X_data, Y_data)