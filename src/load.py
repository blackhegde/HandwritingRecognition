import os, random
import json
import cv2
import tensorflow as tf
import numpy as np
import itertools
import editdistance
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Danh sách các ký tự được hỗ trợ
letters = " !\"#&\\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
MAX_LEN = 70
TIMESTEPS = 240
WIDTH, HEIGHT = 2560, 160
SIZE = WIDTH, HEIGHT
CHAR_DICT = len(letters) + 1

chars = letters

def text_to_labels(text):
    dig_lst = []
    for index, char in enumerate(text):
        try:
            dig_lst.append(letters.index(char))
        except ValueError:
            print(f"Character {char} not in letters")
    return dig_lst

# CTC_lambda
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # bỏ qua 2 bước đầu cho nhiều
    y_pred = y_pred[:, 2:, :]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

class TextImageGenerator:
    def __init__(self, img_dirpath, labels_path, img_w, img_h,
                 batch_size, downsample_factor, idxs, training=True, max_text_len=9):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.idxs = idxs
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath                  # image dir path
        self.labels= json.load(open(labels_path)) if labels_path != None else None
        self.img_dir = os.listdir(self.img_dirpath)     # images list
        if self.idxs is not None:
            self.img_dir = [self.img_dir[idx] for idx in self.idxs]

        self.n = len(self.img_dir)                      # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w, 3), dtype=np.float16)
        self.training = training
        self.texts = []

    def build_data(self):
        print(self.n, " Image Loading start... ", self.img_dirpath)
        for i, img_file in enumerate(self.img_dir):
            # load img vao PIL format
            img = image.load_img(self.img_dirpath + img_file, target_size=SIZE[::-1])
            #chuyen img tu PIL sang numpy
            img = image.img_to_array(img)
            img = preprocess_input(img).astype(np.float16)
            self.imgs[i] = img
            if self.labels != None: 
                self.texts.append(self.labels[img_file])
                #padding text to time steps
                self.texts = tf.keras.preprocessing.sequence.pad_sequences(self.texts, maxlen=TIMESTEPS, padding='post', value = 0)
            else:
                #valid mode
                self.texts.append('')
        print("Image Loading finish...")

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]].astype(np.float32), self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            X_data = np.zeros([self.batch_size, self.img_w, self.img_h, 3], dtype=np.float32)     # (bs, 128, 64, 1)
            Y_data = np.zeros([self.batch_size, self.max_text_len], dtype=np.float32)             # (bs, 9)
            input_length = np.ones((self.batch_size, 1), dtype=np.float32) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1), dtype=np.float32)           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.transpose((1, 0, 2))
                
                X_data[i] = img
                Y_data[i,:len(text)] = text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1)
                'label_length': label_length  # (bs, 1)
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1)
            yield (inputs, outputs)