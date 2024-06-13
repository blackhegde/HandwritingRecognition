import os
import numpy as np
import tensorflow as tf
import argparse

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, LSTM, Lambda, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from load_data import CHAR_DICT, MAX_LEN, ctc_lambda_func, SIZE, TextImageGenerator, ProgressCallback

def maxpooling(base_model):
    model = Sequential(name='vgg16')
    for layer in base_model.layers[:-1]:
        if 'pool' in layer.name:
            pooling_layer = MaxPooling2D(pool_size=(2, 2), name=layer.name)
            model.add(pooling_layer)
        else:
            model.add(layer)
    return model

def build_model(input_shape, training, finetune):
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')
    
    #Convolution layer
    # base_model = tf.keras.applications.VGG16(weights = None, include_top = False) #Not include 3 fully-conected layer
    # base_model = maxpooling(base_model)
    # inner = base_model(inputs)
    inner = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    inner = Conv2D(64, (3, 3), padding='same', activation='relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    inner = Conv2D(128, (3, 3), padding='same', activation='relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)

    base_model = Model(inputs=inputs, outputs=inner)

    #thay đổi kích thước tensor cho phù hợp các lớp sau
    inner = Reshape(target_shape=(int(inner.shape[1]), -1), name='reshape')(inner)
    
    #Recurent layer
    #fully-connected 512 đơn vị, hàm kích hoạt ReLu
    inner = Dense(512, activation='relu', kernel_initializer='he_normal', name='dense1')(inner) 
    #lớp ngẫu nhiên drop tỉ lệ 0.25, giảm over-fitting
    inner = Dropout(0.25)(inner) 
    #lớp 2 chiều LSTM với 512 đơn vị
    lstm = Bidirectional(LSTM(512, return_sequences=True, kernel_initializer='he_normal', name='lstm1', dropout=0.25, recurrent_dropout=0.25))(inner) 

    y_pred = Dense(CHAR_DICT, activation='softmax', kernel_initializer='he_normal', name='dense2')(lstm)

    #CTC_loss
    labels = Input(name='the_labels', shape=[MAX_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # Neu dang finetune thi khong huan luyen lai
    for layer in base_model.layers:
        layer.trainable = finetune
    
    y_func = tf.keras.Function(inputs = [inputs], outputs = [y_pred])
    # Neu training thi include ctc
    if training:
        Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out).summary()
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out), y_func
    else:
        return Model(inputs=[inputs], outputs=y_pred)
    
def train_kfold(idx, kfold, datapath, labelpath,  epochs, batch_size, lr, finetune):
    #xây dựng và biên dịch mô hình, training bao gồm hàm ctc, finetune để huần luyện mô hình
    model, y_func = build_model((*SIZE, 3), training=True, finetune=finetune)
    #bộ tối ưu hoá Adam với learning rate truyền vào lr
    ada = Adam(learning_rate=lr)
    #biên dịch mô hình với CTC Loss, Lambda bỏ qua giá trị y_true (CTC đã bao gồm nhãn thật bên trong)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

    ##load data
    #chỉ số huấn luyện và kiểm tra, nạp vào bộ nhớ
    train_idx, valid_idx = kfold[idx]
    train_generator = TextImageGenerator(datapath, labelpath, *SIZE, batch_size, 32, train_idx, True, MAX_LEN)
    train_generator.build_data()
    valid_generator  = TextImageGenerator(datapath, labelpath, *SIZE, batch_size, 32, valid_idx, False, MAX_LEN)
    valid_generator.build_data()

    ## callbacks
    #đường dẫn lưu mô hình tốt nhất cho mỗi K-fold
    weight_path = "model/best_{}.weights.h5".format(idx)
    #Lưu lại mô hình có giá trị loss trên tập kiểm tra val_loss tốt nhất
    ckp = tf.keras.callbacks.ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    #Callback hiển thị lại quá trình huấn luyện và đánh giá (edit_distance)
    pr = ProgressCallback(y_func, valid_generator, len(valid_idx))
    #Callback Tensorboard
    tb = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True)
    #Callback dừng quá trình huấn luyện nếu giá trị loss không cải thiện sau 1 số lượng epochs (10)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')
    #Reduce learning rate khi giá trị loss không cải thiện sau 5 epochs
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_delta=1e-8, verbose=1)
    
    #nạp mô hình huấn luyện trước(nếu có)
    if finetune:
        print('load pretrain model')
        model.load_weights(weight_path)
    
    #huấn luyện mô hình với các batch sinh ra từ train-generator.next_bacth
    model.fit(train_generator.next_batch(),
                steps_per_epoch=int(len(train_idx) / batch_size), #Số bước mỗi epochs (số mẫu huấn luyện/ kích thước batch)
                epochs=epochs,
                callbacks=[ckp, pr, earlystop, tb, reduce_lr], #gọi các callback
                validation_data=valid_generator.next_batch(), #dữ liệu kiểm tra 
                validation_steps=int(len(valid_idx) / batch_size))
 
def train(datapath, labelpath, epochs, batch_size, lr, finetune=False):
    #Thiết lập số lượng folds là 5
    nsplits = 5

    #Lấy danh sách các tệp trong thư mục dữ liệu
    nfiles = np.arange(len(os.listdir(datapath)))

    #tạo K-folds
    kfold = list(KFold(nsplits).split(nfiles))
    #huấn luyện mô hình cho mỗi fold
    for idx in range(nsplits):
        train_kfold(idx, kfold, datapath, labelpath, epochs, batch_size, lr, finetune)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default='data/preprocess/samples2/', type=str)
    parser.add_argument("--label", default='data/preprocess/samples2.json', type=str)

    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--finetune', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    train(args.train, args.label, args.epochs, args.batch_size, args.lr, args.finetune)