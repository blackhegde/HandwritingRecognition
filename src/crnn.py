from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, LSTM, Lambda, Bidirectional, Dropout
from keras.layers import BatchNormalization, Activation
from keras import backend as K
from load_data import CHAR_DICT, MAX_LEN
import tensorflow as tf

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_model(input_shape, num_classes):
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')

    # Convolutional layers
    inner = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)

    inner = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)

    inner = Reshape(target_shape=(int(inner.shape[1]), -1), name='reshape')(inner)
    inner = Dense(512, activation='relu', kernel_initializer='he_normal', name='dense1')(inner) 
    inner = Dropout(0.25)(inner) 

    # Recurrent layers
    lstm = Bidirectional(LSTM(512, return_sequences=True, kernel_initializer='he_normal', name='lstm1', dropout=0.25, recurrent_dropout=0.25))(inner) 

    # Transform RNN output to character activations
    y_pred = Dense(CHAR_DICT, activation='softmax', kernel_initializer='he_normal',name='dense2')(lstm)

    # Model(inputs, y_pred).summary()

    labels = Input(name='the_labels', shape=[MAX_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

    return model

def get_model(input_shape, training, finetune):
    inputs = Input(name='the_inputs', shape=input_shape, dtype='float32')
    base_model = applications.VGG16(weights='imagenet', include_top=False)
    inner = base_model(inputs)
    inner = Reshape(target_shape=(int(inner.shape[1]), -1), name='reshape')(inner)
    inner = Dense(512, activation='relu', kernel_initializer='he_normal', name='dense1')(inner) 
    inner = Dropout(0.25)(inner) 
    lstm = Bidirectional(LSTM(512, return_sequences=True, kernel_initializer='he_normal', name='lstm1', dropout=0.25, recurrent_dropout=0.25))(inner) 

    y_pred = Dense(CHAR_DICT, activation='softmax', kernel_initializer='he_normal',name='dense2')(lstm)
    
    labels = Input(name='the_labels', shape=[MAX_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    for layer in base_model.layers:
        layer.trainable = finetune
    
    y_func = K.function([inputs], [y_pred])
    
    if training:
        Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out).summary()
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out), y_func
    else:
        return Model(inputs=[inputs], outputs=y_pred)
