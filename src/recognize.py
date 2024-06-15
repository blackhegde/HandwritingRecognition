import os
import tensorflow as tf
import json
import numpy as np
from crnn import build_model
from load_data import SIZE, MAX_LEN, TextImageGenerator, beamsearch
import glob                                                                 
import argparse


def loadmodel(weight_path):
    model = build_model((*SIZE, 3), training=False, finetune=0)
    model.load_weights(weight_path)
    return model

def predict(model, datapath):
    batch_size = 3
    models = glob.glob('model/best_{}.weights.h5'.format(model))
    test_generator  = TextImageGenerator(datapath, None, *SIZE, batch_size, 32, None, False, MAX_LEN)
    test_generator.build_data()

    y_preds = []
    for weight_path in models:
        
        print('load {}'.format(weight_path))
        model = loadmodel(weight_path)
        X_test = test_generator  #.imgs.transpose((0, 2, 1, 3))
        y_pred = model.predict(X_test, batch_size=3)
        y_preds.append(y_pred)
        decoded_res = beamsearch(y_pred)
        for i in range(len(test_generator.img_dir)):
            print('{}: {}'.format(test_generator.img_dir[test_generator.indexes[i]], decoded_res[i]))
    if len(y_preds) == 0:
        print("No predictions were made.")
    else:
        y_preds = np.prod(y_preds, axis=0)**(1.0/len(y_preds))
    y_texts = beamsearch(y_preds)
    submit = dict(zip(test_generator.img_dir, y_texts))
    with open('submit.json', 'w', encoding='utf-8') as json_file:
        json.dump(submit, json_file, ensure_ascii=False, indent=4)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='/model/', type=str)
    parser.add_argument('--data', default='data/preprocess/test/', type=str)
    parser.add_argument('--device', default=2, type=int)
    args = parser.parse_args()
    
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    predict(args.model, args.data)