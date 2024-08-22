# tfutils.py
import os
import math
from glob import glob
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout,GlobalAveragePooling2D,BatchNormalization,Activation
from collections import defaultdict
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from module.arcface import Arcfacelayer

from tensorflow.keras.applications import EfficientNetB0

class TrainGenerator(Sequence):
    '''
    画像パスとラベルからなるデータセットを渡す
    バッチサイズと画像サイズを指定する
    画像読み込みはマルチプロセスでおこなうよう実装した
    '''
    def __init__(self, x_set, y_set, batch_size, img_size):
        self.x, self.y = shuffle(x_set, y_set)
        self.batch_size = batch_size
        self.img_size = img_size
    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    @staticmethod
    def imread(path, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
        p = np.fromfile(path, dtype)
        img = cv2.imdecode(p, flags)
        return img

    def load_img(self,imgpath):
        '''opencvで読み込む'''
        img = self.imread(imgpath)
        img = cv2.resize(img, dsize=(self.img_size,self.img_size))
        return img

    def __getitem__(self, idx):
        path_list_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        with ThreadPoolExecutor(max_workers=8) as executor:
            # 現在のインデックスからバッチサイズ分だけの画像を読み込む
            futures = [executor.submit(self.load_img, p) for p in path_list_x]
            batch = np.empty((self.batch_size,) + (self.img_size,self.img_size,3))
            for i, future in enumerate(futures):
                batch[i] = future.result()
            batch_x = batch[:len(futures)]

        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y

class ArcfaceTrainGenerator(TrainGenerator):
    def __getitem__(self, idx):
        path_list_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        with ThreadPoolExecutor(max_workers=8) as executor:
            # 現在のインデックスからバッチサイズ分だけの画像を読み込む
            futures = [executor.submit(self.load_img, p) for p in path_list_x]
            batch = np.empty((self.batch_size,) + (self.img_size,self.img_size,3))
            for i, future in enumerate(futures):
                batch[i] = future.result()
            batch_x = batch[:len(futures)]

        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        return (batch_x, batch_y), batch_y

class InferenceGenerator(TrainGenerator):
    def __init__(self, x_set, batch_size, img_size):
        self.x = x_set
        self.batch_size = batch_size
        self.img_size = img_size
    
    def __getitem__(self, idx):
        path_list_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        with ThreadPoolExecutor(max_workers=8) as executor:
            # 現在のインデックスからバッチサイズ分だけの画像を読み込む
            futures = [executor.submit(self.load_img, p) for p in path_list_x]
            batch = np.empty((self.batch_size,) + (self.img_size,self.img_size,3))
            for i, future in enumerate(futures):
                batch[i] = future.result()
            batch_x = batch[:len(futures)]

        return batch_x

class BuildModel:
    '''
    kerasNNモデルの構築クラス
    デフォルトはEfnB0の転移学習
    オーバーライドして使う
    '''
    def __init__(self, n_categories, imagesize):
        self.n_categories = n_categories
        self.imagesize = imagesize
        self.model = None
    
    def load_model(self):
        inputs =  Input(shape=(self.imagesize, self.imagesize, 3))
        self.model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet") 
    
    def modify_model(self):
        # Freeze the pretrained weights
        self.model.trainable = False
        # Rebuild top
        x = GlobalAveragePooling2D(name="avg_pool")(self.model.output)
        x = BatchNormalization()(x)
        top_dropout_rate = 0.2
        x = Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = Dense(self.n_categories, activation="softmax", name="pred")(x)
        inputs = self.model.inputs
        self.model = Model(inputs,outputs)

    def compile(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
        self.model.compile(
                    optimizer = optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                    )
    
    def build(self):
        '''コンパイル後のモデルを返す'''
        self.load_model()
        self.modify_model()
        self.compile()
        return self.model

class BuildArcfaceModel(BuildModel):
    
    def modify_model(self):
        
        # ArcFaceLayerをくっつける
        x = self.model.output
        yinput = Input(shape=(self.n_categories,))
        hidden = GlobalAveragePooling2D()(x)
        # x = Arcfacelayer(self.n_categories, 40, 0.05)([hidden,yinput])
        x = Arcfacelayer(self.n_categories, 40, 0.1)([hidden,yinput])
        prediction = Activation('softmax')(x)
        self.model = Model(inputs=[self.model.input,yinput],outputs=prediction)

   
def display_training_curve(history, val=True, display=True, save_path=""):
    # Plot Accuracy
    plot_list = ['accuracy','loss']
    fig, axes = plt.subplots(1,len(plot_list), tight_layout=True)
    
    for i, plot_name in enumerate(plot_list):
        axes[i].plot(history.history[plot_name])
        axes[i].set_ylabel(plot_name)
        axes[i].set_xlabel('Epoch')
        if val:
            axes[i].plot(history.history[f'val_{plot_name}'])
            axes[i].legend(['Train', 'Val'], loc='upper left')
    if save_path:
        plt.savefig(save_path)  
    if display:    
        plt.show()

def save_mean_vectors(model,X,y,imgsize,batch_size,save_path):
    
    dataset = defaultdict(list)
    for onehot_label, path in zip(y, X):
        label = np.argmax(onehot_label)
        dataset[label].append(path)
    
    mean_vectors_dic = {}
    for label in dataset:
        inf_generator = InferenceGenerator(dataset[label],batch_size,imgsize)
        n_batch = math.ceil(len(dataset[label])/batch_size)
        vectors = model.predict(inf_generator,n_batch,verbose=1)
        mean_vector=vectors.mean(axis=0)
        mean_vectors_dic[label] = mean_vector
    
    sorted_keys = sorted(mean_vectors_dic)
    mean_vectors = [mean_vectors_dic[key] for key in sorted_keys]
    mean_vectors = np.array(mean_vectors)
    np.save(save_path,mean_vectors)    
    
class LoadImagePaths:
    '''
    get_datasetで画像パスとラベルを得る
    Args:
    classnames, dirpath, suffix
    '''
    def __init__(self, classnames, dirpath, suffix):
        self.classnames = classnames
        self.dirpath = dirpath
        self.suffix = suffix
        
    def _get_paths(self, classname, label):
        '''globでパスを取得して、labelを作る'''
        img_paths = glob(os.path.join(self.dirpath,f"{classname}/*{self.suffix}"))
        labels = len(img_paths)*[label]
        return img_paths, labels
    
    def _one_hot_encode(self,y):
        y = y.reshape(-1,1)
        ohe = OneHotEncoder()
        return ohe.fit_transform(y).toarray()

    def get_dataset(self):
        X = []
        y = []
        for label, classname in enumerate(self.classnames):
            img_paths, labels = self._get_paths(classname, label)
            X.extend(img_paths)
            y.extend(labels)
            
        X = np.array(X)
        y = np.array(y)
        y = self._one_hot_encode(y)        
        return X, y
    
def cosine_similarity(x1, x2): 
    if x1.ndim == 1:
        x1 = x1[np.newaxis]
    if x2.ndim == 1:
        x2 = x2[np.newaxis]
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    cosine_sim = np.dot(x1, x2.T)/(x1_norm*x2_norm+1e-10)
    return cosine_sim

def judge(predict_vector, mean_vectors, thresh, classnames):
    """
    predict_vector : shape(1,1028)
    hold_vector : shape(n_class, 1028)
    """
    cos_similarity = cosine_similarity(predict_vector, mean_vectors)[0]
    max_cossim = np.max(cos_similarity)
    max_label = classnames[np.argmax(cos_similarity)]
    
    if max_cossim > thresh:
        pred_label =  max_label
    else:
        pred_label = "unkown"
    
    return pred_label, max_cossim 

def create_dir(folder_path, delete=False):
    os.makedirs(folder_path, exist_ok=True)
    if delete:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)