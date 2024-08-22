# inference.py
import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from module.tfutils import InferenceGenerator, LoadImagePaths, judge, create_dir

# 定数設定
DATADIR = "./cifar10/test"
TRAIN_CLASSNAMES = ["airplane","automobile"]
WEIGHT_PATH = "weight.h5"
MEAN_VECTOR_PATH = "vectors.npy"
IMGSIZE = 224
THRESHOLD = 0.95

# テストデータの読み込み
classnames = os.listdir(DATADIR)
X, _ = LoadImagePaths(classnames,DATADIR,"png").get_dataset()
generator = InferenceGenerator(X,64,IMGSIZE)

# モデル読み込み
model = load_model(WEIGHT_PATH)

# 推論実行
preds = model.predict(generator)

# 画像保存フォルダ作成
for classname in classnames:
    create_dir(os.path.join("output",classname), delete=True)

# 判定＆画像保存
mean_vectors = np.load(MEAN_VECTOR_PATH)
for pred,imgpath in zip(preds,X):
    pred_label, max_cossim = judge(pred,mean_vectors,THRESHOLD,classnames)
    true_label = os.path.basename(os.path.dirname(imgpath))
    filename = os.path.basename(imgpath)
    savename = f"{pred_label}_cossim_{max_cossim:.2f}_{filename}"
    shutil.copyfile(imgpath,os.path.join("output",true_label,savename))
     