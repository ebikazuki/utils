# train.py
import os
from tensorflow.keras.models import Model
from module.tfutils import ArcfaceTrainGenerator, BuildArcfaceModel, LoadImagePaths, display_training_curve, save_mean_vectors


# 定数設定
DATADIR = "./cifar10/train"
CLASSNAMES = ["airplane","automobile"]
WEIGHT_PATH = "weight.h5"
MEAN_VECTOR_PATH = "vectors.npy"
IMGSIZE = 224
EPOCH = 5
BATCH_SIZE = 64

# データセット読み込み(X:path,y:onehot)
X, y = LoadImagePaths(CLASSNAMES,DATADIR,"png").get_dataset()
generator = ArcfaceTrainGenerator(X,y,BATCH_SIZE,IMGSIZE)

# モデル定義
model = BuildArcfaceModel(len(CLASSNAMES),IMGSIZE).build()

# モデル学習
hist = model.fit(generator, epochs=EPOCH)
display_training_curve(hist,val=False)

# モデル保存(arcfaceレイヤ以降を削除して保存,古いVerのTFはget_layer(index=0)でも動く)
save_model=Model(model.get_layer(index=1).input, model.get_layer(index=-4).output)
save_model.save(WEIGHT_PATH)

# 平均ベクトル保存
save_mean_vectors(save_model,X,y,IMGSIZE,BATCH_SIZE,MEAN_VECTOR_PATH)

     