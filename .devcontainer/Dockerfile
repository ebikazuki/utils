FROM tensorflow/tensorflow:latest-gpu-jupyter
USER root

RUN apt-get update && \
    apt-get -y install --reinstall ca-certificates && \
    apt-get -y install software-properties-common && \
    pip install --upgrade pip

# # Install Basic Packages
RUN pip install ipykernel jupyter
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install pandas


# FROM tensorflow/tensorflow:latest-gpu-jupyter

# RUN apt-get update
# # ユーザーを作成
# ARG UID=1000
# RUN useradd -m -u ${UID} user

# # 作成したユーザーに切り替える
# # このユーザーはRUN, CMD, ENTRYPOINT, docker run, exec の実行ユーザ。
# USER ${UID}

# # プロジェクトフォルダを/codeに追加する。ユーザー権限で扱えるようchownオプションを使う。
# # ADDの実行権者はrootなのでオプションが必要。
# ADD  --chown=user:user . /work
# # 作成したフォルダに移動し、パッケージリストをインストールする。
# #WORKDIR /work

# RUN pip install matplotlib
# RUN pip install scikit-learn
# RUN pip install pandas

# # https://www.appliot.co.jp/blog/nvidia%E3%81%AEgpu%E3%82%92docker%E3%81%AEtensorflow%E3%81%A7%E5%8B%95%E3%81%8B%E3%81%99/