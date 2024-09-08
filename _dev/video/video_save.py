import cv2
import glob
import os
import numpy as np
def Recording():
    #撮影時間を秒で指定  3分の場合は180
    time = int(input("enter the number of seconds\n>>"))
    
    #USBカメラの場合は1、パソコン内蔵カメラの場合は0
    print("PC CAMERA:0, USB CAMERA:1")
    setting = int(input("enter 0 or 1\n>>"))
    cap = cv2.VideoCapture(setting)
    
    #fpsを20.0にして撮影したい場合はfps=20.0にします
    fps = 30.0
    
    #カメラの幅を取得
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #カメラの高さを取得
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #動画保存時の形式を設定
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    name = "sample.mov"
    
    #(保存名前、fourcc,fps,サイズ)
    video = cv2.VideoWriter(name, fourcc, fps, (w,h))
 
    #fps×指定した撮影時間の値繰り返す
    print("start")
    roop = int(fps * time)
    for i in range(roop):
        ret, frame = cap.read()#1フレーム読み込み
        video.write(frame)#1フレーム保存する
    
    print("stop")
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    
    return video

Recording() #録画
