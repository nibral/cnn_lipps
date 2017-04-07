import tensorflow as tf
import sys
import os
import numpy as np
import cv2

from face_detector import detect
from train import inference

IMAGE_HEIGHT_PX = 28
IMAGE_WIDTH_PX = 28
IMAGE_COLOR_CHANNELS = 3

# 引数チェック
if len(sys.argv) != 2:
    sys.stderr.write("usage: label_face.py <filename>\n")
    sys.exit(-1)

# 動画を開いてファイル名を取得
input = cv2.VideoCapture(sys.argv[1])
basename, ext = os.path.splitext(os.path.basename(sys.argv[1]))

# placeholder用意
images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT_PX * IMAGE_WIDTH_PX * IMAGE_COLOR_CHANNELS])
keep_prob = tf.placeholder(tf.float32)
# モデル生成
logits = inference(images_placeholder, keep_prob)

# TenforFlowセッション開始
sess = tf.InteractiveSession()
# 変数初期化
sess.run(tf.global_variables_initializer())
# モデルの読み込み
saver = tf.train.Saver()
saver.restore(sess, './model/model.ckpt')

# ラベル
label = ['Shiki', 'Shuko', 'Kanade', 'Frederica', 'Mika']

width = input.get(cv2.CAP_PROP_FRAME_WIDTH)
height = input.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = input.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('output.avi', fourcc, fps, (int(width), int(height)))

# 赤(B,G,R)
COLOR = (0, 0, 255)

i = 1
while True:
    # 1フレーム読み込む
    ret, frame = input.read()
    # ファイル末尾まで来たら終了
    if ret == False:
        break
    
    # 顔部分を切り出して判定
    for (x, y, w, h) in detect(frame):
        clip = frame[y:y+h, x:x+w]

        img = cv2.resize(clip, (IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX))
        img = img.flatten().astype(np.float32) / 255.0
        pred = sess.run(logits, feed_dict={
            images_placeholder: [img],
            keep_prob: 1.0
        })

        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR, 2)
        cv2.putText(frame, label[np.argmax(pred)], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR, 3)
    output.write(frame)
    print(i)
    i += 1
