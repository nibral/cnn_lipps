import sys
import os.path
import cv2

from face_detector import detect

# 引数チェック
if len(sys.argv) != 2:
    sys.stderr.write("usage: clip_face.py <filename>\n")
    sys.exit(-1)

# 動画を開いてファイル名を取得
input = cv2.VideoCapture(sys.argv[1])
basename, ext = os.path.splitext(os.path.basename(sys.argv[1]))

i = 1
count = 0
while True:
    # 1フレーム読み込む
    ret, frame = input.read()
    # ファイル末尾まで来たら終了
    if ret == False:
        break

    # 30フレームごとに処理
    count += 1
    if count % 30 == 0:
        # 顔部分を切り抜いて保存
        for (x, y, w, h) in detect(frame):
            dest = frame[y:y+h, x:x+w]
            cv2.imwrite('./clip/%s_%04d.jpg' % (basename, i), dest)
            i += 1