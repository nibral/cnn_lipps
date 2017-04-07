import tensorflow as tf
import os
import numpy as np
import cv2
import random
import math

IMAGE_HEIGHT_PX = 28
IMAGE_WIDTH_PX = 28
IMAGE_COLOR_CHANNELS = 3
NUM_CLASSES = 5

MAX_STEPS = 1000
BATCH_SIZE = 50

def inference(images_placeholder, keep_prob):
    """ 推測モデルの生成
    引数:
      images_placeholder: 画像
      keep_prob: dropout層で残す率
    返り値:
      y_conv: 各クラスの確率
    """
    # 重みを標準偏差0.1の切断正規分布で初期化
    # 値の範囲を-0.2〜0.2(標準偏差の2倍)に制限する
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    # バイアスを0.1で初期化
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    # 2次元の畳込み処理
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # プーリング処理
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    # 1次元の入力画像をを28x28x3に変形
    x_image = tf.reshape(images_placeholder, [-1, IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX, 3])

    # 畳み込み層1(フィルタサイズ5x5x3x32, ReLU, 入力28x28x3, 出力28x28x32)
    # 32種類のフィルタのそれぞれについて、幅5*高さ5*3色=75次元の値を畳み込んで1つの値にする
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # プーリング層1(2x2の最大値, 入力28x28x32, 出力14x14x32)
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
    
    # 畳み込み層2(フィルタサイズ5x5, ReLU, 入力14x14x32, 出力14x14x64)
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # プーリング層2(2x2の最大値, 入力14x14x64, 出力7x7x64)
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    # 全結合層1(ReLU, 入力7x7x64, 出力1024)
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropoutの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 全結合層2(パラメータ定義, 入力1024, 出力5)
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([5])

    # 全結合層2(softmax)
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 各クラスの確率を返す
    return y_conv

def loss(logits, labels):
    """ 予測誤差の計算
    引数:
        logits: 予測値
        labels: 教師値
    戻り値:
        cross_entropy: 交差エントロピー
    """
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    """ 訓練
    引数:
        loss: 予測誤差
        learning_rate: 学習率
    """
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    """ 予測精度の計算
    引数:
        logits: 予測値
        labels: 教師値
    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

if __name__ == '__main__':
    # データセットの読み込み
    # 学習画像データ
    train_image = []
    # 学習データのラベル
    train_label = []
    # 画像のあるディレクトリ
    train_img_dirs = ['shiki', 'shuko', 'kanade', 'fre', 'mika']
    for i, d in enumerate(train_img_dirs):
        # ./data/以下の各ディレクトリ内のファイル名取得
        files = os.listdir('./data/' + d)
        for f in files:
            if f.startswith('.'):
                continue

            # 画像読み込み
            img = cv2.imread('./data/' + d + '/' + f)
            # 28x28にリサイズ
            img = cv2.resize(img, (IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX))
            # 1列にして
            img = img.flatten().astype(np.float32) / 255.0
            train_image.append(img)

            # 1-of-K符号化でラベルを追加
            tmp = np.zeros(NUM_CLASSES)
            tmp[i] = 1
            train_label.append(tmp)
    # numpy配列に変換
    train_image = np.asarray(train_image)
    train_label = np.asarray(train_label)

    # 処理時間計測
    import time
    start_at = time.time()

    # placeholder用意
    images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT_PX * IMAGE_WIDTH_PX * IMAGE_COLOR_CHANNELS])
    labels_placeholder = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
    keep_prob = tf.placeholder(tf.float32)
    # モデル生成
    logits = inference(images_placeholder, keep_prob)
    # 予測誤差計算
    loss = loss(logits, labels_placeholder)
    # 訓練の処理
    training_op = training(loss, 1e-4)
    # 精度計算の処理
    accuracy_op = accuracy(logits, labels_placeholder)

    # TenforFlowセッション開始
    sess = tf.InteractiveSession()
    # 変数初期化
    sess.run(tf.global_variables_initializer())

    # モデルの保存用意
    saver = tf.train.Saver()
    # TensorBoard初期化
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./log', sess.graph)

    # 訓練
    for step in range(MAX_STEPS):
        seed = int(time.time())
        np.random.seed(seed)
        np.random.shuffle(train_image)
        np.random.seed(seed)
        np.random.shuffle(train_label)

        for i in range(math.ceil(len(train_image) / BATCH_SIZE)):
            batch = BATCH_SIZE * i
            sess.run(training_op, feed_dict={
                images_placeholder: train_image[batch:batch+BATCH_SIZE],
                labels_placeholder: train_label[batch:batch+BATCH_SIZE],
                keep_prob: 0.5
            })

        # 精度計算
        train_accuracy = sess.run(accuracy_op, feed_dict={
            images_placeholder: train_image,
            labels_placeholder: train_label,
            keep_prob: 1.0
        })
        print('step %d, training accuracy %g' % (step, train_accuracy))

        # TensorBoardに追記
        summary = sess.run(merged, feed_dict={
            images_placeholder: train_image,
            labels_placeholder: train_label,
            keep_prob: 1.0
        })
        summary_writer.add_summary(summary, step)

    # テスト
    print('test accuracy %g' % sess.run(accuracy_op, feed_dict={
        images_placeholder: train_image,
        labels_placeholder: train_label,
        keep_prob: 1.0}))
    print('--- %s sec. ---' % (time.time() - start_at))

    # モデル保存
    save_path = saver.save(sess, './model/model.ckpt')
