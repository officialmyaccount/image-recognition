# ライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# 画像を表示するための関数を定義
def show_img(Input, Output):
    plt.subplot(121) # 画像の位置を指定
    plt.imshow(Input) # 画像を表示
    plt.title('Input') # 画像の上にInputと表示
    plt.xticks([]) # x軸のメモリを非表示
    plt.yticks([]) # y軸のメモリを非表示
    
    plt.subplot(122)
    plt.imshow(Output)
    plt.title('Output')
    plt.xticks([])
    plt.yticks([])
print('show_img関数の定義完了')

# 画像の読み込み
original = cv2.imread('dog-4390885_1280.jpg')
img0 = cv2.resize(original, (250, 180))
print('画像の読み込み完了')

# 画像を表示
cv2.imshow('Original Image', original)
cv2.imshow('Resized Image', img0)
print('画像の表示完了(resize)')

# Google Colab以外でOpenCVを用いて表示する
cv2.waitKey(0) #0の場合、ボタンが押されるまでウィンドウを削除しない
cv2.destroyAllWindows() # ウィンドウを閉じる
print('Google colab以外で表示(resize)')

# 画像の色を変更
img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
print('画像の色変更完了')

#画像を表示
show_img(original, img)
print('画像の表示完了(色変更)')

plt.show()
print('Google colab以外で表示(色変更)')

# 画像の移動量を決定して変数に格納
M = np.float32([[1, 0, 50], [0, 1, 30]])
print('画像の移動量の決定')

# 画像を平行移動
moved = cv2.warpAffine(img, M, (250, 180))
print('平行移動の完了')

# 関数を利用して画像を表示
show_img(img, moved)
print('画像の表示完了(画像の移動)')

plt.show()

# 画像の行列、列数を変数に格納
rows, cols = img.shape[:2]

# 画像の回転量の設定
M = cv2.getRotationMatrix2D((cols/2, rows/2), 60, 0.5) # 回転の中心、回転の角度、拡大縮小の度合い

# 画像を開店
rotated = cv2.warpAffine(img, M, (cols, rows))

# 関数を利用して画像を表示
show_img(img, rotated)
plt.show()
