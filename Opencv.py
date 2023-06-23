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
cv2.waitKey(0)
cv2.destroyAllWindows()
print('Google colab以外で表示(resize)')

# 画像の色を変更
img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
print('画像の色変更完了')

#画像を表示
show_img(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), img)
print('画像の表示完了(色変更)')

plt.show()
print('Google colab以外で表示(色変更)')
