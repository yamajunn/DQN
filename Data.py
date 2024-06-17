import cv2
import numpy as np
import pyautogui
import time

# 画面の幅と高さを取得
screen_width, screen_height = pyautogui.size()

image_X = 960
image_Y = 540

frame = 0
time.sleep(5)
while True:
    # 画面をキャプチャ
    screenshot = np.array(pyautogui.screenshot())

    # 画像をリサイズしてグレースケールに変換
    resized_image = cv2.resize(cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY), (image_X, image_Y))

    # マウスカーソルの座標を取得
    x, y = pyautogui.position()

    # 画像サイズと画面サイズの比率を計算
    ratio_x = image_X / screen_width
    ratio_y = image_Y / screen_height

    # マウスカーソルの座標を画像の座標系に変換
    x_image = int(x * ratio_x)
    y_image = int(y * ratio_y)

    # マウスカーソルを描画
    cv2.circle(resized_image, (x_image, y_image), 1, (50, 50, 50), -1)

    # 状態として使用する画像を準備
    state = resized_image.flatten()

    # 画面を表示
    # cv2.imshow("Screen", resized_image)
    cv2.imwrite(f"Datas/Pong/{frame}.png", resized_image)
    frame += 1
    time.sleep(0.2)
    # キー入力を待機
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
