import cv2
import numpy as np
import pyautogui
import tensorflow as tf
import tkinter as tk

# カーソルを左端に移動
pyautogui.moveTo(0, pyautogui.position().y)

# DQNの定義
class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = tf.keras.layers.Dense(24, activation='relu')
        self.fc2 = tf.keras.layers.Dense(24, activation='relu')
        self.fc3 = tf.keras.layers.Dense(action_size, activation='linear')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

# DQNのパラメータ
state_size = 300 * 200  # 画像サイズ
action_size = 2  # 行動の数（例：左に移動、右に移動）

# DQNモデルの生成
model = DQN(state_size, action_size)

# ウィンドウを作成
window = tk.Tk()
window.title("Button Example")

# ボタンを作成
button = tk.Button(window, text="Click Me!")

# ボタンをウィンドウに配置
button.pack()

# ウィンドウを表示
window.mainloop()

# 画面の幅と高さを取得
screen_width, screen_height = pyautogui.size()

while True:
    # 画面をキャプチャ
    screenshot = np.array(pyautogui.screenshot())

    # 画像をリサイズしてグレースケールに変換
    resized_image = cv2.resize(cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY), (300, 200))

    # 状態として使用する画像を準備
    state = resized_image.flatten()

    # DQNにより行動を決定
    action = model.predict(state)

    # DQNの出力に基づいてカーソルを移動させる
    if action == 0:
        pyautogui.move(-10, 0)  # 左に移動
    else:
        pyautogui.move(10, 0)  # 右に移動
