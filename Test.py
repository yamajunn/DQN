import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import pyautogui 
from tensorflow.keras.layers import Input
pyautogui.FAILSAFE = False

# # 動画ファイルからのキャプチャ
# def capture_video(video_file):
#     cap = cv2.VideoCapture(video_file)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     cap.release()
#     return frames

# # 正解データとしての動画ファイルの読み込み
# video_file = "pong.mp4"
# correct_frames = capture_video(video_file)

# # ディープラーニングモデルの定義
# model = models.Sequential([
#     Input(shape=(224, 224, 3)),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(3, activation='softmax')  # アクションの数に応じて変更する
# ])

# # モデルのコンパイル
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ## ゲームのキャプチャと学習
# while True:
#     # 画面をキャプチャ
#     screenshot = np.array(pyautogui.screenshot())
    
#     # 画像をリサイズしてモデルの入力サイズに変換
#     frame = cv2.resize(screenshot, (224, 224))
#     frame = np.expand_dims(frame, axis=0)  # バッチ次元を追加
#     frame = frame / 255.0  # 正規化

#     # モデルを使ってアクションを予測
#     action = np.argmax(model.predict(frame)[0])

#     # 予測したアクションに基づいてゲームをプレイ
#     if action == 0:
#         # 上矢印キーを押す
#         pyautogui.hotkey('up')
#         print("up")
#     elif action == 1:
#         # 下矢印キーを押す
#         pyautogui.hotkey('down')
#         print("down")
#     elif action == 2:
#         # 何もしない
#         print("stey")
#     # ゲーム画面を表示
#     # cv2.imshow("Game Window", frame)

#     # キー入力の待機と終了処理
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break

# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from tensorflow.keras import layers, models, optimizers

# サンプルのモデルを定義（適切なモデルに置き換えてください）
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(224, 224, 3)),
    layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ゲーム画面のキャプチャ（適切なキャプチャ方法に置き換えてください）
def capture_game_screen():
    # 仮の画像を返す
    return np.random.rand(224, 224, 3)

# ランダムな行動を返す関数（適切な行動決定ロジックに置き換えてください）
def select_random_action():
    return np.random.randint(3)

# 報酬の計算（適切な報酬計算方法に置き換えてください）
def calculate_reward():
    return np.random.rand()

# 学習用のデータを蓄積するためのリスト
training_data = []

# ゲームのメインループ
while True:  # 無限ループでゲームをプレイし続ける
    # ゲーム画面のキャプチャ
    game_screen = capture_game_screen()

    # モデルによる行動の予測
    predicted_action = model.predict(np.expand_dims(game_screen, axis=0))[0]
    action = np.argmax(predicted_action)

    # 予測したアクションに基づいてゲームをプレイ
    if action == 0:
        # 上矢印キーを押す
        pyautogui.hotkey('up')
        print("up")
    elif action == 1:
        # 下矢印キーを押す
        pyautogui.hotkey('down')
        print("down")
    elif action == 2:
        # 何もしない
        print("stey")

    # 報酬の計算
    reward = calculate_reward()

    # 学習用のデータを保存
    y_categorical = np.zeros(3)  # カテゴリカルな表現を持つ配列を初期化
    y_categorical[action] = 1  # 対応するアクションのインデックスを1に設定
    training_data.append((game_screen, y_categorical, reward))  # 正解ラベルをone-hotエンコーディングしたものを保存

    # 学習用データを取り出してモデルを学習（例として10ステップごとに学習）
    if len(training_data) >= 10:
        X = np.array([data[0] for data in training_data])
        
        y_categorical = np.array([data[1] for data in training_data])
        model.fit(X, y_categorical)
        training_data = []  # 学習用データをリセット

    # ゲームを終了する条件を設定（適切な終了条件を設定してください）
    if False:  # 例：一定のステップ数を超えたら終了
        break