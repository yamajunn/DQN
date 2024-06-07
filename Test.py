import cv2
import numpy as np
import pyautogui
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# 動画ファイルからフレームのシーケンスをキャプチャ
def capture_video_as_sequences(video_file, sequence_length):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError(f"Couldn't read video stream from file {video_file}")

    sequences = []
    current_sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0  # 正規化

        current_sequence.append(frame)
        if len(current_sequence) == sequence_length:
            sequences.append(current_sequence)
            current_sequence = []

    cap.release()
    return sequences

# ゲーム画面のキャプチャ
def capture_game_frame():
    screenshot = np.array(pyautogui.screenshot())
    frame = cv2.cvtColor(screenshot, cv2.COLOR_RGBA2RGB)  # 4チャンネルを3チャンネルに変換
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0  # 正規化
    return frame

def compute_sequence_reward(current_sequence, target_sequence):
    # シーケンス全体の差分を初期化
    total_diff = 0.0

    # シーケンス内の各フレームに対して差分を計算して合計する
    for current_frame, target_frame in zip(current_sequence, target_sequence):
        frame_diff = np.mean(np.abs(current_frame - target_frame))
        total_diff += frame_diff

    # 各フレームの平均差分を報酬として返す
    return -total_diff / len(current_sequence)

def find_nearest_sequence(current_frame, sequences):
    # 最も差分が少ないシーケンスを見つける
    min_diff = float('inf')
    nearest_sequence = None
    for sequence in sequences:
        sequence_diff = np.mean(np.abs(current_frame - sequence[0]))
        if sequence_diff < min_diff:
            min_diff = sequence_diff
            nearest_sequence = sequence

    return nearest_sequence

# リプレイバッファの定義
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

## DQNモデルの定義
def build_dqn_model(input_shape, num_actions):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# リプレイバッファの定義
replay_buffer = ReplayBuffer(max_size=10000)

# パラメータ
input_shape = (224, 224, 3)
num_actions = 2  # 上矢印、下矢印
batch_size = 16
gamma = 0.99  # 割引率

# DQNモデルの構築
model = build_dqn_model(input_shape, num_actions)

# 正解データとしての動画ファイルの読み込み
video_file = "pong.mp4"
sequence_length = 10  # シーケンスの長さを指定
correct_sequences = capture_video_as_sequences(video_file, sequence_length)
# バッチサイズ
batch_size = 32

# ゲームのキャプチャと学習
while True:
    # 現在のバッチを初期化
    batch_states, batch_actions, batch_rewards, batch_next_states = [], [], [], []

    # バッチを取得
    for _ in range(batch_size):
        current_frame = capture_game_frame()

        # フレームの前処理
        current_frame = np.expand_dims(current_frame, axis=0)  # バッチ次元を追加

        # モデルを使ってアクションを予測
        q_values = model.predict(current_frame)
        action = np.argmax(q_values[0])

        # 予測したアクションに基づいてゲームをプレイ
        if action == 0:
            pyautogui.press('up')
        elif action == 1:
            pyautogui.press('down')
        elif action == 2:
            pass  # 何もしない

        # 正解データとの比較
        nearest_sequence = find_nearest_sequence(current_frame[0], correct_sequences)
        reward = compute_sequence_reward(current_frame, nearest_sequence)

        # 次の状態を取得
        next_frame = capture_game_frame()

        # 経験をバッチに追加
        batch_states.append(current_frame[0])
        batch_actions.append(action)
        batch_rewards.append(reward)
        batch_next_states.append(next_frame)

    # バッチの学習
    states = np.array(batch_states)
    next_states = np.array(batch_next_states)
    rewards = np.array(batch_rewards)
    actions = np.array(batch_actions)

    q_values_next = model.predict(next_states)
    target_q_values = rewards + gamma * np.max(q_values_next, axis=1)

    q_values = model.predict(states)
    for i, action in enumerate(actions):
        q_values[i][action] = target_q_values[i]

    model.fit(states, q_values, epochs=1, verbose=0)

    # ゲーム画面を表示
    # cv2.imshow("Game Window", current_frame[0])

    # キー入力の待機と終了処理
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# 終了処理
cv2.destroyAllWindows()
