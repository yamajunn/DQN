import cv2
import numpy as np
import pyautogui
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from concurrent.futures import ThreadPoolExecutor

pyautogui.FAILSAFE = False  # Fail-safe機能を無効化

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
            sequences.append(current_sequence.copy())
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

# DQNモデルの定義
def build_dqn_model(input_shape, num_actions):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D((1, 2, 2)),
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D((1, 2, 2)),
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def compute_sequence_reward(current_sequence, target_sequence):
    total_diff = 0.0
    for current_frame, target_frame in zip(current_sequence, target_sequence):
        frame_diff = np.mean(np.abs(current_frame - target_frame))
        total_diff += frame_diff
    reward = -total_diff / len(current_sequence) + 1.0
    return reward

def find_nearest_sequence(current_sequence, sequences):
    min_diff = float('inf')
    nearest_sequence = None
    for sequence in sequences:
        total_diff = 0.0
        for current_frame, target_frame in zip(current_sequence, sequence):
            frame_diff = np.mean(np.abs(current_frame - target_frame))
            total_diff += frame_diff
        average_diff = total_diff / len(current_sequence)
        if average_diff < min_diff:
            min_diff = average_diff
            nearest_sequence = sequence
    return nearest_sequence

# リプレイバッファの定義
replay_buffer = ReplayBuffer(max_size=10000)

# パラメータ
sequence_length = 5  # シーケンスの長さ
input_shape = (sequence_length, 224, 224, 3)
num_actions = 3  # 上矢印、下矢印、停止
batch_size = 8
gamma = 0.99  # 割引率

# DQNモデルの構築
model = build_dqn_model(input_shape, num_actions)

# 正解データとしての動画ファイルの読み込み
video_file = "pong.mp4"
correct_sequences = capture_video_as_sequences(video_file, sequence_length)

# 並列処理用のスレッドプールを作成
executor = ThreadPoolExecutor(max_workers=2)

# ゲームのキャプチャと学習
current_sequence = []

def capture_and_predict():
    while True:
        current_frame = capture_game_frame()
        current_sequence.append(current_frame)
        if len(current_sequence) < sequence_length:
            continue
        current_input = np.expand_dims(current_sequence, axis=0)  # バッチ次元を追加
        q_values = model.predict(current_input)
        action = np.argmax(q_values[0])
        if action == 0:
            pyautogui.keyDown('up')
            pyautogui.keyUp('up')
            print("up")
        elif action == 1:
            pyautogui.keyDown('down')
            pyautogui.keyUp('down')
            print("down")
        elif action == 2:
            print("stey")
        nearest_sequence = find_nearest_sequence(current_sequence, correct_sequences)
        reward = compute_sequence_reward(current_sequence, nearest_sequence)
        next_frame = capture_game_frame()
        next_sequence = current_sequence[1:] + [next_frame]
        replay_buffer.add((current_sequence.copy(), action, reward, next_sequence.copy()))
        current_sequence.pop(0)

def train_model():
    while True:
        if replay_buffer.size() >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states = zip(*batch)
            states = np.array(states)
            next_states = np.array(next_states)
            rewards = np.array(rewards)
            actions = np.array(actions)
            q_values_next = model.predict(next_states)
            target_q_values = rewards + gamma * np.max(q_values_next, axis=1)
            q_values = model.predict(states)
            for i, action in enumerate(actions):
                q_values[i][action] = target_q_values[i]
            model.fit(states, q_values, epochs=1, verbose=0)

executor.submit(capture_and_predict)
executor.submit(train_model)

# 終了処理
executor.shutdown(wait=True)
cv2.destroyAllWindows()
