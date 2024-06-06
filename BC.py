import numpy as np
import pygame
import sys
import csv

# デモンストレーションデータの読み込み
demo_data = []
with open("mouse_trajectory.csv", mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = map(int, row)
        demo_data.append((x, y))

# データの前処理
X_train = np.array(demo_data[:-1])  # 入力: 最後の点を除いた全ての点
y_train = np.array(demo_data[1:])  # 出力: 全ての点の次の点

# モデルの作成と訓練（線形回帰を例として）
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# 点の初期位置
start_point = X_train[0]

# 点の初期位置から始めて永久に点を回し続ける
current_point = start_point
points = [current_point]

pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 予測して次の点を追加
    next_point = model.predict([current_point])[0]
    points.append(next_point)
    current_point = next_point

    # 線を描画
    if len(points) > 1:
        pygame.draw.lines(screen, (255, 255, 255), False, points, 5)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
