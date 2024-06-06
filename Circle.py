import pygame
import sys
import csv

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    running = True
    points = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                points = [pygame.mouse.get_pos()]

        if len(points) > 0:
            points.append(pygame.mouse.get_pos())
            pygame.draw.lines(screen, (255, 255, 255), False, points, 5)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    # 保存するファイル名
    filename = "mouse_trajectory.csv"

    # マウスの座標をCSVファイルに保存
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for point in points:
            writer.writerow(point)

if __name__ == '__main__':
    main()
