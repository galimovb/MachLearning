import pygame
import numpy as np

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)

screen = pygame.display.set_mode((900, 700))
screen.fill(WHITE)

# Глобальные переменные
points = []  # Список для хранения координат точек
flags = []   # Флажки для точек

# Параметры для алгоритма
eps = 50  # Радиус окрестности
min_samples = 3  # Минимальное количество соседей

# Функция для отображения точек
def draw_points():
    for i, point in enumerate(points):
        pygame.draw.circle(screen, flags[i], point, 5)
    pygame.display.flip()

# Функция для подсчета соседей каждой точки
def count_neighbors(point_index):
    green_neighbor = False
    count = 0
    for i in range(len(points)):
        if i != point_index and np.linalg.norm(np.array(points[point_index]) - np.array(points[i])) <= 50:
            count += 1
            if flags[i] == GREEN:
                green_neighbor = True
    return count, green_neighbor


run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Левая кнопка мыши
                pos = pygame.mouse.get_pos()
                points.append(pos)
                flags.append(BLACK)
                draw_points()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:  # Клавиша Enter
                for i in range(len(points)):
                    neighbors_count, has_green_neighbor = count_neighbors(i)
                    if has_green_neighbor and 1 <= neighbors_count <= 2:
                        flags[i] = YELLOW
                    elif neighbors_count >= min_samples:
                        flags[i] = GREEN
                    else:
                        flags[i] = RED
                draw_points()

pygame.quit()
