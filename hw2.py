import pygame
import numpy as np
from sklearn.cluster import KMeans

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

                for i in range(len(points)):
                    green_neighbor = False
                    count = 0
                    for j in range(len(points)):
                        if j != i and np.linalg.norm(np.array(points[i]) - np.array(points[j])) <= eps:
                            count += 1
                            if flags[j] == GREEN:
                                green_neighbor = True

                    if green_neighbor and 1 <= count <= 2:
                        flags[i] = YELLOW
                    elif count >= min_samples:
                        flags[i] = GREEN
                    else:
                        flags[i] = RED

                for i, point in enumerate(points):
                    pygame.draw.circle(screen, flags[i], point, 5)
                pygame.display.flip()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:  # Нажатие клавиши Enter
                unique_colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(len(set(flags)))]

                X = np.array(points)
                kmeans = KMeans(n_clusters=len(set(flags)), random_state=0).fit(X)

                for i, label in enumerate(kmeans.labels_):
                    flags[i] = unique_colors[label]

                for i, point in enumerate(points):
                    pygame.draw.circle(screen, flags[i], point, 5)
                pygame.display.flip

pygame.quit()