import pygame
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)

screen = pygame.display.set_mode((900, 700))
screen.fill(WHITE)


points = []
flags = []

eps = 50  # Радиус окрестности
min_samples = 3  # Минимальное количество соседей

def dbscan(points, eps, min_samples):
    labels = [0] * len(points)  # 0 означает не посещенную точку
    cluster_id = 0

    def region_query(point):
        neighbors = []
        for i in range(len(points)):
            if np.linalg.norm(np.array(point) - np.array(points[i])) <= eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(point_id, neighbors, cluster_id):
        labels[point_id] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_point_id = neighbors[i]
            if labels[neighbor_point_id] == 0:
                labels[neighbor_point_id] = cluster_id
                neighbor_neighbors = region_query(points[neighbor_point_id])
                if len(neighbor_neighbors) >= min_samples:
                    neighbors += neighbor_neighbors
            i += 1

    for i in range(len(points)):
        if labels[i] == 0:
            neighbors = region_query(points[i])
            if len(neighbors) >= min_samples:
                cluster_id += 1
                expand_cluster(i, neighbors, cluster_id)

    return labels

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
                # Выдать флажки по алгоритму DBSCAN
                labels = dbscan(points, eps, min_samples)
                unique_colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(max(labels)+1)]

                for i, label in enumerate(labels):
                    if label == 0:
                        flags[i] = RED  # Шумовые точки (красный)
                    else:
                        flags[i] = unique_colors[label]

                for i, point in enumerate(points):
                    pygame.draw.circle(screen, flags[i], point, 5)
                pygame.display.flip()

pygame.quit()
