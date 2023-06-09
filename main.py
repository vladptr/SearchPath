import cv2
import numpy as np
import heapq


class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f


def find_yellow_and_red_circles(image_path):

    image = cv2.imread(image_path)


    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])


    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])


    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)


    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)


    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)


    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    yellow_centers = []
    for contour in yellow_contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        yellow_centers.append(center)


    red_centers = []
    for contour in red_contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        red_centers.append(center)


    graph = {}
    all_centers = yellow_centers + red_centers
    for center in all_centers:
        graph[center] = []


    for i, center in enumerate(all_centers):
        distances = []
        for j, other_center in enumerate(all_centers):
            if i != j:
                distance = np.linalg.norm(np.array(center) - np.array(other_center))
                distances.append((distance, other_center))
        distances.sort()
        for distance, other_center in distances[:2]:
            graph[center].append(other_center)


    start = yellow_centers[0]
    end = yellow_centers[-1]
    path = astar(graph, start, end)


    for i in range(len(path) - 1):
        cv2.line(image, path[i], path[i + 1], (0, 255, 0), 2)

    #вивід шляху в вікно
    cv2.imshow('Path', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def heuristic(node, goal):
    return np.linalg.norm(np.array(node.position) - np.array(goal.position))


def astar(graph, start, end):
    open_list = []
    closed_list = []

    start_node = Node(start)
    goal_node = Node(end)

    heapq.heappush(open_list, (start_node.f, start_node))
    while open_list:
        current_node = heapq.heappop(open_list)[1]

        if current_node.position == goal_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_list.append(current_node)

        for neighbor_position in graph[current_node.position]:
            neighbor_node = Node(neighbor_position, current_node)

            if neighbor_node in closed_list:
                continue

            neighbor_node.g = current_node.g + heuristic(neighbor_node, current_node)
            neighbor_node.h = heuristic(neighbor_node, goal_node)
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            if neighbor_node in open_list:
                open_node = next((node for _, node in open_list if node == neighbor_node), None)
                if neighbor_node.g < open_node.g:
                    open_list.remove((open_node.f, open_node))
                    heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
            else:
                heapq.heappush(open_list, (neighbor_node.f, neighbor_node))

    return None


image_path = "Q:\\SCreens\\aeroimg3.jpg"
find_yellow_and_red_circles(image_path)