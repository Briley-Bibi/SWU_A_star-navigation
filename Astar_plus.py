import numpy as np
import heapq

class Node:
    def __init__(self, parent=None, position=None, env_quality=0):
        self.parent = parent
        self.position = position
        self.g = 0  # 距离成本
        self.h = 0  # 启发式成本
        self.f = 0  # 总成本
        self.environmental_quality = env_quality  # 环境质量指数

    def __lt__(self, other):
        return self.f < other.f


def find_nearest_walkable(maze, start):
    from collections import deque
    queue = deque([start])
    visited = set()
    visited.add(start)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        current = queue.popleft()
        if maze[current[0], current[1]] != 0:
            return current
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1] and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return start  # Fallback to the original start if no walkable space is found, though unlikely

def astar(maze, start, end, landscape_mask, alpha, beta):
    start = (round(start[0]), round(start[1]))
    end = (round(end[0]), round(end[1]))

    if maze[start[0], start[1]] == 0:
        start = find_nearest_walkable(maze, start)
    if maze[end[0], end[1]] == 0:
        end = find_nearest_walkable(maze, end)

    start_node = Node(None, start, landscape_mask[start[0], start[1]])
    end_node = Node(None, end, landscape_mask[end[0], end[1]])
    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    closed_list = set()

    # 归一化处理，如果您希望允许用户自由控制这两个参数，可以考虑移除这一行
    alpha, beta = alpha / (alpha + beta), beta / (alpha + beta)

    while open_list:
        current_node = heapq.heappop(open_list)[1]
        closed_list.add(current_node.position)

        if current_node.position == end_node.position:
            return reconstruct_path(current_node)

        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            node_position = (current_node.position[0] + direction[0], current_node.position[1] + direction[1])

            if 0 <= node_position[0] < maze.shape[0] and 0 <= node_position[1] < maze.shape[1] and not (node_position in closed_list) and maze[node_position[0], node_position[1]] != 0:
                new_node = Node(current_node, node_position, landscape_mask[node_position[0], node_position[1]])
                new_node.g = current_node.g + 1
                new_node.h = np.linalg.norm(np.array(end_node.position) - np.array(new_node.position))
                new_node.f = alpha * new_node.g + beta * (1 - new_node.environmental_quality) * new_node.h
                if all(new_node.position != n[1].position or new_node.f < n[1].f for n in open_list):
                    heapq.heappush(open_list, (new_node.f, new_node))

    return []

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.position)
        node = node.parent
    return path[::-1]