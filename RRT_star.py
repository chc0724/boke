import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None

def get_random_node():
    # 生成一个随机节点
    return Node(np.random.uniform(0, 100), np.random.uniform(0, 100))

def distance(n1, n2):
    # 计算两个节点之间的距离
    return np.hypot(n1.x - n2.x, n1.y - n2.y)

def nearest(nodes, random_node):
    # 找到最近的节点
    return min(nodes, key=lambda node: distance(node, random_node))

def steer(from_node, to_node, extend_length=float("inf")):
    # 在给定的长度内延伸到目标节点
    dist = distance(from_node, to_node)
    if dist <= extend_length:
        new_node = Node(to_node.x, to_node.y)
    else:
        theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node = Node(from_node.x + extend_length * np.cos(theta),
                        from_node.y + extend_length * np.sin(theta))
    new_node.cost = from_node.cost + distance(from_node, new_node)
    new_node.parent = from_node
    return new_node

def rewire(nodes, new_node, radius):
    # 重新连接已有节点
    for node in nodes:
        if distance(node, new_node) < radius:
            new_cost = new_node.cost + distance(node, new_node)
            if new_cost < node.cost:
                node.parent = new_node
                node.cost = new_cost

def is_collision(node1, node2, obstacles):
    # 检查路径是否与障碍物相交
    for (ox, oy, size) in obstacles:
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        a = dx * dx + dy * dy
        b = 2 * (dx * (node1.x - ox) + dy * (node1.y - oy))
        c = ox * ox + oy * oy
        c += node1.x * node1.x + node1.y * node1.y
        c -= 2 * (ox * node1.x + oy * node1.y)
        c -= size * size
        bb4ac = b * b - 4 * a * c

        if bb4ac < 0:
            continue

        sqrtbb4ac = np.sqrt(bb4ac)
        t1 = (-b + sqrtbb4ac) / (2 * a)
        t2 = (-b - sqrtbb4ac) / (2 * a)

        if 0 <= t1 <= 1 or 0 <= t2 <= 1:
            return True

    return False

def rrt_star(start, goal, iterations, radius, obstacles, goal_sample_rate=5, extend_length=5.0):
    nodes = [start]
    for _ in range(iterations):
        if np.random.rand() > goal_sample_rate / 100.0:
            random_node = get_random_node()
        else:
            random_node = goal

        nearest_node = nearest(nodes, random_node)
        new_node = steer(nearest_node, random_node, extend_length)

        if is_collision(nearest_node, new_node, obstacles):
            continue

        new_node.cost = nearest_node.cost + distance(nearest_node, new_node)
        nodes.append(new_node)
        rewire(nodes, new_node, radius)

        # 检查是否从新节点可以到达目标
        if distance(new_node, goal) <= extend_length:
            final_node = steer(new_node, goal)
            if not is_collision(new_node, final_node, obstacles):
                final_node.cost = new_node.cost + distance(new_node, final_node)
                final_node.parent = new_node
                nodes.append(final_node)
                rewire(nodes, final_node, radius)
                if distance(final_node, goal) == 0:
                    break

    return nodes

def extract_path(node):
    # 提取路径
    path = []
    while node:
        path.append(node)
        node = node.parent
    return path[::-1]

# Visualization and example usage
start = Node(0, 0)
goal = Node(100, 100)
iterations = 1000
radius = 5.0

# 定义障碍物
obstacles = [(20, 20, 5), (50, 50, 10), (80, 80, 5)]

nodes = rrt_star(start, goal, iterations, radius, obstacles)
path = extract_path(nodes[-1]) if distance(nodes[-1], goal) == 0 else []

plt.figure()
for node in nodes:
    if node.parent:
        plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "g-")
if path:
    plt.plot([node.x for node in path], [node.y for node in path], "r-", linewidth=2)
plt.plot(start.x, start.y, "ro")
plt.plot(goal.x, goal.y, "ro")

# 绘制障碍物
for (ox, oy, size) in obstacles:
    circle = plt.Circle((ox, oy), size, color='b', fill=True)
    plt.gca().add_patch(circle)

plt.grid(True)
plt.axis("equal")
plt.show()
