import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None


def get_random_node():
    return Node(np.random.uniform(0, 100), np.random.uniform(0, 100))


def distance(n1, n2):
    return np.hypot(n1.x - n2.x, n1.y - n2.y)


def nearest(nodes, random_node):
    return min(nodes, key=lambda node: distance(node, random_node))


def steer(from_node, to_node, extend_length=float("inf")):
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
    for node in nodes:
        if distance(node, new_node) < radius:
            new_cost = new_node.cost + distance(node, new_node)
            if new_cost < node.cost:
                node.parent = new_node
                node.cost = new_cost


def is_collision(node1, node2, obstacles):
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


def dynamic_rrt(start, goal, iterations, max_radius, obstacles, goal_sample_rate=5):
    nodes = [start]
    for _ in range(iterations):
        # 随机选择目标或生成一个随机节点
        if np.random.rand() > goal_sample_rate / 100.0:
            random_node = get_random_node()
        else:
            random_node = goal
        nearest_node = nearest(nodes, random_node)
        distance_to_nearest = distance(nearest_node, random_node)
        # 根据环境特征动态调整节点扩展的步长
        # 这里可以根据不同的条件动态调整步长，比如障碍物的密集程度、最近节点到目标的距离等
        extend_length = min(distance_to_nearest, max_radius)
        # 在限定距离内朝着随机节点扩展
        new_node = steer(nearest_node, random_node, extend_length)
        if is_collision(nearest_node, new_node, obstacles):
            continue
        # 更新成本并连接新节点到树
        new_node.cost = nearest_node.cost + distance(nearest_node, new_node)
        nodes.append(new_node)
        rewire(nodes, new_node, max_radius)

        if distance(new_node, goal) <= max_radius:
            final_node = steer(new_node, goal)
            if not is_collision(new_node, final_node, obstacles):
                final_node.cost = new_node.cost + distance(new_node, final_node)
                final_node.parent = new_node
                nodes.append(final_node)
                rewire(nodes, final_node, max_radius)
                if distance(final_node, goal) == 0:
                    break
    return nodes



def extract_path(node):
    path = []
    while node:
        path.append(node)
        node = node.parent
    return path[::-1]


def plot_path(nodes, start, goal, obstacles):
    plt.figure()
    for node in nodes:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "g-")

    final_path = extract_path(nodes[-1]) if distance(nodes[-1], goal) == 0 else []

    if final_path:
        plt.plot([node.x for node in final_path], [node.y for node in final_path], "r-", linewidth=2)

    plt.plot(start.x, start.y, "ro")
    plt.plot(goal.x, goal.y, "ro")

    for (ox, oy, size) in obstacles:
        circle = plt.Circle((ox, oy), size, color='b', fill=True)
        plt.gca().add_patch(circle)

    plt.grid(True)
    plt.axis("equal")
    plt.show()


start = Node(0, 0)
goal = Node(100, 100)
iterations = 1000
max_radius = 10.0  # 修改最大半径以适应更多障碍物
goal_sample_rate = 5

# 定义障碍物
obstacles = [(20, 20, 5), (50, 50, 10), (80, 80, 5), (30, 70, 8), (70, 30, 8), (60, 20, 5)]

nodes = dynamic_rrt(start, goal, iterations, max_radius, obstacles, goal_sample_rate)
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
