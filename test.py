import numpy as np
import matplotlib.pyplot as plt

# 创建网格世界并随机初始化智能体位置
h = 10
w = 5
world = np.zeros((h, w), dtype=int)
agent_row, agent_col = h//2, w//2
world[agent_row, agent_col] = 1


def init_plot(world, h, w, agent_row, agent_col):
    fig, ax = plt.subplots()
    for i in range(h):
        for j in range(w):
            if world[i, j] == 0:
                ax.add_artist(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False))
            else:
                ax.add_artist(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='gray', edgecolor='black'))
            if i == agent_row and j == agent_col:
                ax.add_artist(plt.Circle((j, i), 0.2, color='red'))

    ax.set_xlim([-0.5, w - 0.5])
    ax.set_ylim([-0.5, h - 0.5])
    ax.set_aspect('equal')
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.grid(True)
    # plt.ion() # 开启交互式绘图，使图形窗口保持打开状态
    plt.draw() # 重新绘制网格世界和智能体位置
    plt.pause(0.5) # 暂停0.5秒
    return ax


def update_plot(world, ax, action, agent_row, agent_col):
    h, w = world.shape
    r_boundary = -1
    update = True
    world[agent_row, agent_col] = 0  # 将先前的位置改为 _
    ax.add_artist(plt.Rectangle((agent_col - 0.5, agent_row - 0.5), 1, 1, facecolor='white', edgecolor='black'))
    if action == 'down':
        if agent_row == 0:
            update = False
            reward = r_boundary
        else:
            agent_row -= 1
    if action == 'up':
        if agent_row == h - 1:
            update = False
            reward = r_boundary
        else:
            agent_row += 1
    if action == 'left':
        if agent_col == 0:
            update = False
            reward = r_boundary
        else:
            agent_col -= 1
    if action == 'right':
        if agent_col == w - 1:
            update = False
            reward = r_boundary
        else:
            agent_col += 1

    world[agent_row, agent_col] = 1  # 更新智能体新的位置
    ax.add_artist(plt.Rectangle((agent_col - 0.5, agent_row - 0.5), 1, 1, facecolor='gray', edgecolor='black'))
    ax.add_artist(plt.Circle((agent_col, agent_row), 0.2, color='red'))
    plt.draw() # 重新绘制网格世界和智能体位置
    plt.pause(0.5) # 暂停0.5秒
    return agent_row, agent_col


ax = init_plot(world, h, w, agent_row, agent_col)
agent_row, agent_col = update_plot(world, ax, 'left', agent_row, agent_col)
agent_row, agent_col = update_plot(world, ax, 'up', agent_row, agent_col)
agent_row, agent_col = update_plot(world, ax, 'up', agent_row, agent_col)
# plt.ioff() # 关闭交互式绘图
plt.show() # 显示最终的网格世界和智能体状态
