"""
要在 PyCharm 中可视化智能体在网格世界中的移动，您需要创建一个循环来迭代模拟时间步骤。在每个时间步骤中，您可以更新智能体的位置并重新绘制网格世界和智能体位置。

以下是一些示例代码，可以帮助您开始：
"""
import numpy as np
import matplotlib.pyplot as plt

# 创建网格世界并随机初始化智能体位置
world = [['_' for i in range(10)] for j in range(10)]
agent_row, agent_col = 5, 5
world[agent_row][agent_col] = 'A'

# world_size = 10
# world = np.zeros(world_size, world_size)
# agent_row, agent_col = world_size//2, world_size//2
# world[agent_row, agent_col] = 1

# 绘制网格世界和智能体位置
fig, ax = plt.subplots()
for i in range(len(world)):
    for j in range(len(world[0])):
        if world[i][j] == '_':
            ax.add_artist(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False))
        else:
            ax.add_artist(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='gray', edgecolor='black'))
        if i == agent_row and j == agent_col:
            ax.add_artist(plt.Circle((j, i), 0.2, color='red'))

ax.set_xlim([-0.5, len(world[0]) - 0.5])
ax.set_ylim([-0.5, len(world) - 0.5])
ax.set_aspect('equal')
ax.set_xticks(range(len(world[0])))
ax.set_yticks(range(len(world)))
ax.set_xticklabels([])
ax.set_yticklabels([])
# ax.grid(True)
# plt.ion() # 开启交互式绘图，使图形窗口保持打开状态
#
# 模拟智能体向上移动5步然后向右移动3步的过程
for t in range(5):
    # world[agent_row][agent_col] = '_' # 将先前的位置改为 _
    # agent_row -= 1
    # world[agent_row][agent_col] = 'A' # 更新智能体新的位置
    # ax.clear() # 清除之前的网格世界和智能体状态
    # for i in range(len(world)):
    #     for j in range(len(world[0])):
    #         if world[i][j] == '_':
    #             ax.add_artist(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False))
    #         else:
    #             ax.add_artist(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='gray', edgecolor='black'))
    #         if i == agent_row and j == agent_col:
    #             ax.add_artist(plt.Circle((j, i), 0.2, color='red'))
    world[agent_row][agent_col] = '_'  # 将先前的位置改为 _
    ax.add_artist(plt.Rectangle((agent_col - 0.5, agent_row - 0.5), 1, 1, facecolor='white', edgecolor='black'))
    agent_row -= 1
    world[agent_row][agent_col] = 'A'  # 更新智能体新的位置
    ax.add_artist(plt.Rectangle((agent_col - 0.5, agent_row - 0.5), 1, 1, facecolor='gray', edgecolor='black'))
    ax.add_artist(plt.Circle((agent_col, agent_row), 0.2, color='red'))

    # ax.set_xlim([-0.5, len(world[0]) - 0.5])
    # ax.set_ylim([-0.5, len(world) - 0.5])
    # ax.set_aspect('equal')
    # ax.set_xticks(range(len(world[0])))
    # ax.set_yticks(range(len(world)))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.grid(True)
    plt.draw() # 重新绘制网格世界和智能体位置
    plt.pause(0.5) # 暂停0.5秒
#
# for t in range(3):
#     world[agent_row][agent_col] = '_'
#     agent_col += 1
#     world[agent_row][agent_col] = 'A'
#     ax.clear()
#     for i in range(len(world)):
#         for j in range(len(world[0])):
#             if world[i][j] == '_':
#                 ax.add_artist(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False))
#             else:
#                 ax.add_artist(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='gray', edgecolor='black'))
#             if i == agent_row and j == agent_col:
#                 ax.add_artist(plt.Circle((j, i), 0.2, color='red'))
#
#     ax.set_xlim([-0.5, len(world[0]) - 0.5])
#     ax.set_ylim([-0.5, len(world) - 0.5])
#     ax.set_aspect('equal')
#     ax.set_xticks(range(len(world[0])))
#     ax.set_yticks(range(len(world)))
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.grid(True)
#     plt.draw()
#     plt.pause(0.5)
#
# plt.ioff() # 关闭交互式绘图
plt.show() # 显示最终的网格世界和智能体状态

"""
这段示例代码将模拟一个代理在网格世界中向上移动5步，然后向右移动3步的过程，并显示其轨迹。请注意 `plt.ion()` 和 `plt.ioff()` 命令开启和关闭交互式绘图，使图形窗口保持打开状态，并在每个时间步骤中使用 `ax.clear()` 清除之前的网格世界和智能体位置，然后在更新后重新绘制它们。

希望这可以帮助您开始在 PyCharm 中可视化智能体的移动。
"""
