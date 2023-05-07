import numpy as np
import matplotlib.pyplot as plt


class GridWorld:
    def __init__(self, height, width):
        self.height, self.width = height, width
        self.start_row = 1 #self.height//2
        self.start_col = 3 #self.width//2
        self.cur_row = self.start_row
        self.cur_col = self.start_col
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.grid[self.start_row, self.start_col] = 1
        self.goal_row = self.height-1
        self.goal_col = self.width-1
        self.r_target = 1
        self.r_boundary = -1
        self.r_default = 0
        self.arrow_x, self.arrow_y = 0, 0
        self.prev_col, self.prev_row = self.cur_col, self.cur_row
        self.t = 0
        self.discount_factor = 0.9
        self.state = np.zeros((self.height, self.width), dtype=float)
        self.actions = ['up', 'right', 'down', 'left', 'keep']

        fig, self.ax = plt.subplots()
        self.bg_plot()
        self.ax.set_title(self.t)
        self.ax.add_artist(plt.Rectangle((self.cur_col - 0.5, self.cur_row - 0.5), 1, 1, facecolor='gray'))
        self.ax.add_artist(plt.Circle((self.cur_col, self.cur_row), 0.2, color='red'))
        # ax.grid(True)
        # plt.ion() # 开启交互式绘图，使图形窗口保持打开状态
        plt.draw()  # 重新绘制网格世界和智能体位置
        plt.pause(0.5)  # 暂停0.5秒

    def bg_plot(self):
        self.ax.clear()
        for i in range(self.height):
            for j in range(self.width):
                self.ax.add_artist(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False))
        self.ax.add_artist(
            plt.Rectangle((self.goal_col - 0.5, self.goal_row - 0.5), 1, 1, facecolor='pink', edgecolor='black'))
        self.ax.set_xlim([-0.5, self.width - 0.5])
        self.ax.set_ylim([-0.5, self.height - 0.5])
        self.ax.set_aspect('equal')
        self.ax.set_xticks(range(self.width))
        self.ax.set_yticks(range(self.height))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

    def reset(self):
        self.cur_row = self.start_row
        self.cur_col = self.start_col

    def check_area(self):
        if self.cur_row == self.goal_row and self.cur_col == self.goal_col:
            reward = self.r_target
        else:
            reward = self.r_default
        return reward

    def step(self, action):
        self.t += 1
        self.grid[self.cur_row, self.cur_col] = 0  # 将先前的位置改为0
        self.prev_col, self.prev_row = self.cur_col, self.cur_row
        if action == 'down':
            self.arrow_x, self.arrow_y = 0, -1
            if self.cur_row == 0:
                reward = self.r_boundary
            else:
                self.cur_row -= 1
                reward = self.check_area()
        if action == 'up':
            self.arrow_x, self.arrow_y = 0, 1
            if self.cur_row == self.height - 1:
                reward = self.r_boundary
            else:
                self.cur_row += 1
                reward = self.check_area()
        if action == 'left':
            self.arrow_x, self.arrow_y = -1, 0
            if self.cur_col == 0:
                reward = self.r_boundary
            else:
                self.cur_col -= 1
                reward = self.check_area()
        if action == 'right':
            self.arrow_x, self.arrow_y = 1, 0
            if self.cur_col == self.width - 1:
                reward = self.r_boundary
            else:
                self.cur_col += 1
                reward = self.check_area()
        if action == 'keep':
            self.arrow_x, self.arrow_y = None, None
            reward = self.check_area()
        self.grid[self.cur_row, self.cur_col] = 1  # 更新智能体新的位置
        self.update_plot()
        return reward

    def reward_table(self):
        """for actions dimension: up, right, down, left, keep"""
        reward_table = np.ones((self.height, self.width, len(self.actions))) * self.r_default
        reward_table[0, :, 0] = self.r_boundary
        reward_table[-1, :, 2] = self.r_boundary
        reward_table[:, 0, 3] = self.r_boundary
        reward_table[:, -1, 1] = self.r_boundary
        reward_table[self.goal_row, self.goal_col, 5] = self.r_target
        if self.goal_row != (self.height - 1):
            reward_table[self.goal_row-1, self.goal_col, 0] = self.r_target
        if self.goal_row != 0:
            reward_table[self.goal_row, self.goal_col, 1] = self.r_target

    def update_plot(self):
        self.bg_plot()
        self.ax.add_artist(plt.Rectangle((self.cur_col - 0.5, self.cur_row - 0.5), 1, 1, facecolor='gray', edgecolor='black'))
        self.ax.add_artist(plt.Circle((self.cur_col, self.cur_row), 0.2, color='red'))
        if self.arrow_x is not None:
            self.ax.quiver(self.prev_col, self.prev_row, self.arrow_x, self.arrow_y, color='blue', scale=4)
        else:
            self.ax.add_artist(plt.Circle((self.prev_col, self.prev_row), 0.1, color='blue'))
        self.ax.set_title(self.t)
        plt.draw()  # 重新绘制网格世界和智能体位置
        plt.pause(1)  # 暂停0.5秒


class Agent:
    def __init__(self):
        self.actions = ['up', 'right', 'down', 'left', 'keep']

    def act(self, state):

        action = np.random.choice(self.actions)
        return action


env = GridWorld(5, 5)
agent = Agent()
for t in range(1):
    env.step(agent.act(None))


plt.show()



# # 创建网格世界并随机初始化智能体位置
# h = 10
# w = 5
# world = np.zeros((h, w), dtype=int)
# agent_row, agent_col = h//2, w//2
# world[agent_row, agent_col] = 1
#
#
# def init_plot(world, h, w, agent_row, agent_col):
#     fig, ax = plt.subplots()
#     for i in range(h):
#         for j in range(w):
#             if world[i, j] == 0:
#                 ax.add_artist(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False))
#             else:
#                 ax.add_artist(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='gray', edgecolor='black'))
#             if i == agent_row and j == agent_col:
#                 ax.add_artist(plt.Circle((j, i), 0.2, color='red'))
#
#     ax.set_xlim([-0.5, w - 0.5])
#     ax.set_ylim([-0.5, h - 0.5])
#     ax.set_aspect('equal')
#     ax.set_xticks(range(w))
#     ax.set_yticks(range(h))
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     # ax.grid(True)
#     # plt.ion() # 开启交互式绘图，使图形窗口保持打开状态
#     plt.draw() # 重新绘制网格世界和智能体位置
#     plt.pause(0.5) # 暂停0.5秒
#     return ax
#
#
# def update_plot(world, ax, action, agent_row, agent_col):
#     h, w = world.shape
#     r_boundary = -1
#     update = True
#     world[agent_row, agent_col] = 0  # 将先前的位置改为 _
#     ax.add_artist(plt.Rectangle((agent_col - 0.5, agent_row - 0.5), 1, 1, facecolor='white', edgecolor='black'))
#     if action == 'down':
#         if agent_row == 0:
#             update = False
#             reward = r_boundary
#         else:
#             agent_row -= 1
#     if action == 'up':
#         if agent_row == h - 1:
#             update = False
#             reward = r_boundary
#         else:
#             agent_row += 1
#     if action == 'left':
#         if agent_col == 0:
#             update = False
#             reward = r_boundary
#         else:
#             agent_col -= 1
#     if action == 'right':
#         if agent_col == w - 1:
#             update = False
#             reward = r_boundary
#         else:
#             agent_col += 1
#
#     world[agent_row, agent_col] = 1  # 更新智能体新的位置
#     ax.add_artist(plt.Rectangle((agent_col - 0.5, agent_row - 0.5), 1, 1, facecolor='gray', edgecolor='black'))
#     ax.add_artist(plt.Circle((agent_col, agent_row), 0.2, color='red'))
#     plt.draw() # 重新绘制网格世界和智能体位置
#     plt.pause(0.5) # 暂停0.5秒
#     return agent_row, agent_col
#
#
# ax = init_plot(world, h, w, agent_row, agent_col)
# agent_row, agent_col = update_plot(world, ax, 'left', agent_row, agent_col)
# agent_row, agent_col = update_plot(world, ax, 'up', agent_row, agent_col)
# agent_row, agent_col = update_plot(world, ax, 'up', agent_row, agent_col)
# # plt.ioff() # 关闭交互式绘图
# plt.show() # 显示最终的网格世界和智能体状态
