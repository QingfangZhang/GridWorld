import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('TkAgg')
matplotlib.rcParams['figure.max_open_warning'] = 100


class GridWorld:
    def __init__(self, height, width):
        self.height, self.width = height, width
        self.start_row = 1 #self.height//2
        self.start_col = 3 #self.width//2
        self.cur_row = self.start_row
        self.cur_col = self.start_col
        self.target_row = 3
        self.target_col = 2
        self.forbidden_pixels = [(1, 1), (1, 2), (2, 2), (3, 1), (4, 1), (3, 3)]
        self.bg_color = 0.2
        self.forbidden_color = 1
        self.agent_color = 0.8
        self.grid = np.ones((self.height, self.width), dtype=int) * self.bg_color
        # self.grid[self.cur_row, self.cur_col] = self.agent_color
        for idx in self.forbidden_pixels:
            self.grid[idx[0], idx[1]] = self.forbidden_color
        # self.grid[self.target_row, self.target_col] = self.target_color
        self.r_target = 1
        self.r_boundary = -1
        self.r_default = 0
        self.r_forbidden = -1
        self.arrow_x, self.arrow_y = 0, 0
        self.prev_col, self.prev_row = self.cur_col, self.cur_row
        self.t = 0
        self.discount_factor = 0.9
        self.state = np.zeros((self.height*self.width, 1), dtype=float)
        self.actions = ['up', 'right', 'down', 'left', 'center']

        fig, self.ax = plt.subplots()
        self.bg_plot(self.ax, self.grid)
        self.ax.set_title(self.t)
        # self.ax.add_artist(plt.Rectangle((self.cur_col - 0.5, self.cur_row - 0.5), 1, 1, facecolor='gray'))
        # self.ax.add_artist(plt.Circle((self.cur_col, self.cur_row), 0.2, color='red'))
        # plt.ion() # 开启交互式绘图，使图形窗口保持打开状态
        plt.draw()  # 重新绘制网格世界和智能体位置
        plt.pause(0.5)  # 暂停0.5秒

    def bg_plot(self, ax, grid):
        ax.clear()
        ax.imshow(grid, vmin=0, vmax=1)
        ax.add_artist(
            plt.Rectangle((self.target_col - 0.5, self.target_row - 0.5), 1, 1, facecolor='pink', edgecolor='black'))
        ax.set_xlim([-0.5, self.width - 0.5])
        ax.set_ylim([self.height - 0.5, -0.5])
        ax.set_aspect('equal')
        ax.set_xticks(range(self.width), [])
        ax.set_yticks(range(self.height), [])
        ax.vlines(np.arange(0.5, self.width - 0.5), -0.5, self.height - 0.5, color='black')
        ax.hlines(np.arange(0.5, self.height - 0.5), -0.5, self.width - 0.5, color='black')

    def reset(self):
        self.cur_row = self.start_row
        self.cur_col = self.start_col

    def check_area(self):
        if self.cur_row == self.target_row and self.cur_col == self.target_col:
            reward = self.r_target
        else:
            reward = self.r_default
        return reward

    def step(self, action):
        self.t += 1
        self.grid[self.cur_row, self.cur_col] = self.bg_color  # 将先前的位置改为0
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
        if action == 'center':
            self.arrow_x, self.arrow_y = None, None
            reward = self.check_area()
        self.grid[self.cur_row, self.cur_col] = self.agent_color  # 更新智能体新的位置
        self.update_plot()
        return reward

    def get_reward_pixel(self, reward_table, pixel_row, pixel_col, reward):
        if pixel_row != 0:
            reward_table[2, pixel_row-1, pixel_col] = reward
        if pixel_row != (self.height - 1):
            reward_table[0, pixel_row+1, pixel_col] = reward
        if pixel_col != 0:
            reward_table[1, pixel_row, pixel_col - 1] = reward
        if pixel_col != (self.width - 1):
            reward_table[3, pixel_row, pixel_col + 1] = reward
        return reward_table

    def get_reward_table(self):
        """for actions dimension: up, right, down, left, center"""
        reward_table = np.ones((len(self.actions), self.height, self.width)) * self.r_default
        reward_table[0, 0, :] = self.r_boundary
        reward_table[2, -1, :] = self.r_boundary
        reward_table[3, :, 0] = self.r_boundary
        reward_table[1, :, -1] = self.r_boundary
        reward_table[4, self.target_row, self.target_col] = self.r_target
        reward_table = self.get_reward_pixel(reward_table, self.target_row, self.target_col, self.r_target)
        for idx in self.forbidden_pixels:
            reward_table = self.get_reward_pixel(reward_table, idx[0], idx[1], self.r_forbidden)
        return reward_table.reshape(len(self.actions), -1)

        # method2
        # reward_table1 = reward_table
        # tic1 = time.time()
        # reward_table = np.ones((len(self.actions), self.height, self.width)) * self.r_default
        # for i in range(self.height):
        #     for j in range(self.width):
        #         if i == 0:
        #             reward_table[0, i, j] = -1
        #         if i == self.height - 1:
        #             reward_table[2, i, j] = -1
        #         if j == 0:
        #             reward_table[3, i, j] = -1
        #         if j == self.width - 1:
        #             reward_table[1, i, j] = -1
        #         if i == self.target_row and j == self.target_col:
        #             reward_table[4, i, j] = 1
        #         if i == self.target_row and j == self.target_col-1:
        #             reward_table[1, i, j] = 1
        #         if i == self.target_row and j == self.target_col+1:
        #             reward_table[3, i, j] = 1
        #         if i == self.target_row-1 and j == self.target_col:
        #             reward_table[2, i, j] = 1
        #         if i == self.target_row+1 and j == self.target_col:
        #             reward_table[0, i, j] = 1
        # toc1 = time.time()
        # print(toc-tic, toc1-tic1)

    def get_state_transit_prob(self):
        """state_transit_prob shape: actions x (height x width) x (height x width)
           for actions dimension: up, right, down, left, center
           state dimension: first row then second row ...
        """
        state_transit_prob = np.zeros((len(self.actions), self.height * self.width, self.height * self.width))
        state_transit_prob[4, np.arange(self.height*self.width), np.arange(self.height*self.width)] = 1
        state_transit_prob[0, np.arange(self.width), np.arange(self.width)] = 1   # first row, up action transit to itself
        state_transit_prob[0, np.arange(self.width, self.height*self.width),
                           np.arange(self.width, self.height*self.width) - self.width] = 1
        state_transit_prob[2, np.arange(self.height*self.width - self.width), np.arange(self.width, self.height*self.width)] = 1
        state_transit_prob[2, np.arange(self.height*self.width - self.width, self.height*self.width),
                           np.arange(self.height*self.width - self.width, self.height*self.width)] = 1

        mask = np.arange(self.width-1, self.height*self.width, self.width)
        other_idx = np.delete(np.arange(self.height*self.width), mask)
        state_transit_prob[1, mask, mask] = 1
        state_transit_prob[1, other_idx, other_idx+1] = 1

        mask = np.arange(0, self.height*self.width, self.width)
        other_idx = np.delete(np.arange(self.height*self.width), mask)
        state_transit_prob[3, mask, mask] = 1
        state_transit_prob[3, other_idx, other_idx-1] = 1
        return state_transit_prob

    def value_iteration(self):
        reward = self.get_reward_table()
        state_transit_prob = self.get_state_transit_prob()
        k = 0
        state_value_list = [self.state]
        # self.plot_policy_and_state(None, self.state, k)
        while True and k < 200:
            action_value = reward + self.discount_factor * np.matmul(state_transit_prob,
                                                                  np.tile(self.state, (len(self.actions), 1, 1))).squeeze()
            policy = np.argmax(action_value, axis=0)
            new_state = np.amax(action_value, axis=0).reshape(-1, 1)
            k += 1
            state_value_list.append(new_state)
            # self.plot_policy_and_state(policy, new_state, k)
            if np.linalg.norm(new_state-self.state) < 0.001:
                self.state = new_state
                break
            else:
                self.state = new_state
        self.plot_policy_and_state(policy, self.state, k)
        plt.figure()
        state_value = np.concatenate(state_value_list, axis=1)
        s = np.diff(state_value, axis=1)
        plt.plot(np.linalg.norm(s, axis=0))
        plt.show()

    def plot_policy_and_state(self, policy, state, k):
        fig1, ax1=plt.subplots(1, 2, figsize=(10, 5))
        grid1 = np.ones((self.height, self.width)) * self.bg_color
        self.bg_plot(ax1[0], self.grid)
        ax1[0].set_title(f'Policy, k={k}')
        if policy is not None:
            policy = policy.reshape((self.height, self.width))
            # 生成 x 和 y 矩阵，分别表示每个点的横向和纵向坐标
            x, y = np.meshgrid(np.arange(policy.shape[1]), np.arange(policy.shape[0]))
            u = np.zeros_like(policy)
            v = np.zeros_like(policy)
            for i in range(policy.shape[0]):
                for j in range(policy.shape[1]):
                    if self.actions[policy[i, j]] == 'up':
                        u[i, j] = 0
                        v[i, j] = 1
                    elif self.actions[policy[i, j]] == 'down':
                        u[i, j] = 0
                        v[i, j] = -1
                    elif self.actions[policy[i, j]] == 'left':
                        u[i, j] = -1
                        v[i, j] = 0
                    elif self.actions[policy[i, j]] == 'right':
                        u[i, j] = 1
                        v[i, j] = 0
                    else:  # center
                        u[i, j] = 0
                        v[i, j] = 0
            ax1[0].quiver(x, y, u, v, scale=10, color='red')

        self.bg_plot(ax1[1], self.grid)
        ax1[1].set_title(f'state value, k={k}')
        state = state.reshape((self.height, self.width))
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                ax1[1].text(j, i, "{:.3f}".format(state[i, j]), ha='center', va='center', color='k')

    def update_plot(self):
        self.bg_plot(self.ax, self.grid)
        # self.ax.add_artist(plt.Circle((self.cur_col, self.cur_row), 0.2, color='red'))
        if self.arrow_x is not None:
            self.ax.quiver(self.prev_col, self.prev_row, self.arrow_x, self.arrow_y, color='blue', scale=4)
        else:
            self.ax.add_artist(plt.Circle((self.prev_col, self.prev_row), 0.1, color='blue'))
        self.ax.set_title(self.t)
        plt.draw()  # 重新绘制网格世界和智能体位置
        plt.pause(1)  # 暂停0.5秒


class Agent:
    def __init__(self):
        self.actions = ['up', 'right', 'down', 'left', 'center']

    def act(self, state):

        action = np.random.choice(self.actions)
        return action


env = GridWorld(5, 5)
env.value_iteration()
# agent = Agent()
# for t in range(10):
#     env.step(agent.act(None))
# plt.show()
