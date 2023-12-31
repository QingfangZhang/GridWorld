import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import time
import copy
import torch
from torch import nn
import torch.nn.functional as F
matplotlib.use('TkAgg')
matplotlib.rcParams['figure.max_open_warning'] = 100


class GridWorld:
    def __init__(self, height, width, target_pixels, forbidden_pixels, r_target, r_boundary, r_default, r_forbidden,
                 start_pos=(0, 0)):
        self.height, self.width = height, width
        self.start_row = start_pos[0]
        self.start_col = start_pos[1]
        self.cur_row = self.start_row
        self.cur_col = self.start_col
        self.target_row = target_pixels[0]
        self.target_col = target_pixels[1]
        self.forbidden_pixels = forbidden_pixels  # 以top left作为(0, 0), (row, col)
        self.bg_color = 0.2
        self.forbidden_color = 1
        self.agent_color = 0.8
        self.grid = np.ones((self.height, self.width), dtype=int) * self.bg_color
        # self.grid[self.cur_row, self.cur_col] = self.agent_color
        for idx in self.forbidden_pixels:
            self.grid[idx[0], idx[1]] = self.forbidden_color
        # self.grid[self.target_row, self.target_col] = self.target_color
        self.r_target = r_target
        self.r_boundary = r_boundary
        self.r_default = r_default
        self.r_forbidden = r_forbidden
        self.arrow_x, self.arrow_y = 0, 0
        self.prev_col, self.prev_row = self.cur_col, self.cur_row
        self.t = 0
        self.discount_factor = 0.9
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
            if self.cur_row == self.height - 1:
                reward = self.r_boundary
            else:
                self.cur_row += 1
                reward = self.check_area()
        if action == 'up':
            self.arrow_x, self.arrow_y = 0, 1
            if self.cur_row == 0:
                reward = self.r_boundary
            else:
                self.cur_row -= 1
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
            reward_table[4, idx[0], idx[1]] = self.r_forbidden
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
        reward = self.get_reward_table()  # (n_actions, height * weight)
        state_transit_prob = self.get_state_transit_prob()  # (n_actions, (height * width), (height * width))
        k = 0
        v0 = np.zeros((self.height * self.width, 1), dtype=float)
        state_value_list = [v0]
        # self.plot_policy_and_state(None, v0, k)
        while k < 200:
            action_value = reward + self.discount_factor * \
                           np.matmul(state_transit_prob, np.tile(v0, (len(self.actions), 1, 1))).squeeze()
            greedy_action_idx = np.argmax(action_value, axis=0)
            policy = np.zeros_like(action_value)
            policy[greedy_action_idx, np.arange(policy.shape[1])] = 1  # (n_actions, height * weight)
            v = np.amax(action_value, axis=0).reshape(-1, 1)
            k += 1
            state_value_list.append(v)
            # self.plot_policy_and_state(policy, v, k)
            if np.linalg.norm(v-v0) < 0.001:
                break
            else:
                v0 = v
        self.plot_policy_and_state(policy, v, k)
        plt.figure()
        state_value = np.concatenate(state_value_list, axis=1)
        s = np.diff(state_value, axis=1)
        plt.plot(np.linalg.norm(s, axis=0))

    def policy_iteration(self, ini_policy, truncate_time=10):
        reward = self.get_reward_table()  # (n_actions, height * weight)
        state_transit_prob = self.get_state_transit_prob()  # (n_actions, (height * width), (height * width))
        policy = ini_policy
        state_value_list = []
        k = 0
        pre_v = np.zeros((self.height * self.width, 1))
        self.plot_policy_and_state(policy, None, k)
        while k < 200:
            # policy evaluation
            greedy_action_idx = np.argmax(policy, axis=0)
            r = np.array([reward[greedy_action_idx[s], s] for s in range(policy.shape[1])]).reshape(-1, 1)  # ((height * width), 1)
            s = np.vstack([np.reshape(state_transit_prob[greedy_action_idx[s], s, :], (1, -1))
                           for s in range(policy.shape[1])])  # ((height * width), (height * width))
            v0 = pre_v
            j = 0
            while j < truncate_time:
                v = r + self.discount_factor * np.matmul(s, v0)  # ((height * width), 1)
                if np.linalg.norm((v - v0)) < 0.001:
                    print(f"j is {j}")
                    break
                else:
                    v0 = v
                j += 1
            # policy improvement
            action_value = reward + self.discount_factor * \
                           np.matmul(state_transit_prob, np.tile(v, (len(self.actions), 1, 1))).squeeze()
            greedy_action_idx = np.argmax(action_value, axis=0)
            policy = np.zeros_like(action_value)
            policy[greedy_action_idx, np.arange(policy.shape[1])] = 1  # (n_actions, height * weight)
            # self.plot_policy_and_state(policy, v, k)
            state_value_list.append(v)
            if np.linalg.norm(v - pre_v) < 0.001:
                break
            else:
                pre_v = v
            k += 1
        self.plot_policy_and_state(policy, v, k)
        plt.figure()
        state_value = np.concatenate(state_value_list, axis=1)
        s = np.diff(state_value, axis=1)
        plt.plot(np.linalg.norm(s, axis=0))
        return policy, v, k

    def policy_evaluation(self, policy, truncate_time=10000):
        reward_table = self.get_reward_table()  # (n_actions, height * weight)
        state_transit_prob = self.get_state_transit_prob()  # (n_actions, (height * width), (height * width))
        v0 = np.zeros((self.height * self.width, 1))
        r = np.sum(reward_table * policy, axis=0).reshape(-1, 1)  # ((height * width), 1)
        na, ns, ns = state_transit_prob.shape
        s = np.sum((state_transit_prob.reshape(na*ns, ns) * np.tile(policy.reshape(-1, 1), (1, ns)))
                   .reshape(na, ns, ns), axis=0).squeeze()  # ((height * width), (height * width))
        j = 0
        while j < truncate_time:
            v = r + self.discount_factor * np.matmul(s, v0)  # ((height * width), 1)
            if np.linalg.norm((v - v0)) < 0.001:
                # print(f"policy_evaluation iter num: {j}")
                break
            else:
                v0 = v
            j += 1
        return v  # ((height * width), 1)

    def MC_epsilon_greedy(self, policy):
        e = 0.1
        episode_step = 50000
        n_actions = len(self.actions)
        n_states = self.height * self.width
        assert policy.shape == (n_actions, n_states)  # (n_actions, (height * width))
        reward_table = self.get_reward_table()  # (n_actions, height * weight)
        state_transit_prob = self.get_state_transit_prob()  # (n_actions, (height * width), (height * width))
        k = 0
        action_value_pre = np.zeros((n_actions, n_states))
        pre_v = self.policy_evaluation(policy)
        self.plot_policy_and_state(policy, pre_v, k)
        state_value_list = [pre_v]
        while k < 10000:
            # generate episode
            s = [np.random.randint(0, n_states - 1)]
            a = [np.random.randint(0, n_actions - 1)]
            for i in range(episode_step-1):
                s.append(np.argmax(state_transit_prob[a[-1], s[-1], :].squeeze()))
                a.append(np.random.choice(np.arange(n_actions), p=policy[:, s[-1]].reshape(-1)))
            # for i in range(len(a)):
            #     self.step(self.actions[a[i]])
            # calculate return
            g = 0
            returns = np.zeros((n_actions, n_states))
            counts = np.zeros((n_actions, n_states))
            for i in range(len(s)-1, -1, -1):
                r = reward_table[a[i], s[i]]
                g = r + self.discount_factor * g
                returns[a[i], s[i]] += g
                counts[a[i], s[i]] += 1
            action_value = np.where(counts != 0, np.divide(returns, counts), action_value_pre)
            action_value_pre = action_value
            policy = np.ones((n_actions, n_states)) * (e / n_actions)
            greedy_action_idx = np.argmax(action_value, axis=0)
            policy[greedy_action_idx, np.arange(policy.shape[1])] = 1 - (e / n_actions)*(n_actions-1)
            v = self.policy_evaluation(policy)
            # self.plot_policy_and_state(policy, v, k)
            state_value_list.append(v)
            if np.linalg.norm(v - pre_v) < 0.001:
                print(f'MC_epsilon_greedy iteration number: {k}')
                break
            else:
                pre_v = v
            k += 1

        self.plot_policy_and_state(policy, v, k)
        plt.figure()
        state_value = np.concatenate(state_value_list, axis=1)
        s = np.diff(state_value, axis=1)
        plt.plot(np.linalg.norm(s, axis=0))
        return policy, v, k

    def Sarsa(self, init_policy):
        reward_table = self.get_reward_table()  # (n_actions, height * weight)
        state_transit_prob = self.get_state_transit_prob()  # (n_actions, (height * width), (height * width))
        policy = init_policy
        s_end = self.target_row * self.width + self.target_col
        q = np.zeros((len(self.actions), self.height * self.width))  # (n_actions, n_states)
        n_actions = len(self.actions)
        episode_num = 500
        episode_length = np.zeros(episode_num)
        total_reward = np.zeros(episode_num)
        e = 0.1
        alpha = 0.1
        for i in range(episode_num):
            s = self.start_row * self.width + self.start_col
            j = 0
            while s != s_end:
                a = np.random.choice(len(self.actions), p=policy[:, s].reshape(-1))
                r1 = reward_table[a, s]
                s1 = np.argmax(state_transit_prob[a, s, :].squeeze())
                # a1 = np.random.choice(len(self.actions), p=policy[:, s1].reshape(-1))  # for Sarsa
                # q[a, s] = q[a, s] - alpha * (q[a, s] - (r1 + self.discount_factor * q[a1, s1]))  # for Sarsa
                vt = np.sum(q[:, s1] * policy[:, s1], axis=0).squeeze()  # for expected Sarsa
                q[a, s] = q[a, s] - alpha * (q[a, s] - (r1 + self.discount_factor * vt))  # for expected Sarsa
                # for on-policy Q-learning
                # q[a, s] = q[a, s] - alpha * (q[a, s] - (r1 + self.discount_factor * np.max(q[:, s1], axis=0)))
                greedy_a = np.argmax(q[:, s], axis=0)
                policy[:, s] = e / n_actions
                policy[greedy_a, s] = 1 - e / n_actions * (n_actions -1)
                total_reward[i] += r1
                j += 1
                s = s1
            episode_length[i] = j

        v = self.policy_evaluation(policy)
        self.plot_policy_and_state(policy, v, k=500)
        fig, axe = plt.subplots(2, 1)
        axe[0].plot(total_reward)
        axe[0].set_xlabel('episode num')
        axe[0].set_ylabel('total reward')
        axe[1].plot(episode_length)
        axe[1].set_xlabel('episode num')
        axe[1].set_ylabel('episode length')

    def QLearning(self, policy_be):
        n_actions = len(self.actions)
        n_states = self.height * self.width
        reward_table = self.get_reward_table()  # (n_actions, height * weight)
        state_transit_prob = self.get_state_transit_prob()  # (n_actions, (height * width), (height * width))
        q = np.zeros((len(self.actions), self.height * self.width))  # (n_actions, n_states)
        policy_target = copy.deepcopy(policy_be)
        pre_v = self.policy_evaluation(policy_target)
        k = 0
        self.plot_policy_and_state(policy_target, pre_v, k)
        _, optimal_v, _ = self.policy_iteration(policy_target)
        s = [np.random.randint(0, n_states - 1)]
        a = [np.random.randint(0, n_actions - 1)]
        alpha = 0.1
        state_value_list = [pre_v]
        v_diff_optimal = []
        while k <= 100000:
            r1 = reward_table[a, s]
            s1 = np.argmax(state_transit_prob[a, s, :].squeeze())
            q[a, s] = q[a, s] - alpha * (q[a, s] - (r1 + self.discount_factor * np.max(q[:, s1], axis=0)))
            greedy_a = np.argmax(q[:, s], axis=0)
            policy_target[:, s] = 0
            policy_target[greedy_a, s] = 1
            s = s1
            a = np.random.choice(np.arange(n_actions), p=policy_be[:, s].reshape(-1))
            v = self.policy_evaluation(policy_target)
            state_value_list.append(v)
            v_diff_optimal.append(np.linalg.norm(v - optimal_v))
            if np.linalg.norm(v - optimal_v) < 0.001:
                print(f'Q_learning iteration number: {k}')
                break
            else:
                pre_v = v
            k += 1

        self.plot_policy_and_state(policy_target, v, k)
        plt.figure()
        state_value = np.concatenate(state_value_list, axis=1)
        s = np.diff(state_value, axis=1)
        plt.plot(np.linalg.norm(s, axis=0))
        plt.figure()
        plt.plot(v_diff_optimal)
        return policy_target, v, k

    def get_batch(self, s, a, r, s1, batch_size):
        idx = torch.randint(0, len(s), (batch_size, ))
        s_batch = torch.tensor([s[i] for i in idx], dtype=torch.float32).reshape((-1, 1))
        a_batch = torch.tensor([a[i] for i in idx])
        r_batch = torch.tensor([r[i] for i in idx], dtype=torch.float32)
        s1_batch = torch.tensor([s1[i] for i in idx], dtype=torch.float32).reshape((-1, 1))
        s_batch = torch.cat((s_batch // self.width, s_batch % self.width), dim=1) / self.width
        s1_batch = torch.cat((s1_batch // self.width, s1_batch % self.width), dim=1) / self.width
        return s_batch, a_batch, r_batch, s1_batch

    def DQN(self, policy_be):
        @torch.no_grad()
        def get_policy():
            main_net.eval()
            s = torch.arange(n_states, dtype=torch.float32).reshape((-1, 1))
            s = torch.cat((s // self.width, s % self.width), dim=1) / self.width
            y_pred = main_net(s)
            a = torch.argmax(y_pred, dim=1)
            policy = np.zeros((n_actions, n_states))
            policy[a, np.arange(n_states)] = 1
            v = self.policy_evaluation(policy)
            main_net.train()
            return policy, v

        n_actions = len(self.actions)
        n_states = self.height * self.width
        reward_table = self.get_reward_table()  # (n_actions, height * weight)
        state_transit_prob = self.get_state_transit_prob()  # (n_actions, (height * width), (height * width))
        pre_v = self.policy_evaluation(policy_be)
        self.plot_policy_and_state(policy_be, pre_v, k=0)
        _, optimal_v, _ = self.policy_iteration(policy_be)
        s = [np.random.randint(0, n_states - 1)]
        a = [np.random.randint(0, n_actions - 1)]
        r = [reward_table[a[-1], s[-1]]]
        v_diff_optimal = []
        k = 0
        while k < 1000:
            s.append(np.argmax(state_transit_prob[a[-1], s[-1], :].squeeze()))
            a.append(np.random.choice(np.arange(n_actions), p=policy_be[:, s[-1]].reshape(-1)))
            r.append(reward_table[a[-1], s[-1]])
            k += 1
        main_net = Model(2, n_actions)
        target_net = Model(2, n_actions)
        target_net.load_state_dict(main_net.state_dict())
        lr = 0.05
        batch_size = 100
        optimizer = torch.optim.Adam(main_net.parameters(), lr=lr)
        loss = nn.MSELoss()
        num_iter = 1000
        losslist = []
        main_net.train()
        for i in range(num_iter):
            sb, ab, rb, s1b = self.get_batch(s=s[:-1], a=a[:-1], r=r[:-1], s1=s[1:], batch_size=batch_size)
            y_all = main_net(sb)
            y = y_all[torch.arange(batch_size), ab]
            with torch.no_grad():
                yt_all = target_net(s1b)
                yt = rb + self.discount_factor * torch.max(yt_all, dim=1)[0]
            l = loss(y, yt)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if i % 50 == 0:
                target_net.load_state_dict(main_net.state_dict())
            losslist.append(l.detach())
            policy, v = get_policy()
            v_diff_optimal.append(np.linalg.norm(v - optimal_v))
        plt.figure()
        plt.plot(losslist), plt.title('loss')
        plt.figure()
        plt.plot(v_diff_optimal), plt.title('v_diff_optimal')
        policy, v = get_policy()
        self.plot_policy_and_state(policy, v, k=num_iter)
        plt.show()

    def plot_policy_and_state(self, policy, state, k):
        fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
        self.bg_plot(ax1[0], self.grid)
        ax1[0].set_title(f'Policy, k={k}')
        if policy is not None:
            assert policy.shape == (len(self.actions), self.height * self.width)
            # 生成 x 和 y 矩阵，分别表示每个点的横向和纵向坐标
            x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            x, y = x.reshape(-1), y.reshape(-1)
            u = np.array([0, 1, 0, -1, 0])  # ['up', 'right', 'down', 'left', 'center']
            v = np.array([1, 0, -1, 0, 0])
            for i in range(len(x)):
                xx = np.ones(len(u), dtype=int) * x[i]
                yy = np.ones(len(u), dtype=int) * y[i]
                uu = u * policy[:, i].reshape(-1)
                vv = v * policy[:, i].reshape(-1)
                ax1[0].quiver(xx, yy, uu, vv, scale=10, color='red')

        if state is not None:
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
        plt.pause(0.5)  # 暂停0.5秒


class Model(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        # self.apply(self.init_weights)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

class Agent:
    def __init__(self):
        self.actions = ['up', 'right', 'down', 'left', 'center']

    def act(self, state):

        action = np.random.choice(self.actions)
        return action


GridWorld_height = 5
GridWorld_width = 5
env = GridWorld(GridWorld_height, GridWorld_width, target_pixels=(3, 2),
                forbidden_pixels=[(1, 1), (1, 2), (2, 2), (3, 1), (4, 1), (3, 3)],
                r_target=1, r_boundary=-10, r_default=0, r_forbidden=-10)
# # env.value_iteration()
ini_policy = np.ones((5, GridWorld_height * GridWorld_width)) * 0.2  # (n_actions, height * width)
# ini_policy = np.ones((5, GridWorld_height * GridWorld_width))*0.02
# ini_policy[1, :] = 0.92
# ini_policy = np.ones((5, GridWorld_height * GridWorld_width))*0.1
# ini_policy[1, :] = 0.6

# policy = env.policy_iteration(ini_policy, truncate_time=10)
# plt.show()

# env.QLearning(ini_policy)
env.DQN(ini_policy)

# policy, v, k = env.MC_epsilon_greedy(ini_policy)
# np.savez('policy_MC_epsilon_greedy_0.1_1.npz', policy, v, k)
# data = np.load('policy_MC_epsilon_greedy_0.1_1.npz')
# policy = data['arr_0']
# v = data['arr_1']
# k = data['arr_2']
# env.plot_policy_and_state(policy, v, k)
# plt.show()

# 同一policy, 不同epsilon的state value不同
# e = 0.5
# greedy_action_idx = np.argmax(policy, axis=0)
# policy = np.ones((5, 25)) * (e / 5)
# policy[greedy_action_idx, np.arange(policy.shape[1])] = 1 - (e / 5) * (5 - 1)
# v = env.policy_evaluation(policy)
# env.plot_policy_and_state(policy, v, k=0)
# plt.show()

# agent = Agent()
# for t in range(10):
#     env.step(agent.act(None))
# plt.show()

# GridWorld_height = 5
# GridWorld_width = 5
# env = GridWorld(GridWorld_height, GridWorld_width, target_pixels=(3, 2),
#                 forbidden_pixels=[(1, 1), (1, 2), (2, 2), (3, 1), (4, 1), (3, 3)],
#                 r_target=0, r_boundary=-10, r_default=-1, r_forbidden=-10, start_pos=(0, 0))
# init_policy = np.ones((5, GridWorld_height * GridWorld_width)) * 0.2  # (n_actions, height * width)
# env.Sarsa(init_policy)
