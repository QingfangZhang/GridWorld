import numpy as np
import matplotlib.pyplot as plt


class Gridworld:
    """
    属性:
        size：Gridworld 的大小（默认值为 4）
        actions：可用的动作列表（向上、向下、向左和向右）
        start_pos：起始位置
        goal_pos：目标位置
        current_state：当前状态，初始为起始位置。
    方法：
        reset() 重置环境，并将当前状态设置为初始位置。
        step(action) 使用给定的动作更新当前状态，计算即时回报，并返回新状态、回报和完成状态。
        render() 在终端打印出当前状态。
    """
    def __init__(self, size=4):
        self.size = size
        self.actions = ["up", "down", "left", "right"]
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.current_state = self.start_pos

    def reset(self):
        self.current_state = self.start_pos

    def step(self, action):
        if action not in self.actions:
            raise ValueError("Invalid action")

        # update current state based on the chosen action
        next_state = self.current_state
        if action == "up":
            next_state = (max(0, self.current_state[0] - 1), self.current_state[1])
        elif action == "down":
            next_state = (min(self.size - 1, self.current_state[0] + 1), self.current_state[1])
        elif action == "left":
            next_state = (self.current_state[0], max(0, self.current_state[1] - 1))
        elif action == "right":
            next_state = (self.current_state[0], min(self.size - 1, self.current_state[1] + 1))

        done = False
        reward = 0
        if next_state == self.goal_pos:  # reached the goal
            done = True
            reward = 1

        self.current_state = next_state

        return self.current_state, reward, done

    def render(self):
        env = np.zeros((self.size, self.size))
        env[self.start_pos] = 0.5
        env[self.goal_pos] = 0.7
        env[self.current_state] = 0.9
        for row in env:
            print("|", end="")
            for col in row:
                if col == 0.5:
                    print(" S ", end="|")
                elif col == 0.7:
                    print(" G ", end="|")
                elif col == 0.9:
                    print(" * ", end="|")
                else:
                    print("   ", end="|")
            print("")

    def plot(self,  ax, agent_pos=None):
        grid = np.zeros((self.size, self.size))
        grid[self.start_pos] = 0.5
        grid[self.goal_pos] = 0.7
        if agent_pos:
            grid[agent_pos] = 0.9

        # fig, ax = plt.subplots()
        ax.imshow(grid, cmap='viridis_r')
        ax.set_xticks(np.arange(-0.5, self.size, 1))
        ax.set_yticks(np.arange(-0.5, self.size, 1))
        ax.grid(axis='both')

        plt.pause(0.5)


class Agent:
    def __init__(self):
        self.actions = ["up", "down", "left", "right"]

    def act(self, state):
        action = np.random.choice(self.actions)
        return action


if __name__ == '__main__':
    # Create a Gridworld and Agent
    env = Gridworld()
    agent = Agent()

    # Render the initial state of the environment
    fig, ax = plt.subplots()
    env.plot(ax, agent_pos=env.current_state)

    for i in range(1):
        action = agent.act(env.current_state)
        next_state, reward, done = env.step(action)
        env.plot(ax, agent_pos=next_state)
        env.render()
    plt.show()
