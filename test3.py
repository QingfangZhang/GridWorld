import matplotlib.pyplot as plt
import numpy as np

# a = np.ones((5, 5)) *0.2
# a[1, 1] = 0.8
# a[2, 2] = 0.7
# a[1, 3] = 1
# # 自定义刻度位置
# x_ticks = [0.5, 1.5, 2.5, 3.5]
# y_ticks = [0.5, 1.5, 2.5, 3.5]
#
# # 创建一个新的图形对象 fig 和 axes 对象 ax，并设置长宽比为 1:1
# fig, ax = plt.subplots()
# ax.imshow(a, vmin=0, vmax=1)
# # 去掉背景主网格线
# # ax.grid(False)
# ax.vlines(np.arange(0.5, 4.5), -0.5, 4.5, color='black')
# ax.hlines(np.arange(0.5, 4.5), -0.5, 4.5, color='black')
# # 给刻度位置添加次要坐标轴网格线
# # ax.set_xticks(x_ticks, minor=True)
# # ax.set_yticks(y_ticks, minor=True)
# # ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
#
# # 指定 x 轴和 y 轴的范围，并保留边框
# ax.set_xlim(-0.5, 4.5)
# ax.set_ylim(-0.5, 4.5)
# # ax.spines['left'].set_visible(True)
# # ax.spines['bottom'].set_visible(True)
# # ax.spines['right'].set_visible(True)
# # ax.spines['top'].set_visible(True)
#
# # 把整个坐标系内的数字都隐藏
# plt.xticks(range(5), [])
# plt.yticks(range(5), [])
#
# # 显示图形
# plt.show()

a = np.ones((5, 5))*0.2
a[1, 3] = 0.8
fig, ax = plt.subplots()
ax.imshow(a, vmin=0, vmax=1)
plt.show()
