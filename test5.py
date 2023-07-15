import matplotlib.pyplot as plt
import numpy as np

# # 创建图形对象和子图对象
# fig, ax = plt.subplots()
#
# # 定义箭头起始点的坐标
# x = np.array([2, 2, 2])
# y = np.array([2, 2, 2])
#
# # 定义箭头的长度和方向
# u = np.array([0, -1, 1])
# v = np.array([1, 0, 0])
# scale = np.array([1, 2, 4])
#
# # 绘制箭头
# ax.quiver(x, y, u, v, scale=scale)
#
# # 设置坐标轴范围
# ax.set_xlim(0, 4)
# ax.set_ylim(0, 4)
#
# # 显示图形
# plt.show()

x = [0, 0, 0, 0]
y = [0, 0, 0, 0]
u = np.array([0, 0, 1, -1])* np.array([0.02, 0.02, 0.92, 0.02])
v = np.array([1, -1, 0, 0])* np.array([0.02, 0.02, 0.92, 0.02])
scale = np.array([1/0.02, 1/0.02, 1/0.92, 1/0.02])

fig, ax = plt.subplots()

scale_large = 10
scale_small = 0.5

# 使用较大的 scale 值绘制箭头
ax.quiver(x, y, u, v, color='red', label='Large Scale')

# 使用较小的 scale 值绘制箭头
# ax.quiver(x, y, u, v, scale=scale_small, color='blue', label='Small Scale')

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.legend()

plt.show()

# import numpy as np
#
# # 定义二维函数
# def f(x, y):
#     return x**2 + y**2
#
# # 定义一维数组
# x = np.array([1, 2, 3])
# y = np.array([4, 5, 6, 7])
#
# # 使用 meshgrid 生成二维坐标矩阵
# X, Y = np.meshgrid(x, y)
#
# # 计算函数在网格点上的取值
# Z = f(X, Y)
#
# print("Z 坐标矩阵：")
# print(Z)
# print(X)
# print(Y)
#
#
# # 绘制三维曲面图
# fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.plot_surface(X, Y, Z)
# # plt.show()
# plt.imshow(Z)
# plt.colorbar()
# plt.show()