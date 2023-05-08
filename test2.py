import numpy as np
import matplotlib.pyplot as plt


# a = np.zeros((4, 4))
# a[1, 3] = 1
# fig, ax = plt.subplots()
# ax.imshow(a)
# ax.add_artist(plt.Circle((3, 1), 0.1, color='blue'))
#
# plt.show()
# a = np.zeros((5,5))
# plt.imshow(a)
# # plt.grid(True)
# plt.quiver(2, 2, 1, 0, color='red', scale=4)
# plt.show()

#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 创建数据
# x = np.arange(0.5, 10, 1)
# y = np.arange(0.5, 10, 1)
# u = np.ones(x.shape)
# v = np.zeros(y.shape)
#
# # 改变箭头长度
# u *= 1
#
# # 绘制箭头
# fig, ax = plt.subplots()
#
# # 关闭自动缩放
# ax.set_xlim([0, 10])
# ax.set_ylim([0, 10])
#
# # 绘制箭头
# ax.quiver(x, y, u, v, scale=5)
#
# plt.show()
#
#

# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.arange(0, 5)
# y = x * 2
# u = np.ones(x.shape)
# v = np.zeros(y.shape)
# c = [0.1,0.6,0.9, 0.3,0.4]
#
# fig, ax = plt.subplots()
# ax.grid(linestyle='--')
#
# # scale 值分别为 1, 2, 4
# for scale_val in [1, 2, 4]:
#     ax.quiver(x, y, u, v, c, scale=scale_val, label=f'scale={scale_val}')
#
# ax.legend()
# plt.show()


# # 创建一个 5 × 5 的二维数组，用于存储每个元素点的坐标
# arr = np.array([[1,2,3,4,5],
#                 [6,7,8,9,10],
#                 [11,12,13,14,15],
#                 [16,17,18,19,20],
#                 [21,22,23,24,25]])
#
# # 定义箭头的长度和颜色
# scale = 30
# color = 'red'
#
# # 生成 x 和 y 矩阵，分别表示每个点的横向和纵向坐标
# x, y = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
#
# # 随机生成箭头的方向
# u = np.zeros_like(arr)
# v = np.zeros_like(arr)
# directions = ['up', 'down', 'left', 'right', 'center']
# for i in range(arr.shape[0]):
#     for j in range(arr.shape[1]):
#         # 对于每个点，从五个方向中随机选择一个作为箭头方向
#         direction = np.random.choice(directions)
#         if direction == 'up':
#             u[i,j] = 0
#             v[i,j] = 1
#         elif direction == 'down':
#             u[i,j] = 0
#             v[i,j] = -1
#         elif direction == 'left':
#             u[i,j] = -1
#             v[i,j] = 0
#         elif direction == 'right':
#             u[i,j] = 1
#             v[i,j] = 0
#         else:  # center
#             u[i,j] = 0
#             v[i,j] = 0
#
# # 绘制箭头
# fig, ax = plt.subplots()
# ax.quiver(1, 1, 0, 0, scale=5, color=color)
#
# plt.show()

# 创建一个 5 × 5 的二维数组
arr = np.array([[1,2,3,4,5],
                [6,7,8,9,10],
                [11,12,13,14,15],
                [16,17,18,19,20],
                [21,22,23,24,25]])

# 绘制矩阵
fig, ax = plt.subplots()
im = ax.imshow(arr)

# 在每个元素位置上添加数值标签
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        ax.text(j, i, arr[i,j], ha='center', va='center', color='w')

plt.show()