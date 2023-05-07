import numpy as np
import matplotlib.pyplot as plt


a = np.zeros((4, 4))
a[1, 3] = 1
fig, ax = plt.subplots()
ax.imshow(a)
ax.add_artist(plt.Circle((3, 1), 0.1, color='blue'))

plt.show()
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