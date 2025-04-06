import numpy as np
import matplotlib.pyplot as plt

# Hai vector bạn muốn vẽ
a = np.array([2, 3])
b = np.array([4, 1])

# Vẽ gốc toạ độ
origin = np.array([0, 0])

# Tạo biểu đồ
plt.figure(figsize=(6, 6))
plt.quiver(*origin, *a, color='r', scale=1, scale_units='xy', angles='xy', label='Vector a')
plt.quiver(*origin, *b, color='b', scale=1, scale_units='xy', angles='xy', label='Vector b')

# Cài đặt trục
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal')

# Tính cosine similarity
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
plt.title(f'Cosine similarity = {cos_sim:.2f}')
plt.legend()
plt.show()
