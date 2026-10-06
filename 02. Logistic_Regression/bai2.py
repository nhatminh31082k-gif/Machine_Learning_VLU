# -*- coding: utf-8 -*-
"""Bài tập 2: Vẽ đồ thị hàm sigmoid

Yêu cầu:
- Vẽ đồ thị hàm sigmoid với z chạy từ -8 tới 8
- Lưu thành tệp sigmoid.png
- Trên cùng hình vẽ thêm một đường ngang tại mức 0.5 và một đường dọc tại z = 0
- Gợi ý: dùng np.linspace, plt.plot, plt.axhline, plt.axvline và plt.savefig
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    """Hàm sigmoid ép giá trị thực z về khoảng (0, 1)."""
    return 1 / (1 + np.exp(-z))


# Tạo dãy giá trị z từ -8 tới 8
z = np.linspace(-8, 8, 400)
sigma_z = sigmoid(z)

# Thiết lập kích thước đồ thị
plt.figure(figsize=(8, 5), dpi=150)

# Vẽ đường cong hàm sigmoid
plt.plot(z, sigma_z, label=r"$\sigma(z) = \frac{1}{1 + e^{-z}}$", color="navy", linewidth=2.5)

# Vẽ đường ngang tại mức 0.5 và đường dọc tại z = 0
plt.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label=r"Ngưỡng 0.5 ($\sigma(0) = 0.5$)")
plt.axvline(0, color="gray", linestyle=":", linewidth=1.2, label="Trục đối xứng z = 0")

# Đánh dấu điểm tâm (0, 0.5)
plt.scatter([0], [0.5], color="red", zorder=5, s=60)
plt.annotate(r"$\sigma(0) = 0.5$", xy=(0, 0.5), xytext=(0.8, 0.42),
             arrowprops=dict(arrowstyle="->", color="red"),
             fontsize=11, color="red", fontweight="bold")

# Trang trí đồ thị
plt.title("Đồ thị hàm Sigmoid", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("z", fontsize=12)
plt.ylabel(r"$\sigma(z)$", fontsize=12)
plt.xlim(-8.5, 8.5)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower right", fontsize=10)

plt.tight_layout()

# Lưu đồ thị ra tệp sigmoid.png (ở thư mục gốc và baitap02)
output_file = "sigmoid.png"
plt.savefig(output_file)

os.makedirs("baitap02", exist_ok=True)
plt.savefig("baitap02/sigmoid.png")

plt.close()

print(f"Da ve va luu thanh cong do thi ham sigmoid vao tep: {output_file} va baitap02/{output_file}")
