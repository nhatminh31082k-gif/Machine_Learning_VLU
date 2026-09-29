# -*- coding: utf-8 -*-
"""
Bài tập 2: Vẽ biểu đồ phân tán theo số phòng
Vẽ biểu đồ phân tán với trục hoành là cột so_phong và trục tung là cột gia.
Đặt tên cho hai trục và đặt tiêu đề cho hình.
Lưu hình thành tệp bai2.png rồi viết một câu nhận xét về xu hướng nhìn thấy.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Đảm bảo hiển thị tốt trên mọi môi trường terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Xác định đường dẫn tệp dữ liệu tương đối linh hoạt
duong_dan = "data/gia_nha.csv"
if not os.path.exists(duong_dan):
    if os.path.exists("../data/gia_nha.csv"):
        duong_dan = "../data/gia_nha.csv"
    elif os.path.exists("gia_nha.csv"):
        duong_dan = "gia_nha.csv"
    elif os.path.exists("../gia_nha.csv"):
        duong_dan = "../gia_nha.csv"

# 1. Đọc dữ liệu
df = pd.read_csv(duong_dan)

# 2. Chuẩn bị dữ liệu vẽ biểu đồ
so_phong = df["so_phong"]
gia_ban = df["gia"]

# 3. Khởi tạo và vẽ biểu đồ phân tán (scatter plot)
plt.figure(figsize=(8, 6))
plt.scatter(so_phong, gia_ban, color="#1f77b4", alpha=0.75, edgecolors="k", s=60, label="60 can ho")

# Thiết lập tiêu đề và nhãn các trục
plt.title("Bieu do phan tan: Gia nha theo so phong ngu", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("So phong ngu (phong)", fontsize=12)
plt.ylabel("Gia ban (ty dong)", fontsize=12)
plt.xticks([1, 2, 3, 4])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left")

# 4. Lưu hình ra tệp bai2.png trước khi hiển thị
# Lưu cả ở thư mục hiện hành và thư mục baitap01 để tiện nộp bài
plt.savefig("bai2.png", dpi=150, bbox_inches="tight")
if os.path.exists("baitap01"):
    plt.savefig("baitap01/bai2.png", dpi=150, bbox_inches="tight")

print("=== KET QUA BAI TAP 2 ===")
print("Da luu bieu do phan tan vao tep bai2.png.")
print()
print("Nhan xet ve xu huong:")
print("- Khi so phong ngu tang len thi gia nha trung binh co xu huong tang theo ti le thuan (can 1 phong co gia trung binh khoang 4.58 ty, can 4 phong co gia trung binh khoang 8.92 ty).")
print("- Tuy nhien, vi so phong la bien roi rac (chi nhan gia tri 1, 2, 3, 4) nen cac diem tap trung thanh tung cot thang dung; o cung mot so phong, muc gia van dao dong kha rong do phu thuoc them vao dien tich va tuoi nha.")

# Dong hinh sau khi xu ly xong
plt.close()
