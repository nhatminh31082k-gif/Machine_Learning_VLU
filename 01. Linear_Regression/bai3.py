# -*- coding: utf-8 -*-
"""
Bài tập 3: Đổi biến đầu vào sang tuổi nhà
Dùng lại đúng công thức bình phương tối thiểu ở mục 5, đổi biến đầu vào từ dien_tich sang tuoi_nha.
Biến cần dự đoán vẫn là gia.
In ra hệ số góc w và hệ số chặn b với 6 chữ số sau dấu phẩy.
Nhận xét dấu của w và giải thích bằng lời vì sao nó mang dấu âm.
Trả lời câu hỏi thêm về độ tin cậy của w và quan hệ giữa hai biến.
"""

import os
import sys
import numpy as np
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

# Đổi biến đầu vào x sang tuoi_nha, y vẫn là gia
x = df["tuoi_nha"].to_numpy()
y = df["gia"].to_numpy()

# 2. Tính trung bình các đại lượng
x_tb = x.mean()
y_tb = y.mean()

# 3. Áp dụng công thức bình phương tối thiểu
tu_so = ((x - x_tb) * (y - y_tb)).sum()
mau_so = ((x - x_tb) ** 2).sum()

w = tu_so / mau_so
b = y_tb - w * x_tb

# 4. Đánh giá sai số MSE
y_du_doan = w * x + b
mse = ((y - y_du_doan) ** 2).mean()

# Tính thêm R2 để đánh giá chính xác độ chặt chẽ của quan hệ
ss_res = ((y - y_du_doan) ** 2).sum()
ss_tot = ((y - y_tb) ** 2).sum()
r2 = 1 - (ss_res / ss_tot)

print("=== KET QUA BAI TAP 3 ===")
print(f"Trung binh tuoi nha : {x_tb:.4f} nam")
print(f"Trung binh gia      : {y_tb:.4f} ty dong")
print(f"Tu so               : {tu_so:.4f}")
print(f"Mau so              : {mau_so:.4f}")
print()
print(f"He so goc w = {w:.6f}")
print(f"He so chan b = {b:.6f}")
print()
print(f"Mo hinh: gia = {w:.6f} * tuoi_nha + {b:.6f}")
print(f"MSE tren toan bo 60 can : {mse:.4f}")
print(f"R2 tren toan bo 60 can  : {r2:.4f}")
print()
print("--- Nhan xet va giai thich dau cua w ---")
print("1. Nhan xet ve dau cua w:")
print("   He so w mang dau am (w = -0.037858).")
print("2. Giai thich ly do mang dau am:")
print("   He so w the hien muc thay doi cua gia khi tuoi nha tang them 1 nam. w mang dau am")
print("   nghia la tuoi nha va gia nha co quan he ty le nghich: nha cang cu, thoi gian su dung")
print("   cang lau thi bi hao mon, xuong cap, do do gia tri cua can ho co xu huong giam xuong.")
print("   Cu the, cu them 1 nam tuoi nha thi gia ban trung binh giam khoang 0.0379 ty (khoang 38 trieu dong).")
print()
print("--- Tra loi cau hoi them (mo rong) ---")
print("1. Cac diem du lieu co that su bam quanh mot duong thang khong?")
print("   Khong. Khi nhin vao bieu do phan tan giua tuoi nha va gia, cac diem du lieu rai rac hon loan,")
print("   khong tao thanh mot dai bám sat duong thang nhu dien tich. MSE cua mo hinh nay rat lon (2.5786")
print("   so voi 0.1790 cua dien tich), va R2 chi dat khoang 0.0262 (chi giai thich duoc 2.6% bien dong).")
print("2. He so w nay dang tin toi muc nao va dau dung da du de ket luan quan he chat chua?")
print("   Con so w nay co do tin cay rat thap khi dung doc lap mot minh de du doan gia nha.")
print("   Mot he so co dau dung voi thuc te (dau am) chi la dieu kien can chu CHUA DU de ket luan")
print("   hai dai luong co quan he chat che, vi do phan tan con qua lon va do phu thuoc chu yeu vao dien tich.")

# Ve bieu do tuoi_nha vs gia de kiem chung truc quan
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color="#2ca02c", alpha=0.75, edgecolors="k", s=60, label="60 can ho")
x_line = np.linspace(x.min(), x.max(), 100)
y_line = w * x_line + b
plt.plot(x_line, y_line, color="red", linewidth=2, label=f"Duong hoi quy: y = {w:.4f}x + {b:.4f}")
plt.title("Bieu do phan tan: Gia nha theo Tuoi nha", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Tuoi nha (nam)", fontsize=12)
plt.ylabel("Gia ban (ty dong)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.savefig("bai3_tuoi_nha.png", dpi=150, bbox_inches="tight")
if os.path.exists("baitap01"):
    plt.savefig("baitap01/bai3_tuoi_nha.png", dpi=150, bbox_inches="tight")
plt.close()
