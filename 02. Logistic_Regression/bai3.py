# -*- coding: utf-8 -*-
"""Bài tập 3: Dự đoán cho một bạn cụ thể

Yêu cầu:
- Khớp lại mô hình một biến như ở mục 4.
- Viết một hàm du_doan(gio) nhận vào số giờ ôn và in ra ba thứ:
  1. Giá trị z
  2. Xác suất qua môn
  3. Nhãn theo ngưỡng 0.5
- Gọi hàm đó với các giá trị: 3, 8, 12.89, 18 và 26 giờ.
- Giải thích vì sao với 12.89 giờ thì xác suất ra gần đúng 0.5.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Đọc dữ liệu và khớp mô hình một biến như mục 4
data_path = "data/sinh_vien.csv" if os.path.exists("data/sinh_vien.csv") else "sinh_vien.csv"
df = pd.read_csv(data_path)

X = df[["gio_on"]]
y = df["qua_mon"]

mo_hinh = LogisticRegression()
mo_hinh.fit(X, y)

w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])

print("=" * 70)
print("BAI TAP 3: DU DOAN CHO MOT BAN CU THE")
print("=" * 70)
print(f"He so mo hinh khop duoc: w = {w:.6f}, b = {b:.6f}")
print(f"Moc phan van (z = 0): -b/w = {-b/w:.4f} gio")
print("-" * 70)


def du_doan(gio):
    """Nhan vao so gio on va in ra gia tri z, xac suat qua mon va nhan du doan."""
    z = w * gio + b
    p = 1 / (1 + np.exp(-z))
    nhan = 1 if p >= 0.5 else 0
    print(f"Gio on: {gio:5.2f}h | z = {z:7.4f} | Xac suat qua = {p:.4f} ({p*100:6.2f}%) | Nhan = {nhan} ({'Qua' if nhan == 1 else 'Rot'})")
    return z, p, nhan


print("Ket qua du doan cho cac moc gio on:")
cac_moc_gio = [3, 8, 12.89, 18, 26]
for gio in cac_moc_gio:
    du_doan(gio)

print("-" * 70)
print("Giai thich vi sao voi 12.89 gio thi xac suat ra gan dung 0.5:")
print(f"1. Cong thuc tuyen tinh: z = w * gio + b = {w:.6f} * 12.89 + ({b:.6f}) = {w * 12.89 + b:.6f} ~= 0.")
print("2. Moc 12.89 gio chinh la nghiem xap xi cua phuong trinh z = 0 (tuc gio = -b/w ~= 12.8938 gio).")
print("3. Theo dinh nghia ham sigmoid: sigma(z) = 1 / (1 + e^(-z)). Khi z ~= 0 thi e^0 = 1,")
print("   dan toi sigma(0) = 1 / (1 + 1) = 0.5 (50%).")
print("   Do do, tai 12.89 gio on, xac suat qua mon tinh ra xap xi dung 0.5 (0.4996).")
print("=" * 70)
