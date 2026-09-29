# -*- coding: utf-8 -*-
"""
Bài tập 1: Lọc ra nhóm căn hộ lớn
Đọc bộ dữ liệu gia_nha.csv, sau đó in ra hai con số:
1. Số căn có diện tích lớn hơn 100 mét vuông.
2. Giá trung bình của riêng nhóm căn đó, tính bằng tỷ đồng.
"""

import os
import sys
import pandas as pd

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

# 1. Đọc bộ dữ liệu
df = pd.read_csv(duong_dan)

# 2. Lọc ra nhóm căn hộ có diện tích lớn hơn 100 m2
nhom_can_lon = df[df["dien_tich"] > 100]

# 3. Tính số căn và giá trung bình của nhóm
so_can_lon = len(nhom_can_lon)
gia_trung_binh_nhom_lon = nhom_can_lon["gia"].mean()

# 4. In kết quả ra màn hình
print("=== KET QUA BAI TAP 1 ===")
print(f"Tong so can trong bo du lieu: {len(df)}")
print(f"1. So can co dien tich > 100 m2: {so_can_lon} can")
print(f"2. Gia trung binh cua nhom can lon (> 100 m2): {gia_trung_binh_nhom_lon:.4f} ty dong ({gia_trung_binh_nhom_lon:.3f} ty dong)")
print()
print("Danh sach cac can ho co dien tich > 100 m2:")
print(nhom_can_lon[["dien_tich", "so_phong", "tuoi_nha", "gia"]].to_string(index=False))
