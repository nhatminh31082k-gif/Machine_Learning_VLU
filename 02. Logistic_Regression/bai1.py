# -*- coding: utf-8 -*-
"""Bài tập 1: Thống kê theo nhóm

Yêu cầu:
- Đọc bộ dữ liệu rồi in ra ba con số:
  1. Số bạn có điểm giữa kỳ từ 7 trở lên
  2. Tỷ lệ qua môn của riêng nhóm đó
  3. Tỷ lệ qua môn của nhóm còn lại
- Nhận xét một câu về việc điểm giữa kỳ có phân biệt được hai nhóm hay không.
"""

import os
import pandas as pd

# Đọc dữ liệu (hỗ trợ cả đường dẫn từ thư mục gốc và thư mục hiện tại)
data_path = "data/sinh_vien.csv" if os.path.exists("data/sinh_vien.csv") else "sinh_vien.csv"
df = pd.read_csv(data_path)

# 1. Lọc nhóm có điểm giữa kỳ từ 7 trở lên và nhóm còn lại
nhom_diem_cao = df[df["diem_giua_ky"] >= 7.0]
nhom_diem_thap = df[df["diem_giua_ky"] < 7.0]

# Tính toán ba con số
so_ban_diem_cao = len(nhom_diem_cao)
ty_le_qua_nhom_cao = nhom_diem_cao["qua_mon"].mean()
ty_le_qua_nhom_thap = nhom_diem_thap["qua_mon"].mean()

print("=" * 60)
print("BAI TAP 1: THONG KE THEO NHOM")
print("=" * 60)
print(f"1. So ban co diem giua ky tu 7 tro len : {so_ban_diem_cao} ban")
print(f"2. Ty le qua mon cua nhom diem >= 7     : {ty_le_qua_nhom_cao:.4f} ({ty_le_qua_nhom_cao * 100:.2f}%)")
print(f"3. Ty le qua mon cua nhom con lai (< 7) : {ty_le_qua_nhom_thap:.4f} ({ty_le_qua_nhom_thap * 100:.2f}%)")
print()

# Nhận xét
print("Nhan xet:")
print("Diem giua ky co kha nang phan biet rat ro ret giua hai nhom qua mon va rot mon,")
print("vi sinh vien co diem giua ky tu 7 tro len co ty le qua mon vuot troi (90.32%)")
print("so voi nhom duoi 7 diem (chi dat 48.31%, chenh lech gan gap doi).")
print("=" * 60)
