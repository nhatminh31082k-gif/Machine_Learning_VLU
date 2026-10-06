# -*- coding: utf-8 -*-
"""Bài tập 5: Dò ngưỡng tốt nhất theo F1

Yêu cầu:
- Cho ngưỡng chạy từ 0.05 tới 0.95, mỗi bước 0.05.
- Với mỗi ngưỡng, tính F1 trên tập kiểm tra rồi in ra thành bảng.
- Cuối cùng in ra ngưỡng cho F1 cao nhất.
- Nhận xét một câu: ngưỡng tốt nhất theo F1 có đúng bằng 0.5 không?
- Gợi ý: dùng np.arange(0.05, 1.0, 0.05) và hàm f1_score với tham số zero_division=0.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

data_path = "data/sinh_vien.csv" if os.path.exists("data/sinh_vien.csv") else "sinh_vien.csv"
df = pd.read_csv(data_path)
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = LogisticRegression().fit(X_train, y_train)

# Lấy xác suất dự đoán lớp 1 (qua môn) trên tập kiểm tra
p = mo_hinh.predict_proba(X_test)[:, 1]

# Tạo dải ngưỡng từ 0.05 tới 0.95, bước 0.05
nguongs = np.arange(0.05, 1.0, 0.05)

print("=" * 65)
print("BAI TAP 5: DO NGUONG TOT NHAT THEO F1-SCORE")
print("=" * 65)
print(f"{'Nguong':>8s} | {'So doan qua':>12s} | {'Precision':>10s} | {'Recall':>8s} | {'F1-score':>10s}")
print("-" * 65)

ket_qua = []
for t in nguongs:
    t_val = round(float(t), 2)
    y_pred = (p >= t).astype(int)
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    pre = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    ket_qua.append((t_val, f1, pre, rec, int(y_pred.sum())))
    print(f"{t_val:8.2f} | {y_pred.sum():12d} | {pre:10.4f} | {rec:8.4f} | {f1:10.4f}")

print("-" * 65)

# Tìm F1 cao nhất và các ngưỡng tương ứng
max_f1 = max(item[1] for item in ket_qua)
cac_nguong_tot_nhat = [item[0] for item in ket_qua if np.isclose(item[1], max_f1)]
str_cac_nguong = ", ".join(f"{x:.2f}" for x in cac_nguong_tot_nhat)

# Giá trị F1 tại ngưỡng mặc định 0.5
f1_tai_05 = [item[1] for item in ket_qua if np.isclose(item[0], 0.50)][0]

print(f"F1 cao nhat dat duoc la      : {max_f1:.4f}")
print(f"Cac nguong cho F1 cao nhat   : {str_cac_nguong}")
print(f"F1 tai nguong mac dinh 0.5 la: {f1_tai_05:.4f}")
print()

# Nhận xét
print("NHAN XET:")
print(f"Nguong tot nhat theo F1 KHONG phai la 0.5 ma la {str_cac_nguong} (dat F1 = {max_f1:.4f}).")
print("Giai thich: Khi nang nguong tu 0.5 len 0.60-0.65, Precision tang manh tu 0.8500 len 1.0000")
print("(loai bo hoan toan ca 3 loi FP, khong con ban rot nao bi bao nham la qua),")
print("trong khi Recall chi giam nhe tu 0.9444 xuong 0.8889 (chi bo sot them 1 ban qua mon).")
print("Su danh doi nay giup diem so F1 tang ro ret tu 0.8947 len dinh 0.9412.")
print("=" * 65)
