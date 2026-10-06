# -*- coding: utf-8 -*-
"""Bài tập 6: Đổi lớp dương rồi chấm lại

Yêu cầu:
- Ở mục 5 ta luôn lấy lớp dương là qua môn (lớp 1).
- Chấm điểm cho lớp rớt môn (lớp 0) bằng cách thêm tham số pos_label=0
  vào precision_score và recall_score.
- In ra precision và recall của lớp 0 rồi so với con số của lớp 1 trong tài liệu.
- Giải thích bằng lời vì sao hai bộ số khác nhau, dù mô hình và dữ liệu không đổi chút nào.
"""

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

data_path = "data/sinh_vien.csv" if os.path.exists("data/sinh_vien.csv") else "sinh_vien.csv"
df = pd.read_csv(data_path)
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = LogisticRegression().fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

# Ma tran nham lan
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Cham diem voi lop duong la lop 0 (rot mon)
precision_lop_0 = precision_score(y_test, y_pred, pos_label=0)
recall_lop_0 = recall_score(y_test, y_pred, pos_label=0)

# Cham diem voi lop duong la lop 1 (qua mon - nhu trong tai lieu)
precision_lop_1 = precision_score(y_test, y_pred, pos_label=1)
recall_lop_1 = recall_score(y_test, y_pred, pos_label=1)

print("=" * 70)
print("BAI TAP 6: DOI LOP DUONG (POS_LABEL=0) VA CHAM LAI")
print("=" * 70)
print(f"Ma tran nham lan tren tap kiem tra (30 sinh vien):")
print(f"  TN = {tn} (Doan rot, that su rot)")
print(f"  FP = {fp} (Doan qua, that ra rot)")
print(f"  FN = {fn} (Doan rot, that ra qua)")
print(f"  TP = {tp} (Doan qua, that su qua)")
print("-" * 70)

print("KET QUA SO SANH GIUA HAI LOP DUONG:")
print(f"1. Lop 0 (Rot mon - pos_label=0):")
print(f"   Precision = {precision_lop_0:.4f} (9 / (9 + 1))")
print(f"   Recall    = {recall_lop_0:.4f} (9 / (9 + 3))")
print()
print(f"2. Lop 1 (Qua mon - pos_label=1, trong tai lieu):")
print(f"   Precision = {precision_lop_1:.4f} (17 / (17 + 3))")
print(f"   Recall    = {recall_lop_1:.4f} (17 / (17 + 1))")
print("-" * 70)

print("GIAI THICH CHI TIET VI SAO HAI BO SO KHAC NHAU:")
print("Du mo hinh va du lieu khong doi chut nao, hai bo so Precision va Recall khac nhau vi:")
print()
print("1. SU THAY DOI VAI TRO TRONG MA TRAN NHAM LAN:")
print("   - Khi chon lop 1 la duong (positive): 'Duong' nghia la Qua mon.")
print("     + TP = 17 (doan qua, that su qua), FP = 3 (doan qua, nhung rot)")
print("     + FN = 1  (doan rot, nhung qua),   TN = 9 (doan rot, that su rot)")
print("   - Khi chon lop 0 la duong (positive): 'Duong' nghia la Rot mon.")
print("     Vai tro trong ma tran bi hoan doi:")
print("     + TP_moi = TN_cu = 9  (doan rot, that su rot)")
print("     + FP_moi = FN_cu = 1  (doan rot, nhung thuc ra qua)")
print("     + FN_moi = FP_cu = 3  (doan qua, nhung thuc ra rot)")
print("     + TN_moi = TP_cu = 17 (doan qua, that su qua)")
print()
print("2. Y NGHIA THUC TE CUA HAI BO CHI SO:")
print("   - Voi lop 0 (Rot mon):")
print("     + Precision = 9 / (9 + 1) = 0.9000: Trong 10 ban mo hinh bao 'Rot', co 9 ban")
print("       rot that -> Mo hinh rat chac chan khi bao rot (do chinh xac dat 90%).")
print("     + Recall = 9 / (9 + 3) = 0.7500: Trong 12 ban thuc su bi rot, mo hinh chi bat")
print("       duoc 9 ban, bo sot 3 ban rot (do bao phu chi dat 75%).")
print("   - Voi lop 1 (Qua mon):")
print("     + Precision = 17 / (17 + 3) = 0.8500: Trong 20 ban mo hinh bao 'Qua', co 17 ban")
print("       qua that (do chinh xac dat 85%, co 3 ban bao nham).")
print("     + Recall = 17 / (17 + 1) = 0.9444: Trong 18 ban thuc su qua mon, mo hinh bat")
print("       duoc 17 ban (do bao phu rat cao 94.44%, chi bo sot 1 ban).")
print()
print("3. KET LUAN BAN CHAT:")
print("   Precision va Recall la cac thuoc do KHONG DOI XUNG (asymmetric metrics), phu thuoc")
print("   chat che vao viec ta dat muc tieu quan tam vao lop nao lam lop duong.")
print("   Nguong mac dinh 0.5 hien tai dang khien mo hinh co thien huong 'de tinh' cho viec")
print("   qua mon (uu tien doan lop 1), dan toi Recall lop 1 rat cao (0.9444), nhung lai lam")
print("   Recall lop 0 bi thap (0.7500).")
print("=" * 70)
