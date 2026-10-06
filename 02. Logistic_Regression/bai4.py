# -*- coding: utf-8 -*-
"""Bài tập 4: Tự tính bốn thước đo

Yêu cầu:
- Chép tệp c4_danh_gia.py sang baitap02 rồi bỏ hết các hàm accuracy_score,
  precision_score, recall_score, f1_score của scikit-learn đi.
- Tự tính bốn con số đó chỉ từ bốn số TP, TN, FP, FN bằng công thức ở mục 5.
- So kết quả tự tính với kết quả thư viện in ra trong tài liệu, hai bên phải khớp.
"""

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data_path = "data/sinh_vien.csv" if os.path.exists("data/sinh_vien.csv") else "sinh_vien.csv"
df = pd.read_csv(data_path)
X = df[["gio_on"]]
y = df["qua_mon"]

# stratify=y giữ đúng tỷ lệ qua và rớt ở cả hai phần sau khi chia
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("=" * 65)
print("BAI TAP 4: TU TINH BON THUOC DO DUA TREN MA TRAN NHAM LAN")
print("=" * 65)
print("So sinh vien de hoc      :", len(X_train))
print("So sinh vien de kiem tra :", len(X_test))
print()

mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)

y_pred = mo_hinh.predict(X_test)

# Thứ tự bốn ô do scikit-learn quy định là TN, FP, FN, TP
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("Ma tran nham lan:")
print(f"  TN = {tn:2d}  (Doan rot, that su rot - Dung)")
print(f"  FP = {fp:2d}  (Doan qua, that ra rot - Sai)")
print(f"  FN = {fn:2d}  (Doan rot, that ra qua - Sai)")
print(f"  TP = {tp:2d}  (Doan qua, that su qua - Dung)")
print("-" * 65)

# Tự tính 4 thước đo bằng công thức toán học từ 4 ô:
n = len(y_test)  # n = tp + tn + fp + fn
accuracy = (tp + tn) / n
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print("KET QUA TU TINH BANG CONG THUC:")
print(f"  Accuracy  = (TP + TN) / n        = ({tp} + {tn}) / {n}   = {accuracy:.4f}")
print(f"  Precision = TP / (TP + FP)       = {tp} / ({tp} + {fp}) = {precision:.4f}")
print(f"  Recall    = TP / (TP + FN)       = {tp} / ({tp} + {fn}) = {recall:.4f}")
print(f"  F1-score  = 2*Pre*Rec/(Pre+Rec)  = 2*{precision:.4f}*{recall:.4f}/({precision:.4f}+{recall:.4f}) = {f1:.4f}")
print("-" * 65)

# So sánh với kết quả thư viện trong tài liệu (Mục 5.2 & Mục 5.3)
doc_accuracy = 0.8667
doc_precision = 0.8500
doc_recall = 0.9444
doc_f1 = 0.8947

print("SO SANH VOI KET QUA THU VIEN SCIKIT-LEARN TRONG TAI LIEU:")
print(f"  Accuracy  : Tu tinh = {accuracy:.4f} | Tai lieu = {doc_accuracy:.4f} -> Khop: {round(accuracy, 4) == doc_accuracy}")
print(f"  Precision : Tu tinh = {precision:.4f} | Tai lieu = {doc_precision:.4f} -> Khop: {round(precision, 4) == doc_precision}")
print(f"  Recall    : Tu tinh = {recall:.4f} | Tai lieu = {doc_recall:.4f} -> Khop: {round(recall, 4) == doc_recall}")
print(f"  F1-score  : Tu tinh = {f1:.4f} | Tai lieu = {doc_f1:.4f} -> Khop: {round(f1, 4) == doc_f1}")
print("-" * 65)

print("GIAI THICH Y NGHIA BON THUOC DO:")
print("1. Accuracy (0.8667): Do chinh xac tong the. Trong 30 sinh vien kiem tra,")
print("   mo hinh doan dung 26 ban (17 qua + 9 rot), dat ty le ~86.67%.")
print("2. Precision (0.8500): Do chinh xac duong. Trong so 20 sinh vien mo hinh du doan")
print("   la qua mon, co 17 ban qua that su (dat 85.00%). Co 3 ban bi bao nham (FP).")
print("3. Recall (0.9444): Do bao phu/thu hoi. Trong so 18 sinh vien thuc su qua mon,")
print("   mo hinh tim ra va bao dung duoc 17 ban (dat 94.44%). Chi bo sot 1 ban (FN).")
print("4. F1-score (0.8947): Trung binh dieu hoa giua Precision va Recall, giup danh gia")
print("   can bang ca hai yeu to tren khi khong uu tien rieng le Precision hay Recall.")
print("=" * 65)
