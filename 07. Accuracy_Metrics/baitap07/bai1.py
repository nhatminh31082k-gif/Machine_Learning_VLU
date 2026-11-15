# -*- coding: utf-8 -*-
"""
Bài tập 1: Tự viết hàm đánh giá
Yêu cầu:
- Viết hàm danh_gia(y_that, y_doan) trả về accuracy, precision, recall và F1,
  chỉ dùng numpy.
- Chạy hàm cho mô hình logistic và cho mô hình lười ở mục 2, so với các hàm
  của scikit-learn.
- Trả lời: Với mô hình lười, precision có tính được không? Hàm của em xử lý
  trường hợp chia cho 0 thế nào?
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def danh_gia(y_that, y_doan):
    """Tính accuracy, precision, recall và F1 chỉ bằng numpy."""
    that = np.asarray(y_that)
    doan = np.asarray(y_doan)

    tp = int(np.sum((that == 1) & (doan == 1)))
    fp = int(np.sum((that == 0) & (doan == 1)))
    fn = int(np.sum((that == 1) & (doan == 0)))
    tn = int(np.sum((that == 0) & (doan == 0)))

    tong = tp + fp + fn + tn
    accuracy = (tp + tn) / tong if tong > 0 else 0.0
    # Xử lý chia cho 0: nếu không dự đoán mẫu dương nào (tp + fp == 0)
    # hoặc không có mẫu dương thật (tp + fn == 0) thì trả về 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return accuracy, precision, recall, f1


def main():
    df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
    X = df.drop(columns="bo_hoc")
    y = df["bo_hoc"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    luoi = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    du_luoi = luoi.predict(X_te)

    lr = make_pipeline(StandardScaler(), LogisticRegression())
    lr.fit(X_tr, y_tr)
    du_lr = lr.predict(X_te)

    print("=" * 72)
    print("BAI TAP 1: TU VIET HAM DANH GIA CHI DUNG NUMPY")
    print("=" * 72)

    for ten, du in [("Hoi quy logistic", du_lr), ("Mo hinh luoi", du_luoi)]:
        acc_tay, p_tay, r_tay, f1_tay = danh_gia(y_te, du)

        acc_sk = accuracy_score(y_te, du)
        p_sk = precision_score(y_te, du, zero_division=0)
        r_sk = recall_score(y_te, du, zero_division=0)
        f1_sk = f1_score(y_te, du, zero_division=0)

        print(f"\n--- {ten} ---")
        print(f"  Ham tu viet  : acc={acc_tay:.4f}  P={p_tay:.4f}  "
              f"R={r_tay:.4f}  F1={f1_tay:.4f}")
        print(f"  Scikit-learn : acc={acc_sk:.4f}  P={p_sk:.4f}  "
              f"R={r_sk:.4f}  F1={f1_sk:.4f}")
        khop = (
            np.isclose(acc_tay, acc_sk)
            and np.isclose(p_tay, p_sk)
            and np.isclose(r_tay, r_sk)
            and np.isclose(f1_tay, f1_sk)
        )
        print(f"  Kiem tra khop: {'KHOP HOAN TOAN' if khop else 'KHONG KHOP'}")

    print("\n" + "-" * 72)
    print("GIAI THICH:")
    print("1. Voi mo hinh luoi, precision co tinh duoc khong?")
    print("   - Mo hinh luoi luon doan lop 0 (khong bo hoc) cho ca 300 sinh vien.")
    print("   - Vi vay TP = 0 va FP = 0, dan den mau so TP + FP = 0.")
    print("   - Ve mat toan hoc, precision = 0 / 0 la KHONG XAC DINH (khong tinh duoc")
    print("     theo phep chia thong thuong). Khi goi precision_score cua scikit-learn")
    print("     mac dinh se bao canh bao UndefinedMetricWarning.")
    print("2. Ham danh_gia xu ly truong hop chia cho 0 the nao?")
    print("   - Truoc khi chia, ham kiem tra dieu kien mau so > 0:")
    print("     + Neu (tp + fp) == 0 thi gan precision = 0.0.")
    print("     + Neu (tp + fn) == 0 thi gan recall = 0.0.")
    print("     + Neu (precision + recall) == 0 thi gan f1 = 0.0.")
    print("   - Cach quy uoc nay tuong duong tham so zero_division=0 cua scikit-learn,")
    print("     tranh loi ZeroDivisionError hoac gia tri NaN.")
    print("=" * 72)


if __name__ == "__main__":
    main()
