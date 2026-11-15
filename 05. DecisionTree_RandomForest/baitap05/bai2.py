# -*- coding: utf-8 -*-
"""Bài tập 2: Tự tìm câu hỏi gốc trên dữ liệu thật (450 khách tập học).

Mở rộng f1_gini.py thành hàm tim_cau_hoi(X, y) thử mọi cột và mọi ngưỡng,
trả về cột và ngưỡng làm Gini giảm nhiều nhất.

Chạy từ thư mục gốc: python baitap05/bai2.py
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def gini(nhan):
    """Tính chỉ số Gini của một tập nhãn nhị phân (0 và 1)."""
    if len(nhan) == 0:
        return 0.0
    p1 = np.mean(nhan)
    p0 = 1.0 - p1
    return 1.0 - p1 ** 2 - p0 ** 2


def tim_cau_hoi(X, y):
    """Thử mọi cột và mọi ngưỡng có thể trên tập dữ liệu.

    Trả về:
      cot_tot       : tên cột tốt nhất
      nguong_tot    : ngưỡng chia tốt nhất
      gini_giam_max : lượng Gini giảm nhiều nhất
      chi_tiet_cot  : dict chứa thông tin câu hỏi tốt nhất của từng cột
    """
    y_arr = np.asarray(y)
    n = len(y_arr)
    g_goc = gini(y_arr)

    cot_tot = None
    nguong_tot = None
    gini_giam_max = -1.0
    chi_tiet_cot = {}

    for cot in X.columns:
        vals = X[cot].to_numpy()
        muc = np.sort(np.unique(vals))

        best_nguong_cot = None
        best_giam_cot = -1.0
        best_gchia_cot = 1.0

        # Thử mọi ngưỡng nằm giữa 2 giá trị liền kề
        for a, b in zip(muc[:-1], muc[1:]):
            nguong = (a + b) / 2.0
            di_trai = (vals <= nguong)
            trai = y_arr[di_trai]
            phai = y_arr[~di_trai]

            g_chia = (len(trai) / n) * gini(trai) + (len(phai) / n) * gini(phai)
            g_giam = g_goc - g_chia

            if g_giam > best_giam_cot:
                best_giam_cot = g_giam
                best_nguong_cot = nguong
                best_gchia_cot = g_chia

            if g_giam > gini_giam_max:
                gini_giam_max = g_giam
                cot_tot = cot
                nguong_tot = nguong

        chi_tiet_cot[cot] = {
            "nguong": best_nguong_cot,
            "gini_sau_chia": best_gchia_cot,
            "gini_giam": best_giam_cot,
            "so_nguong_thu": len(muc) - 1,
        }

    return cot_tot, nguong_tot, gini_giam_max, chi_tiet_cot


def main():
    print("=" * 70)
    print("BAI TAP 2: TU TIM CAU HOI GOC TREN DU LIEU THAT (450 KHACH TAP HOC)")
    print("=" * 70)

    # 1. Đọc dữ liệu và chia tập học/kiểm tra
    df = pd.read_csv("data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    n_samples = len(X_tr)
    g_goc = gini(y_tr)
    print(f"So luong khach tap hoc: {n_samples}")
    print(f"Ty le tra dung han     : {np.mean(y_tr):.4f}")
    print(f"Gini tai nut goc       : {g_goc:.4f}")
    print()

    # 2. Tìm câu hỏi tốt nhất bằng hàm tự viết tim_cau_hoi
    cot_tot, nguong_tot, gini_giam_max, chi_tiet = tim_cau_hoi(X_tr, y_tr)

    print("--- KET QUA THU TUNG COT TREN TAP HOC ---")
    print(f"{'Ten cot':<20} | {'So nguong':<10} | {'Nguong tot nhat':<16} | {'Gini sau chia':<14} | {'Gini giam':<10}")
    print("-" * 75)
    for cot, info in chi_tiet.items():
        print(f"{cot:<20} | {info['so_nguong_thu']:<10} | {info['nguong']:<16.4f} | {info['gini_sau_chia']:<14.4f} | {info['gini_giam']:<10.4f}")
    print()

    print("=" * 70)
    print("CAU HOI TOT NHAT DUOC CHON (Gini giam nhieu nhat):")
    print(f"  -> Cot       : '{cot_tot}'")
    print(f"  -> Nguong    : {nguong_tot:.4f} (<= {nguong_tot:.2f})")
    print(f"  -> Gini giam : {gini_giam_max:.4f}")
    print()

    # 3. So sánh với Scikit-learn DecisionTreeClassifier(max_depth=1)
    cay_1_tang = DecisionTreeClassifier(max_depth=1, random_state=0)
    cay_1_tang.fit(X_tr, y_tr)

    print("--- Scikit-learn DecisionTreeClassifier(max_depth=1) ---")
    print(export_text(cay_1_tang, feature_names=list(X.columns), decimals=3, show_weights=True))

    skl_feature = X.columns[cay_1_tang.tree_.feature[0]]
    skl_threshold = cay_1_tang.tree_.threshold[0]
    print(f"Scikit-learn chon cot: '{skl_feature}', nguong: {skl_threshold:.4f}")
    print()

    # 4. Đối chiếu kết quả
    khop = (cot_tot == skl_feature) and np.isclose(nguong_tot, skl_threshold)
    print("=" * 70)
    print("NHAN XET VA KET LUAN:")
    if khop:
        print("[XAC NHAN] Ket qua hoan toan TRUNG KHOP voi scikit-learn va tai lieu lab:")
        print(f"           Cau hoi goc la: '{cot_tot} <= {nguong_tot:.2f}' (thu nhap voi nguong 15.05).")
    else:
        print("[CANH BAO] Ket qua chua trung khop.")
    print("=" * 70)


if __name__ == "__main__":
    main()
