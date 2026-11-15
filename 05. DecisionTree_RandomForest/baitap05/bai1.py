# -*- coding: utf-8 -*-
"""Bài tập 1: Tự vẽ cây quyết định và giải thích cơ chế phân chia nút.

Chạy từ thư mục gốc: python baitap05/bai1.py
"""

import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("=" * 70)
    print("BAI TAP 1: VE CAY QUYET DINH VA GIAI THICH CO CHE CHIA NUT")
    print("=" * 70)

    # 1. Đọc dữ liệu khách vay và chia tập dữ liệu
    df = pd.read_csv("data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]

    feature_names = list(X.columns)
    class_names = ["tre han", "dung han"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    # 2. Huấn luyện cây 5 lá (mục 3)
    cay_5_la = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
    cay_5_la.fit(X_tr, y_tr)

    # Vẽ cây 5 lá và lưu thành cay_nam_la.png
    plt.figure(figsize=(14, 8), dpi=300)
    plot_tree(
        cay_5_la,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title("Cay quyet dinh 5 la (max_leaf_nodes=5)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cay_nam_la.png")
    plt.savefig("baitap05/cay_nam_la.png")
    plt.close()
    print("[+] Da ve va luu cay 5 la vao cay_nam_la.png")

    # 3. Huấn luyện cây có độ sâu 2 (max_depth=2)
    cay_sau_2 = DecisionTreeClassifier(max_depth=2, random_state=0)
    cay_sau_2.fit(X_tr, y_tr)

    # Vẽ cây sâu 2 và lưu thành cay_sau_2.png
    plt.figure(figsize=(12, 7), dpi=300)
    plot_tree(
        cay_sau_2,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=11,
    )
    plt.title("Cay quyet dinh sau 2 (max_depth=2)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("baitap05/cay_sau_2.png")
    plt.close()
    print("[+] Da ve va luu cay sau 2 vao baitap05/cay_sau_2.png")
    print()

    # In cây dạng chữ
    print("--- Cay sau 2 tang dang chu (export_text) ---")
    print(export_text(cay_sau_2, feature_names=feature_names, show_weights=True, decimals=3))

    # 4. Trả lời câu hỏi
    print("=" * 70)
    print("TRA LOI CAU HOI:")
    print("1. Hai la ben trai cua cay nay co cung ket luan khong?")
    print("   -> CO, ca hai la ben trai deu co ket luan la lop 0 ('tre han'):")
    print("      - La 1 (thu_nhap <= 15.05 va ty_le_no <= 0.615):")
    print("        weights = [68, 53] -> lop da so la 0 ('tre han').")
    print("      - La 2 (thu_nhap <= 15.05 va ty_le_no > 0.615):")
    print("        weights = [114, 14] -> lop da so la 0 ('tre han').")
    print()
    print("2. Vi sao cay van chia nut do ra lam hai?")
    print("   -> Vi cay quyet dinh (thuat toan CART) khong chon cau hoi dua tren viec")
    print("      lop da so co thay doi hay khong, ma dua tren chi so bat thuan khiet Gini:")
    print("      - Tai nut cha (thu_nhap <= 15.05): co 249 khach [182 tre, 67 dung],")
    print("        Gini = 1 - (182/249)^2 - (67/249)^2 = 0.3926.")
    print("      - Khi chia theo 'ty_le_no <= 0.615':")
    print("        + Nhanh trai: 121 khach [68 tre, 53 dung], Gini = 0.4921.")
    print("        + Nhanh phai: 128 khach [114 tre, 14 dung], Gini = 0.1947 (rat thuan khiet, 89% tre han).")
    print("        + Gini trung binh co trong so sau chia: G_chia = (121/249)*0.4921 + (128/249)*0.1947 = 0.3392.")
    print("        + Muc Gini giam duoc la: 0.3926 - 0.3392 = 0.0534 > 0.")
    print("      - Phep chia nay giup tach han mot nhom 128 khach co ty le tre han cuc cao (89%),")
    print("        lam tang manh do tu tin / xac suat du doan (predict_proba), giam do hon loan")
    print("        cua du lieu, nen cay van thuc hien phep chia.")
    print("=" * 70)


if __name__ == "__main__":
    main()
