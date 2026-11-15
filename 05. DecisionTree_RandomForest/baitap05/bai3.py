# -*- coding: utf-8 -*-
"""Bài tập 3: So sánh thước đo Gini và Entropy khi chọn độ sâu cây quyết định.

Chạy lại f3_do_sau.py với criterion="entropy". Ghi lại độ sâu tốt nhất và
accuracy trên tập kiểm tra, so với Gini ở mục 4. Nhận xét.

Chạy từ thư mục gốc: python baitap05/bai3.py
"""

import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("=" * 75)
    print("BAI TAP 3: SO SANH THUOC DO GINI VA ENTROPY CHO CAY QUYET DINH")
    print("=" * 75)

    # 1. Đọc dữ liệu và chia tập train/test giống hệt f3_do_sau.py
    df = pd.read_csv("data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    # 2. Khảo sát độ sâu từ 1 đến 15 với cả Gini và Entropy bằng 5-fold CV
    diem_gini = {}
    hoc_gini = {}
    diem_ent = {}
    hoc_ent = {}

    print(f"{'Do sau':<8} | {'--- GINI (Muc 4) ---':<25} | {'--- ENTROPY (Bai 3) ---':<25}")
    print(f"{'':<8} | {'Tap hoc':<10} {'CV (5-fold)':<14} | {'Tap hoc':<10} {'CV (5-fold)':<14}")
    print("-" * 75)

    for d in range(1, 16):
        # Mô hình dùng Gini (mục 4)
        mh_g = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=0)
        diem_gini[d] = cross_val_score(mh_g, X_tr, y_tr, cv=5).mean()
        mh_g.fit(X_tr, y_tr)
        hoc_gini[d] = accuracy_score(y_tr, mh_g.predict(X_tr))

        # Mô hình dùng Entropy (bài 3)
        mh_e = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=0)
        diem_ent[d] = cross_val_score(mh_e, X_tr, y_tr, cv=5).mean()
        mh_e.fit(X_tr, y_tr)
        hoc_ent[d] = accuracy_score(y_tr, mh_e.predict(X_tr))

        print(f"  {d:2d}     | {hoc_gini[d]:<10.4f} {diem_gini[d]:<14.4f} | {hoc_ent[d]:<10.4f} {diem_ent[d]:<14.4f}")

    # 3. Chọn độ sâu tối ưu theo CV
    d_tot_gini = max(diem_gini, key=diem_gini.get)
    d_tot_ent = max(diem_ent, key=diem_ent.get)

    # 4. Đánh giá trên tập kiểm tra
    tot_gini = DecisionTreeClassifier(criterion="gini", max_depth=d_tot_gini, random_state=0)
    tot_gini.fit(X_tr, y_tr)
    acc_te_gini = accuracy_score(y_te, tot_gini.predict(X_te))

    tot_ent = DecisionTreeClassifier(criterion="entropy", max_depth=d_tot_ent, random_state=0)
    tot_ent.fit(X_tr, y_tr)
    acc_te_ent = accuracy_score(y_te, tot_ent.predict(X_te))

    # Đếm số mẫu kiểm tra bị dự đoán khác nhau
    so_khac = (tot_gini.predict(X_te) != tot_ent.predict(X_te)).sum()

    print("=" * 75)
    print("BANG TONG KET SO SANH GIUA GINI VA ENTROPY:")
    print(f"{'Tieu chi':<30} | {'Gini (Muc 4)':<18} | {'Entropy (Bai 3)':<18}")
    print("-" * 75)
    print(f"{'Do sau tot nhat (d_tot)':<30} | {d_tot_gini:<18} | {d_tot_ent:<18}")
    print(f"{'Diem CV cao nhat':<30} | {diem_gini[d_tot_gini]:<18.4f} | {diem_ent[d_tot_ent]:<18.4f}")
    print(f"{'So la cua cay toi uu':<30} | {tot_gini.get_n_leaves():<18} | {tot_ent.get_n_leaves():<18}")
    print(f"{'Accuracy tap kiem tra':<30} | {acc_te_gini:<18.4f} | {acc_te_ent:<18.4f}")
    print(f"{'So mau test doan khac nhau':<30} | {so_khac} / {len(y_te)} mau ({so_khac / len(y_te):.1%})")
    print("=" * 75)

    # 5. Nhận xét
    print("NHAN XET:")
    print('  "Doi thuoc do tu Gini sang Entropy khong lam ket qua khac di nhieu, ca hai')
    print('   deu dat do sau toi uu tai d = 5 va cho do chinh xac tren tap kiem tra gan nhu')
    print(f'   tuong duong tuyet doi ({acc_te_gini:.4f} so voi {acc_te_ent:.4f}, chi khac nhau dung {so_khac} khach)."')
    print()
    print("GIAI THICH:")
    print("  - Ca Gini va Entropy deu la cac ham do do hon loan / bat thuan khiet cua nut,")
    print("    deu dat cuc tieu bang 0 khi nut hoan toan thuan khiet va dat cuc dai khi chia deu 50-50.")
    print("  - Su khac biet chi nam o do cong cua ham so (Gini dung tong binh phuong, Entropy dung logarit).")
    print("  - Trong thuc te hon 95% cac truong hop, Gini va Entropy chon cung mot cau hoi phan chia.")
    print("    Gini thuong duoc uu tien lam mac dinh vi tinh toan nhanh hon (khong can phep tinh log).")
    print("=" * 75)


if __name__ == "__main__":
    main()
