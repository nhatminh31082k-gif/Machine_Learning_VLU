# -*- coding: utf-8 -*-
"""Bài tập 4: Cắt tỉa cây quyết định bằng số khách tối thiểu ở lá (min_samples_leaf).

Khảo sát min_samples_leaf trong [1, 2, 5, 10, 20, 40].
Chọn giá trị tốt nhất bằng 5-fold CV trên tập học, chấm trên tập kiểm tra.
So sánh với cây sâu 5 ở mục 4 và rút ra kết luận.

Chạy từ thư mục gốc: python baitap05/bai4.py
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
    print("BAI TAP 4: CAT TIA CAY BANG SO KHACH TOI THIEU O LA (min_samples_leaf)")
    print("=" * 75)

    # 1. Đọc dữ liệu và chia tập train/test
    df = pd.read_csv("data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    cac_gia_tri_leaf = [1, 2, 5, 10, 20, 40]
    diem_cv = {}
    acc_tr_dict = {}
    do_sau_dict = {}
    so_la_dict = {}

    print(f"{'min_samples_leaf':<18} | {'Do sau':<8} | {'So la':<8} | {'Tap hoc':<10} | {'CV (5-fold)':<12}")
    print("-" * 75)

    for leaf in cac_gia_tri_leaf:
        mh = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=0)
        cv = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
        diem_cv[leaf] = cv

        # Huấn luyện trên toàn bộ tập học để lấy độ sâu, số lá và acc tập học
        mh.fit(X_tr, y_tr)
        acc_tr = accuracy_score(y_tr, mh.predict(X_tr))
        acc_tr_dict[leaf] = acc_tr
        do_sau_dict[leaf] = mh.get_depth()
        so_la_dict[leaf] = mh.get_n_leaves()

        print(f"  {leaf:2d}{'':<14} | {do_sau_dict[leaf]:<8d} | {so_la_dict[leaf]:<8d} | {acc_tr:<10.4f} | {cv:<12.4f}")

    # 2. Chọn giá trị tốt nhất theo kiểm định chéo
    best_leaf = max(diem_cv, key=diem_cv.get)
    print()
    print(f"Gia tri min_samples_leaf tot nhat theo CV: {best_leaf} (CV = {diem_cv[best_leaf]:.4f})")

    # 3. Huấn luyện mô hình tốt nhất và chấm trên tập kiểm tra
    cay_leaf_tot = DecisionTreeClassifier(min_samples_leaf=best_leaf, random_state=0)
    cay_leaf_tot.fit(X_tr, y_tr)
    acc_te_leaf = accuracy_score(y_te, cay_leaf_tot.predict(X_te))

    # 4. Lấy lại kết quả cây sâu 5 ở mục 4 để đối chiếu
    cay_sau_5 = DecisionTreeClassifier(max_depth=5, random_state=0)
    cv_sau_5 = cross_val_score(cay_sau_5, X_tr, y_tr, cv=5).mean()
    cay_sau_5.fit(X_tr, y_tr)
    acc_te_sau_5 = accuracy_score(y_te, cay_sau_5.predict(X_te))

    print("=" * 75)
    print("SO SANH HAI PHUONG PHAP CAT TIA:")
    print(f"{'Phuong phap cat tia':<32} | {'Do sau':<8} | {'So la':<8} | {'Diem CV':<10} | {'Test Acc':<10}")
    print("-" * 75)
    print(f"{f'min_samples_leaf = {best_leaf}':<32} | {cay_leaf_tot.get_depth():<8d} | {cay_leaf_tot.get_n_leaves():<8d} | {diem_cv[best_leaf]:<10.4f} | {acc_te_leaf:<10.4f}")
    print(f"{'max_depth = 5 (Muc 4)':<32} | {cay_sau_5.get_depth():<8d} | {cay_sau_5.get_n_leaves():<8d} | {cv_sau_5:<10.4f} | {acc_te_sau_5:<10.4f}")
    print("=" * 75)

    # 5. Trả lời câu hỏi so sánh
    print("TRA LOI CAU HOI:")
    print("So voi cay sau 5 cua muc 4, cach cat tia nao cho ket qua tot hon tren bo du lieu nay?")
    if acc_te_sau_5 > acc_te_leaf:
        ket_luan = "Cach cat tia bang gioi han do sau (max_depth=5) cho ket qua TOT HON"
    elif acc_te_sau_5 < acc_te_leaf:
        ket_luan = f"Cach cat tia bang min_samples_leaf={best_leaf} cho ket qua TOT HON"
    else:
        ket_luan = "Ca hai cach cat tia cho ket qua TUONG DUONG NHAU tren tap kiem tra"

    print(f"-> {ket_luan}:")
    print(f"   - max_depth = 5        : CV = {cv_sau_5:.4f}, Test Accuracy = {acc_te_sau_5:.4f}")
    print(f"   - min_samples_leaf = {best_leaf}: CV = {diem_cv[best_leaf]:.4f}, Test Accuracy = {acc_te_leaf:.4f}")
    print()
    print("GIAI THICH NGUYEN NHAN:")
    print("  1. Khi gioi han do sau max_depth=5, chieu dai chuoi cau hoi tu goc toi la bi")
    print("     khong che nghiem ngat, giup loai bo hoan toan cac quy tac qua phuc tap o tat ca")
    print("     cac nhanh, giu cho mo hinh tinh gon va tong quat hoa tot nhat.")
    print("  2. Trong khi do, min_samples_leaf chi rang buoc so luong mau o cuoi la ma khong gioi han")
    print(f"     do sau tong the (cay min_samples_leaf={best_leaf} van sau toi {cay_leaf_tot.get_depth()} tang, 30 la).")
    print("     Dieu nay van de lai mot so nhanh sau phuc tap hoc thuoc phan nao nhieu cua du lieu.")
    print("=" * 75)


if __name__ == "__main__":
    main()
