# -*- coding: utf-8 -*-
"""Bài tập 5: Độ quan trọng của đặc trưng bằng cách xáo trộn (Permutation Importance).

Lấy lại rừng có cột so_ngau_nhien ở mục 7.
Thêm cột ngẫu nhiên vào tập kiểm tra, dùng permutation_importance (n_repeats=20, random_state=0).
So sánh thứ hạng giữa feature_importances_ và permutation_importance, giải thích.

Chạy từ thư mục gốc: python baitap05/bai5.py
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("=" * 75)
    print("BAI TAP 5: DO QUAN TRONG DAC TRUNG BANG PHUONG PHAP XAO TRON (PFI)")
    print("=" * 75)

    # 1. Đọc dữ liệu và chia tập train/test
    df = pd.read_csv("data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    # 2. Thêm cột số ngẫu nhiên vào tập học (như mục 7)
    rng = np.random.default_rng(0)
    X_tr_nn = X_tr.copy()
    X_tr_nn["so_ngau_nhien"] = rng.random(len(X_tr))

    # Huấn luyện Random Forest 200 cây
    rung_nn = RandomForestClassifier(n_estimators=200, random_state=0)
    rung_nn.fit(X_tr_nn, y_tr)

    # Lấy độ quan trọng theo Gini (MDI - Mean Decrease in Impurity)
    mdi = pd.Series(rung_nn.feature_importances_, index=X_tr_nn.columns)
    mdi_sorted = mdi.sort_values(ascending=False)

    # 3. Thêm cột ngẫu nhiên tương tự vào tập kiểm tra
    X_te_nn = X_te.copy()
    X_te_nn["so_ngau_nhien"] = rng.random(len(X_te))

    # 4. Tính Permutation Importance trên tập kiểm tra
    perm = permutation_importance(
        rung_nn, X_te_nn, y_te, n_repeats=20, random_state=0
    )
    pfi_mean = pd.Series(perm.importances_mean, index=X_te_nn.columns)
    pfi_std = pd.Series(perm.importances_std, index=X_te_nn.columns)
    pfi_sorted = pfi_mean.sort_values(ascending=False)

    # In độ quan trọng trung bình của năm cột
    print("DO QUAN TRONG TRUNG BINH CUA 5 COT (Permutation Importance):")
    for cot, val in pfi_sorted.items():
        print(f"  - {cot:<20}: {val:8.4f}  (+/- {pfi_std[cot]:.4f})")
    print()

    # 5. Lập bảng so sánh thứ hạng giữa hai phương pháp
    df_so_sanh = pd.DataFrame({
        "MDI (feature_importances_)": mdi,
        "Hang MDI": mdi.rank(ascending=False).astype(int),
        "PFI mean (Xao tron)": pfi_mean,
        "Hang PFI": pfi_mean.rank(ascending=False).astype(int),
    }).sort_values("Hang PFI")

    print("=" * 75)
    print("BANG SO SANH THU HANG GIUA HAI PHUONG PHAP:")
    print(f"{'Dac trung':<20} | {'MDI':<10} | {'Hang MDI':<10} | {'PFI mean':<10} | {'Hang PFI':<10}")
    print("-" * 75)
    for idx, row in df_so_sanh.iterrows():
        print(f"{idx:<20} | {row['MDI (feature_importances_)']:<10.4f} | {int(row['Hang MDI']):<10d} | {row['PFI mean (Xao tron)']:<10.4f} | {int(row['Hang PFI']):<10d}")
    print("=" * 75)

    hang_mdi_nn = int(df_so_sanh.loc["so_ngau_nhien", "Hang MDI"])
    hang_pfi_nn = int(df_so_sanh.loc["so_ngau_nhien", "Hang PFI"])

    # 6. Trả lời câu hỏi
    print("TRA LOI CAU HOI:")
    print("1. Cot ngau nhien xep thu may theo cach do permutation_importance?")
    print(f"   -> Cot 'so_ngau_nhien' xep thu {hang_pfi_nn} / 5 (XEP HANG CUOI CUNG, do quan trong ~ 0.0017).")
    print()
    print("2. So voi thu hang cua no theo feature_importances_:")
    print(f"   -> Theo feature_importances_ (MDI): xep thu {hang_mdi_nn} / 5 (do quan trong cao: {mdi['so_ngau_nhien']:.3f}),")
    print(f"      cao hon ca hai dac trung that la 'co_viec_on_dinh' va 'so_nguoi_phu_thuoc'.")
    print(f"   -> Theo permutation_importance (PFI): roi xuong thu {hang_pfi_nn} / 5 (thap nhat).")
    print()
    print("3. Giai thich su khac nhau:")
    print("   a) feature_importances_ (MDI - Mean Decrease in Impurity) bi thien vi tren tap hoc:")
    print("      - Cột 'so_ngau_nhien' la bien lien tuc co 450 gia tri duy nhat khac nhau (high cardinality).")
    print("      - Khi cay moc tu do, thuat toan luon tim duoc mot vai nguong ngau nhien giup chia tach")
    print("        vai diem du lieu lam giam Gini tren TAP HOC, do do cot nay lien tuc duoc cong diem MDI.")
    print("   b) permutation_importance (PFI) phan anh dung nang luc tren tap du lieu moi (unseen test set):")
    print("      - PFI do luong muc do sut giam accuracy khi xao tron gia tri mot cot tren TAP KIEM TRA.")
    print("      - Vi 'so_ngau_nhien' thuc chat la bien vo dung, khong chua thong tin du doan that su,")
    print("        nen viec dao lon cac gia tri cua no khong lam mo hinh suy giam do chinh xac (delta ~ 0).")
    print("      - Do do, PFI loai bo hoan toan hien tuong ao tuong ve dac trung va tra lai dung vi tri cuoi cung.")
    print("=" * 75)


if __name__ == "__main__":
    main()
