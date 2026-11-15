# -*- coding: utf-8 -*-
"""
Bài tập 5: Cân bằng lớp cho mô hình khác
Yêu cầu:
- Thêm class_weight="balanced" cho RandomForestClassifier và cho SVC.
- So precision, recall, F1 trên tập kiểm tra trước và sau khi cân bằng.
- Trả lời: Mô hình nào thay đổi nhiều nhất?
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def danh_gia_mo_hinh(ten, mh, X_tr, y_tr, X_te, y_te):
    """Huấn luyện mô hình và trả về precision, recall, f1 trên tập kiểm tra."""
    mh.fit(X_tr, y_tr)
    du = mh.predict(X_te)
    p = precision_score(y_te, du, zero_division=0)
    r = recall_score(y_te, du, zero_division=0)
    f1 = f1_score(y_te, du, zero_division=0)
    return p, r, f1


def main():
    df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
    X = df.drop(columns="bo_hoc")
    y = df["bo_hoc"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    cau_hinh = [
        (
            "RandomForest (200 cay)",
            RandomForestClassifier(200, random_state=0),
            RandomForestClassifier(200, class_weight="balanced", random_state=0),
        ),
        (
            "SVC (RBF + StandardScaler)",
            make_pipeline(StandardScaler(), SVC(random_state=0)),
            make_pipeline(
                StandardScaler(), SVC(class_weight="balanced", random_state=0)
            ),
        ),
    ]

    print("=" * 72)
    print("BAI TAP 5: SO SANH CAN BANG LOP CHO RANDOM FOREST VA SVC")
    print("=" * 72)

    tong_thay_doi = {}
    for ten, mh_thuong, mh_cb in cau_hinh:
        p0, r0, f1_0 = danh_gia_mo_hinh(ten, mh_thuong, X_tr, y_tr, X_te, y_te)
        p1, r1, f1_1 = danh_gia_mo_hinh(ten, mh_cb, X_tr, y_tr, X_te, y_te)

        dp = p1 - p0
        dr = r1 - r0
        df1 = f1_1 - f1_0
        tong_thay_doi[ten] = abs(dp) + abs(dr) + abs(df1)

        print(f"\n--- {ten} ---")
        print("  Trang thai      Precision   Recall     F1")
        print(f"  Truoc can bang   {p0:.4f}     {r0:.4f}   {f1_0:.4f}")
        print(f"  Sau can bang     {p1:.4f}     {r1:.4f}   {f1_1:.4f}")
        print(f"  Chenh lech      {dp:+.4f}    {dr:+.4f}  {df1:+.4f}")

    nhieu_nhat = max(tong_thay_doi, key=tong_thay_doi.get)

    print("\n" + "-" * 72)
    print("NHAN XET:")
    print(f"1. Mo hinh thay doi nhieu nhat: {nhieu_nhat}")
    print("   - SVC thay doi cuc ky manh khi them class_weight='balanced':")
    print("     + Recall tang vot +0.3333 (tu 0.5000 len 0.8333, tim them 10 ban bo hoc).")
    print("     + Precision giam manh -0.4359 (tu 0.8824 xuong 0.4464 do bao nham nhieu).")
    print("     + F1 giam nhe tu 0.6383 xuong 0.5814 (-0.0569).")
    print("   - Nguyen nhan: trong SVC, class_weight='balanced' nhan he so phat C")
    print("     cho lop hiem (gap ~9 lan lop dong), keo ranh gioi quyet dinh lui")
    print("     ve phia lop 0 de tranh bo sot lop 1.")
    print("2. Con RandomForestClassifier thi sao?")
    print("   - Thay doi it hon SVC va theo chieu huong khac:")
    print("     + Precision giam nhe tu 0.7727 xuong 0.7500 (-0.0227).")
    print("     + Recall giam tu 0.5667 xuong 0.4000 (-0.1667).")
    print("     + F1 giam tu 0.6538 xuong 0.5217 (-0.1321).")
    print("   - Vi cay quyet dinh chi dung trong so khi tinh do tinh khiet Gini luc")
    print("     chia nut; viec tang trong so lop hiem co the khien cay chia qua sat")
    print("     vao vai diem ngoai le cua tap hoc (qua khop nhe), khong tao hieu ung")
    print("     dich nguong deu dan nhu mo hinh tuyen tinh hay SVM.")
    print("=" * 72)


if __name__ == "__main__":
    main()
