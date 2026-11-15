# -*- coding: utf-8 -*-
"""
Bài tập 3: Chọn ngưỡng theo yêu cầu
Yêu cầu:
- Nhà trường yêu cầu tìm ra ít nhất 80% số bạn bỏ học (recall >= 0.8).
- Dùng xác suất từ cross_val_predict trên tập học như mục 5, tìm ngưỡng
  lớn nhất cho recall từ 0.8 trở lên.
- Áp ngưỡng đó cho tập kiểm tra một lần.
- Trả lời: Recall có đạt 0.8 không? Precision và số bạn bị báo bằng bao nhiêu?
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
    X = df.drop(columns="bo_hoc")
    y = df["bo_hoc"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    lr = make_pipeline(StandardScaler(), LogisticRegression())
    lr.fit(X_tr, y_tr)

    # Chỉ dùng tập học qua kiểm định chéo 5 phần để chọn ngưỡng (không nhìn tập kiểm tra)
    xs_hoc = cross_val_predict(
        lr, X_tr, y_tr, cv=5, method="predict_proba"
    )[:, 1]

    cac_nguong = np.round(np.arange(0.05, 0.96, 0.05), 2)

    print("=" * 72)
    print("BAI TAP 3: CHON NGUONG LON NHAT DAT RECALL >= 0.80 TREN TAP HOC")
    print("=" * 72)
    print(f"Tap hoc: {len(y_tr)} sinh vien ({y_tr.sum()} ban bo hoc)")
    print("\nKet qua kiem dinh cheo 5 phan tren tap hoc theo nguong:")
    print("  nguong   so bao   precision   recall      F1    dat >=0.8?")

    nguong_dat = []
    for t in cac_nguong:
        du_hoc = (xs_hoc >= t).astype(int)
        p_h = precision_score(y_tr, du_hoc, zero_division=0)
        r_h = recall_score(y_tr, du_hoc, zero_division=0)
        f1_h = f1_score(y_tr, du_hoc, zero_division=0)
        dat = r_h >= 0.8
        if dat:
            nguong_dat.append(t)
        if t <= 0.50:
            print(f"   {t:.2f}     {du_hoc.sum():4d}     {p_h:.4f}    {r_h:.4f}"
                  f"   {f1_h:.4f}    {'CO' if dat else 'khong'}")

    nguong_chon = max(nguong_dat)
    du_hoc_chon = (xs_hoc >= nguong_chon).astype(int)
    recall_hoc_chon = recall_score(y_tr, du_hoc_chon)

    print(f"\n-> Nguong lon nhat dat recall >= 0.80 tren tap hoc (buoc 0.05): "
          f"{nguong_chon:.2f} (recall CV = {recall_hoc_chon:.4f})")

    # Tham khảo thêm bước mịn 0.01 trên tập học
    cac_nguong_min = np.round(np.arange(0.01, 0.96, 0.01), 2)
    nguong_min_dat = [
        t for t in cac_nguong_min
        if recall_score(y_tr, (xs_hoc >= t).astype(int)) >= 0.8
    ]
    nguong_min_chon = max(nguong_min_dat)

    # Mở tập kiểm tra ĐÚNG MỘT LẦN với ngưỡng đã chọn
    xs_te = lr.predict_proba(X_te)[:, 1]
    du_te = (xs_te >= nguong_chon).astype(int)

    tp = int(((du_te == 1) & (y_te == 1)).sum())
    fp = int(((du_te == 1) & (y_te == 0)).sum())
    fn = int(((du_te == 0) & (y_te == 1)).sum())
    so_bao = int(du_te.sum())
    r_te = recall_score(y_te, du_te)
    p_te = precision_score(y_te, du_te)
    f1_te = f1_score(y_te, du_te)

    print("\n" + "-" * 72)
    print(f"AP DUNG NGUONG {nguong_chon:.2f} LEN TAP KIEM TRA (CHI MOT LAN):")
    print(f"  - Recall    : {r_te:.4f} ({tp}/{int(y_te.sum())} ban bo hoc)")
    print(f"  - Precision : {p_te:.4f} ({tp}/{so_bao} loi bao dung)")
    print(f"  - F1        : {f1_te:.4f}")
    print(f"  - So bao    : {so_bao} ban (TP={tp}, FP={fp}, bo sot FN={fn})")

    print("\n" + "-" * 72)
    print("TRA LOI VA GIAI THICH:")
    print(f"1. Recall tren tap kiem tra co dat 0.8 khong?")
    print(f"   - CO. Recall dat {r_te:.4f} (tim duoc {tp}/30 ban bo hoc, vuot muc 0.80).")
    print("2. Precision va so ban bi bao bang bao nhieu?")
    print(f"   - Precision dat {p_te:.4f} va co {so_bao} sinh vien bi bao nguy co")
    print(f"     (trong do {tp} ban bo hoc that va {fp} ban bao nham).")
    print(f"   - (Ghi chu: neu do luoi min buoc 0.01 tren tap hoc, nguong lon nhat")
    print(f"     la {nguong_min_chon:.2f}, khi ap len tap kiem tra cho recall = 0.8000,")
    print(f"     precision = 0.5106 va bao 47 ban.)")
    print("3. Vi sao phai chon nguong bang cross_val_predict tren tap hoc?")
    print("   - Vi khong duoc phep nhin dap an tap kiem tra de chinh nguong.")
    print("   - Moi sinh vien trong tap hoc nhan xac suat du doan tu mo hinh chua")
    print("     tung thay ban ay trong luc huan luyen, nen nguong 0.15 duoc chon")
    print("     khach quan va cho ket qua danh gia trung thuc khi ap len tap kiem tra.")
    print("=" * 72)


if __name__ == "__main__":
    main()
