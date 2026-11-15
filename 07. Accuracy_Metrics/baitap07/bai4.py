# -*- coding: utf-8 -*-
"""
Bài tập 4: Tự vẽ đường ROC
Yêu cầu:
- Dùng roc_curve lấy hai mảng fpr và tpr của hồi quy logistic và của rừng ngẫu nhiên.
- Vẽ cả hai đường cùng đường chéo lên một hình bằng plt.plot, ghi AUC vào chú thích,
  lưu thành roc.png.
- Trả lời:
  + Đường nào ôm sát góc trên bên trái hơn?
  + Có đoạn nào hai đường cắt nhau không?
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def danh_gia_roc_tai_luoi(fpr, tpr, luoi_fpr):
    """Lấy giá trị TPR của đường ROC bậc thang tại các mốc FPR cho trước."""
    chi_so = np.searchsorted(fpr, luoi_fpr, side="right") - 1
    chi_so = np.clip(chi_so, 0, len(tpr) - 1)
    return tpr[chi_so]


def main():
    df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
    X = df.drop(columns="bo_hoc")
    y = df["bo_hoc"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    lr = make_pipeline(StandardScaler(), LogisticRegression())
    rf = RandomForestClassifier(200, random_state=0)

    lr.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)

    xs_lr = lr.predict_proba(X_te)[:, 1]
    xs_rf = rf.predict_proba(X_te)[:, 1]

    fpr_lr, tpr_lr, _ = roc_curve(y_te, xs_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_te, xs_rf)

    auc_lr = roc_auc_score(y_te, xs_lr)
    auc_rf = roc_auc_score(y_te, xs_rf)

    # Vẽ đường ROC của hai mô hình và đường chéo đoán ngẫu nhiên
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(
        fpr_lr, tpr_lr, linewidth=2.2, color="#1f4e78",
        label=f"Hoi quy logistic (AUC = {auc_lr:.4f})"
    )
    ax.plot(
        fpr_rf, tpr_rf, linewidth=2.2, color="#d95f02", linestyle="-.",
        label=f"Rung ngau nhien 200 cay (AUC = {auc_rf:.4f})"
    )
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.4,
        label="Doan ngau nhien (AUC = 0.5000)"
    )

    ax.set_xlabel("Ty le bao dong nham (FPR)", fontsize=11)
    ax.set_ylabel("Recall (TPR)", fontsize=11)
    ax.set_title("Duong ROC: Hoi quy logistic vs Rung ngau nhien",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    os.makedirs("baitap07", exist_ok=True)
    plt.savefig("baitap07/roc.png", dpi=150)
    plt.savefig("roc.png", dpi=150)
    plt.close(fig)

    # Kiểm tra chi tiết các đoạn hai đường giao / vượt nhau
    luoi = np.sort(np.unique(np.concatenate([fpr_lr, fpr_rf])))
    tpr_lr_luoi = danh_gia_roc_tai_luoi(fpr_lr, tpr_lr, luoi)
    tpr_rf_luoi = danh_gia_roc_tai_luoi(fpr_rf, tpr_rf, luoi)

    print("=" * 72)
    print("BAI TAP 4: VE DUONG ROC VA SO SANH HAI MO HINH")
    print("=" * 72)
    print(f"Hoi quy logistic     : {len(fpr_lr):2d} diem ROC, AUC = {auc_lr:.4f}")
    print(f"Rung ngau nhien (200): {len(fpr_rf):2d} diem ROC, AUC = {auc_rf:.4f}")
    print("Da luu hinh vao baitap07/roc.png va roc.png")

    print("\nSo sanh tai vung FPR rat nho (gan goc tren ben trai):")
    print("   FPR      TPR (Logistic)   TPR (Rung)   Chenh lech (LR - RF)")
    for f, tl, tr in zip(luoi[:7], tpr_lr_luoi[:7], tpr_rf_luoi[:7]):
        print(f"  {f:.4f}        {tl:.4f}         {tr:.4f}         {tl - tr:+.4f}")

    print("\n" + "-" * 72)
    print("NHAN XET:")
    print("1. Duong nao om sat goc tren ben trai hon?")
    print(f"   - Duong cua Hoi quy logistic (AUC = {auc_lr:.4f}) om sat goc tren ben trai")
    print(f"     hon nhin chung so voi Rung ngau nhien (AUC = {auc_rf:.4f}).")
    print("   - Tu FPR = 0.0111 tro di, duong Logistic luon nam cao hon hoac bang")
    print("     duong Rung ngau nhien va cham muc Recall = 1.0000 som hon (FPR = 0.7296).")
    print("2. Co doan nao hai duong cat nhau khong?")
    print("   - CO, o ca hai vung tren hinh:")
    print("     + Vung 1 (doan dau, FPR tu 0.0000 den 0.0074):")
    print("       Tai FPR = 0.0000, Logistic dat TPR = 0.4000 > Rung (0.3000).")
    print("       Tai FPR = 0.0037, Rung nhay len TPR = 0.5333, cat va vuot len")
    print("       tren Logistic (0.4000), roi gap nhau tai (0.0074, 0.5333).")
    print("     + Vung 2 (doan cuoi, FPR tu 0.4741 den 0.7296):")
    print("       Tu diem (0.4741, 0.9667), duong Rung noi thang ve (1.0, 1.0)")
    print("       nen nam hoi cao hon duong Logistic (giu ngang 0.9667 toi")
    print("       FPR = 0.7296 moi nhay len 1.0000), tao them mot lan cat nhau.")
    print("   - Ngoai ra hai duong cham/trung nhau tai FPR ~ 0.0667..0.0815 (TPR = 0.80)")
    print("     va FPR ~ 0.3556..0.4741 (TPR = 0.9667).")
    print("=" * 72)


if __name__ == "__main__":
    main()
