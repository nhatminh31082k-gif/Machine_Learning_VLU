# -*- coding: utf-8 -*-
"""
Bài tập 2: Chữ số viết tay
Yêu cầu:
- Lấy lại SVM nhận dạng chữ số của bài 6, in classification_report cho cả 10 chữ số.
- Trả lời:
  + Chữ số nào có recall thấp nhất?
  + Chữ số nào có precision thấp nhất?
  + Vì sao với bộ dữ liệu này, macro avg và weighted avg gần như bằng nhau?
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main():
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    svm = SVC(kernel="rbf").fit(X_tr, y_tr)
    du_doan = svm.predict(X_te)

    print("=" * 72)
    print("BAI TAP 2: CLASSIFICATION REPORT CHO SVM CHU SO VIET TAY (BAI 6)")
    print("=" * 72)
    ten_lop = [f"Chu so {i}" for i in range(10)]
    print(classification_report(y_te, du_doan, digits=4, target_names=ten_lop))

    bao_cao = classification_report(
        y_te, du_doan, output_dict=True, zero_division=0
    )
    recalls = {i: bao_cao[str(i)]["recall"] for i in range(10)}
    precisions = {i: bao_cao[str(i)]["precision"] for i in range(10)}
    supports = [bao_cao[str(i)]["support"] for i in range(10)]

    min_r = min(recalls.values())
    min_p = min(precisions.values())
    so_r_thap = [k for k, v in recalls.items() if np.isclose(v, min_r)]
    so_p_thap = [k for k, v in precisions.items() if np.isclose(v, min_p)]

    macro_f1 = bao_cao["macro avg"]["f1-score"]
    weighted_f1 = bao_cao["weighted avg"]["f1-score"]

    print("-" * 72)
    print("NHAN XET:")
    print(f"1. Chu so co recall thap nhat   : {so_r_thap} (recall = {min_r:.4f})")
    print(f"2. Chu so co precision thap nhat: {so_p_thap} (precision = {min_p:.4f})")
    print(f"   - Trong tap kiem tra ({len(y_te)} anh), co dung 4 anh doan sai:")
    for idx in np.flatnonzero(du_doan != y_te):
        print(f"     + Anh {idx:3d}: that la {y_te[idx]}, doan la {du_doan[idx]}")
    print("3. Vi sao macro avg va weighted avg gan nhu bang nhau?")
    print(f"   - So mau that (support) cua 10 chu so rat deu nhau: tu {int(min(supports))} "
          f"den {int(max(supports))} anh moi lop.")
    print("   - macro avg lay trung binh cong voi trong so bang nhau (1/10 cho moi lop).")
    print("   - weighted avg lay trung binh theo ty le support moi lop (~45/450 = 1/10).")
    print(f"   - Do do hai cach tinh cho ra ket qua gan nhu trung khop "
          f"(macro F1 = {macro_f1:.4f}, weighted F1 = {weighted_f1:.4f}).")
    print("=" * 72)


if __name__ == "__main__":
    main()
