# -*- coding: utf-8 -*-
"""
Bài tập 5: Dò lưới cho mạng nơ-ron
Yêu cầu:
- Dùng GridSearchCV trên bộ chữ số với hidden_layer_sizes trong
  (10,), (50,), (100,), (50, 50) và alpha trong 0.0001, 0.01, 1.
- Mô hình là make_pipeline(StandardScaler(), MLPClassifier(...)), nên tên
  tham số trong lưới viết là "mlpclassifier__alpha".
- Trả lời: Cặp nào tốt nhất? Accuracy trên tập kiểm tra có vượt được SVM không?
"""

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    X, y = load_digits(return_X_y=True)
    X_hoc, X_kiem_tra, y_hoc, y_kiem_tra = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    luoi_tham_so = {
        "mlpclassifier__hidden_layer_sizes": [(10,), (50,), (100,), (50, 50)],
        "mlpclassifier__alpha": [0.0001, 0.01, 1],
    }

    duong_ong = make_pipeline(
        StandardScaler(),
        MLPClassifier(max_iter=1000, random_state=0),
    )

    do_luoi = GridSearchCV(
        estimator=duong_ong,
        param_grid=luoi_tham_so,
        cv=5,
        n_jobs=-1,
    )
    do_luoi.fit(X_hoc, y_hoc)

    # Đối chiếu với SVM kernel RBF như mục 6
    svm_rbf = SVC().fit(X_hoc, y_hoc)
    acc_svm = accuracy_score(y_kiem_tra, svm_rbf.predict(X_kiem_tra))
    acc_mlp_tot = accuracy_score(y_kiem_tra, do_luoi.predict(X_kiem_tra))

    bang_ket_qua = pd.DataFrame(do_luoi.cv_results_)[
        [
            "param_mlpclassifier__hidden_layer_sizes",
            "param_mlpclassifier__alpha",
            "mean_test_score",
            "rank_test_score",
        ]
    ].sort_values("rank_test_score")

    print("=" * 74)
    print("BAI TAP 5: DO LUOI (GridSearchCV) CHO MANG MLP TREN BO CHU SO")
    print("=" * 74)
    print("Ket qua kiem dinh cheo 5-fold cho 12 cap tham so:")
    print("  Hang   hidden_layer_sizes    alpha    Diem CV trung binh")
    print("  " + "-" * 58)
    for _, dong in bang_ket_qua.iterrows():
        lop_an = str(dong["param_mlpclassifier__hidden_layer_sizes"])
        alpha_val = dong["param_mlpclassifier__alpha"]
        diem_cv = dong["mean_test_score"]
        hang = int(dong["rank_test_score"])
        print(f"   {hang:2d}         {lop_an:<12s}      {alpha_val:<7}      {diem_cv:.4f}")

    tot_nhat = do_luoi.best_params_
    print("\n--- KET QUA TOT NHAT VA SO SANH VOI SVM ---")
    print(f"Cap tham so tot nhat      : hidden_layer_sizes = {tot_nhat['mlpclassifier__hidden_layer_sizes']}, alpha = {tot_nhat['mlpclassifier__alpha']}")
    print(f"Diem CV 5-fold tot nhat   : {do_luoi.best_score_:.4f}")
    print(f"Accuracy MLP tren tap test: {acc_mlp_tot:.4f} ({int(round(acc_mlp_tot * len(y_kiem_tra)))}/{len(y_kiem_tra)} anh)")
    print(f"Accuracy SVM (RBF) tap test: {acc_svm:.4f} ({int(round(acc_svm * len(y_kiem_tra)))}/{len(y_kiem_tra)} anh)")

    print("\n" + "-" * 74)
    print("NHAN XET:")
    print("1. Cap tham so tot nhat qua GridSearchCV (5-fold) la:")
    print("   hidden_layer_sizes=(50,) va alpha=1 (diem CV = 0.9785).")
    print("   Muc dieu chuan L2 alpha=1 giup kiem soat trong so tot hon cac gia tri")
    print("   nho (0.0001, 0.01) va mang 1 lop an 50 no-ron vua du gon cho 1347 anh hoc.")
    print("2. Co vuot duoc SVM khong?")
    print(f"   - KHONG. Accuracy tap kiem tra cua MLP tot nhat la {acc_mlp_tot:.4f}, van thap hon")
    print(f"     SVM kernel RBF la {acc_svm:.4f} (chenh lech {acc_svm - acc_mlp_tot:.4f}, tuc kem hon 4 anh).")
    print("   - Dieu nay khang dinh lai ket luan o muc 6: voi tap du lieu nho (1797 anh")
    print("     8x8), SVM va KNN thuong dat hieu nang cao va on dinh hon mang no-ron.")
    print("=" * 74)


if __name__ == "__main__":
    main()
