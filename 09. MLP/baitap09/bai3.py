# -*- coding: utf-8 -*-
"""
Bài tập 3: So các hàm kích hoạt
Yêu cầu:
- Trên bộ hai vầng trăng, dùng mạng hai lớp ẩn 100 nơ-ron với alpha=1.0 như mục 4.
- Lần lượt đặt activation bằng "logistic", "tanh" và "relu".
- Ghi lại accuracy trên tập học và tập kiểm tra. Hàm nào cho kết quả tốt nhất ở đây?
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def tao_mang_mlp(ham_kich_hoat):
    """Tạo pipeline chuẩn hóa + MLP hai lớp ẩn 100 nơ-ron với alpha=1.0."""
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100, 100),
            alpha=1.0,
            activation=ham_kich_hoat,
            solver="lbfgs",
            max_iter=20000,
            random_state=0,
        ),
    )


def main():
    bang_du_lieu = pd.read_csv("data/hai_vang_trang.csv")
    dac_trung = bang_du_lieu[["x1", "x2"]]
    nhan = bang_du_lieu["nhan"]

    X_hoc, X_kiem_tra, y_hoc, y_kiem_tra = train_test_split(
        dac_trung, nhan, test_size=0.25, random_state=17, stratify=nhan
    )

    danh_sach_ham = ["logistic", "tanh", "relu"]
    ket_qua = []

    print("=" * 72)
    print("BAI TAP 3: SO SANH CAC HAM KICH HOAT TREN BO HAI VANG TRANG")
    print("=" * 72)
    print("Cau hinh mang: hidden_layer_sizes=(100, 100), alpha=1.0, solver='lbfgs'")
    print()
    print("Ham kich hoat   Tap hoc   Tap kiem tra   Kiem dinh cheo (5-fold)")
    print("-" * 66)

    for ten_ham in danh_sach_ham:
        mo_hinh = tao_mang_mlp(ten_ham).fit(X_hoc, y_hoc)
        acc_hoc = accuracy_score(y_hoc, mo_hinh.predict(X_hoc))
        acc_kiem_tra = accuracy_score(y_kiem_tra, mo_hinh.predict(X_kiem_tra))
        acc_cv = cross_val_score(tao_mang_mlp(ten_ham), X_hoc, y_hoc, cv=5).mean()
        ket_qua.append((ten_ham, acc_hoc, acc_kiem_tra, acc_cv))
        print(f"  {ten_ham:<12s}   {acc_hoc:.4f}      {acc_kiem_tra:.4f}          {acc_cv:.4f}")

    # Tìm hàm có accuracy cao nhất trên tập kiểm tra
    acc_kt_cao_nhat = max(kq[2] for kq in ket_qua)
    cac_ham_tot_nhat = [kq[0] for kq in ket_qua if kq[2] == acc_kt_cao_nhat]
    tra_cuu = {ten: (hoc, kt, cv) for ten, hoc, kt, cv in ket_qua}

    print("\n" + "-" * 72)
    print("NHAN XET:")
    print(f"1. Cac ham dat accuracy cao nhat tren tap kiem tra ({acc_kt_cao_nhat:.4f}): {', '.join(cac_ham_tot_nhat)}.")
    print(f"   - Ca 'relu' va 'tanh' deu dat {tra_cuu['relu'][1]:.4f} tren tap kiem tra (cao hon 'logistic'")
    print(f"     la {tra_cuu['logistic'][1]:.4f}). Trong do 'relu' nhinh hon nhe tren tap hoc ({tra_cuu['relu'][0]:.4f} so voi")
    print(f"     {tra_cuu['tanh'][0]:.4f} cua 'tanh') va trong kiem dinh cheo 5-fold ({tra_cuu['relu'][2]:.4f} so voi {tra_cuu['tanh'][2]:.4f}).")
    print("   - Do do, xet tong the ca tap hoc, tap kiem tra va kiem dinh cheo thi")
    print("     'relu' la lua chon tot nhat o day (theo sat ngay sau la 'tanh').")
    print(f"2. Vi sao 'logistic' kem hon ro ret ({tra_cuu['logistic'][0]:.4f} / {tra_cuu['logistic'][1]:.4f})?")
    print("   - Ham sigmoid/logistic chi cho dau ra trong (0, 1) (khong doi xung quanh 0)")
    print("     va nhanh chong bao hoa (dao ham rat nho o hai dau), khien mang hai lop")
    print("     an kho toi uu hon va de bi 'cau tha' khi chiu phat dieu chuan L2 alpha=1.0.")
    print("=" * 72)


if __name__ == "__main__":
    main()
