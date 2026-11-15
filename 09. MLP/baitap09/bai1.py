# -*- coding: utf-8 -*-
"""
Bài tập 1: Tự dự đoán bằng trọng số đã học
Yêu cầu:
- Học lại mạng 8 nơ-ron ẩn trên bộ XOR như mục 3.
- Lấy mlp.coefs_ và mlp.intercepts_, rồi tự tính lan truyền tiến bằng numpy
  cho 5 điểm đầu của tập kiểm tra: lớp ẩn dùng ReLU, lớp ra dùng sigmoid.
- So kết quả với mlp.predict_proba, cột thứ hai. Hai kết quả phải khớp tới
  nhiều chữ số thập phân.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def ham_relu(gia_tri_z):
    """Hàm kích hoạt ReLU: giữ nguyên giá trị dương, chặn giá trị âm về 0."""
    return np.maximum(0.0, gia_tri_z)


def ham_sigmoid(gia_tri_z):
    """Hàm kích hoạt sigmoid: đưa giá trị thực về khoảng xác suất (0, 1)."""
    return 1.0 / (1.0 + np.exp(-gia_tri_z))


def main():
    # 1. Đọc bộ dữ liệu XOR và chia tập học / tập kiểm tra đúng như mục 3
    bang_du_lieu = pd.read_csv("data/xor.csv")
    dac_trung = bang_du_lieu[["x1", "x2"]]
    nhan = bang_du_lieu["nhan"]

    X_hoc, X_kiem_tra, y_hoc, y_kiem_tra = train_test_split(
        dac_trung, nhan, test_size=0.25, random_state=17, stratify=nhan
    )

    # 2. Huấn luyện mạng MLP 1 lớp ẩn 8 nơ-ron (solver="lbfgs", mặc định ReLU)
    mo_hinh_mlp = MLPClassifier(
        hidden_layer_sizes=(8,),
        solver="lbfgs",
        max_iter=5000,
        random_state=0,
    )
    mo_hinh_mlp.fit(X_hoc, y_hoc)

    # 3. Lấy trọng số và hệ số chặn đã học từ mô hình
    trong_so_an = mo_hinh_mlp.coefs_[0]          # Kích thước (2, 8)
    he_so_chan_an = mo_hinh_mlp.intercepts_[0]   # Kích thước (8,)
    trong_so_ra = mo_hinh_mlp.coefs_[1]          # Kích thước (8, 1)
    he_so_chan_ra = mo_hinh_mlp.intercepts_[1]   # Kích thước (1,)

    # 4. Lấy 5 điểm đầu của tập kiểm tra và tự tính lan truyền tiến bằng numpy
    nam_diem_dau = X_kiem_tra.iloc[:5].to_numpy()
    nhan_that_nam_diem = y_kiem_tra.iloc[:5].to_numpy()

    # Bước 4a: Tính lớp ẩn (tổ hợp tuyến tính rồi qua hàm ReLU)
    tong_lop_an = nam_diem_dau @ trong_so_an + he_so_chan_an
    dau_ra_lop_an = ham_relu(tong_lop_an)

    # Bước 4b: Tính lớp ra (tổ hợp tuyến tính từ lớp ẩn rồi qua hàm sigmoid)
    tong_lop_ra = dau_ra_lop_an @ trong_so_ra + he_so_chan_ra
    xac_suat_tu_tinh = ham_sigmoid(tong_lop_ra).ravel()

    # 5. Lấy xác suất lớp 1 từ scikit-learn (cột thứ hai của predict_proba)
    xac_suat_sklearn = mo_hinh_mlp.predict_proba(X_kiem_tra.iloc[:5])[:, 1]
    sai_so_tung_diem = np.abs(xac_suat_tu_tinh - xac_suat_sklearn)
    sai_so_lon_nhat = float(sai_so_tung_diem.max())

    # In kết quả chi tiết
    print("=" * 76)
    print("BAI TAP 1: TU DU DOAN BANG TRONG SO DA HOC CUA MANG MLP (8 NO-RON AN)")
    print("=" * 76)
    print(f"Kich thuoc W1 (coefs_[0])      : {trong_so_an.shape}")
    print(f"Kich thuoc b1 (intercepts_[0]) : {he_so_chan_an.shape}")
    print(f"Kich thuoc W2 (coefs_[1])      : {trong_so_ra.shape}")
    print(f"Kich thuoc b2 (intercepts_[1]) : {he_so_chan_ra.shape}")
    print(f"Ham kich hoat lop an           : {mo_hinh_mlp.activation}")
    print(f"Ham kich hoat lop ra           : {mo_hinh_mlp.out_activation_}")
    print(f"Accuracy toan tap kiem tra     : {accuracy_score(y_kiem_tra, mo_hinh_mlp.predict(X_kiem_tra)):.4f}")

    print("\n--- SO SANH 5 DIEM DAU TAP KIEM TRA ---")
    print("Diem      x1       x2    Nhan   Tu tinh (numpy)   predict_proba[:,1]   Sai so tuyet doi")
    for chi_so in range(5):
        x1_val, x2_val = nam_diem_dau[chi_so]
        print(
            f"  {chi_so + 1}   {x1_val:7.4f}  {x2_val:7.4f}    {nhan_that_nam_diem[chi_so]}     "
            f"{xac_suat_tu_tinh[chi_so]:15.12f}    {xac_suat_sklearn[chi_so]:17.12f}     {sai_so_tung_diem[chi_so]:.3e}"
        )

    print(f"\nSai lech lon nhat giua tu tinh va scikit-learn: {sai_so_lon_nhat:.3e}")
    print(f"Ket luan kiem tra: {'KHOP CHINH XAC' if np.allclose(xac_suat_tu_tinh, xac_suat_sklearn, atol=1e-12) else 'CO SAI LECH'}")

    print("\n" + "-" * 76)
    print("GIAI THICH CHI TIET BEN TRONG MANG NO-RON:")
    print("1. Cau truc tham so sau khi hoc:")
    print("   - Du lieu XOR co 2 dac trung (x1, x2), lop an co 8 no-ron nen W1 co")
    print("     kich thuoc (2, 8) (16 trong so) va b1 co 8 he so chan.")
    print("   - Bai toan phan loai nhi phan (nhan 0 hoac 1) chi can 1 no-ron dau ra")
    print("     nen W2 co kich thuoc (8, 1) (8 trong so) va b2 co 1 he so chan.")
    print("2. Qua trinh lan truyen tien cho lo 5 diem X (kich thuoc 5 x 2):")
    print("   - Buoc lop an : Z1 = X @ W1 + b1 -> ma tran 5 x 8, moi cot la tong")
    print("     co trong so cua mot no-ron an. Sau do H = max(0, Z1) (ReLU) de be")
    print("     gay tinh tuyen tinh, giup mang tach duoc 4 goc cua bai toan XOR.")
    print("   - Buoc lop ra : z2 = H @ W2 + b2 -> vecto 5 x 1 tong hop dau ra cua")
    print("     8 no-ron an. Vi la phan loai nhi phan, scikit-learn ap dung ham")
    print("     sigmoid(z2) = 1 / (1 + exp(-z2)) de bien z2 thanh xac suat P(y=1|x).")
    print("3. Vi sao khop toi tung chu so thap phan voi predict_proba[:, 1]?")
    print("   - Ham predict_proba tra ve 2 cot: [1 - p, p], trong do cot thu hai")
    print("     chinh la xac suat lop 1 ma ta vua tu tinh bang cong thuc sigmoid.")
    print("   - Sai so ~10^-16 chi la sai so lam tron dau phay dong (float64) trong")
    print("     tinh toan may tinh, chung to ta da mo phong dung 100% phep tinh cua")
    print("     MLPClassifier khi du doan.")
    print("=" * 76)


if __name__ == "__main__":
    main()
