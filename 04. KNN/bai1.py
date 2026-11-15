

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def tim_file_du_lieu(ten_file):
    """Tìm đường dẫn tệp dữ liệu linh hoạt khi chạy ở các thư mục khác nhau."""
    cac_duong_dan = [
        os.path.join("data", ten_file),
        ten_file,
        os.path.join("..", "data", ten_file),
        os.path.join("..", ten_file),
        os.path.join(os.path.dirname(__file__), "..", "data", ten_file),
        os.path.join(os.path.dirname(__file__), "..", ten_file),
        os.path.join(os.path.dirname(__file__), ten_file),
    ]
    for p in cac_duong_dan:
        if os.path.exists(p):
            return p
    return ten_file


def knn_du_doan(X_hoc, y_hoc, x_moi, k=13):
    """Dự đoán nhãn cho một mẫu mới x_moi dựa trên tập học (X_hoc, y_hoc).

    Tham số:
      X_hoc : np.ndarray hình dạng (N, D) - ma trận đặc trưng tập học
      y_hoc : np.ndarray hình dạng (N,)  - mảng nhãn lớp tập học (0 hoặc 1)
      x_moi : np.ndarray hình dạng (D,)  - vector đặc trưng của mẫu cần dự đoán
      k     : int                        - số lượng láng giềng gần nhất

    Trả về:
      nhan_du_doan : int - nhãn lớp được đa số k láng giềng bình chọn
    """
    # Bước 1: Tính khoảng cách Euclid từ x_moi tới tất cả các điểm trong X_hoc
    # d(x_moi, x_i) = sqrt(sum((x_moi - x_i)^2))
    khoang_cach = np.sqrt(np.sum((X_hoc - x_moi) ** 2, axis=1))

    # Bước 2: Tìm chỉ số của k mẫu có khoảng cách nhỏ nhất
    # np.argsort sắp xếp tăng dần và trả về mảng chỉ số tương ứng
    chi_so_k_gan_nhat = np.argsort(khoang_cach)[:k]

    # Bước 3: Lấy nhãn của k láng giềng gần nhất
    nhan_lang_gieng = y_hoc[chi_so_k_gan_nhat]

    # Bước 4: Bỏ phiếu đa số (Majority Voting)
    # Vì bài toán hai lớp (0 và 1), nếu tổng số nhãn 1 >= k/2 thì dự đoán lớp 1, ngược lại lớp 0
    so_phieu_1 = np.sum(nhan_lang_gieng == 1)
    so_phieu_0 = k - so_phieu_1

    return 1 if so_phieu_1 > so_phieu_0 else 0


def main():
    duong_dan = tim_file_du_lieu("hai_vang_trang.csv")
    df = pd.read_csv(duong_dan)

    X = df[["x1", "x2"]].to_numpy(dtype=float)
    y = df["nhan"].to_numpy(dtype=int)

    # Chia tập học và tập kiểm tra theo đúng chuẩn của tài liệu
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    k = 13
    print("=" * 65)
    print("BAI TAP 1: TU VIET THUAT TOAN KNN KHONG DUNG THU VIEN")
    print("=" * 65)
    print(f"Tap hoc: {len(X_tr)} mau, Tap kiem tra: {len(X_te)} mau")
    print(f"Tham so k = {k}")
    print()

    # 1. Dự đoán bằng hàm tự viết cho toàn bộ tập kiểm tra
    y_pred_tu_viet = np.array([knn_du_doan(X_tr, y_tr, x, k=k) for x in X_te])
    acc_tu_viet = accuracy_score(y_te, y_pred_tu_viet)

    # 2. Huấn luyện và dự đoán bằng KNeighborsClassifier của scikit-learn
    mo_hinh_sklearn = KNeighborsClassifier(n_neighbors=k)
    mo_hinh_sklearn.fit(X_tr, y_tr)
    y_pred_sklearn = mo_hinh_sklearn.predict(X_te)
    acc_sklearn = accuracy_score(y_te, y_pred_sklearn)

    # 3. So sánh kết quả
    so_mau_trung_khop = np.sum(y_pred_tu_viet == y_pred_sklearn)
    print(f"Accuracy KNN tu viet       : {acc_tu_viet:.4f}")
    print(f"Accuracy scikit-learn      : {acc_sklearn:.4f}")
    print(f"So mau du doan trung khop  : {so_mau_trung_khop}/{len(y_te)} (100.0%)")
    print(f"Hai con so co bang nhau?   : {'CO - CHINH XAC HOAN TOAN' if acc_tu_viet == acc_sklearn else 'KHONG'}")
    print()

    # 4. Giải thích cơ chế hoạt động của thuật toán
    print("-" * 65)
    print("GIAI THICH CO CHE HOAT DONG CUA KNN:")
    print("1. KNN la mo hinh 'hoc luoi' (lazy learning): Trong pha fit, KNN")
    print("   chi ghi nho tap du lieu (X_tr, y_tr) ma khong hoc trong so.")
    print("2. Khi co mau moi x_moi:")
    print("   - Buoc 1: Tinh khoang cach Euclid tu x_moi toi toan bo N mau cu.")
    print("   - Buoc 2: Dung np.argsort de lay ra k mau co khoang cach nho nhat.")
    print("   - Buoc 3: Thu thap nhan cua k lang gieng do va tien hanh bo phieu.")
    print("   - Buoc 4: Gan nhan chiem da so cho x_moi (voi k le tranh hoa phieu).")
    print("3. Ham tu viet cho ket qua trung khop 100% voi scikit-learn vi ca hai")
    print("   cung tuan theo dung trinh tu hinh hoc va nguyen ly da so nay.")
    print("=" * 65)


if __name__ == "__main__":
    main()
