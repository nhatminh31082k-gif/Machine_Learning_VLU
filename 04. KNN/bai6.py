

import sys
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("=" * 75)
    print("BAI TAP 6: DO THOI GIAN .FIT VA .PREDICT CUA KNN (LAZY LEARNING)")
    print("=" * 75)

    # Khởi tạo bộ sinh số ngẫu nhiên có hạt giống cố định
    rng = np.random.default_rng(0)

    # 1,000 mẫu kiểm tra ngẫu nhiên (2 đặc trưng)
    so_mau_test = 1000
    X_test = rng.random((so_mau_test, 2))

    cac_co_mau = [1_000, 10_000, 100_000]
    ket_qua = []

    print(f"Khoi tao tap kiem tra co dinh: {so_mau_test:,} mau, 2 dac trung.")
    print("Dang do thoi gian cho tung co du lieu...")
    print()

    for N in cac_co_mau:
        # Sinh dữ liệu học ngẫu nhiên
        X_train = rng.random((N, 2))
        y_train = rng.integers(0, 2, size=N)

        mo_hinh = KNeighborsClassifier(n_neighbors=5)

        # Đo thời gian huấn luyện .fit
        t_bat_dau_fit = time.perf_counter()
        mo_hinh.fit(X_train, y_train)
        thoi_gian_fit = time.perf_counter() - t_bat_dau_fit

        # Đo thời gian dự đoán .predict cho 1000 mẫu mới
        t_bat_dau_pred = time.perf_counter()
        mo_hinh.predict(X_test)
        thoi_gian_pred = time.perf_counter() - t_bat_dau_pred

        ket_qua.append({
            "N": N,
            "fit": thoi_gian_fit,
            "pred": thoi_gian_pred,
        })

    # In bảng kết quả
    print("1. BANG KET QUA THOI GIAN THUC THI:")
    print("-" * 75)
    print(f"{'Co tap hoc (N)':<18} | {'Thoi gian .fit (s)':<22} | {'Thoi gian .predict (s)':<24}")
    print("-" * 75)
    for r in ket_qua:
        print(f"{r['N']:<18,d} | {r['fit']:<22.6f} | {r['pred']:<24.6f}")
    print("-" * 75)
    print()

    # 2. Giải thích bản chất "Học lười" (Lazy Learning)
    print("2. GIAI THICH VI SAO KNN DUOC GOI LA MO HINH HOC LUOI (LAZY LEARNING):")
    print()
    print("  a) Giai doan huan luyen (.fit) cuc nhanh:")
    print("     - Cac mo hinh 'hoc chu dong' (Eager Learning) nhu Linear Regression, Logistic")
    print("       Regression hay Neural Network phai toi uu ham mat mat, tinh dao ham, cap nhat")
    print("       hang nghin tham so qua nhieu vong lap, ton rat nhieu thoi gian huan luyen.")
    print("     - Nguoc lai, KNN KHONG HOC BAT CU THAM SO NAO. Luc .fit, no chi luu tru toan bo")
    print("       tap du lieu vao bo nho (RAM) hoac dung cay chi muc (KD-tree/Ball-tree).")
    print(f"       Vi the thoi gian .fit cuc ky ngan (chi ton ~{ket_qua[0]['fit']:.4f}s cho 1,000 mau).")
    print()
    print("  b) Giai doan du doan (.predict) ganh toan bo khoi luong tinh toan:")
    print("     - Khi gap mot mau moi can du doan, KNN moi bat dau lam viec: no phai tinh khoang cach")
    print("       tu mau moi do toi TAT CA cac mau trong bo nho, roi sap xep tim k phan tu gan nhat.")
    print("     - Voi tap du lieu cang lon, so phep tinh khoang cach cang nhieu.")
    print("     - Doi voi mo hinh tham so (sau khi da fit xong), du doan 1 mau chi la phep tinh")
    print("       tich vo huong nhanh tuc thi O(1), trong khi KNN phai ton chi phi lon tren toan bo tap du lieu.")
    print()
    print("  -> KET LUAN: 'Hoc luoi' nghia la tri hoan moi cong viec tinh toan cho den khi thuc su")
    print("     phai dua ra du doan. Uu diem la huan luyen tuc thi, de cap nhat du lieu moi;")
    print("     Nhuoc diem la du doan cham va ton rat nhieu bo nho RAM khi du lieu lon.")
    print("=" * 75)


if __name__ == "__main__":
    main()
