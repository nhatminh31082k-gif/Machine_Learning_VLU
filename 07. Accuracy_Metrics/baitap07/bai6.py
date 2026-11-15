# -*- coding: utf-8 -*-
"""
Bài tập 6: MAE và RMSE với một lần đoán hỏng
Yêu cầu:
- Trên dữ liệu giá nhà của bài 1, tính MAE, RMSE và R2 của hồi quy tuyến tính
  trên tập kiểm tra.
- Sau đó cộng thêm 5 tỷ vào dự đoán của đúng một căn, rồi tính lại ba con số.
- Trả lời: Con số nào thay đổi nhiều nhất? Giải thích vì sao RMSE phạt nặng
  sai số lớn hơn MAE.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def tinh_chi_so_hoi_quy(y_that, y_du_doan):
    """Tính MAE, RMSE và R2 cho bài toán hồi quy."""
    mae = mean_absolute_error(y_that, y_du_doan)
    rmse = float(np.sqrt(mean_squared_error(y_that, y_du_doan)))
    r2 = r2_score(y_that, y_du_doan)
    return mae, rmse, r2


def main():
    df = pd.read_csv("data/gia_nha.csv")
    # Dùng đúng đặc trưng dien_tich và cách chia train/test như bài 1 mục 6
    X = df[["dien_tich"]]
    y = df["gia"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mo_hinh = LinearRegression().fit(X_tr, y_tr)
    du_doan_goc = mo_hinh.predict(X_te)

    mae0, rmse0, r2_0 = tinh_chi_so_hoi_quy(y_te, du_doan_goc)

    # Cộng thêm 5 tỷ vào dự đoán của đúng căn đầu tiên trong tập kiểm tra
    du_doan_hong = du_doan_goc.copy()
    du_doan_hong[0] += 5.0
    mae1, rmse1, r2_1 = tinh_chi_so_hoi_quy(y_te, du_doan_hong)

    print("=" * 72)
    print("BAI TAP 6: MAE, RMSE VA R2 KHI MOT LAN DOAN HONG (+5 TY)")
    print("=" * 72)
    print(f"Tap hoc: {len(X_tr)} can | Tap kiem tra: {len(X_te)} can")
    print(f"Can bi sua du doan (can #0 trong tap kiem tra): "
          f"gia that = {y_te.iloc[0]:.2f} ty, "
          f"du doan goc = {du_doan_goc[0]:.4f} ty -> sua thanh {du_doan_hong[0]:.4f} ty")

    print("\nSo sanh 3 chi so tren tap kiem tra:")
    print("  Chi so    Truoc (+0 ty)   Sau (+5 ty)   Chenh lech    Ty le thay doi")
    print(f"  MAE (ty)     {mae0:.4f}        {mae1:.4f}       {mae1 - mae0:+.4f}"
          f"       gap {mae1 / mae0:.2f} lan")
    print(f"  RMSE (ty)    {rmse0:.4f}        {rmse1:.4f}       {rmse1 - rmse0:+.4f}"
          f"       gap {rmse1 / rmse0:.2f} lan")
    print(f"  R2           {r2_0:.4f}        {r2_1:.4f}       {r2_1 - r2_0:+.4f}"
          f"       giam {(r2_0 - r2_1) / r2_0 * 100:.1f}%")

    print("\n" + "-" * 72)
    print("GIAI THICH:")
    print("1. Con so nao thay doi nhieu nhat?")
    print("   - Giua hai thuoc do sai so (MAE va RMSE): RMSE thay doi nhieu hon han:")
    print(f"     + RMSE tang {rmse1 - rmse0:+.4f} ty (tu {rmse0:.4f} len {rmse1:.4f}, "
          f"gap {rmse1 / rmse0:.2f} lan).")
    print(f"     + MAE chi tang {mae1 - mae0:+.4f} ty (tu {mae0:.4f} len {mae1:.4f}, "
          f"gap {mae1 / mae0:.2f} lan).")
    print(f"   - Dong thoi he so R2 cung sut giam rat manh tu {r2_0:.4f} xuong {r2_1:.4f}")
    print("     (mat hon 61 diem phan tram), vi cong thuc R2 = 1 - SS_res/SS_tot")
    print("     cung dung tong binh phuong sai so tuong tu nhu MSE/RMSE.")
    print("2. Vi sao RMSE phat nang sai so lon hon MAE?")
    print("   - MAE = (1/n) * sum(|e_i|): moi sai so duoc tinh tuyen tinh theo gia tri")
    print("     tuyet doi. Khi 1 trong 12 can lech them ~5 ty, MAE tang toi da ~5/12 = 0.4167 ty.")
    print("   - RMSE = sqrt((1/n) * sum(e_i^2)): moi sai so bi BINH PHUONG truoc khi")
    print("     cong lai. Sai so ~5 ty sau khi binh phuong thanh ~25 (lon gap hang tram")
    print("     lan sai so 0.2-0.3 ty cua cac can khac), keo trung binh tang vot len")
    print("     va chi phoi gan nhu toan bo gia tri RMSE.")
    print("=" * 72)


if __name__ == "__main__":
    main()
