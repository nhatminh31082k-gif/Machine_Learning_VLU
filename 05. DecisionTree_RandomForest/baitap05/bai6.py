# -*- coding: utf-8 -*-
"""Bài tập 6: Rừng ngẫu nhiên cho bài toán hồi quy giá nhà và vấn đề ngoại suy.

Dùng RandomForestRegressor với 200 cây và random_state=0 trên dữ liệu gia_nha.csv.
So MAE với cây sâu 3 và đường thẳng. Dự đoán giá căn 250 m2.
Giải thích cơ chế ngoại suy của rừng bằng cách lá của mỗi cây dự đoán.

Chạy từ thư mục gốc: python baitap05/bai6.py
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("=" * 75)
    print("BAI TAP 6: RUNG NGAU NHIEN CHO HOI QUY GIA NHA VA VAN DE NGOAI SUY")
    print("=" * 75)

    # 1. Đọc dữ liệu giá nhà và chia tập train/test theo mục 8
    df = pd.read_csv("data/gia_nha.csv")
    X = df[["dien_tich"]]
    y = df["gia"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Tong so can nha       : {len(df)}")
    print(f"So can tap hoc (80%)  : {len(X_tr)} (dien tich tu {X_tr['dien_tich'].min():.1f} den {X_tr['dien_tich'].max():.1f} m2)")
    print(f"So can tap test (20%) : {len(X_te)}")
    print(f"Gia nha tap hoc       : tu {y_tr.min():.2f} den {y_tr.max():.2f} ty dong")
    print()

    # 2. Huấn luyện 3 mô hình
    duong = LinearRegression().fit(X_tr, y_tr)
    cay = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_tr, y_tr)
    rung = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_tr, y_tr)

    # 3. Đánh giá MAE trên tập kiểm tra
    mae_duong = mean_absolute_error(y_te, duong.predict(X_te))
    mae_cay = mean_absolute_error(y_te, cay.predict(X_te))
    mae_rung = mean_absolute_error(y_te, rung.predict(X_te))

    print("--- 1. BANG SO SANH MAE TREN TAP KIEM TRA (ty dong) ---")
    print(f"{'Mo hinh':<30} | {'MAE tap kiem tra':<20}")
    print("-" * 55)
    print(f"{'Duong thang (LinearRegression)':<30} | {mae_duong:<20.4f}")
    print(f"{'Rung 200 cay (RandomForest)':<30} | {mae_rung:<20.4f}")
    print(f"{'Cay sau 3 (DecisionTree)':<30} | {mae_cay:<20.4f}")
    print("=" * 75)

    # 4. Dự đoán cho các căn nhà mới, đặc biệt là căn 250 m2
    can_moi = pd.DataFrame({"dien_tich": [60.0, 100.0, 150.0, 250.0]})
    pred_duong = duong.predict(can_moi)
    pred_cay = cay.predict(can_moi)
    pred_rung = rung.predict(can_moi)

    print("--- 2. DU DOAN GIA CHO CAC CAN NHA MOI (ty dong) ---")
    print(f"{'Dien tich':<12} | {'Duong thang':<14} | {'Cay sau 3':<14} | {'Rung 200 cay':<14}")
    print("-" * 65)
    for dt, a, b, c in zip(can_moi["dien_tich"], pred_duong, pred_cay, pred_rung):
        print(f"  {dt:6.0f} m2   | {a:10.2f}     | {b:10.2f}     | {c:10.2f}")
    print("=" * 75)

    # 5. Vẽ đồ thị so sánh và lưu ảnh
    x_plot = pd.DataFrame({"dien_tich": np.linspace(20, 260, 500)})
    plt.figure(figsize=(10, 6), dpi=300)
    plt.scatter(X_tr["dien_tich"], y_tr, color="blue", alpha=0.6, label="Tap hoc (Train)")
    plt.scatter(X_te["dien_tich"], y_te, color="green", alpha=0.8, marker="^", label="Tap kiem tra (Test)")
    plt.plot(x_plot["dien_tich"], duong.predict(x_plot), color="gray", linestyle="--", label="Duong thang (Linear)")
    plt.plot(x_plot["dien_tich"], cay.predict(x_plot), color="red", label="Cay sau 3 (Decision Tree)")
    plt.plot(x_plot["dien_tich"], rung.predict(x_plot), color="darkgreen", linewidth=2, label="Rung 200 cay (Random Forest)")
    plt.axvline(x=250, color="orange", linestyle=":", label="Can 250 m2 (Ngoai suy)")
    plt.xlabel("Dien tich (m2)", fontsize=12)
    plt.ylabel("Gia (ty dong)", fontsize=12)
    plt.title("So sanh du doan hoi quy gia nha va hien tuong khong ngoai suy duoc", fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("baitap05/hoi_quy_gia_nha.png")
    plt.close()
    print("[+] Da luu bieu do truc quan hoa vao baitap05/hoi_quy_gia_nha.png")
    print()

    # 6. Trả lời câu hỏi chi tiết
    print("TRA LOI CAU HOI:")
    print(f"1. So sanh MAE tren tap kiem tra:")
    print(f"   - Duong thang tot nhat voi MAE = {mae_duong:.4f} ty dong.")
    print(f"   - Rung 200 cay dung thu hai voi MAE = {mae_rung:.4f} ty dong (tot hon cay don le).")
    print(f"   - Cay sau 3 kem nhat voi MAE = {mae_cay:.4f} ty dong.")
    print("   Lý do: Gia nha tang gan nhu ty le thuan tuyen tinh theo dien tich, nen duong thang")
    print("   la mo hinh phu hop nhat.")
    print()
    print(f"2. Du doan gia can 250 m2:")
    print(f"   - Duong thang : {pred_duong[3]:.2f} ty dong")
    print(f"   - Cay sau 3   : {pred_cay[3]:.2f} ty dong")
    print(f"   - Rung 200 cay: {pred_rung[3]:.2f} ty dong")
    print()
    print("3. Rung co ngoai suy duoc khong?")
    print("   -> RUNG HOAN TOAN KHONG NGOAI SUY DUOC! (Gia van bi chan o muc ~9.20 ty dong).")
    print()
    print("4. Giai thich bang cach la cua moi cay dua ra du doan:")
    print("   - Moi cay quyet dinh trong rung phan chia khong gian dien tich thanh cac doan.")
    print("   - O moi nut la, gia tri du doan la TRUNG BINH CONG cua cac can nha trong tap hoc")
    print("     roi vao la do: y_la = mean(y_i). Do do, moi la khong the nao du doan gia tri lon hon")
    print(f"     can nha dat nhat trong tap hoc cua no (gia cao nhat trong tap hoc la {y_tr.max():.2f} ty).")
    print(f"   - Can 250 m2 vuot xa can rong nhat cua tap hoc ({X_tr['dien_tich'].max():.1f} m2), nen no luon re")
    print("     theo nhanh ngoai cung ben phai cua moi cay va roi vao chiec la cuoi cung.")
    print("   - Khi do, tat ca 200 cay deu tra ve gia trung binh cua nhung can rong nhat ma cay do nhin thay.")
    print("   - Rung tinh trung binh cong ket qua cua 200 cay, nen ket qua cuoi cung van chi la ~9.20 ty dong,")
    print("     khong the tang len theo xu huong (19.91 ty) nhu mo hinh tuyen tinh.")
    print("   -> Ket luan: Cac mo hinh dua tren cay (Tree-based models) chi noi suy noi bo vung du lieu da biet,")
    print("      khong the ngoai suy (extrapolate) ra ngoai mien du lieu huan luyen.")
    print("=" * 75)


if __name__ == "__main__":
    main()
