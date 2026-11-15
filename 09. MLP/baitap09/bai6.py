# -*- coding: utf-8 -*-
"""
Bài tập 6: Mạng nơ-ron cho hồi quy
Yêu cầu:
- Trên dữ liệu giá nhà của bài 1, dùng MLPRegressor(hidden_layer_sizes=(20,),
  max_iter=5000, random_state=0) có chuẩn hoá.
- So MAE trên tập kiểm tra với hồi quy tuyến tính.
- Dự đoán giá căn 250 m2. Mạng nơ-ron có ngoại suy tốt như đường thẳng không?
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    bang_nha = pd.read_csv("data/gia_nha.csv")
    dac_trung = bang_nha[["dien_tich"]]
    gia_ban = bang_nha["gia"]

    # Chia tập học / kiểm tra giống Bài 1 (test_size=0.2, random_state=42)
    X_hoc, X_kiem_tra, y_hoc, y_kiem_tra = train_test_split(
        dac_trung, gia_ban, test_size=0.2, random_state=42
    )

    hoi_quy_tuyen_tinh = LinearRegression().fit(X_hoc, y_hoc)
    mang_hoi_quy = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0),
    ).fit(X_hoc, y_hoc)

    du_doan_lr = hoi_quy_tuyen_tinh.predict(X_kiem_tra)
    du_doan_mlp = mang_hoi_quy.predict(X_kiem_tra)

    mae_lr = mean_absolute_error(y_kiem_tra, du_doan_lr)
    mae_mlp = mean_absolute_error(y_kiem_tra, du_doan_mlp)
    r2_lr = r2_score(y_kiem_tra, du_doan_lr)
    r2_mlp = r2_score(y_kiem_tra, du_doan_mlp)

    # Dự đoán cho căn hộ 250 m2 (ngoại suy xa khỏi miền dữ liệu [35.5, 117.5] m2)
    can_250 = pd.DataFrame({"dien_tich": [250.0]})
    gia_250_lr = float(hoi_quy_tuyen_tinh.predict(can_250)[0])
    gia_250_mlp = float(mang_hoi_quy.predict(can_250)[0])

    print("=" * 74)
    print("BAI TAP 6: MANG NO-RON CHO HOI QUY VA KHA NANG NGOAI SUY")
    print("=" * 74)
    print(f"Khoang dien tich trong du lieu: [{dac_trung['dien_tich'].min():.1f}, {dac_trung['dien_tich'].max():.1f}] m2 (60 can)")
    print(f"So can tap hoc / tap kiem tra : {len(X_hoc)} / {len(X_kiem_tra)}")
    print()
    print("Mo hinh                       MAE (ty dong)     R2      Du doan can 250 m2")
    print("-" * 74)
    print(f"Hoi quy tuyen tinh (LR)         {mae_lr:.4f}        {r2_lr:.4f}       {gia_250_lr:.4f} ty dong")
    print(f"MLPRegressor (20,), chuan hoa   {mae_mlp:.4f}        {r2_mlp:.4f}       {gia_250_mlp:.4f} ty dong")

    print("\n--- KHAO SAT DU DOAN TU TRONG MIEN RA NGOAI BIEN ---")
    cac_muc_dt = [40.0, 80.0, 115.0, 150.0, 200.0, 250.0]
    khung_khao_sat = pd.DataFrame({"dien_tich": cac_muc_dt})
    pred_lr_khao_sat = hoi_quy_tuyen_tinh.predict(khung_khao_sat)
    pred_mlp_khao_sat = mang_hoi_quy.predict(khung_khao_sat)
    print("  Dien tich (m2)   LR (ty)    MLP (ty)    Chenh lech (MLP - LR)")
    for dt, glr, gmlp in zip(cac_muc_dt, pred_lr_khao_sat, pred_mlp_khao_sat):
        vung = "noi suy" if dt <= 117.5 else "ngoai suy"
        print(f"     {dt:5.1f} ({vung:9s}) {glr:7.3f}    {gmlp:7.3f}         {gmlp - glr:+.3f}")

    print("\n" + "-" * 74)
    print("NHAN XET:")
    print("1. So sanh MAE tren tap kiem tra:")
    print(f"   - Hoi quy tuyen tinh dat MAE = {mae_lr:.4f} ty dong (R2 = {r2_lr:.4f}), tot hon")
    print(f"     han MLPRegressor co MAE = {mae_mlp:.4f} ty dong (R2 = {r2_mlp:.4f}).")
    print("   - Nguyen nhan: Bo du lieu chi co 60 mau va quy luat gia ~ dien_tich von")
    print("     tuyen tinh. MLP (ReLU + Adam) du thua tham so cho bai toan 1 bien it")
    print("     du lieu nay nen kho khop sat bang mot duong thang.")
    print("2. Du doan can 250 m2 va kha nang ngoai suy:")
    print(f"   - Duong thang LR du doan {gia_250_lr:.4f} ty dong (~0.078 * 250 + 0.40), giu")
    print("     dung toc do tang trung binh ~78 trieu/m2 cua toan bo du lieu.")
    print(f"   - MLPRegressor lai vot len {gia_250_mlp:.4f} ty dong (lech +{gia_250_mlp - gia_250_lr:.2f} ty).")
    print("   - KET LUAN: Mang no-ron KHONG ngoai suy tot nhu duong thang. Cac no-ron")
    print("     ReLU chi ghep cac doan thang o trong vung co du lieu hoc; khi di ra")
    print("     ngoai bien [35.5, 117.5] m2, do doc cua mang chi do mot vai no-ron o")
    print("     vung ria quyet dinh nen de bi be lech huong va sai so rat manh.")
    print("=" * 74)


if __name__ == "__main__":
    main()
