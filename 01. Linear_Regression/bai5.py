# -*- coding: utf-8 -*-
"""
Bài tập 5: Thử hai tốc độ học khác nhau trong Gradient Descent
Chạy lại gradient descent với hai tốc độ học:
1. toc_do_hoc = 0.001 (quá nhỏ, hội tụ rất chậm)
2. toc_do_hoc = 1.02  (quá lớn, bước nhảy vọt qua đáy làm phân kỳ)
Ghi lại giá trị MSE ở vòng 200 và giải thích nguyên nhân.
Mở rộng: Khảo sát tốc độ học 0.5, 0.9 và tìm ranh giới ổn định.
"""

import os
import sys
import numpy as np
import pandas as pd

# Đảm bảo hiển thị tốt trên mọi môi trường terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Xác định đường dẫn tệp dữ liệu tương đối linh hoạt
duong_dan = "data/gia_nha.csv"
if not os.path.exists(duong_dan):
    if os.path.exists("../data/gia_nha.csv"):
        duong_dan = "../data/gia_nha.csv"
    elif os.path.exists("gia_nha.csv"):
        duong_dan = "gia_nha.csv"
    elif os.path.exists("../gia_nha.csv"):
        duong_dan = "../gia_nha.csv"

# 1. Đọc dữ liệu
df = pd.read_csv(duong_dan)
x_goc = df["dien_tich"].to_numpy()
y = df["gia"].to_numpy()

# 2. Chuẩn hoá dữ liệu (Z-score standardization)
x_tb = x_goc.mean()
x_do_lech = x_goc.std()
x = (x_goc - x_tb) / x_do_lech
n = len(x)
so_vong = 200

def chay_gradient_descent(toc_do_hoc, in_chi_tiet=True):
    """Chạy gradient descent với một tốc độ học cụ thể."""
    w, b = 0.0, 0.0
    lich_su_mse = {}
    
    if in_chi_tiet:
        print(f"--- Chay Gradient Descent voi toc_do_hoc = {toc_do_hoc} ---")
        print("Vong      w           b              MSE")
        
    for vong in range(1, so_vong + 1):
        y_du_doan = w * x + b
        chenh_lech = y_du_doan - y
        
        # Tính độ dốc (gradient)
        grad_w = (2 / n) * (chenh_lech * x).sum()
        grad_b = (2 / n) * chenh_lech.sum()
        
        # Cập nhật ngược hướng gradient
        w -= toc_do_hoc * grad_w
        b -= toc_do_hoc * grad_b
        
        if vong in (1, 2, 5, 10, 25, 50, 100, 200):
            mse = (((w * x + b) - y) ** 2).mean()
            lich_su_mse[vong] = mse
            if in_chi_tiet:
                print(f"{vong:4d} {w:11.4f} {b:11.4f} {mse:16.4f}")
                
    w_goc = w / x_do_lech
    b_goc = b - w * x_tb / x_do_lech
    return w_goc, b_goc, lich_su_mse[200]

# Kiểm tra nếu người dùng truyền tham số dòng lệnh (ví dụ: python bai5.py 0.001)
if len(sys.argv) > 1:
    try:
        lr_arg = float(sys.argv[1])
        w_g, b_g, mse_200 = chay_gradient_descent(lr_arg, in_chi_tiet=True)
        print(f"\nMSE o vong 200: {mse_200:.4f}")
        print(f"He so goc quy doi: w = {w_g:.6f}, b = {b_g:.6f}")
        sys.exit(0)
    except ValueError:
        pass

# Mặc định chạy cả 2 tốc độ học theo yêu cầu của đề bài
print("================== BAI TAP 5 ==================")
w1, b1, mse1 = chay_gradient_descent(0.001, in_chi_tiet=True)
print()
w2, b2, mse2 = chay_gradient_descent(1.02, in_chi_tiet=True)
print()

print("=== TONG KET SO SANH O VONG LAP THU 200 ===")
print(f"1. Voi toc_do_hoc = 0.001 : MSE o vong 200 = {mse1:.4f}")
print(f"2. Voi toc_do_hoc = 1.02  : MSE o vong 200 = {mse2:.4f} (khoang {mse2:,.0f})")
print(f"3. Doi chieu toc_do_hoc = 0.1 (tai lieu): MSE = 0.1790")
print()
print("=== GIAI THICH VI SAO HAI CON SO KHAC NHAU ===")
print("1. Voi toc_do_hoc = 0.001:")
print("   - Buoc di qua ngan (chi bang 1/100 so voi 0.1). O moi vong lap, w va b chi nhich")
print("     tung chut mot. Do do sau 200 vong, mo hinh van chua kip bo toi day thung lung")
print("     (MSE van con o muc 20.2175, cao hon rat nhieu so voi day 0.1790).")
print("   - Thuat toan van dang hoi tu dung huong nhung can hang nghin vong nua moi toi day.")
print()
print("2. Voi toc_do_hoc = 1.02:")
print("   - Buoc di qua dai, vuot qua bien do on dinh cua ham loi. O moi vong lap, buoc nhay")
print("     khong chi bo qua diem cuc tieu ma con vang sang suon ben kia o do cao lon hon.")
print("   - Sai so bi khuech dai theo cap so nhan qua tung vong (o vong 25 MSE la 317, vong 100")
print("     la 113,845 va den vong 200 tang vot len 290,391,782). Day la hien tuong phan ky (divergence).")
print("   - Con so khong lo nay la ket qua tinh toan dung ve mat so hoc cua hien tuong no gradient, khong phai do may loi.")
print()
print("=== PHAN MO RONG: KHAO SAT TOC DO HOC 0.5, 0.9 VA RANH GIOI ===")
for lr_test in [0.1, 0.5, 0.9, 0.99, 1.0, 1.01]:
    _, _, mse_val = chay_gradient_descent(lr_test, in_chi_tiet=False)
    print(f"  - toc_do_hoc = {lr_test:4.2f} -> MSE vong 200 = {mse_val:14.4f}")

print("\nKet luan ve ranh gioi on dinh:")
print("- Vi du lieu da duoc chuan hoa (x co phuong sai = 1), ma tran Hessian cua ham mat mat MSE")
print("  co cac gia tri rieng bang 2 (dao ham bac hai theo w va b deu bang 2).")
print("- Dieu kien hoi tu cua Gradient Descent la: |1 - 2 * toc_do_hoc| < 1  <=>  0 < toc_do_hoc < 1.0.")
print("- Do do:")
print("  + Khi toc_do_hoc < 1.0 (nhu 0.1, 0.5, 0.9): thuat toan deu hoi tu ve dung day MSE = 0.1790.")
print("  + Khi toc_do_hoc = 1.0: he so |1 - 2| = |-1| = 1, sai so bi dao dong tuan hoan mai quanh day (MSE = 44.8113).")
print("  + Khi toc_do_hoc > 1.0 (nhu 1.01, 1.02): he so > 1, sai so bung no theo cap so nhan (phan ky).")
print("  => Ranh gioi giua hoi tu va phan ky nam CHINH XAC quanh con so 1.0!")
