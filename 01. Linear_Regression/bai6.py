# -*- coding: utf-8 -*-
"""
Bài tập 6: Viết hàm dự đoán giá nhà có cảnh báo ngoại suy
Viết hàm du_doan_gia(dien_tich) sử dụng mô hình một biến với:
  w = 0.078367
  b = 0.401752
Cảnh báo nếu diện tích nằm ngoài khoảng dữ liệu đã học [35.5, 117.5] m2.
Chạy thử với ba căn: 60, 80 và 200 m2.
"""

import sys

# Đảm bảo hiển thị tốt trên mọi môi trường terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Hai hệ số tối ưu đã tìm được từ Mục 5 của bài thực hành
HE_SO_GOC_W = 0.078367
HE_SO_CHAN_B = 0.401752

# Giới hạn diện tích quan sát được trong bộ dữ liệu 60 căn hộ
DIEN_TICH_MIN = 35.5
DIEN_TICH_MAX = 117.5


def du_doan_gia(dien_tich):
    """
    Dự đoán giá căn hộ từ diện tích theo mô hình hồi quy tuyến tính một biến:
        y_du_doan = w * dien_tich + b
    Nếu diện tích nằm ngoài khoảng [35.5, 117.5] m2, hàm sẽ in dòng cảnh báo ngoại suy.
    """
    if dien_tich < DIEN_TICH_MIN or dien_tich > DIEN_TICH_MAX:
        print(f"  [CANH BAO] Dien tich {dien_tich:.1f} m2 nam ngoai khoang du lieu da hoc [{DIEN_TICH_MIN}, {DIEN_TICH_MAX}] m2! Du doan mang tinh chat ngoai suy (extrapolation) va do tin cay khong cao.")
        
    gia_du_doan = HE_SO_GOC_W * dien_tich + HE_SO_CHAN_B
    return gia_du_doan


print("=== KET QUA BAI TAP 6 ===")
print("Mo hinh hoi quy: gia = 0.078367 * dien_tich + 0.401752")
print(f"Khoang du lieu hop le: tu {DIEN_TICH_MIN} m2 den {DIEN_TICH_MAX} m2\n")

danh_sach_dien_tich = [60.0, 80.0, 200.0]

for dt in danh_sach_dien_tich:
    print(f"--- Du doan cho can ho rong {dt:.0f} m2 ---")
    gia = du_doan_gia(dt)
    print(f"  -> Gia du doan: {gia:.4f} ty dong (lam tron: {gia:.3f} ty dong)")
    print()

print("--- Giai thich y nghia cua canh bao ---")
print("1. Can 80 m2 nam trong khoang du lieu [35.5, 117.5] m2, du doan ra 6.671 ty dong,")
print("   dung khop voi ket qua tinh toan o Muc 5.")
print("2. Can 200 m2 nam rat xa tat ca 60 can ho trong du lieu mau (can lon nhat chi 117.5 m2).")
print("   Mo hinh chi dang keo dai duong thang mot cach may moc (ngoai suy). Ngoai thuc te,")
print("   can ho 200 m2 thuong la penthouse/duplex cao cap voi cau truc gia va phan khuc hoan toan khac.")
print("   Do do, dong canh bao giup nguoi su dung khong bi nham lan ve do tin cay cua con so du doan.")
