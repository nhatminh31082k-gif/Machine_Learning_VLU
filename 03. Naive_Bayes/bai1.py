# -*- coding: utf-8 -*-
"""Bài tập 1: Định lý Bayes với ba nguyên nhân.


"""

import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def bayes_ba_nguyen_nhan(tien_nghiem, kha_nang):
    """Tính xác suất hậu nghiệm cho 3 nguyên nhân bằng định lý Bayes.

    tien_nghiem : list/tuple gồm 3 giá trị P(A_i) (tổng bằng 1)
    kha_nang    : list/tuple gồm 3 giá trị P(B | A_i)

    Trả về:
      hau_nghiem : list gồm 3 giá trị P(A_i | B)
      mau_so     : xác suất toàn phần P(B)
    """
    assert len(tien_nghiem) == len(kha_nang) == 3, "Cần đúng 3 nguyên nhân"

    # Tử số cho từng nhánh: P(B | A_i) * P(A_i)
    tu_so = [kn * tn for tn, kn in zip(tien_nghiem, kha_nang)]

    # Mẫu số: công thức xác suất toàn phần P(B) = sum(P(B | A_i) * P(A_i))
    mau_so = sum(tu_so)

    # Xác suất hậu nghiệm: P(A_i | B) = tu_so / mau_so
    hau_nghiem = [t / mau_so for t in tu_so]

    return hau_nghiem, mau_so


def main():
    cac_nhom = ["Hoc deu", "Hoc don", "It hoc"]
    tien_nghiem = [0.50, 0.30, 0.20]   # P(Nhom_i)
    kha_nang = [0.95, 0.70, 0.30]      # P(Qua mon | Nhom_i)

    print("=" * 60)
    print("BAI TAP 1: DINH LY BAYES VOI BA NGUYEN NHAN")
    print("=" * 60)
    print("Thong tin ban dau:")
    for nhom, tn, kn in zip(cac_nhom, tien_nghiem, kha_nang):
        print(f"  - Nhom {nhom:8s}: Tien nghiem P(Nhom) = {tn:.2f}, "
              f"Ty le qua mon P(Qua | Nhom) = {kn:.2f}")
    print()

    # Tính toán
    hau_nghiem, p_qua = bayes_ba_nguyen_nhan(tien_nghiem, kha_nang)

    print(f"Xac suat toan phan sinh vien qua mon P(Qua mon):")
    print(f"  P(Qua mon) = 0.50*0.95 + 0.30*0.70 + 0.20*0.30")
    print(f"             = {p_qua:.4f} (tuc {p_qua * 100:.2f}%)")
    print()

    print("Xac suat hau nghiem khi biet sinh vien da qua mon:")
    for nhom, p_hn in zip(cac_nhom, hau_nghiem):
        print(f"  - P({nhom:8s} | Qua mon) = {p_hn:.4f} ({p_hn * 100:.2f}%)")
    print()

    tong_hau_nghiem = sum(hau_nghiem)
    print(f"Kiem tra tong xac suat hau nghiem: {tong_hau_nghiem:.6f}")
    assert abs(tong_hau_nghiem - 1.0) < 1e-9, "Tong ba xac suat phai bang 1!"
    print("-> Tong ba con so chinh xac bang 1 (dat yeu cau).")
    print("=" * 60)


if __name__ == "__main__":
    main()
