# -*- coding: utf-8 -*-
"""
Bài tập 2: Một mạng XOR khác
Yêu cầu:
- Kiểm chứng bằng numpy rằng mạng sau cũng tính đúng XOR:
    h1 = max(0, x1 - x2)
    h2 = max(0, x2 - x1)
    y  = h1 + h2
- Giải thích bằng lời vì sao mạng này đúng.
"""

import numpy as np


def ham_relu(gia_tri_z):
    """Hàm kích hoạt ReLU: max(0, z)."""
    return np.maximum(0.0, gia_tri_z)


def main():
    # 1. Thiết lập bảng chân trị đầu vào X và nhãn XOR chuẩn để đối chiếu
    dau_vao_xor = np.array(
        [[0.0, 0.0],
         [0.0, 1.0],
         [1.0, 0.0],
         [1.0, 1.0]]
    )
    nhan_xor_chuan = np.array([0.0, 1.0, 1.0, 0.0])

    # 2. Biểu diễn mạng dưới dạng ma trận trọng số và hệ số chặn bằng numpy
    #    h1 = max(0,  1*x1 - 1*x2 + 0)
    #    h2 = max(0, -1*x1 + 1*x2 + 0)
    trong_so_an = np.array(
        [[ 1.0, -1.0],
         [-1.0,  1.0]]
    )
    he_so_chan_an = np.array([0.0, 0.0])

    #    y = 1*h1 + 1*h2 + 0
    trong_so_ra = np.array([1.0, 1.0])
    he_so_chan_ra = 0.0

    # 3. Tính lan truyền tiến cho cả 4 điểm cùng lúc bằng phép nhân ma trận
    dau_ra_an = ham_relu(dau_vao_xor @ trong_so_an + he_so_chan_an)
    dau_ra_y = dau_ra_an @ trong_so_ra + he_so_chan_ra
    khop_hoan_toan = bool(np.array_equal(dau_ra_y, nhan_xor_chuan))

    print("=" * 72)
    print("BAI TAP 2: KIEM CHUNG MOT MANG GIAI XOR KHAC BANG NUMPY")
    print("=" * 72)
    print("Cong thuc mang:")
    print("  h1 = max(0, x1 - x2)")
    print("  h2 = max(0, x2 - x1)")
    print("  y  = h1 + h2")
    print("\n--- BANG KIEM CHUNG TREN 4 DAU VAO XOR ---")
    print("  x1    x2    x1-x2   x2-x1    h1     h2      y    XOR chuan")
    for chi_so in range(len(dau_vao_xor)):
        x1_val, x2_val = dau_vao_xor[chi_so]
        h1_val, h2_val = dau_ra_an[chi_so]
        y_val = dau_ra_y[chi_so]
        chuan_val = nhan_xor_chuan[chi_so]
        print(
            f"  {x1_val:.0f}     {x2_val:.0f}     {x1_val - x2_val:4.0f}    {x2_val - x1_val:4.0f}   "
            f"  {h1_val:.0f}      {h2_val:.0f}      {y_val:.0f}        {chuan_val:.0f}"
        )

    print(f"\nKet qua vecto Y: {dau_ra_y}")
    print(f"Kiem tra khop XOR: {'DUNG HOAN TOAN (4/4)' if khop_hoan_toan else 'SAI'}")

    print("\n" + "-" * 72)
    print("GIAI THICH BANG LOI VI SAO MANG NAY DUNG:")
    print("1. Y nghia dai so (tri tuyet doi |x1 - x2|):")
    print("   - Ham ReLU max(0, z) chi giu lai phan duong va xoa phan am ve 0.")
    print("   - Do do h1 = max(0, x1 - x2) chi khac 0 khi x1 > x2, con h2 = max(0, x2 - x1)")
    print("     chi khac 0 khi x2 > x1. Khi cong lai, y = h1 + h2 = |x1 - x2|.")
    print("   - Voi hai bit nhi phan 0 va 1: |x1 - x2| bang 0 khi hai bit giong nhau")
    print("     va bang 1 khi hai bit khac nhau -- chinh xac la dinh nghia phep XOR.")
    print("2. Phan tich tung truong hop dau vao:")
    print("   - (0, 0) va (1, 1): hai so bang nhau nen x1 - x2 = 0 va x2 - x1 = 0;")
    print("     ca hai no-ron an deu tat (h1 = 0, h2 = 0), suy ra y = 0.")
    print("   - (1, 0): x1 - x2 = +1 > 0 nen h1 = 1 bat len, con x2 - x1 = -1 < 0 bi")
    print("     ReLU chan ve h2 = 0, suy ra y = 1 + 0 = 1.")
    print("   - (0, 1): nguoc lai, h1 = 0 tat di, h2 = 1 bat len, suy ra y = 0 + 1 = 1.")
    print("3. Y nghia hinh hoc:")
    print("   - Duong thang x1 - x2 = 0 (tuc x1 = x2) chua ca hai diem (0, 0) va (1, 1).")
    print("   - No-ron h1 bat len o nua mat phang ben phia x1 > x2 (chua diem (1, 0));")
    print("     no-ron h2 bat len o nua mat phang doi dien x2 > x1 (chua diem (0, 1)).")
    print("   - Nho ham ReLU be gap mat phang tai duong cheo x1 = x2 thanh hinh chu V,")
    print("     mang tach duoc hai diem (0, 0), (1, 1) (y = 0) khoi (0, 1), (1, 0) (y = 1),")
    print("     dieu ma mot duong thang don le (mot no-ron) khong the lam duoc.")
    print("=" * 72)


if __name__ == "__main__":
    main()
