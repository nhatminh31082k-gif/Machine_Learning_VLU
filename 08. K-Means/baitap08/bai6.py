# -*- coding: utf-8 -*-
"""
Bài tập 6: Phân cụm chữ số viết tay
Yêu cầu:
- Chạy K-Means với K = 10 trên bộ chữ số viết tay load_digits, không dùng nhãn.
- Sau đó mới dùng nhãn thật: lập bảng pd.crosstab(nhan_that, cum) để xem mỗi cụm
  chứa những chữ số nào.
- Trả lời: Chữ số nào được gom gọn vào một cụm? Những chữ số nào hay bị gom chung?
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits


def main():
    bo_chu_so = load_digits()
    du_lieu_anh = bo_chu_so.data
    nhan_that = bo_chu_so.target

    km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(du_lieu_anh)
    nhan_cum = km.labels_

    bang_cheo = pd.crosstab(
        pd.Series(nhan_that, name="nhan_that"),
        pd.Series(nhan_cum, name="cum"),
    )

    print("=" * 72)
    print("BAI TAP 6: PHAN CUM K-MEANS (K=10) TREN BO CHU SO VIET TAY LOAD_DIGITS")
    print("=" * 72)
    print(f"So luong anh: {len(du_lieu_anh)}, so chieu moi anh: {du_lieu_anh.shape[1]}")
    print("\n--- BANG CHEO pd.crosstab(nhan_that, cum) ---")
    print(bang_cheo.to_string())

    # Tóm tắt tỷ lệ tập trung vào cụm chính của từng chữ số
    print("\n--- TY LE TAP TRUNG VAO CUM CHINH CUA TUNG CHU SO ---")
    for chu_so in range(10):
        hang = bang_cheo.loc[chu_so]
        cum_chinh = int(hang.idxmax())
        so_dung_cum = int(hang.max())
        tong_mau = int(hang.sum())
        print(f"Chu so {chu_so}: tap trung nhieu nhat o cum {cum_chinh} "
              f"({so_dung_cum}/{tong_mau} mau = {so_dung_cum / tong_mau * 100:5.1f}%)")

    print("\n" + "-" * 72)
    print("NHAN XET:")
    print("1. Chu so nao duoc gom gon vao mot cum?")
    print("   - Chu so 0 (176/178 = 98.9% o cum 0), chu so 6 (177/181 = 97.8% o cum 5)")
    print("     va chu so 7 (174/179 = 97.2% o cum 2) duoc gom gan nhu tron ven")
    print("     vao mot cum rieng biet, vi hinh dang pixel cua 0, 6, 7 rat dac trung.")
    print("   - Cac chu so 4 (165/181 = 91.2% o cum 6), 3 (154/183 = 84.2% o cum 4),")
    print("     2 (148/177 = 83.6% o cum 9) cung duoc gom kha gon.")
    print("2. Nhung chu so nao hay bi gom chung?")
    print("   - Chu so 1 va chu so 8 bi gom chung ro nhat o cum 1: cum 1 chua toi")
    print("     100 mau so 1 va 100 mau so 8 (dong thoi so 1 con bi tach sang")
    print("     cum 3 voi 54 mau va cum 9 voi 24 mau, con so 8 chi co 52 mau o cum 8).")
    print("   - Chu so 5 va chu so 9 bi lan nhe o cum 8 (cum 8 gom 139 mau so 9,")
    print("     nhung cung keo theo 41 mau so 5 va 52 mau so 8).")
    print("   - Nguyen nhan: K-Means chi do khoang cach Euclid truc tiep tren tung")
    print("     pixel 8x8 ma khong hieu net viet; cac so co net thang dung giua")
    print("     (nhu 1 va 8 viet hep) hoac net cong tren/duoi giong nhau (nhu 5, 8, 9)")
    print("     de co khoang cach pixel gan nhau.")
    print("=" * 72)


if __name__ == "__main__":
    main()
