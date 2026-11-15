# -*- coding: utf-8 -*-
"""
Bài tập 5: DBSCAN trên hai vầng trăng
Yêu cầu:
- Dùng DBSCAN(eps=e, min_samples=5) trong sklearn.cluster trên bộ hai vầng trăng,
  với e lần lượt bằng 0.2, 0.3 và 0.4.
- Với mỗi giá trị, đếm số cụm tìm được và số điểm bị gán nhãn -1, tức điểm nhiễu.
- Trả lời: Có giá trị nào tách được hai vầng trăng tốt hơn K-Means không?
  Nhận xét vai trò của eps.
"""

import pandas as pd
from sklearn.cluster import DBSCAN, KMeans


def main():
    du_lieu_trang = pd.read_csv("data/hai_vang_trang.csv")
    toa_do = du_lieu_trang[["x1", "x2"]]
    nhan_that = du_lieu_trang["nhan"]

    # Đối chiếu lại tỷ lệ khớp của K-Means (mục 5.3)
    nhan_km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(toa_do)
    khop_km = max((nhan_km == nhan_that).mean(), (nhan_km != nhan_that).mean())

    print("=" * 72)
    print("BAI TAP 5: DBSCAN TREN DU LIEU HAI VANG TRANG")
    print("=" * 72)
    print(f"Tong so diem du lieu : {len(du_lieu_trang)}")
    print(f"K-Means (K=2)        : 2 cum, ty le khop nhan that = {khop_km:.4f}")
    print()
    print("  eps   so_cum   so_diem_nhieu (-1)   ty_le_khop_nhan_that")

    for eps in [0.2, 0.3, 0.4]:
        db = DBSCAN(eps=eps, min_samples=5).fit(toa_do)
        nhan_db = db.labels_

        # Nhãn -1 là nhiễu, các nhãn >= 0 là cụm
        so_cum = len(set(nhan_db) - {-1})
        so_nhieu = int((nhan_db == -1).sum())

        if so_cum == 2:
            hop_le = nhan_db != -1
            k0 = (nhan_db[hop_le] == nhan_that[hop_le]).mean()
            ty_le_khop = f"{max(k0, 1 - k0):.4f}"
        elif so_cum == 1:
            # Mọi điểm không nhiễu đều vào chung 1 cụm
            khop_c0 = (nhan_db == nhan_that).mean()
            khop_c1 = ((nhan_db == 0) & (nhan_that == 1)).mean()
            ty_le_khop = f"{max(khop_c0, khop_c1):.4f} (gop 1 cum)"
        else:
            ty_le_khop = "Khong ap dung"

        print(f"  {eps:.1f}   {so_cum:6d}   {so_nhieu:20d}   {ty_le_khop}")

    print("\n" + "-" * 72)
    print("NHAN XET:")
    print("1. Co gia tri nao tach duoc hai vang trang tot hon K-Means khong?")
    print("   - KHONG. Ca ba gia tri eps = 0.2, 0.3 va 0.4 deu chi tim ra DUY NHAT")
    print("     1 cum (gom ca hai vang trang lam mot), khop nhan that kem hon")
    print("     ca muc 0.7500 cua K-Means.")
    print("   - Nguyen nhan: bo du lieu hai_vang_trang.csv co do nhieu tuong doi")
    print("     cao, cac diem o ria giua hai vang trang nam gan nhau. Khi eps >= 0.2")
    print("     (voi min_samples=5), ban kinh lan can du rong de 'bac cau' noi")
    print("     thong hai vang trang thanh mot cum lien thong duy nhat.")
    print("2. Vai tro cua tham so eps:")
    print("   - eps la ban kinh vung lan can de xem cac diem co du gan nhau de")
    print("     noi thanh cum hay khong:")
    print("     + Khi eps tang (0.2 -> 0.3 -> 0.4), nhieu diem ria lot vao ban")
    print("       kinh cua vung dac nen so diem nhieu (-1) giam dan (40 -> 10 -> 3),")
    print("       nhung cac cum gan nhau cang de bi dinh lien thanh 1 cum.")
    print("     + Nguoc lai, neu giam eps xuong qua nho (< 0.2) de tranh bac cau")
    print("       thi mat do diem khong du dat min_samples=5, hai vang trang se bi")
    print("       vo vun thanh nhieu cum nho va rat nhieu diem nhieu.")
    print("=" * 72)


if __name__ == "__main__":
    main()
