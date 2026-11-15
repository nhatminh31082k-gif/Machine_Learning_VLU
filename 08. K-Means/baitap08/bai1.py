# -*- coding: utf-8 -*-
"""
Bài tập 1: Tự tính silhouette
Yêu cầu:
- Viết hàm silhouette_mot_diem(X, nhan, i) tính a, b và s của điểm thứ i
  theo đúng định nghĩa ở mục 4, chỉ dùng numpy.
- Chạy hàm cho mọi khách hàng với K = 5, lấy trung bình, rồi so với
  silhouette_score.
- Trả lời: Khách nào có s nhỏ nhất? Khách đó nằm ở đâu trên Hình 3?
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


def silhouette_mot_diem(X, nhan, i):
    """Tính a, b và chỉ số silhouette s của điểm thứ i bằng numpy."""
    cum_hien_tai = nhan[i]
    # Khoảng cách Euclid từ điểm i tới tất cả các điểm trong tập dữ liệu
    khoang_cach = np.sqrt(((X - X[i]) ** 2).sum(axis=1))

    # Các điểm cùng cụm với điểm i (loại trừ chính điểm i)
    cung_cum = (nhan == cum_hien_tai)
    cung_cum[i] = False

    # Nếu cụm chỉ có duy nhất 1 điểm thì silhouette quy ước bằng 0
    if not np.any(cung_cum):
        return 0.0, 0.0, 0.0

    # a: khoảng cách trung bình tới các điểm cùng cụm
    a = float(khoang_cach[cung_cum].mean())

    # b: khoảng cách trung bình nhỏ nhất tới các điểm của một cụm khác
    cac_cum_khac = [c for c in np.unique(nhan) if c != cum_hien_tai]
    kc_cum_khac = [float(khoang_cach[nhan == c].mean()) for c in cac_cum_khac]
    b = min(kc_cum_khac)

    # s = (b - a) / max(a, b)
    mau_so = max(a, b)
    s = (b - a) / mau_so if mau_so > 0 else 0.0
    return a, b, s


def main():
    bang_khach = pd.read_csv("data/khach_hang.csv")
    dac_trung = bang_khach[["thu_nhap", "diem_chi_tieu"]]
    du_lieu_chuan = StandardScaler().fit_transform(dac_trung)

    mo_hinh = KMeans(n_clusters=5, n_init=10, random_state=0).fit(du_lieu_chuan)
    nhan_cum = mo_hinh.labels_

    # Tính a, b, s cho từng khách hàng
    danh_sach_a = []
    danh_sach_b = []
    danh_sach_s = []
    for idx in range(len(du_lieu_chuan)):
        a_i, b_i, s_i = silhouette_mot_diem(du_lieu_chuan, nhan_cum, idx)
        danh_sach_a.append(a_i)
        danh_sach_b.append(b_i)
        danh_sach_s.append(s_i)

    mang_s = np.array(danh_sach_s)
    trung_binh_tu_tinh = float(mang_s.mean())
    trung_binh_sklearn = float(silhouette_score(du_lieu_chuan, nhan_cum))
    mau_sklearn = silhouette_samples(du_lieu_chuan, nhan_cum)
    sai_so_max = float(np.abs(mang_s - mau_sklearn).max())

    # Tìm khách hàng có s nhỏ nhất
    vi_tri_min = int(np.argmin(mang_s))
    khach_min = bang_khach.iloc[vi_tri_min]
    cum_min = int(nhan_cum[vi_tri_min])

    # Xác định cụm láng giềng gần nhất của khách có s nhỏ nhất
    kc_min = np.sqrt(((du_lieu_chuan - du_lieu_chuan[vi_tri_min]) ** 2).sum(axis=1))
    cum_gan_nhat = min(
        [int(c) for c in np.unique(nhan_cum) if c != cum_min],
        key=lambda c: float(kc_min[nhan_cum == c].mean()),
    )

    print("=" * 72)
    print("BAI TAP 1: TU TINH CHI SO SILHOUETTE BANG NUMPY")
    print("=" * 72)
    print(f"So khach hang          : {len(du_lieu_chuan)}")
    print(f"Silhouette trung binh (tu tinh)     : {trung_binh_tu_tinh:.6f}")
    print(f"Silhouette trung binh (scikit-learn): {trung_binh_sklearn:.6f}")
    print(f"Sai lech lon nhat tung diem         : {sai_so_max:.2e}")
    print(f"Kiem tra khop                       : {'KHOP HOAN TOAN' if np.isclose(trung_binh_tu_tinh, trung_binh_sklearn) else 'KHONG KHOP'}")

    print("\n--- KHACH HANG CO SILHOUETTE NHO NHAT ---")
    print(f"Chi so dong (0-based)  : {vi_tri_min} (dong du lieu thu {vi_tri_min + 1})")
    print(f"Thong tin khach        : tuoi={int(khach_min['tuoi'])}, "
          f"thu_nhap={khach_min['thu_nhap']:.1f} trieu, "
          f"diem_chi_tieu={int(khach_min['diem_chi_tieu'])}")
    print(f"Thuoc cum              : {cum_min} (cum thu nhap vua, chi tieu vua)")
    print(f"Cum lang gieng gan nhat: {cum_gan_nhat} (cum thu nhap cao, chi tieu it)")
    print(f"Gia tri chi tiet       : a = {danh_sach_a[vi_tri_min]:.4f}, "
          f"b = {danh_sach_b[vi_tri_min]:.4f}, s = {mang_s[vi_tri_min]:.4f}")

    print("\n" + "-" * 72)
    print("GIAI THICH:")
    print("1. Y nghia cua a, b va s:")
    print("   - a la khoang cach trung binh tu khach do toi cac khach khac trong")
    print("     cung cum (do do gan ket noi cum; a cang nho cum cang chat).")
    print("   - b la khoang cach trung binh toi cum hang xom gan nhat (do do")
    print("     tach biet voi cum khac; b cang lon thi cang tach xa cum khac).")
    print("   - s = (b - a) / max(a, b) nam trong [-1, 1]. Khi s gan 1 nghia la")
    print("     a << b (khach nam gon trong cum cua minh). Khi s gan 0 nghia la")
    print("     a xap xi b (khach nam ngay vung ranh gioi giua hai cum).")
    print("2. Khach co s nho nhat la ai va nam o dau tren Hinh 3?")
    print(f"   - Do la khach hang thu {vi_tri_min} (thu nhap {khach_min['thu_nhap']:.1f} trieu, "
          f"diem chi tieu {int(khach_min['diem_chi_tieu'])}), s = {mang_s[vi_tri_min]:.4f}.")
    print("   - Tren Hinh 3 (cum giua duoc danh so la Cum 3, mau xanh ngoc),")
    print("     khach nay nam lech han ve phia ria ben phai - phia duoi cua cum")
    print("     giua, troi ra khoang trong giua cum thu nhap vua/chi vua (tam 17.4,")
    print("     49.3) va cum thu nhap cao/chi it (Cum 4 mau cam, tam 30.1, 20.0).")
    print("   - Vi nam o vung bien giua hai cum nay, khoang cach trung binh toi")
    print(f"     cum minh (a = {danh_sach_a[vi_tri_min]:.3f}) gan bang khoang cach toi cum lang gieng")
    print(f"     (b = {danh_sach_b[vi_tri_min]:.3f}), khien chi so s rat nho (xap xi 0.029).")
    print("=" * 72)


if __name__ == "__main__":
    main()
