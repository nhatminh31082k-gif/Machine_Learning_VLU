# -*- coding: utf-8 -*-
"""
Bài tập 4: Lưu ảnh đã nén
Yêu cầu:
- Dựa vào k5_nen_anh.py, tạo ảnh 16 màu rồi lưu bằng plt.imsave(..., anh_moi),
  với anh_moi là mảng có dạng (427, 640, 3). Lưu cả ảnh gốc, rồi so dung lượng
  hai tệp.
- Làm lại với một bức ảnh của chính em, đọc bằng plt.imread. Chú ý ảnh PNG có
  thể có kênh thứ tư là độ trong suốt, phải bỏ kênh đó trước khi phân cụm.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image


def nen_anh_k_mau(mang_anh, so_mau=16, so_diem_hoc=5000):
    """Nén màu ảnh về `so_mau` màu bằng K-Means và trả về ảnh mới dạng [0, 1]."""
    # Nếu là ảnh PNG 4 kênh RGBA thì bỏ kênh alpha (độ trong suốt) thứ tư
    if mang_anh.ndim == 3 and mang_anh.shape[2] == 4:
        mang_anh = mang_anh[:, :, :3]

    # Chuẩn hóa giá trị pixel về khoảng [0.0, 1.0]
    if np.issubdtype(mang_anh.dtype, np.integer) or mang_anh.max() > 1.0:
        anh_chuan = mang_anh.astype(np.float64) / 255.0
    else:
        anh_chuan = mang_anh.astype(np.float64)

    cao, rong, _ = anh_chuan.shape
    cac_diem = anh_chuan.reshape(-1, 3)

    # Học K-Means trên mẫu ngẫu nhiên cho nhanh rồi gán nhãn cho toàn bộ điểm ảnh
    rng = np.random.default_rng(0)
    so_mau_hoc = min(so_diem_hoc, len(cac_diem))
    diem_hoc = cac_diem[rng.choice(len(cac_diem), so_mau_hoc, replace=False)]

    km = KMeans(n_clusters=so_mau, n_init=4, random_state=0).fit(diem_hoc)
    nhan = km.predict(cac_diem)
    anh_moi = np.clip(km.cluster_centers_[nhan].reshape(cao, rong, 3), 0.0, 1.0)
    return anh_chuan, anh_moi


def main():
    print("=" * 72)
    print("BAI TAP 4: NEN MAU ANH VA SO SANH DUNG LUONG TEP PNG")
    print("=" * 72)

    # --- Phần 1: Ảnh mẫu china.jpg ---
    anh_china = load_sample_image("china.jpg")
    _, china_16_mau = nen_anh_k_mau(anh_china, so_mau=16)

    duong_dan_goc = "baitap08/anh_goc.png"
    duong_dan_16 = "baitap08/anh_16_mau.png"
    plt.imsave(duong_dan_goc, anh_china)
    plt.imsave(duong_dan_16, china_16_mau)

    kb_goc = os.path.getsize(duong_dan_goc) / 1024.0
    kb_16 = os.path.getsize(duong_dan_16) / 1024.0
    ty_le_china = kb_goc / kb_16

    print("\n--- PHAN 1: ANH MAU CHINA.JPG ---")
    print(f"Kich thuoc mang anh_moi : {china_16_mau.shape}")
    print(f"Dung luong anh goc PNG  : {kb_goc:7.1f} KB ({duong_dan_goc})")
    print(f"Dung luong anh 16 mau   : {kb_16:7.1f} KB ({duong_dan_16})")
    print(f"Giam dung luong tep PNG : nho hon {ty_le_china:.2f} lan "
          f"(giam {(1 - kb_16 / kb_goc) * 100:.1f}%)")

    # --- Phần 2: Ảnh cá nhân RGBA PNG đọc bằng plt.imread ---
    duong_dan_ca_nhan = "baitap08/anh_cua_em.png"
    anh_doc = plt.imread(duong_dan_ca_nhan)
    so_kenh_ban_dau = anh_doc.shape[2] if anh_doc.ndim == 3 else 1

    _, ca_nhan_16_mau = nen_anh_k_mau(anh_doc, so_mau=16)
    duong_dan_ca_nhan_16 = "baitap08/anh_cua_em_16_mau.png"
    plt.imsave(duong_dan_ca_nhan_16, ca_nhan_16_mau)

    kb_ca_nhan_goc = os.path.getsize(duong_dan_ca_nhan) / 1024.0
    kb_ca_nhan_16 = os.path.getsize(duong_dan_ca_nhan_16) / 1024.0
    ty_le_ca_nhan = kb_ca_nhan_goc / kb_ca_nhan_16

    print("\n--- PHAN 2: ANH CA NHAN (PNG CO KENH TRONG SUOT) ---")
    print(f"Shape doc tu plt.imread : {anh_doc.shape} ({so_kenh_ban_dau} kenh RGBA)")
    print(f"Shape sau khi bo kenh 4 : {ca_nhan_16_mau.shape} (3 kenh RGB)")
    print(f"Dung luong anh goc PNG  : {kb_ca_nhan_goc:7.1f} KB ({duong_dan_ca_nhan})")
    print(f"Dung luong anh 16 mau   : {kb_ca_nhan_16:7.1f} KB ({duong_dan_ca_nhan_16})")
    print(f"Giam dung luong tep PNG : nho hon {ty_le_ca_nhan:.2f} lan "
          f"(giam {(1 - kb_ca_nhan_16 / kb_ca_nhan_goc) * 100:.1f}%)")

    print("\n" + "-" * 72)
    print("NHAN XET:")
    print("- PNG nen khong mat du lieu (lossless) bang thuat toan loc va DEFLATE.")
    print("  Khi anh chi con 16 mau dong nhat theo tung vung, du lieu lap lai")
    print("  nhieu hon rat nhieu nen tep PNG cua anh 16 mau nhe hon han anh goc.")
    print("- Voi anh PNG doc bang plt.imread, can kiem tra shape[2] == 4 de bo")
    print("  kenh thu tu (alpha - do trong suot) truoc khi reshape(-1, 3) cho")
    print("  K-Means, tranh tinh khoang cach tren kenh trong suot hoac loi shape.")
    print("=" * 72)


if __name__ == "__main__":
    main()
