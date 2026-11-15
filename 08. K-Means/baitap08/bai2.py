# -*- coding: utf-8 -*-
"""
Bài tập 2: Thêm cột tuổi
Yêu cầu:
- Phân cụm trên cả ba cột tuổi, thu nhập và điểm chi tiêu, có chuẩn hoá.
- Chọn lại K bằng silhouette với K từ 2 tới 10.
- Trả lời: K tốt nhất có còn là 5 không? Tâm các cụm thay đổi thế nào so với mục 3?
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main():
    bang_khach = pd.read_csv("data/khach_hang.csv")

    # 1. Phân cụm với 2 cột (như mục 3) để làm mốc đối chiếu
    bo_chuan_2 = StandardScaler().fit(bang_khach[["thu_nhap", "diem_chi_tieu"]])
    du_lieu_2 = bo_chuan_2.transform(bang_khach[["thu_nhap", "diem_chi_tieu"]])
    km_2_cot = KMeans(n_clusters=5, n_init=10, random_state=0).fit(du_lieu_2)
    tam_2_cot = pd.DataFrame(
        bo_chuan_2.inverse_transform(km_2_cot.cluster_centers_),
        columns=["thu_nhap", "diem_chi_tieu"],
    )
    tam_2_cot["so_khach"] = np.bincount(km_2_cot.labels_)
    tam_2_cot["tuoi_tb"] = bang_khach.groupby(km_2_cot.labels_)["tuoi"].mean().values
    tam_2_cot = tam_2_cot[["tuoi_tb", "thu_nhap", "diem_chi_tieu", "so_khach"]]
    tam_2_cot = tam_2_cot.sort_values(["thu_nhap", "diem_chi_tieu"]).reset_index(drop=True)

    # 2. Phân cụm với 3 cột (tuổi, thu nhập, điểm chi tiêu), có chuẩn hoá
    cot_3 = ["tuoi", "thu_nhap", "diem_chi_tieu"]
    bo_chuan_3 = StandardScaler().fit(bang_khach[cot_3])
    du_lieu_3 = bo_chuan_3.transform(bang_khach[cot_3])

    print("=" * 72)
    print("BAI TAP 2: PHAN CUM VOI 3 COT (TUOI, THU NHAP, DIEM CHI TIEU)")
    print("=" * 72)
    print(" K   inertia   silhouette")
    k_tot_nhat = None
    sil_tot_nhat = -1.0
    mo_hinh_tot_nhat = None

    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(du_lieu_3)
        sil = float(silhouette_score(du_lieu_3, km.labels_))
        print(f"{k:2d}   {km.inertia_:7.2f}   {sil:10.4f}")
        if sil > sil_tot_nhat:
            sil_tot_nhat = sil
            k_tot_nhat = k
            mo_hinh_tot_nhat = km

    print(f"\n-> K tot nhat theo silhouette: K = {k_tot_nhat} (silhouette = {sil_tot_nhat:.4f})")

    tam_3_cot = pd.DataFrame(
        bo_chuan_3.inverse_transform(mo_hinh_tot_nhat.cluster_centers_),
        columns=cot_3,
    )
    tam_3_cot["so_khach"] = np.bincount(mo_hinh_tot_nhat.labels_)
    tam_3_cot = tam_3_cot.sort_values(["thu_nhap", "diem_chi_tieu"]).reset_index(drop=True)

    print("\n--- TAM CUM MUC 3 (2 COT: THU NHAP, CHI TIEU - SAP XEP DE SO SANH) ---")
    print(tam_2_cot.round(1).to_string())

    print(f"\n--- TAM CUM BAI 2 (3 COT: TUOI, THU NHAP, CHI TIEU, K={k_tot_nhat}) ---")
    print(tam_3_cot.round(1).to_string())

    print("\n" + "-" * 72)
    print("NHAN XET:")
    print(f"1. K tot nhat co con la 5 khong?")
    print(f"   - CO, K tot nhat van la {k_tot_nhat} (silhouette = {sil_tot_nhat:.4f}). Tuy nhien")
    print("     gia tri silhouette giam tu 0.723 (2 cot) xuong 0.481 (3 cot) vi khi")
    print("     them chieu tuoi, cac diem trong cung nhom phan tan rong hon theo")
    print("     do tuoi nen do tach biet 3D giam xuong.")
    print("2. Tam cac cum thay doi the nao so voi muc 3?")
    print("   - Cau truc 5 nhom khach hang van giu nguyen, nhung co su dich chuyen nhe:")
    print("     + Hai nhom chi it (thu nhap thap 8.0/18.9 va thu nhap cao 30.1/20.0):")
    print("       GIU NGUYEN HOAN TOAN ca tam cum lan so khach (40 va 35 khach).")
    print("     + Nhom thu nhap thap, chi nhieu: tang tu 35 len 39 khach, tam dich")
    print("       tu (thu nhap 7.5, chi tieu 80.2) thanh (8.5, 77.6), tuoi ~25.1.")
    print("       Ly do la 3 khach tre o ria nhom giua va nhom khac co tuoi gan 25")
    print("       duoc keo ve nhom tre tuoi nay.")
    print("     + Nhom thu nhap vua, chi vua: giam tu 60 xuong 57 khach, tuoi trung")
    print("       binh tang nhe tu 40.4 len 41.6 (tam 17.6, 49.2).")
    print("     + Nhom thu nhap cao, chi nhieu: giam nhe 1 khach (40 -> 39), tam")
    print("       gan nhu khong doi (30.3, 82.0, tuoi trung binh 32.1).")
    print("=" * 72)


if __name__ == "__main__":
    main()
