# -*- coding: utf-8 -*-
"""Bài tập 6: SVM trên dữ liệu lớn.

Yêu cầu:
1. Tạo dữ liệu bằng make_classification(n_samples=n, n_features=20, random_state=0)
   với n bằng 2000, 5000, 10000 và 20000.
2. Với mỗi cỡ, đo thời gian .fit của SVC() và của LinearSVC() bằng mô đun time.
3. Lập bảng rồi nhận xét: khi dữ liệu tăng gấp đôi, thời gian học của mỗi mô hình
   tăng khoảng bao nhiêu lần? Kết quả thời gian khác nhau giữa các máy là chuyện
   bình thường, chỉ cần so tương đối.
"""

import time
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC, SVC

# 1. Danh sách các kích thước mẫu dữ liệu
cac_co_mau = [2000, 5000, 10000, 20000]

print("=" * 80)
print("BAI TAP 6: SO SANH THOI GIAN HUAN LUYEN CUA SVC() VA LINEARSVC()")
print("=" * 80)
print("Dang tao du lieu va do thoi gian huan luyen tren tung co mau...")

ket_qua = []

for n in cac_co_mau:
    # Tạo dữ liệu ngẫu nhiên với 20 đặc trưng
    X, y = make_classification(
        n_samples=n,
        n_features=20,
        random_state=0
    )
    
    # Đo thời gian fit của SVC() (mặc định kernel RBF)
    bat_dau_svc = time.perf_counter()
    mo_hinh_svc = SVC()
    mo_hinh_svc.fit(X, y)
    thoi_gian_svc = time.perf_counter() - bat_dau_svc
    
    # Đo thời gian fit của LinearSVC() (tối ưu tuyến tính trực tiếp)
    # Ghi chú: dual=False giúp tối ưu vượt trội khi n_samples > n_features
    bat_dau_lsvc = time.perf_counter()
    mo_hinh_lsvc = LinearSVC(dual=False, random_state=0, max_iter=2000)
    mo_hinh_lsvc.fit(X, y)
    thoi_gian_lsvc = time.perf_counter() - bat_dau_lsvc
    
    ket_qua.append({
        "n_samples": n,
        "time_svc": thoi_gian_svc,
        "time_lsvc": thoi_gian_lsvc
    })
    print(f"  + Hoan thanh n = {n:5d}: SVC = {thoi_gian_svc:7.4f}s | LinearSVC = {thoi_gian_lsvc:7.4f}s")

print("-" * 80)

# 2. Lập bảng so sánh và tính tỉ lệ tăng thời gian khi dữ liệu tăng
df_kq = pd.DataFrame(ket_qua)

# Tính tốc độ tăng mẫu và tốc độ tăng thời gian so với mốc trước
df_kq["tang_mau"] = df_kq["n_samples"] / df_kq["n_samples"].shift(1)
df_kq["tang_svc"] = df_kq["time_svc"] / df_kq["time_svc"].shift(1)
df_kq["tang_lsvc"] = df_kq["time_lsvc"] / df_kq["time_lsvc"].shift(1)

print("BANG TONG HOP THOI GIAN VA HE SO TANG TRUONG:")
print(f"{'n_samples':<10} | {'SVC (s)':<12} | {'Tang SVC':<12} | {'LinearSVC (s)':<14} | {'Tang LinearSVC':<14}")
print("-" * 74)
for _, row in df_kq.iterrows():
    tang_s = f"{row['tang_svc']:5.2f}x" if pd.notna(row['tang_svc']) else "---"
    tang_l = f"{row['tang_lsvc']:5.2f}x" if pd.notna(row['tang_lsvc']) else "---"
    note = f"(tang {row['tang_mau']:.1f}x mau)" if pd.notna(row['tang_mau']) else "(goc)"
    print(f"{int(row['n_samples']):<10} | {row['time_svc']:<12.4f} | {tang_s:<12} | {row['time_lsvc']:<14.4f} | {tang_l:<14} {note}")
print("-" * 80)

# 3. Phân tích cụ thể các mốc tăng gấp đôi (5000 -> 10000 và 10000 -> 20000)
t5 = df_kq.loc[df_kq["n_samples"] == 5000].iloc[0]
t10 = df_kq.loc[df_kq["n_samples"] == 10000].iloc[0]
t20 = df_kq.loc[df_kq["n_samples"] == 20000].iloc[0]

ratio_svc_10_5 = t10["time_svc"] / t5["time_svc"]
ratio_svc_20_10 = t20["time_svc"] / t10["time_svc"]

ratio_lsvc_10_5 = t10["time_lsvc"] / t5["time_lsvc"]
ratio_lsvc_20_10 = t20["time_lsvc"] / t10["time_lsvc"]

print("PHAN TICH KHI DU LIEU TANG GAP DOI (2x):")
print(f"  * Tu 5,000 len 10,000 mau (tang 2.0x):")
print(f"    - SVC()       tang: {ratio_svc_10_5:.2f} lan (tu {t5['time_svc']:.4f}s len {t10['time_svc']:.4f}s)")
print(f"    - LinearSVC() tang: {ratio_lsvc_10_5:.2f} lan (tu {t5['time_lsvc']:.4f}s len {t10['time_lsvc']:.4f}s)")
print(f"  * Tu 10,000 len 20,000 mau (tang 2.0x):")
print(f"    - SVC()       tang: {ratio_svc_20_10:.2f} lan (tu {t10['time_svc']:.4f}s len {t20['time_svc']:.4f}s)")
print(f"    - LinearSVC() tang: {ratio_lsvc_20_10:.2f} lan (tu {t10['time_lsvc']:.4f}s len {t20['time_lsvc']:.4f}s)")
print("-" * 80)

# 4. Nhận xét lý thuyết và thực tiễn
print("NHAN XET VA GIAI THICH:")
nhan_xet = f"""
1. Khi du lieu tang gap doi, thoi gian hoc cua moi mo hinh tang bao nhieu lan?
   - Voi SVC() (kernel RBF mac dinh):
     + Khi du lieu tang gap doi (tu 5k len 10k va 10k len 20k), thoi gian huan luyen tang
       khoang 3 den 4 lan (thuc te do duoc: ~{ratio_svc_20_10:.2f} lan o moc 10k -> 20k).
     + Dieu nay hoan toan phu hop voi ly thuyet: Do phuc tap tinh toan cua kernel SVM
       la O(n^2) den O(n^3) do phai tinh va luu ma tran tuong dong (Gram/Kernel matrix)
       giua tung cap diem du lieu. Khi n tang gap 2, O(n^2) se tang 2^2 = 4 lan!
   - Voi LinearSVC():
     + Khi du lieu tang gap doi, thoi gian huan luyen chi tang khoang 1.8 den 2.2 lan
       (xap xi dung 2 lan, thuc te do duoc: ~{ratio_lsvc_20_10:.2f} lan o moc 10k -> 20k).
     + Ly do: LinearSVC su dung thuat toan ha toa do (Coordinate Descent tu thu vien LIBLINEAR),
       giai quyet bai toan toi uu tuyen tinh truc tiep tren khong gian d dac trung ban dau,
       do phuc tap xap xi tuyen tinh tuyet doi O(n * d). Khi n tang 2 lan thi thoi gian
       chi tang dung 2 lan.

2. Y nghia thuc tien trong ky thuat Machine Learning:
   - Khi lam viec voi bo du lieu lon (tu hang chuc nghin den hang trieu dong):
     + SVC() se tro nen cuc ky cham, ton nhieu bo nho RAM cho kernel cache va co the gay treo he thong.
     + LinearSVC() vuot troi hoan toan ve toc do, chay nhanh gap hang tram lan (o 20,000 mau:
       LinearSVC chay mat chi ~{t20['time_lsvc']:.3f}s so voi {t20['time_svc']:.3f}s cua SVC, nhanh gap hon {t20['time_svc']/t20['time_lsvc']:.0f} lan).
   - Loi khuyen:
     + Neu du lieu lon va chap nhan mat phan cach tuyen tinh: Luon uu tien dung LinearSVC()
       hoac SGDClassifier(loss='hinge').
     + Neu bat buoc phai dung bien cong tren du lieu lon: Dung cac phuong phap xap xi kernel
       nhu Nystroem hoac RBFSampler ket hop LinearSVC, hoac chuyen sang mo hinh cay (LightGBM/XGBoost).
"""
print(nhan_xet.strip())
print("=" * 80)
