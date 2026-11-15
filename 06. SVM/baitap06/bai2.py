# -*- coding: utf-8 -*-
"""Bài tập 2: Thêm một điểm.

Yêu cầu:
1. Thêm vào 24 điểm của mục 2 một điểm lớp 1 tại (-1, -1), học lại và in w1, w2, b.
2. Sau đó bỏ điểm đó, thay bằng một điểm lớp 1 tại (4.5, 4), học lại và in lần nữa.
3. Lần nào đường ranh giới thay đổi? Giải thích bằng khái niệm vectơ hỗ trợ.
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# 1. Tập dữ liệu gốc 24 điểm từ mục 2
X_goc, y_goc = make_blobs(n_samples=24, centers=[[2, 2], [6, 6]],
                          cluster_std=0.9, random_state=0)
y_goc = 1 - y_goc

svm_goc = SVC(kernel="linear", C=1000)
svm_goc.fit(X_goc, y_goc)
w_goc = svm_goc.coef_[0]
b_goc = svm_goc.intercept_[0]

print("=" * 70)
print("BAI TAP 2: ANH HUONG CUA DIEM MOI TOI DUONG RANH GIOI SVM")
print("=" * 70)
print("MO HINH GOC (24 diem):")
print(f"  w1 = {w_goc[0]:.4f}, w2 = {w_goc[1]:.4f}, b = {b_goc:.4f}")
print(f"  Chi so cac vecto ho tro: {[int(i) for i in svm_goc.support_]}")
print(f"  So vecto ho tro: {len(svm_goc.support_)}")
print("-" * 70)

# 2. Lần 1: Thêm điểm lớp 1 tại (-1, -1)
diem_a = np.array([[-1.0, -1.0]])
lop_a = np.array([1])
X_lan1 = np.vstack([X_goc, diem_a])
y_lan1 = np.concatenate([y_goc, lop_a])

svm_lan1 = SVC(kernel="linear", C=1000)
svm_lan1.fit(X_lan1, y_lan1)
w_lan1 = svm_lan1.coef_[0]
b_lan1 = svm_lan1.intercept_[0]

f_a_cu = svm_goc.decision_function(diem_a)[0]
la_sv_a = (len(X_lan1) - 1) in svm_lan1.support_

print("LAN 1: Them diem lop 1 tai (-1, -1):")
print(f"  w1 = {w_lan1[0]:.4f}, w2 = {w_lan1[1]:.4f}, b = {b_lan1:.4f}")
print(f"  Chi so cac vecto ho tro: {[int(i) for i in svm_lan1.support_]}")
print(f"  Gia tri f((-1, -1)) theo mo hinh goc: {f_a_cu:+.4f} (rat lon > 1)")
print(f"  Diem (-1, -1) co tro thanh vecto ho tro khong? {'CO' if la_sv_a else 'KHONG'}")
thay_doi_1 = not (np.allclose(w_goc, w_lan1, atol=1e-3) and np.isclose(b_goc, b_lan1, atol=1e-3))
print(f"  => Ket luan Lan 1: Duong ranh gioi {'THAY DOI' if thay_doi_1 else 'HOAN TOAN KHONG DOI'}")
print("-" * 70)

# 3. Lần 2: Bỏ (-1, -1), thay bằng điểm lớp 1 tại (4.5, 4)
diem_b = np.array([[4.5, 4.0]])
lop_b = np.array([1])
X_lan2 = np.vstack([X_goc, diem_b])
y_lan2 = np.concatenate([y_goc, lop_b])

svm_lan2 = SVC(kernel="linear", C=1000)
svm_lan2.fit(X_lan2, y_lan2)
w_lan2 = svm_lan2.coef_[0]
b_lan2 = svm_lan2.intercept_[0]

f_b_cu = svm_goc.decision_function(diem_b)[0]
la_sv_b = (len(X_lan2) - 1) in svm_lan2.support_

print("LAN 2: Bo (-1, -1), thay bang diem lop 1 tai (4.5, 4):")
print(f"  w1 = {w_lan2[0]:.4f}, w2 = {w_lan2[1]:.4f}, b = {b_lan2:.4f}")
print(f"  Chi so cac vecto ho tro: {[int(i) for i in svm_lan2.support_]}")
print(f"  Gia tri f((4.5, 4)) theo mo hinh goc: {f_b_cu:+.4f} (am, vi pham ran gioi cu)")
print(f"  Diem (4.5, 4) co tro thanh vecto ho tro khong? {'CO' if la_sv_b else 'KHONG'}")
thay_doi_2 = not (np.allclose(w_goc, w_lan2, atol=1e-3) and np.isclose(b_goc, b_lan2, atol=1e-3))
print(f"  => Ket luan Lan 2: Duong ranh gioi {'THAY DOI RO RET' if thay_doi_2 else 'KHONG DOI'}")
print("-" * 70)

# 4. Trả lời câu hỏi và giải thích bằng khái niệm vectơ hỗ trợ
print("CAU TRA LOI VA GIAI THICH:")
tra_loi = """
1. Lan nao duong ranh gioi thay doi?
   - Lan 1 (them diem (-1, -1)): Duong ranh gioi KHONG thay doi.
     He so w va b giu nguyen tuyet doi (w1 = -0.8314, w2 = -0.3549, b = 4.8204).
   - Lan 2 (them diem (4.5, 4.0)): Duong ranh gioi THAY DOI MANH.
     He so doi han thanh w1 = -2.6030, w2 = -0.8068, b = 15.9409.

2. Giai thich bang khai niem vecto ho tro (Support Vectors):
   - Ban chat cua SVM la duong ranh gioi duoc xac dinh DUY NHAT boi cac vecto ho tro
     (nhung diem nam tren mep le hoac vi pham le). Cac diem con lai nam sau ben ngoai
     le khong he dong gop gi vao ham muc tieu va he so nhan Lagrange cua chung bang 0.
   - O Lan 1: Diem (-1, -1) thuoc lop 1 nam rat xa ranh gioi ve phia lop 1 (f = +6.00 > 1),
     nam sau ben ngoai mep le f(x) = 1. Vi diem nay khong vi pham le va khong cham vao
     mep le, no KHONG tro thanh vecto ho tro. Do do, bai toan toi uu khong thay doi
     rang buoc tich cuc, duong ranh gioi va do rong le duoc giu nguyen 100%.
   - O Lan 2: Diem (4.5, 4.0) thuoc lop 1 nhung lai nam lan sau vao vung cua lop 0
     (theo mo hinh cu, f(4.5, 4.0) = -0.34 < 0, tuc bi phan loai sai va vi pham le tram trong).
     Khi them diem nay vao, no tro thanh mot vecto ho tro moi (chi so 24 trong tap moi).
     Diem nay tao ra mot rang buoc moi buoc SVM phai xoay va doi mat phang phan tach
     de cach ly no, dan toi w va b thay doi hoan toan.
"""
print(tra_loi.strip())
print("=" * 70)
