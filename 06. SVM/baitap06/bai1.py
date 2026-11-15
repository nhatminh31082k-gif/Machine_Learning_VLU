# -*- coding: utf-8 -*-
"""Bài tập 1: Tính tay khoảng cách tới ranh giới.

Yêu cầu:
1. Dùng w1, w2, b in ra ở mục 2, tự tính f(x) cho hai điểm (3, 3) và (5, 4.5)
   bằng phép nhân và cộng, không dùng decision_function.
2. Khoảng cách từ một điểm tới đường ranh giới bằng |f(x)| / sqrt(w1^2 + w2^2).
   Tính khoảng cách của hai điểm trên, và của ba vectơ hỗ trợ.
3. Vì sao khoảng cách của ba vectơ hỗ trợ bằng đúng một nửa độ rộng lề?
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# 1. Tái lập mô hình SVM tuyến tính ở mục 2 trên 24 điểm
X, y = make_blobs(n_samples=24, centers=[[2, 2], [6, 6]],
                  cluster_std=0.9, random_state=0)
y = 1 - y  # Lớp 1 là cụm dưới bên trái

svm = SVC(kernel="linear", C=1000)
svm.fit(X, y)

w = svm.coef_[0]
w1, w2 = w[0], w[1]
b = svm.intercept_[0]
do_dai_w = np.sqrt(w1 ** 2 + w2 ** 2)
do_rong_le = 2 / do_dai_w

print("=" * 65)
print("BAI TAP 1: TINH TAY KHOANG CACH TOI RANH GIOI")
print("=" * 65)
print(f"He so hoc duoc tu mo hinh:")
print(f"  w1 = {w1:.6f}, w2 = {w2:.6f}, b = {b:.6f}")
print(f"  Do dai vecto phap tuyen ||w|| = {do_dai_w:.6f}")
print(f"  Do rong cua le (margin = 2/||w||) = {do_rong_le:.6f}")
print(f"  Nua do rong cua le (margin/2)    = {do_rong_le / 2:.6f}")
print("-" * 65)


# 2. Hàm tự tính f(x) bằng phép nhân và cộng thông thường
def tinh_f_tay(x1, x2):
    """Tính f(x) = w1*x1 + w2*x2 + b bằng phép tính thủ công."""
    return w1 * x1 + w2 * x2 + b


def tinh_khoang_cach_tay(x1, x2):
    """Tính khoảng cách d = |f(x)| / ||w||."""
    return abs(tinh_f_tay(x1, x2)) / do_dai_w


# 3. Tính cho hai điểm mới: (3.0, 3.0) và (5.0, 4.5)
diem_moi = [(3.0, 3.0), (5.0, 4.5)]
print("1. KET QUA TINH TAY CHO HAI DIEM MOI:")
for x1, x2 in diem_moi:
    f_tay = tinh_f_tay(x1, x2)
    d_tay = tinh_khoang_cach_tay(x1, x2)
    f_scikit = svm.decision_function([[x1, x2]])[0]
    lop_doan = 1 if f_tay > 0 else 0
    print(f"  Diem ({x1:.1f}, {x2:.1f}):")
    print(f"    - f(x) tinh tay   = {w1:.4f} * {x1} + ({w2:.4f}) * {x2} + ({b:.4f}) = {f_tay:+.4f}")
    print(f"    - f(x) scikit-learn = {f_scikit:+.4f} (khop 100%)")
    print(f"    - Khoang cach d   = |{f_tay:+.4f}| / {do_dai_w:.4f} = {d_tay:.4f}")
    print(f"    - Du doan lop     = {lop_doan} (|f| {'lon' if abs(f_tay) > 1 else 'nho'} hon 1 => {'ngoai le' if abs(f_tay) > 1 else 'trong le'})")
print("-" * 65)

# 4. Tính khoảng cách của ba vectơ hỗ trợ
print("2. KHOANG CACH CUA BA VECTO HO TRO TOI DUONG RANH GIOI:")
for idx in svm.support_:
    x_sv = X[idx]
    y_sv = y[idx]
    f_sv = tinh_f_tay(x_sv[0], x_sv[1])
    d_sv = tinh_khoang_cach_tay(x_sv[0], x_sv[1])
    print(f"  Vecto ho tro idx {idx:2d}: toa do ({x_sv[0]:.2f}, {x_sv[1]:.2f}), lop {y_sv}")
    print(f"    + f(x) tinh tay = {f_sv:+.4f} (|f(x)| = {abs(f_sv):.4f})")
    print(f"    + Khoang cach d = {d_sv:.4f}")
print("-" * 65)

# 5. Giải thích câu hỏi lý thuyết
print("3. GIAI THICH VI SAO KHOANG CACH 3 VECTO HO TRO DUNG BANG NUA DO RONG LE:")
giai_thich = """
- Duong ranh gioi quyet dinh cua SVM la mat phang: f(x) = w1*x1 + w2*x2 + b = 0.
- Hai mep le cua SVM duoc xac dinh boi hai mat phang f(x) = +1 va f(x) = -1.
- Theo dinh nghia cua SVM, cac vecto ho tro (Support Vectors) la cac diem nam
  chinh xac tren hai mep le nay, nghia la moi vecto ho tro deu co |f(x)| = 1.
- Khoang cach tu mot diem bat ky toi ranh gioi f(x) = 0 la: d = |f(x)| / ||w||.
- Voi cac vecto ho tro, vi |f(x)| = 1 nen khoang cach cua chung la:
    d_sv = 1 / ||w||.
- Trong khi do, do rong toan bo con duong le (margin) la khoang cach giua hai
  mep le f(x) = 1 va f(x) = -1, co cong thuc:
    do_rong_le = 2 / ||w||.
- Vi duong ranh gioi f(x) = 0 nam chinh giua hai mep le doi xung, nen:
    d_sv = 1 / ||w|| = (2 / ||w||) / 2 = do_rong_le / 2.
=> Do do, khoang cach tu bat ky vecto ho tro nao toi duong ranh gioi deu bang
   dung mot nua do rong cua le!
"""
print(giai_thich.strip())
print("=" * 65)
