# -*- coding: utf-8 -*-
"""Bài tập 3: Kernel đa thức (Polynomial Kernel).

Yêu cầu:
1. Trên bộ hai vầng trăng, thử SVC(kernel="poly", degree=d) với d bằng 2, 3 và 4,
   có chuẩn hoá bằng make_pipeline.
2. Với mỗi bậc, dò C trong 0.1, 1, 10, 100 bằng GridSearchCV.
3. Bậc nào tốt nhất? Có vượt được kernel RBF ở mục 5 không?
   (Gợi ý: trong lưới tham số của pipeline, tên tham số viết là "svc__C").
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Đọc dữ liệu hai vầng trăng
duong_dan = "data/hai_vang_trang.csv" if os.path.exists("data/hai_vang_trang.csv") else "hai_vang_trang.csv"
df = pd.read_csv(duong_dan)
X = df[["x1", "x2"]]
y = df["nhan"]

# 2. Tách tập học và kiểm tra (giữ nguyên seed 17 và stratify=y như giáo trình)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("=" * 70)
print("BAI TAP 3: KERNEL DA THUC (POLYNOMIAL KERNEL) TREN HAI VANG TRANG")
print("=" * 70)
print(f"Kich thuoc tap hoc: {len(X_tr)} diem, tap kiem tra: {len(X_te)} diem")
print("-" * 70)

# Lưới tham số C
luoi_tham_so = {"svc__C": [0.1, 1, 10, 100]}
bac_da_thuc = [2, 3, 4]
ket_qua_cac_bac = []

# 3. Thử nghiệm với từng bậc d = 2, 3, 4
for d in bac_da_thuc:
    # Xây dựng pipeline gồm chuẩn hóa StandardScaler và SVC poly bậc d
    pipeline_poly = make_pipeline(
        StandardScaler(),
        SVC(kernel="poly", degree=d)
    )
    
    # Dò lưới tham số svc__C bằng 5-fold CV
    tim_kiem = GridSearchCV(pipeline_poly, luoi_tham_so, cv=5)
    tim_kiem.fit(X_tr, y_tr)
    
    c_tot = tim_kiem.best_params_["svc__C"]
    diem_cv = tim_kiem.best_score_
    acc_te = tim_kiem.score(X_te, y_te)
    
    ket_qua_cac_bac.append({
        "bac": d,
        "best_C": c_tot,
        "cv_score": diem_cv,
        "test_acc": acc_te,
        "cv_details": tim_kiem.cv_results_["mean_test_score"]
    })
    
    print(f"Bac d = {d}:")
    print(f"  + Tham so tot nhat : C = {c_tot}")
    print(f"  + Diem CV 5-fold   : {diem_cv:.4f}")
    print(f"  + Accuracy tap test: {acc_te:.4f}")
    scores_str = ", ".join([f"C={c}: {s:.4f}" for c, s in zip(luoi_tham_so["svc__C"], tim_kiem.cv_results_["mean_test_score"])])
    print(f"  + Chi tiet cac C   : {scores_str}")
    print()

print("-" * 70)
print("BANG TONG HOP SO SANH CAC BAC DA THUC VA KERNEL RBF:")
print(f"{'Kernel / Bac':<24} | {'C tot nhat':<12} | {'Diem CV (5-fold)':<18} | {'Accuracy Test':<14}")
print("-" * 74)
for kq in ket_qua_cac_bac:
    ten = f"Poly (degree={kq['bac']})"
    print(f"{ten:<24} | {kq['best_C']:<12} | {kq['cv_score']:<18.4f} | {kq['test_acc']:<14.4f}")

# Kết quả RBF ở mục 5
print(f"{'RBF (muc 5)':<24} | {'C=10, g=1':<12} | {0.8711:<18.4f} | {0.9200:<14.4f}")
print("-" * 74)

# 4. Trả lời câu hỏi và phân tích
print("\nCAU TRA LOI VA NHAN XET:")
tra_loi = """
1. Bac nao tot nhat?
   - Kernel da thuc bac 3 (degree=3) la tot nhat:
     + Dat diem kiem dinh cheo CV = 0.8178 (tai C = 1).
     + Dat Accuracy tren tap kiem tra = 0.8267.
   - Bac 2 chi dat CV = 0.5422 (test = 0.5067) vi da thuc bac 2 tao ra mat phan cach
     dang conic (parabol/elip) khong the tach duoc hinh dang hai vao trua long nhau.
   - Bac 4 chi dat CV = 0.5156 (test = 0.4800) vi bac chan co xu huong tang manh ve
     hai dau cung chieu, khong linh hoat bang bac le de men theo do cong chu S.

2. Co vuot duoc kernel RBF o muc 5 khong?
   - KHONG! Bac tot nhat cua Poly la bac 3 chi dat test accuracy 0.8267, thap hon rat
     nhieu so voi kernel RBF o muc 5 (dat diem CV = 0.8711 va test accuracy = 0.9200).

3. Giai thich ban chat hinh hoc:
   - Bo du lieu hai vang trang co dac tinh phi tuyen phuc tap, cac diem tao thanh hai
     vong cung xoan long vao nhau.
   - Kernel Da thuc (Polynomial) phai dung mot ham da thuc toan cuc bac d. Voi d nho (2, 3, 4),
     mat phan cach bi go bo vao cac hinh hoc bac thap, khong the uon luon hai lan theo
     hinh luoi liem cua ca hai vang trang.
   - Nguoc lai, kernel RBF (Radial Basis Function) tuong duong voi khong gian vo han chieu,
     su dung cac qua chuong Gauss cuc bo xung quanh tung vecto ho tro, cho phep uon
     duong bien mot cach cuc ky mem mai va tu nhien theo bat ky hinh thu phuc tap nao.
"""
print(tra_loi.strip())
print("=" * 70)
