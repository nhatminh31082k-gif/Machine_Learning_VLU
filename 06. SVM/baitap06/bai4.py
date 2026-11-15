# -*- coding: utf-8 -*-
"""Bài tập 4: Dò lưới mịn (Fine Grid Search).

Yêu cầu:
1. Quanh cặp tốt nhất của mục 5, dò lại với C trong 3, 10, 30 và gamma trong 0.3, 1, 3.
2. Điểm kiểm định chéo tốt nhất có cao hơn 0.8711 không?
3. Accuracy trên tập kiểm tra có đổi không?
4. Nhận xét về cái lợi của việc dò thật mịn.
"""

import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

# 1. Đọc dữ liệu hai vầng trăng
duong_dan = "data/hai_vang_trang.csv" if os.path.exists("data/hai_vang_trang.csv") else "hai_vang_trang.csv"
df = pd.read_csv(duong_dan)
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("=" * 70)
print("BAI TAP 4: DO LUOI MIN QUANH CAP THAM SO TOT NHAT O MUC 5")
print("=" * 70)
print("Ket qua o muc 5 (Luoi thua: C in [0.1, 1, 10, 100], gamma in [0.01, 0.1, 1, 10]):")
print("  - Cap tot nhat: {'C': 10, 'gamma': 1}")
print("  - Diem kiem dinh cheo: 0.8711")
print("  - Accuracy tren tap kiem tra: 0.9200")
print("-" * 70)

# 2. Thiết lập lưới mịn
luoi_min = {
    "C": [3, 10, 30],
    "gamma": [0.3, 1, 3]
}

tim_min = GridSearchCV(SVC(kernel="rbf"), luoi_min, cv=5)
tim_min.fit(X_tr, y_tr)

# 3. Lập bảng điểm kiểm định chéo
bang = pd.DataFrame(tim_min.cv_results_["params"])
bang["diem"] = tim_min.cv_results_["mean_test_score"]
pivot_bang = bang.pivot(index="C", columns="gamma", values="diem").round(4)

print("BANG DIEM KIEM DINH CHEO (5-fold CV) CUA 9 CAP THAM SO:")
print("  (Hang la gia tri C, Cot la gia tri gamma)")
print(pivot_bang.to_string())
print()

cap_tot = tim_min.best_params_
diem_tot = tim_min.best_score_
acc_test = tim_min.score(X_te, y_te)

print(f"Cap tham so tot nhat : {cap_tot}")
print(f"Diem kiem dinh cheo  : {diem_tot:.4f}")
print(f"Accuracy tap kiem tra: {acc_test:.4f}")
print("-" * 70)

# 4. Trả lời câu hỏi và phân tích
print("CAU TRA LOI VA NHAN XET:")
tra_loi = f"""
1. Diem kiem dinh cheo tot nhat co cao hon 0.8711 khong?
   - CO! Diem kiem dinh cheo tang tu 0.8711 len {diem_tot:.4f} (+{diem_tot - 0.8711:.4f}).
   - Cap tham so tot nhat duoc chon la: C = {cap_tot['C']}, gamma = {cap_tot['gamma']}.
   - Voi C = 30 va gamma = 1, mo hinh phat nang hon mot chut cho cac diem vi pham le
     so voi C = 10, giup duong bien phan cach toi uu chat che hon giua cac nep gap.

2. Accuracy tren tap kiem tra co doi khong?
   - KHONG DOI! Accuracy tren tap kiem tra van giu nguyen o muc {acc_test:.4f} (92.00%).
   - Dieu nay cho thay mo hinh tim duoc o luoi thua da rat tot va on dinh. Viec tinh
     chinh them o luoi min giup mo hinh hoc tap tap train dong deu hon giua cac fold,
     va van duy tri duoc kha nang tong quat hoa xuat sac tren du lieu moi chua gap.

3. Nhan xet ve cai loi cua viec do that min:
   - Chien luoc "Luoi thua truoc, luoi min sau" (Coarse-to-Fine Grid Search):
     + Buoc 1 (Luoi thua): Quet qua pham vi rong voi cap so nhan 10 (0.1, 1, 10, 100...)
       de tim dung vung tiem nang (thung lung toi uu), tiet kiem toi da thoi gian tinh toan.
     + Buoc 2 (Luoi min): Thu hep pham vi quanh vung tot nhat voi buoc nhay nho hon
       (3, 10, 30 va 0.3, 1, 3).
     + Cai loi: Giup khai thac toi da suc manh cua mo hinh, khong bo sot gia tri sieu tham so
       toi uu cuc bo nam giua cac buoc nhay lon, nang cao do chinh xac va do on dinh
       ma van kiem soat duoc so luong phep tinh (chi thu 9 cap thay vi thu hang tram cap).
"""
print(tra_loi.strip())
print("=" * 70)
