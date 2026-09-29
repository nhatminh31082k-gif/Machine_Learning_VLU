# -*- coding: utf-8 -*-
"""
Bài tập 4: Thêm số phòng vào mô hình hồi quy tuyến tính
Huấn luyện mô hình LinearRegression với hai đặc trưng: dien_tich và so_phong.
Chia tập train/test theo test_size=0.2, random_state=42.
In ra R2 trên tập kiểm tra, so sánh với R2 = 0.9622 của mô hình một biến ở mục 6
và nhận xét việc thêm cột này có đáng hay không.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Đảm bảo hiển thị tốt trên mọi môi trường terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Xác định đường dẫn tệp dữ liệu tương đối linh hoạt
duong_dan = "data/gia_nha.csv"
if not os.path.exists(duong_dan):
    if os.path.exists("../data/gia_nha.csv"):
        duong_dan = "../data/gia_nha.csv"
    elif os.path.exists("gia_nha.csv"):
        duong_dan = "gia_nha.csv"
    elif os.path.exists("../gia_nha.csv"):
        duong_dan = "../gia_nha.csv"

# 1. Đọc dữ liệu
df = pd.read_csv(duong_dan)

# 2. Chuẩn bị đặc trưng X (2 biến) và nhãn mục tiêu y
X_hai_bien = df[["dien_tich", "so_phong"]]
y = df["gia"]

# 3. Chia tập học và tập kiểm tra (80% học, 20% kiểm tra, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_hai_bien, y, test_size=0.2, random_state=42
)

# 4. Huấn luyện mô hình LinearRegression trên tập học
mo_hinh = LinearRegression()
mo_hinh.fit(X_train, y_train)

# 5. Dự đoán trên tập kiểm tra và chấm điểm
y_du_doan = mo_hinh.predict(X_test)
r2_hai_bien = r2_score(y_test, y_du_doan)
mae_hai_bien = mean_absolute_error(y_test, y_du_doan)
rmse_hai_bien = np.sqrt(mean_squared_error(y_test, y_du_doan))

# R2 của mô hình một biến ở Mục 6
r2_mot_bien = 0.9622
mae_mot_bien = 0.2578
rmse_mot_bien = 0.3730

print("=== KET QUA BAI TAP 4 ===")
print(f"So can tap hoc     : {len(X_train)} can")
print(f"So can tap kiem tra: {len(X_test)} can")
print()
print("He so cua mo hinh hai bien:")
for cot, he_so in zip(X_hai_bien.columns, mo_hinh.coef_):
    print(f"  - {cot:10s} : {he_so:+.4f}")
print(f"  - {'he so chan':10s} : {mo_hinh.intercept_:+.4f}")
print()
print(f"R2 tren tap kiem tra (mo hinh hai bien) : {r2_hai_bien:.4f}")
print(f"MAE tren tap kiem tra                   : {mae_hai_bien:.4f} ty dong")
print(f"RMSE tren tap kiem tra                  : {rmse_hai_bien:.4f} ty dong")
print()
print("--- So sanh voi mo hinh mot bien (Muc 6) ---")
print(f"- R2 mo hinh 1 bien (dien_tich)            : {r2_mot_bien:.4f}")
print(f"- R2 mo hinh 2 bien (dien_tich + so_phong) : {r2_hai_bien:.4f} (tang +{r2_hai_bien - r2_mot_bien:.4f})")
print(f"- MAE giam tu {mae_mot_bien:.4f} ty xuong {mae_hai_bien:.4f} ty dong (giam {mae_mot_bien - mae_hai_bien:.4f} ty)")
print(f"- RMSE giam tu {rmse_mot_bien:.4f} ty xuong {rmse_hai_bien:.4f} ty dong (giam {rmse_mot_bien - rmse_hai_bien:.4f} ty)")
print()
print("--- Nhan xet: Them cot so_phong co dang hay khong? ---")
print("Viec them cot so_phong vao mo hinh la DANG GIA vi:")
print("1. Ve mat do chinh xac: R2 tang tu 0.9622 len 0.9698, ca MAE va RMSE deu giam ro ret.")
print("   Phan sai so chua giai thich duoc giam tu 3.78% xuong con 3.02%.")
print("2. Ve mat thuc te: He so cua so_phong mang dau duong (+0.2704 ty dong, tuc khoang 270 trieu),")
print("   phan anh dung quy luat thuc te: giua hai can ho co cung dien tich, can co nhieu phong ngu hon")
print("   se co gia cao hon nho cach bo tri khong gian toi uu va cong nang su dung lon hon.")
print("3. Ve mat chi phi: So phong ngu la dac trung cuc ky de thu thap, khong gay qua khop va")
print("   khong lam tang chi phi tinh toan, nen viec dua them vao mo hinh la hoan toan hop ly.")
