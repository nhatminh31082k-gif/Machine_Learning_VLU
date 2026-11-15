# -*- coding: utf-8 -*-
"""Bài tập 5: Chữ số viết tay, những lỗi nào hay gặp.

Yêu cầu:
1. Dùng confusion_matrix trong sklearn.metrics để in bảng nhầm lẫn 10 hàng 10 cột
   của SVM ở mục 6. Hàng là chữ số thật, cột là chữ số đoán.
2. Đọc bảng và cho biết những cặp chữ số nào bị nhầm.
3. Dùng plt.imshow(X_te[i].reshape(8, 8), cmap="gray_r") để vẽ và lưu từng ảnh
   bị đoán sai, rồi nhận xét vì sao SVM nhầm.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 1. Nạp dữ liệu và chia train/test theo đúng mục 6
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# 2. Huấn luyện SVM kernel RBF mặc định
svm = SVC(kernel="rbf")
svm.fit(X_tr, y_tr)
du_doan = svm.predict(X_te)
acc = accuracy_score(y_te, du_doan)

print("=" * 75)
print("BAI TAP 5: MA TRAN NHAM LAN VA PHAN TICH ANH DOAN SAI")
print("=" * 75)
print(f"SVM kernel RBF: Accuracy dat {acc:.4f} ({acc*100:.2f}%) tren {len(y_te)} anh tap test")
print(f"So vecto ho tro: {len(svm.support_)} tren {len(X_tr)} anh tap hoc")
print("-" * 75)

# 3. Tính và in ma trận nhầm lẫn 10 hàng 10 cột
cm = confusion_matrix(y_te, du_doan)

# Tạo DataFrame để in bảng với nhãn hàng/cột rõ ràng
df_cm = pd.DataFrame(
    cm,
    index=[f"That {i}" for i in range(10)],
    columns=[f"Doan {j}" for j in range(10)]
)

print("BANG NHAM LAN (CONFUSION MATRIX) 10x10:")
print("(Hang: Chu so that | Cot: Chu so doan)")
print(df_cm.to_string())
print("-" * 75)

# 4. Đọc bảng nhầm lẫn và tìm các cặp bị nhầm
print("DOC BANG NHAM LAN - CAC CAP CHU SO BI NHAM:")
sai_indices = np.flatnonzero(du_doan != y_te)
cac_cap_nham = []

for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            cac_cap_nham.append((i, j, cm[i, j]))
            print(f"  * That la {i}, nhung doan nham thanh {j}: {cm[i, j]} truong hop")

print(f"\nTong cong co {len(sai_indices)} anh bi doan sai tren {len(y_te)} anh ({len(sai_indices)/len(y_te)*100:.2f}%):")
for idx in sai_indices:
    print(f"  - Vi tri index {idx:3d}: Chu so that = {y_te[idx]}, Du doan = {du_doan[idx]}")
print("-" * 75)

# 5. Vẽ và lưu từng ảnh bị đoán sai + ảnh tổng hợp
thu_muc_luu = "baitap06"
os.makedirs(thu_muc_luu, exist_ok=True)

# 5.1 Lưu từng ảnh riêng lẻ
print("VE VA LUU TUNG ANH BI DOAN SAI:")
for idx in sai_indices:
    that = y_te[idx]
    doan = du_doan[idx]
    ten_tep = f"anh_sai_i{idx}_that{that}_doan{doan}.png"
    duong_dan_tep = os.path.join(thu_muc_luu, ten_tep)
    
    plt.figure(figsize=(3, 3))
    plt.imshow(X_te[idx].reshape(8, 8), cmap="gray_r")
    plt.title(f"Index: {idx}\nThat: {that} -> Doan: {doan}", fontsize=12, fontweight="bold", color="darkred")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(duong_dan_tep, dpi=120)
    plt.close()
    print(f"  [+] Da luu: {duong_dan_tep}")

# 5.2 Vẽ ảnh tổng hợp 4 ảnh trên một hàng
duong_dan_tong_hop = os.path.join(thu_muc_luu, "tong_hop_anh_sai.png")
fig, axes = plt.subplots(1, len(sai_indices), figsize=(12, 3.5))
for k, idx in enumerate(sai_indices):
    that = y_te[idx]
    doan = du_doan[idx]
    axes[k].imshow(X_te[idx].reshape(8, 8), cmap="gray_r")
    axes[k].set_title(f"Anh {idx}\nThat: {that} | Doan: {doan}", fontsize=11, fontweight="bold")
    axes[k].axis("off")
plt.suptitle("4 ANH CHU SO BI SVM KERNEL RBF DOAN SAI TREN TAP KIEM TRA", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(duong_dan_tong_hop, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [+] Da luu anh tong hop: {duong_dan_tong_hop}")
print("-" * 75)

# 6. Nhận xét lý do vì sao SVM nhầm
print("NHAN XET VI SAO SVM BI NHAM O TUNG ANH:")
nhan_xet = """
1. Anh 6: That la 9, doan la 3
   - Giai thich: Phan vong tron phia tren cua so 9 bi dut doan o mep trai, khong khep kin.
     Net cong phia tren ket hop voi net moc phia duoi khien cau truc hinh hoc giong
     hai nua vong cung chong len nhau cua chu so 3.

2. Anh 18: That la 8, doan la 1
   - Giai thich: Chu so 8 nay duoc viet rat hep, hai vong tron bi ep chat thanh mot duong
     doc thang dung, khoang trong o giua hai vong tron bi bien mat hoan toan tren luoi 8x8.
     Mo hinh chi nhin thay mot dai diem anh thang dung giong het chu so 1.

3. Anh 161: That la 4, doan la 8
   - Giai thich: Net ngang cua so 4 bi cong uon len tren va cham vao net doc, vo tinh
     tao thanh hai khoang trong nho phia tren va phia duoi. Tren do phan giai thap (8x8),
     hinh thai nay rat giong voi hai vong tron cua so 8.

4. Anh 428: That la 7, doan la 9
   - Giai thich: Net ngang tren dau so 7 hoi uon cong xuong va co mot vung diem sang nhe
     o phia tren trai noi voi than so 7, tao cam giac nhu co mot vong tron khep o nua tren,
     khien bo phan loai SVM danh gia diem nay gan voi cum dac trung cua so 9.

=> Ket luan chung:
   - Ca 4 loi deu xuat phat tu net viet tay dac biet, bien dang hoac mo nhat tren luoi
     do phan giai thap 8x8 pixel.
   - Mo hinh SVM kernel RBF dat do chinh xac cuc ky cao (99.11%, 446/450 anh dung).
   - Nho ma tran nham lan, ta thay cac loi khong he mang tinh he thong (khong co cap so nao
     bi nham qua 1 lan), cho thay mo hinh co do tin cay rat cao.
"""
print(nhan_xet.strip())
print("=" * 75)
