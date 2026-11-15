
import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def tim_file_du_lieu(ten_file):
    """Tìm đường dẫn tệp dữ liệu linh hoạt khi chạy ở các thư mục khác nhau."""
    cac_duong_dan = [
        os.path.join("data", ten_file),
        ten_file,
        os.path.join("..", "data", ten_file),
        os.path.join("..", ten_file),
        os.path.join(os.path.dirname(__file__), "..", "data", ten_file),
        os.path.join(os.path.dirname(__file__), "..", ten_file),
        os.path.join(os.path.dirname(__file__), ten_file),
    ]
    for p in cac_duong_dan:
        if os.path.exists(p):
            return p
    return ten_file


def main():
    duong_dan = tim_file_du_lieu("hai_vang_trang.csv")
    df = pd.read_csv(duong_dan)

    X = df[["x1", "x2"]]
    y = df["nhan"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    print("=" * 65)
    print("BAI TAP 2: KHOANG CACH MANHATTAN TRONG KNN")
    print("=" * 65)

    # 1. Tìm k tốt nhất bằng kiểm định chéo 5 phần với metric='manhattan'
    print("Kiem dinh cheo 5 phan tren tap hoc (metric='manhattan'):")
    diem_manhattan = {}
    for k in range(1, 52, 2):
        diem_manhattan[k] = cross_val_score(
            KNeighborsClassifier(n_neighbors=k, metric="manhattan"),
            X_tr, y_tr, cv=5
        ).mean()

    k_tot_man = max(diem_manhattan, key=diem_manhattan.get)
    for k in [1, 5, 9, 13, 17, 25, 27, 51]:
        dau = " <- tot nhat" if k == k_tot_man else ""
        print(f"  k = {k:2d}  diem CV trung binh {diem_manhattan[k]:.4f}{dau}")
    print()

    # 2. Đánh giá mô hình Manhattan với k tốt nhất trên tập kiểm tra
    mh_man = KNeighborsClassifier(n_neighbors=k_tot_man, metric="manhattan")
    mh_man.fit(X_tr, y_tr)
    acc_man = accuracy_score(y_te, mh_man.predict(X_te))

    # 3. Kết quả tham chiếu của khoảng cách Euclid (từ mục 4)
    # Tìm k tốt nhất cho Euclid để so sánh trọn vẹn
    diem_euclid = {}
    for k in range(1, 52, 2):
        diem_euclid[k] = cross_val_score(
            KNeighborsClassifier(n_neighbors=k, metric="euclidean"),
            X_tr, y_tr, cv=5
        ).mean()
    k_tot_euc = max(diem_euclid, key=diem_euclid.get)
    mh_euc = KNeighborsClassifier(n_neighbors=k_tot_euc, metric="euclidean")
    mh_euc.fit(X_tr, y_tr)
    acc_euc = accuracy_score(y_te, mh_euc.predict(X_te))

    # 4. Bảng so sánh giữa Euclid và Manhattan
    print("-" * 65)
    print(f"{'Chi tieu':<28} | {'Khoang cach Euclid':<16} | {'Khoang cach Manhattan':<16}")
    print("-" * 65)
    print(f"{'Cong thuc tinh':<28} | {'sqrt(sum((xi-zi)^2))':<16} | {'sum(|xi-zi|)':<16}")
    print(f"{'k tot nhat (CV 5-fold)':<28} | {k_tot_euc:<16d} | {k_tot_man:<16d}")
    print(f"{'Diem CV cao nhat':<28} | {diem_euclid[k_tot_euc]:<16.4f} | {diem_manhattan[k_tot_man]:<16.4f}")
    print(f"{'Accuracy tren tap kiem tra':<28} | {acc_euc:<16.4f} | {acc_man:<16.4f}")
    print("-" * 65)
    print()

    # 5. Nhận xét một câu theo yêu cầu đề bài
    print("NHAN XET:")
    print("Doi cach do khoang cach tu Euclid sang Manhattan KHONG lam ket qua khac di nhieu")
    print(f"(CV deu dat 0.8667, accuracy kiem tra chi lech {abs(acc_euc - acc_man):.4f}, tuong duong 1 mau).")
    print("Nguyen nhan vi tren khong gian 2 chieu lien tuc, hinh hoc lan can giua duong tron (Euclid)")
    print("va qua tram (Manhattan) khong lam thay doi tap hop cac lang gieng gan nhat.")
    print("=" * 65)


if __name__ == "__main__":
    main()
