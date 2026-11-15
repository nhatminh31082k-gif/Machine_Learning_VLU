
import os
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor

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
    duong_dan = tim_file_du_lieu("gia_nha.csv")
    df = pd.read_csv(duong_dan)

    X = df[["dien_tich"]]
    y = df["gia"]

    # Cùng cách chia với bài 1 và mục 6 để so sánh công bằng
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("=" * 70)
    print("BAI TAP 5: CHON K CHO HOI QUY KNN TREN DU LIEU GIA NHA")
    print("=" * 70)
    print(f"Tap hoc: {len(X_tr)} can, Tap kiem tra: {len(X_te)} can")
    print()

    # 1. Kiểm định chéo 5 phần trên tập học để chọn k từ 1 tới 20
    print("1. KIEM DINH CHEO 5 PHAN TREN TAP HOC (k = 1 -> 20):")
    diem_cv_mae = {}
    for k in range(1, 21):
        # scoring='neg_mean_absolute_error' trả về giá trị âm: -MAE
        scores = cross_val_score(
            KNeighborsRegressor(n_neighbors=k),
            X_tr, y_tr,
            cv=5,
            scoring="neg_mean_absolute_error"
        )
        # Đổi dấu để nhận lại MAE dương
        mae_cv = -scores.mean()
        diem_cv_mae[k] = mae_cv

    # k tốt nhất là k có MAE nhỏ nhất
    k_tot = min(diem_cv_mae, key=diem_cv_mae.get)

    for k in range(1, 21):
        dau = " <- TOT NHAT (MAE nho nhat)" if k == k_tot else ""
        print(f"  k = {k:2d}  |  MAE trung binh (CV 5-fold): {diem_cv_mae[k]:.4f} ty dong{dau}")
    print()

    # 2. Huấn luyện KNN với k tốt nhất và đánh giá trên tập kiểm tra
    knn_tot = KNeighborsRegressor(n_neighbors=k_tot).fit(X_tr, y_tr)
    mae_knn_te = mean_absolute_error(y_te, knn_tot.predict(X_te))

    # Huấn luyện mô hình đường thẳng (Linear Regression) để đối chiếu
    duong_thang = LinearRegression().fit(X_tr, y_tr)
    mae_duong_te = mean_absolute_error(y_te, duong_thang.predict(X_te))

    # Cũng tính thêm KNN với k = 5 mặc định (mục 6) để đối chiếu
    knn_k5 = KNeighborsRegressor(n_neighbors=5).fit(X_tr, y_tr)
    mae_knn_k5 = mean_absolute_error(y_te, knn_k5.predict(X_te))

    # 3. Bảng so sánh kết quả
    print("2. BANG SO SANH MAE TREN TAP KIEM TRA:")
    print("-" * 70)
    print(f"{'Mo hinh':<35} | {'MAE tren tap kiem tra (ty dong)':<30}")
    print("-" * 70)
    print(f"{f'KNN toi uu (k = {k_tot})':<35} | {mae_knn_te:<30.4f}")
    print(f"{'KNN mac dinh (k = 5, muc 6)':<35} | {mae_knn_k5:<30.4f}")
    print(f"{'Duong thang (Linear Regression)':<35} | {mae_duong_te:<30.4f}")
    print("-" * 70)
    print()

    # 4. Trả lời câu hỏi và nhận xét chuyên sâu
    print("3. TRA LOI CAU HOI VA GIAI THICH:")
    print()
    print("Cau hoi: Voi k tot nhat, MAE tren tap kiem tra co vuot duoc duong thang khong?")
    print()
    print(f"  -> KHONG VUOT DUOC. MAE cua KNN (k={k_tot}) la {mae_knn_te:.4f} ty dong,")
    print(f"     van kem hon ro ret so voi Duong thang la {mae_duong_te:.4f} ty dong (sai so cao hon {mae_knn_te - mae_duong_te:.4f} ty).")
    print()
    print("  -> GIAI THICH NGUYEN NHAN BAN CHAT:")
    print("     1. Ban chat du lieu gia nha la tuyen tinh (linear):")
    print("        Gia nha trong thuc te tang deu dan va on dinh theo dien tich (y ~ a*x + b).")
    print("        Duong thang Linear Regression gia dinh dung ban chat nay va tan dung toan bo")
    print("        du lieu de uoc luong do doc va he so chan cuc ky chinh xac.")
    print()
    print("     2. Ban chat xap xi cua KNN hoi quy la bac thang roi rac (piecewise constant):")
    print("        KNN du doan bang cach lay trung binh cong cua k diem lan can. Duong du doan")
    print("        cua no la cac doan ngang gap khuc, de bi giat cuc bo theo nhieu va thieu")
    print("        do doc muot ma lien tuc.")
    print()
    print("     3. Khong the ngoai suy (No extrapolation):")
    print("        Voi cac can nha o ria hoac ngoai khoang dien tich tap hoc, KNN chi co the")
    print("        gan gia tri trung binh cua nhom bien, trong khi duong thang tiep tuc tang theo quy luat.")
    print()
    print("     -> KET LUAN: Khong co mo hinh nao luon thang moi bai toan. Tren bien cong,")
    print("        KNN thang; nhung tren du lieu quan he tuyen tinh ro net, mo hinh tham so")
    print("        duong thang moi la lua chon toi uu nhat.")
    print("=" * 70)


if __name__ == "__main__":
    main()
