

import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    duong_dan = tim_file_du_lieu("sinh_vien_mo_rong.csv")
    df = pd.read_csv(duong_dan)

    y = df["qua_mon"]
    cot_features = ["gio_on", "diem_giua_ky", "chi_tieu"]

    # 1. Dữ liệu gốc: chi_tieu tính bằng NGHÌN ĐỒNG
    X_nghin = df[cot_features].copy()

    # 2. Dữ liệu mới: đổi chi_tieu sang TRIỆU ĐỒNG (chia 1000)
    X_trieu = df[cot_features].copy()
    X_trieu["chi_tieu"] = X_trieu["chi_tieu"] / 1000.0

    # Chia train/test cùng random_state=17, stratify=y như mục 3
    X_tr_nghin, X_te_nghin, y_tr, y_te = train_test_split(
        X_nghin, y, test_size=0.25, random_state=17, stratify=y
    )
    X_tr_trieu, X_te_trieu, _, _ = train_test_split(
        X_trieu, y, test_size=0.25, random_state=17, stratify=y
    )

    k = 7
    print("=" * 70)
    print("BAI TAP 3: ANH HUONG CUA DOI DON VI DO VA CHUAN HOA")
    print("=" * 70)

    # Xem độ lệch chuẩn của các cột ở 2 đơn vị
    print("1. SO SANH THANG DO (DO LECH CHUAN):")
    std_nghin = X_nghin.std()
    std_trieu = X_trieu.std()
    print(f"  - Ban dau (nghin dong) : gio_on={std_nghin['gio_on']:.2f}, "
          f"diem_giua_ky={std_nghin['diem_giua_ky']:.2f}, chi_tieu={std_nghin['chi_tieu']:.1f}")
    print(f"  - Sau khi chia 1000    : gio_on={std_trieu['gio_on']:.2f}, "
          f"diem_giua_ky={std_trieu['diem_giua_ky']:.2f}, chi_tieu={std_trieu['chi_tieu']:.3f}")
    print()

    # TH1: Không chuẩn hoá, đơn vị nghìn đồng (kết quả mục 3)
    clf_raw_nghin = KNeighborsClassifier(n_neighbors=k).fit(X_tr_nghin, y_tr)
    acc_raw_nghin = accuracy_score(y_te, clf_raw_nghin.predict(X_te_nghin))

    # TH2: Không chuẩn hoá, đơn vị triệu đồng
    clf_raw_trieu = KNeighborsClassifier(n_neighbors=k).fit(X_tr_trieu, y_tr)
    acc_raw_trieu = accuracy_score(y_te, clf_raw_trieu.predict(X_te_trieu))

    # TH3: CÓ chuẩn hoá, đơn vị nghìn đồng
    pipe_nghin = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    pipe_nghin.fit(X_tr_nghin, y_tr)
    acc_pipe_nghin = accuracy_score(y_te, pipe_nghin.predict(X_te_nghin))

    # TH4: CÓ chuẩn hoá, đơn vị triệu đồng
    pipe_trieu = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    pipe_trieu.fit(X_tr_trieu, y_tr)
    acc_pipe_trieu = accuracy_score(y_te, pipe_trieu.predict(X_te_trieu))

    print("2. KET QUA THUC NGHIEM:")
    print(f"  - Muc doan bua ca lop qua              : {y_te.mean():.4f}")
    print(f"  - KHONG chuan hoa, chi tieu NGHIN DONG : {acc_raw_nghin:.4f} (ket qua muc 3)")
    print(f"  - KHONG chuan hoa, chi tieu TRIEU DONG : {acc_raw_trieu:.4f}")
    print(f"  - CO chuan hoa, chi tieu NGHIN DONG    : {acc_pipe_nghin:.4f}")
    print(f"  - CO chuan hoa, chi tieu TRIEU DONG    : {acc_pipe_trieu:.4f}")
    print()

    # 3. Trả lời câu hỏi và giải thích chuyên sâu
    print("-" * 70)
    print("3. GIAI THICH VA TRA LOI CAU HOI:")
    print()
    print("Cau hoi 1: Accuracy thay doi the nao so voi 0.5333 o muc 3?")
    print(f"  -> Accuracy tang vot tu {acc_raw_nghin:.4f} len {acc_raw_trieu:.4f} (+{acc_raw_trieu - acc_raw_nghin:.4f}).")
    print("     Tu muc te hon doan bua (0.5333 < 0.6000), mo hinh tro thanh kha tot (0.8667).")
    print()
    print("Cau hoi 2: Vi sao chi doi don vi ma ket qua lai khac han?")
    print("  -> KNN hoan toan dua vao phep tinh khoang cach Euclid giua cac toa do so thuan tuy.")
    print("     Thuat toan khong biet 'nghin dong' hay 'trieu dong', chi nhin do lech gia tri.")
    print("     - Khi dung 'nghin dong': std(chi_tieu) = 1692, gap ~216 lan gio_on (7.82) va")
    print("       gap ~809 lan diem (2.09). Khoang cach d(x, z) bi cot chi tieu ap dao hoan toan,")
    print("       hai dac trung quan trong bi 'nuot chung', bien KNN thanh bo phan loai chi tieu.")
    print("     - Khi chia 1000 (trieu dong): std(chi_tieu) giam con 1.692, luc nay be hon ca")
    print("       gio_on (7.82) va tuong duong diem (2.09). Khoang cach khong con bi chenh lech")
    print("       don vi lam meo mo, giup hai cot hoc tap phat huy dung vai tro quyet dinh.")
    print()
    print("Cau hoi 3: Neu co chuan hoa thi doi don vi co lam ket qua thay doi khong?")
    print("  -> TUYET DOI KHONG THAY DOI. Ca hai deu dat accuracy 0.8667.")
    print("     Chung minh toan hoc: StandardScaler bien doi z = (x - mean) / std.")
    print("     Khi nhan/chia mot cot voi hang so duong c (o day c = 1/1000):")
    print("       mean_moi = c * mean, std_moi = c * std")
    print("       z_moi = (c*x - c*mean) / (c*std) = (x - mean) / std = z_cu.")
    print("     Toa do khong gian sau chuan hoa la dong nhat 100%, do do khoang cach va thu tu")
    print("     lang gieng khong doi, accuracy giu nguyen tuyet doi.")
    print("=" * 70)


if __name__ == "__main__":
    main()
