
import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

    cac_k = [1, 13, 101]

    print("=" * 70)
    print("BAI TAP 4: LANG GIENG CO TRONG SO (WEIGHTS='DISTANCE')")
    print("=" * 70)
    print(f"Tap hoc: {len(X_tr)} mau, Tap kiem tra: {len(X_te)} mau")
    print()

    # Lưu kết quả để so sánh
    ket_qua = []
    for k in cac_k:
        # 1. Trọng số đều (mặc định - mục 4)
        mh_uniform = KNeighborsClassifier(n_neighbors=k, weights="uniform").fit(X_tr, y_tr)
        acc_tr_u = accuracy_score(y_tr, mh_uniform.predict(X_tr))
        acc_te_u = accuracy_score(y_te, mh_uniform.predict(X_te))

        # 2. Trọng số theo nghịch đảo khoảng cách
        mh_dist = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X_tr, y_tr)
        acc_tr_d = accuracy_score(y_tr, mh_dist.predict(X_tr))
        acc_te_d = accuracy_score(y_te, mh_dist.predict(X_te))

        ket_qua.append({
            "k": k,
            "u_train": acc_tr_u, "u_test": acc_te_u,
            "d_train": acc_tr_d, "d_test": acc_te_d
        })

    # In bảng so sánh
    print("1. BANG SO SANH WEIGHTS='UNIFORM' VS WEIGHTS='DISTANCE':")
    print("-" * 70)
    print(f"{'k':<4} | {'Uniform (Tap hoc)':<18} | {'Uniform (Kiem tra)':<18} | "
          f"{'Distance (Tap hoc)':<18} | {'Distance (Kiem tra)':<18}")
    print("-" * 70)
    for r in ket_qua:
        print(f"{r['k']:<4d} | {r['u_train']:<18.4f} | {r['u_test']:<18.4f} | "
              f"{r['d_train']:<18.4f} | {r['d_test']:<18.4f}")
    print("-" * 70)
    print()

    # Nhận xét và trả lời câu hỏi
    print("2. TRA LOI CAU HOI VA GIAI THICH CHUYEN SAU:")
    print()
    print("Cau hoi: Voi k = 101, trong so theo khoang cach co giup mo hinh bot cau tha khong? Vi sao?")
    print()
    print("  -> CO, TRONG SO THEO KHOANG CACH GIUP MO HINH BOT CAU THA RO RET:")
    print(f"     - Voi weights='uniform' : Accuracy tap kiem tra chi dat {ket_qua[2]['u_test']:.4f}")
    print(f"     - Voi weights='distance': Accuracy tap kiem tra tang len {ket_qua[2]['d_test']:.4f} (+{ket_qua[2]['d_test'] - ket_qua[2]['u_test']:.4f})")
    print()
    print("  -> NGUYEN NHAN TOAN HOC VA HINH HOC:")
    print("     1. Khi k = 101 voi weights='uniform':")
    print("        Tap hoc chi co 225 mau, nen 101 lang gieng chiem toi ~45% toan bo du lieu.")
    print("        Moi phieu bau co gia tri nhu nhau (=1). Do k qua lon, vung lang gieng phai")
    print("        mo rong rat xa, gom ca nhieu diem o tan vang trang doi dien. Cac diem o xa")
    print("        co so luong lon se 'pha loang' va lan at nhung diem o gan, dan den ranh gioi")
    print("        quyet dinh bi san phang, bo qua hoan toan chi tiet uon cong (underfitting/cau tha).")
    print()
    print("     2. Khi k = 101 voi weights='distance':")
    print("        Moi lang gieng co trong so w_i = 1 / d(x, x_i).")
    print("        - Nhung lang gieng o cuc xa co d rat lon -> trong so w_i tien ve 0, hau nhu")
    print("          khong anh huong toi ket qua bo phieu.")
    print("        - Nhung lang gieng o cuc gan co d rat be -> trong so w_i rat lon, co tieng noi")
    print("          ap dao hoan toan.")
    print("        Nho do, du lay toi 101 lang gieng, quyet dinh du doan van duoc dan dat chu yeu")
    print("        boi cac diem lan can gan nhat, giu duoc hinh dang bien cong tu nhien cua du lieu.")
    print()
    print("  -> LUU Y VE TAP HOC:")
    print("     Voi weights='distance', Accuracy tren tap hoc luon bang 1.0000 vi khi danh gia")
    print("     mot diem trong tap hoc, khoang cach toi chinh no bang 0 -> trong so bang vo cung.")
    print("     Do do con so 1.0000 tren tap hoc khong phan anh kha nang tong quat hoa.")
    print("=" * 70)


if __name__ == "__main__":
    main()
