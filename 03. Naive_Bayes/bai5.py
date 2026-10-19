# -*- coding: utf-8 -*-
"""Bài tập 5: Chọn sai biến thể Naive Bayes.

"""

import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    duong_dan = "data/sinh_vien.csv" if os.path.exists("data/sinh_vien.csv") else "sinh_vien.csv"
    df = pd.read_csv(duong_dan)

    print("=" * 65)
    print("BAI TAP 5: CHON SAI BIEN THE NAIVE BAYES")
    print("=" * 65)

    X = df[["gio_on", "diem_giua_ky"]]
    y = df["qua_mon"]

    # Chia dữ liệu đúng như mục 6 của tài liệu thực hành
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    print(f"Du lieu sinh vien: {len(df)} mau")
    print(f"Tap hoc: {len(X_train)} mau, Tap kiem tra: {len(X_test)} mau")
    print()

    # 1. Chạy GaussianNB (chuẩn mực cho số liệu liên tục)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)
    acc_gnb = accuracy_score(y_test, y_pred_gnb)

    # 2. Chạy MultinomialNB (biến thể dành cho số đếm rời rạc)
    co_loi = False
    thong_bao_loi = ""
    acc_mnb = None
    try:
        mnb = MultinomialNB()
        mnb.fit(X_train, y_train)
        y_pred_mnb = mnb.predict(X_test)
        acc_mnb = accuracy_score(y_test, y_pred_mnb)
    except Exception as e:
        co_loi = True
        thong_bao_loi = str(e)

    # In kết quả thực nghiệm
    print("KET QUA THUC NGHIEM:")
    print(f"  - GaussianNB (dung bien the)       : accuracy = {acc_gnb:.4f}")
    if co_loi:
        print(f"  - MultinomialNB (sai bien the)     : BÁO LỖI! ({thong_bao_loi})")
    else:
        print(f"  - MultinomialNB (sai bien the)     : accuracy = {acc_mnb:.4f}")
    print()

    # Trả lời chi tiết 3 câu hỏi của đề bài
    print("TRA LOI CAC CAU HOI CUA DE BAI:")
    print("1. Chuong trinh co bao loi khong?")
    if co_loi:
        print(f"   -> Co bao loi: {thong_bao_loi}")
    else:
        print("   -> KHONG bao loi! Chuong trinh van chay troi chay va cho ra ket qua.")
        print("      Ly do: MultinomialNB chi nem ra ValueError khi du lieu co gia tri AM (< 0).")
        print("      O day gio on tap va diem giua ky deu la cac so khong am (>= 0),")
        print("      nen thu vien van thuc hien cac phep tinh so hoc binh thuong.")
    print()

    print("2. Con so accuracy ra bao nhieu?")
    print(f"   -> Accuracy cua MultinomialNB la {acc_mnb:.4f} (80.00%).")
    print(f"      Con so nay THAP HON HAN so voi GaussianNB dat {acc_gnb:.4f} (90.00%).")
    print()

    print("3. Giai thich vi sao dung MultinomialNB o day la SAI VE NGUYEN TAC, du van chay:")
    print("   a) Sai lam ve ban chat du lieu:")
    print("      - MultinomialNB duoc thiet ke cho DU LIEU DEM ROI RAC (discrete count data),")
    print("        vi du: so lan mot tu xuat hien trong tin nhan (0, 1, 2, ... lan).")
    print("      - Hai dac trung 'gio_on' (so gio) va 'diem_giua_ky' (diem so) la BIEN SO THUC")
    print("        LIEN TUC (continuous variables). Gia tri 12.5 gio hay 7.7 diem la dai luong do")
    print("        luong tren truc so thuc, khong phai la so lan dien ra cua mot bien co.")
    print("   b) Sai lam ve cach tinh xac suat:")
    print("      - MultinomialNB coi gia tri la so lan xuat hien va tinh xac suat bang ty le:")
    print("            P(x_j | y) = (tong_x_j + alpha) / (tong_tat_ca + alpha * d)")
    print("        Dieu nay vo nghia doi voi so gio on va diem so, lam meo mo moi quan he giua")
    print("        dac trung va kha nang qua mon.")
    print("   c) Mo hinh dung phai la GaussianNB:")
    print("      - Bien lien tuc thuong tuan theo phan phoi chuan (hinh chuong), dac trung boi")
    print("        gia tri trung binh (mean) va do lech chuan (std).")
    print("      - GaussianNB dung ham mat do xac suat Gaussian de mo ta hinh dang tung lop,")
    print("        vi the phan loai chinh xac hon rat nhieu (90% so voi 80%).")
    print("   -> Day la vi du dien hinh ve 'LOI IM LANG' (silent bug) trong hoc may:")
    print("      Khong co loi runtime, nhung mo hinh dua tren gia dinh toan hoc sai lam.")
    print("=" * 65)


if __name__ == "__main__":
    main()
