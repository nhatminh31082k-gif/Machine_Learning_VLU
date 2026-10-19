# -*- coding: utf-8 -*-
"""Bài tập 6: Lớp hiếm và xác suất tiên nghiệm (Cái bẫy Accuracy).

"""

import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    duong_dan = "data/tin_nhan.csv" if os.path.exists("data/tin_nhan.csv") else "tin_nhan.csv"
    df_goc = pd.read_csv(duong_dan)

    print("=" * 65)
    print("BAI TAP 6: LOP HIEM VA XAC SUAT TIEN NGHIEM (CAI BAY ACCURACY)")
    print("=" * 65)

    # 1. Tạo bộ dữ liệu mới mất cân bằng nặng
    df_thuong = df_goc[df_goc["la_rac"] == 0]
    df_rac = df_goc[df_goc["la_rac"] == 1].head(20)
    df_mcb = pd.concat([df_thuong, df_rac], ignore_index=True)

    print(f"Bo du lieu goc       : {len(df_goc)} tin (180 thuong, 120 rac, ty le rac: {df_goc['la_rac'].mean():.2%})")
    print(f"Bo du lieu mat can bang: {len(df_mcb)} tin (180 thuong,  20 rac, ty le rac: {df_mcb['la_rac'].mean():.2%})")
    print()

    # 2. Vectorize và chia train/test (tỷ lệ 75/25, random_state=17, stratify theo nhãn)
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X = vec.fit_transform(df_mcb["noi_dung"])
    y = df_mcb["la_rac"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    print(f"Tap hoc      : {X_train.shape[0]} tin (thuong: {(y_train == 0).sum()}, rac: {(y_train == 1).sum()})")
    print(f"Tap kiem tra : {X_test.shape[0]} tin (thuong: {(y_test == 0).sum()}, rac: {(y_test == 1).sum()})")
    print()

    # 3. Huấn luyện MultinomialNB
    mo_hinh = MultinomialNB(alpha=1.0)
    mo_hinh.fit(X_train, y_train)
    y_pred = mo_hinh.predict(X_test)

    # 4. Tính toán các chỉ số đánh giá
    acc = accuracy_score(y_test, y_pred)
    rec_rac = recall_score(y_test, y_pred, pos_label=1)
    prec_rac = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("KET QUA DANH GIA TREN TAP KIEM TRA:")
    print(f"  - Accuracy          = {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  - Recall (lop rac)  = {rec_rac:.4f} ({rec_rac * 100:.2f}%)")
    print(f"  - Precision(lop rac)= {prec_rac:.4f} ({prec_rac * 100:.2f}%)")
    print()
    print("Ma tran nham lan:")
    print(f"  TN = {tn:2d} (thuong giu dung)   FP = {fp:2d} (thuong bi xoa nham)")
    print(f"  FN = {fn:2d} (rac bi lot luoi)   TP = {tp:2d} (rac bi chan dung)")
    print()

    # 5. So sánh với mô hình cơ sở ngớ ngẩn (Dummy / Majority Baseline)
    acc_dummy = (y_test == 0).sum() / len(y_test)
    print(f"Doi chung voi mo hinh doan tat ca la tin thuong (Dummy Baseline):")
    print(f"  - Accuracy cua Dummy Baseline: {acc_dummy:.4f} ({acc_dummy * 100:.2f}%)")
    print(f"  - Recall lop rac cua Dummy   : 0.0000 (khong bat duoc bat ky tin rac nao!)")
    print()

    # 6. Nhận xét chi tiết
    print("NHAN XET VA GIAI THICH:")
    print("1. Accuracy co con phan anh dung chat luong bo loc khong?")
    print("   -> HOAN TOAN KHONG! Day chinh la 'CAI BAY ACCURACY' (Accuracy Paradox).")
    print("   - Accuracy trong rat cao (0.9800 tuc 98%), cao hon ca bo loc ban dau (0.9467).")
    print("   - Tuy nhien, con so 98% nay la ao tuong. Do lop thuong chiem toi 90% du lieu,")
    print("     ngay ca mot mo hinh vo dung luon doan moi thu la 'tin thuong' cung da dat 90% accuracy!")
    print("   - Chi can doan dung lop da so, accuracy da rat cao ma khong he chung to bo loc bat rac gioi.")
    print()
    print("2. Co che cua Naive Bayes voi lop hiem (Xac suat tien nghiem P(y)):")
    print("   - Quy tac quyet dinh cua Naive Bayes la: y_hat = argmax P(y) * prod P(x_j | y).")
    print("   - Trong bo du lieu moi, xac suat tien nghiem cua lop rac P(rac) chi la 10%,")
    print("     trong khi P(thuong) len toi 90% (gap 9 lan).")
    print("   - Tien nghiem lop rac qua nho dong vai tro nhu mot 'rao can rat cao'. De mo hinh")
    print("     dam xep mot tin vao lop rac, bang chung tu tu ngu phai cuc manh de vuot qua")
    print("     khoan bat loi tien nghiem nay.")
    print("   - Hau qua: Mo hinh tro nen qua de dat va thien vi ve tin thuong. Trong 5 tin rac")
    print("     kiem tra, mo hinh da bo lot 1 tin (FN = 1), khien Recall tut xuong con 80.00%")
    print("     (so voi 86.67% o bo du lieu goc).")
    print()
    print("3. Ket luan:")
    print("   - Khi du lieu bi mat can bang nghiem trong, accuracy tro nen vo nghia.")
    print("   - Voi bai toan loc tin rac, bat buoc phai theo doi Recall va Precision cua rieng lop rac,")
    print("     hoac cac chi so nhu F1-Score / PR-AUC de danh gia chat luong thuc su.")
    print("=" * 65)


if __name__ == "__main__":
    main()
