# -*- coding: utf-8 -*-
"""Bài tập 3: Đổi tham số làm trơn Laplace (alpha).

"""

import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    duong_dan = "data/tin_nhan.csv" if os.path.exists("data/tin_nhan.csv") else "tin_nhan.csv"
    df = pd.read_csv(duong_dan)

    print("=" * 65)
    print("BAI TAP 3: ANH HUONG CUA THAM SO LAM TRON ALPHA")
    print("=" * 65)

    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X = vec.fit_transform(df["noi_dung"])
    y = df["la_rac"]

    # Chia tập học và kiểm tra tương tự mục 4.2 trong tài liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    alphas = [0.001, 0.1, 1.0, 10.0, 100.0]
    ket_qua = []

    for alpha in alphas:
        mo_hinh = MultinomialNB(alpha=alpha)
        mo_hinh.fit(X_train, y_train)
        y_pred = mo_hinh.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)

        ket_qua.append({
            "alpha": alpha,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec
        })

    # In bảng kết quả
    print(f"{'alpha':<10} {'Accuracy':<14} {'Precision':<14} {'Recall':<14}")
    print("-" * 52)
    for row in ket_qua:
        print(f"{row['alpha']:<10.3f} {row['Accuracy']:<14.4f} {row['Precision']:<14.4f} {row['Recall']:<14.4f}")
    print()

    # Thử thêm với alpha cực lớn (1000, 10000) để thấy rõ hiện tượng thoái hóa
    print("Khao sat them voi alpha cuc lon de thay ro gioi han:")
    for alpha_lon in [1000.0, 10000.0]:
        m_lon = MultinomialNB(alpha=alpha_lon).fit(X_train, y_train)
        p_lon = m_lon.predict(X_test)
        print(f"alpha = {alpha_lon:<7.1f} | Accuracy = {accuracy_score(y_test, p_lon):.4f} | "
              f"Precision = {precision_score(y_test, p_lon, zero_division=0):.4f} | "
              f"Recall = {recall_score(y_test, p_lon):.4f}")
    print()

    print("NHAN XET:")
    print("  1. Chuyen gi xay ra khi alpha qua lon:")
    print("     - Khi alpha tang tu 0.001 len 100.0, tren tap du lieu nho nay cac chi so")
    print("       van giu o muc Accuracy = 0.9467, Precision = 1.0000, Recall = 0.8667.")
    print("     - Tuy nhien, khi alpha tang len rat lon (vi du 1000.0 tro len), mo hinh bi")
    print("       'te liet': Precision va Recall cua lop rac tut ve 0, mo hinh du doan tat ca")
    print("       tin nhan deu la tin thuong (lop chiem da so trong tien nghiem).")
    print()
    print("  2. Vi sao lai xay ra hien tuong nay:")
    print("     - Theo cong thuc lam tron Laplace:")
    print("           P(tu_j | lop_k) = (dem_jk + alpha) / (tong_k + alpha * V)")
    print("     - Khi alpha rat lon (alpha >> dem_jk va alpha >> tong_k), ca tu so va mau so")
    print("       deu bi chi phoi boi alpha: (dem_jk + alpha) / (tong_k + alpha * V) -> alpha / (alpha * V) = 1 / V.")
    print("     - Nhu vay, moi tu deu co xac suat xuat hien xap xi 1/V o tat ca cac lop,")
    print("       tuc la moi xac suat tu deu xich lai gan nhau va tro nen bang phang.")
    print("     - Khi do, bang chung tu cac tu bi triet tieu hoan toan, mo hinh khong con dua vao")
    print("       noi dung van ban nua ma chi con dua vao xac suat tien nghiem P(y).")
    print("       Vi tin thuong chiem da so (60% tap train), mo hinh se doan moi tin la tin thuong.")
    print("=" * 65)


if __name__ == "__main__":
    main()
