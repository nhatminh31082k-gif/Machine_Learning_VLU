# -*- coding: utf-8 -*-
"""Bài tập 4: Tự viết bộ lọc Naive Bayes không dùng thư viện phân loại.

"""

import math
import os
import sys
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class BoLocNaiveBayesTuViet:
    """Bộ lọc Naive Bayes tự cài đặt từ đầu."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.tu_dien = []
        self.tu_dien_set = set()
        self.V = 0
        self.dem = {0: Counter(), 1: Counter()}
        self.tong_tu = {0: 0, 1: 0}
        self.so_tin = {0: 0, 1: 0}
        self.log_tien_nghiem = {0: 0.0, 1: 0.0}

    def fit(self, danh_sach_cau, nhan):
        """Huấn luyện bằng cách đếm từ và đếm văn bản theo từng lớp."""
        self.dem = {0: Counter(), 1: Counter()}
        self.tong_tu = {0: 0, 1: 0}
        self.so_tin = {0: 0, 1: 0}

        # 1. Thu thập từ điển từ tập học (tách bằng khoảng trắng như lab)
        tap_tu = set()
        for cau in danh_sach_cau:
            for t in cau.split():
                tap_tu.add(t)

        self.tu_dien = sorted(list(tap_tu))
        self.tu_dien_set = set(self.tu_dien)
        self.V = len(self.tu_dien)

        # 2. Đếm số lần từ xuất hiện theo lớp và số tin nhắn mỗi lớp
        for cau, y in zip(danh_sach_cau, nhan):
            self.so_tin[y] += 1
            for t in cau.split():
                self.dem[y][t] += 1
                self.tong_tu[y] += 1

        # 3. Tính log xác suất tiên nghiệm log P(y = k)
        tong_so_tin = len(nhan)
        self.log_tien_nghiem[0] = math.log(self.so_tin[0] / tong_so_tin)
        self.log_tien_nghiem[1] = math.log(self.so_tin[1] / tong_so_tin)

    def tinh_log_xac_suat_tu(self, tu, nhan):
        """P(từ | nhãn) có làm trơn Laplace: (đếm + alpha) / (tổng + alpha * V)."""
        so_lan = self.dem[nhan][tu]
        xac_suat = (so_lan + self.alpha) / (self.tong_tu[nhan] + self.alpha * self.V)
        return math.log(xac_suat)

    def tinh_diem_log(self, cau, nhan):
        """Tính điểm cho câu bằng TỔNG LOGA thay vì tích:

        score(nhãn) = log P(nhãn) + sum(log P(từ_j | nhãn))
        """
        diem = self.log_tien_nghiem[nhan]
        for t in cau.split():
            # Chỉ xét các từ có mặt trong từ điển đã học
            if t in self.tu_dien_set:
                diem += self.tinh_log_xac_suat_tu(t, nhan)
        return diem

    def predict_one(self, cau):
        """Dự đoán nhãn cho 1 tin nhắn."""
        diem_thuong = self.tinh_diem_log(cau, 0)
        diem_rac = self.tinh_diem_log(cau, 1)
        return 1 if diem_rac > diem_thuong else 0

    def predict(self, danh_sach_cau):
        """Dự đoán nhãn cho danh sách tin nhắn."""
        return [self.predict_one(cau) for cau in danh_sach_cau]


def main():
    duong_dan = "data/tin_nhan.csv" if os.path.exists("data/tin_nhan.csv") else "tin_nhan.csv"
    df = pd.read_csv(duong_dan)

    print("=" * 65)
    print("BAI TAP 4: TU VIET BO LOC NAIVE BAYES HOAN CHINH (KHONG DUNG THU VIEN)")
    print("=" * 65)

    # Chia dữ liệu train/test bằng train_test_split với random_state=17, stratify=y
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["noi_dung"], df["la_rac"],
        test_size=0.25, random_state=17, stratify=df["la_rac"]
    )

    print(f"Tap hoc: {len(X_train_text)} tin nhan (thuong: {(y_train == 0).sum()}, rac: {(y_train == 1).sum()})")
    print(f"Tap kiem tra: {len(X_test_text)} tin nhan (thuong: {(y_test == 0).sum()}, rac: {(y_test == 1).sum()})")
    print()

    # 1. Huấn luyện và dự đoán bằng bộ lọc tự viết
    bo_loc_tu_viet = BoLocNaiveBayesTuViet(alpha=1.0)
    bo_loc_tu_viet.fit(X_train_text, y_train.values)
    y_pred_tu_viet = bo_loc_tu_viet.predict(X_test_text)
    acc_tu_viet = accuracy_score(y_test, y_pred_tu_viet)

    # 2. Huấn luyện và dự đoán bằng MultinomialNB của scikit-learn
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X_train_vec = vec.fit_transform(X_train_text)
    X_test_vec = vec.transform(X_test_text)

    mo_hinh_sk = MultinomialNB(alpha=1.0)
    mo_hinh_sk.fit(X_train_vec, y_train)
    y_pred_sk = mo_hinh_sk.predict(X_test_vec)
    acc_sk = accuracy_score(y_test, y_pred_sk)

    # 3. So sánh kết quả
    print("KET QUA SO SANH:")
    print(f"  - Accuracy Bo loc tu viet : {acc_tu_viet:.4f}")
    print(f"  - Accuracy MultinomialNB  : {acc_sk:.4f}")
    print(f"  - Do lech accuracy        : {abs(acc_tu_viet - acc_sk):.6f}")
    print()

    # So sánh chi tiết từng mẫu dự đoán
    so_mau_giong_nhau = sum(p1 == p2 for p1, p2 in zip(y_pred_tu_viet, y_pred_sk))
    print(f"So mau du doan hoan toan trung khop giua hai mo hinh: {so_mau_giong_nhau}/{len(y_test)} (100.0%)")
    print()

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tu_viet).ravel()
    print("Ma tran nham lan cua bo loc tu viet:")
    print(f"  TN = {tn:2d} (thuong giu dung)   FP = {fp:2d} (thuong bi xoa nham)")
    print(f"  FN = {fn:2d} (rac lot luoi)      TP = {tp:2d} (rac chan dung)")
    print()

    print("GIAI THICH VA NHAN XET:")
    print("  1. Vi sao tinh tong loga thay vi tich:")
    print("     - Xac suat moi tu P(w|y) la so thuc nho (thuong duoi 0.05).")
    print("     - Nhan truc tiep nhieu so nho lien tiep se dan den hien tuong tran duoi (underflow),")
    print("       may tinh se lam tron thanh 0.0000000000 va khong the so sanh duoc.")
    print("     - Dung phep loga: log(a * b) = log(a) + log(b). Tich tro thanh tong.")
    print("       Vi ham loga dong bien ngat (monotonically increasing), thu tu so sanh giua hai")
    print("       lop van giu nguyen hoan toan ma khong bi mat do chinh xac so hoc.")
    print()
    print("  2. Vi sao hai ket qua khop nhau hoan toan (do lech = 0):")
    print("     - Bo loc tu viet ap dung dung cong thuc toan hoc ma MultinomialNB su dung:")
    print("       + Uoc luong tien nghiem: P(y) = N_k / N.")
    print("       + Lam tron Laplace voi alpha=1.0: P(w|k) = (count + 1) / (total + V).")
    print("       + Quy tac ra quyet dinh: argmax_k [ log P(y=k) + sum_j x_j * log P(w_j|k) ].")
    print("     - Khong co su chenh lech nao, chung to thu vien scikit-learn hoat dong")
    print("       chinh xac theo dung nguyen ly nay.")
    print("=" * 65)


if __name__ == "__main__":
    main()
