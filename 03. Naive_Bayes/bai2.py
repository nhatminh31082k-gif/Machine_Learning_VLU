# -*- coding: utf-8 -*-
"""Bài tập 2: Từ nào tố cáo mạnh nhất.

"""

import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    duong_dan = "data/tin_nhan.csv" if os.path.exists("data/tin_nhan.csv") else "tin_nhan.csv"
    df = pd.read_csv(duong_dan)

    print("=" * 65)
    print("BAI TAP 2: TIM TU TO CAO MANH NHAT BANG FEATURE_LOG_PROB_")
    print("=" * 65)
    print(f"Tong so tin nhan: {len(df)}")
    print(f"So tin thuong (0): {(df['la_rac'] == 0).sum()}, So tin rac (1): {(df['la_rac'] == 1).sum()}")
    print()

    # Chuyển đổi văn bản thành ma trận túi từ
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X = vec.fit_transform(df["noi_dung"])
    y = df["la_rac"]

    # Huấn luyện MultinomialNB trên toàn bộ 300 tin nhắn
    mo_hinh = MultinomialNB(alpha=1.0)
    mo_hinh.fit(X, y)

    # feature_log_prob_: log P(từ | lớp), dòng 0 là tin thường, dòng 1 là tin rác
    # Hiệu = log P(từ | rác) - log P(từ | thường) = log [P(từ | rác) / P(từ | thường)]
    tu_vung = vec.get_feature_names_out()
    log_prob_thuong = mo_hinh.feature_log_prob_[0]
    log_prob_rac = mo_hinh.feature_log_prob_[1]
    hieu_log = log_prob_rac - log_prob_thuong

    bang_tu = pd.DataFrame({
        "tu": tu_vung,
        "log_p_thuong": log_prob_thuong,
        "log_p_rac": log_prob_rac,
        "hieu_log": hieu_log
    })

    # 5 từ tố cáo tin rác mạnh nhất (hiệu lớn nhất)
    top_rac = bang_tu.sort_values("hieu_log", ascending=False).head(5).reset_index(drop=True)

    # 5 từ cho thấy tin thường mạnh nhất (hiệu nhỏ nhất, âm nhiều nhất)
    top_thuong = bang_tu.sort_values("hieu_log", ascending=True).head(5).reset_index(drop=True)

    print("5 TU TO CAO TIN RAC MANH NHAT (hieu log lon nhat):")
    print(f"  {'STT':<4} {'Tu':<12} {'log P(tu|rac)':<16} {'log P(tu|thuong)':<18} {'Hieu log':<12}")
    print("  " + "-" * 60)
    for i, r in top_rac.iterrows():
        print(f"  {i+1:<4} {r['tu']:<12} {r['log_p_rac']:<16.4f} {r['log_p_thuong']:<18.4f} {r['hieu_log']:<12.4f}")
    print()

    print("5 TU CHO THAY TIN THUONG MANH NHAT (hieu log nho nhat):")
    print(f"  {'STT':<4} {'Tu':<12} {'log P(tu|rac)':<16} {'log P(tu|thuong)':<18} {'Hieu log':<12}")
    print("  " + "-" * 60)
    for i, r in top_thuong.iterrows():
        print(f"  {i+1:<4} {r['tu']:<12} {r['log_p_rac']:<16.4f} {r['log_p_thuong']:<18.4f} {r['hieu_log']:<12.4f}")
    print()

    print("GIAI THICH:")
    print("  - Hieu log duong cang lon (nhu 'mai', 'tay', 'ngay', 'mien', 'trung')")
    print("    nghia la xac suat tu do xuat hien trong tin rac vuot troi so voi tin thuong.")
    print("  - Hieu log am cang sau (nhu 'lop', 'em', 'bai', 'phong', 'bao')")
    print("    nghia la tu do xuat hien ap dao trong tin thuong so voi tin rac.")
    print("=" * 65)


if __name__ == "__main__":
    main()
