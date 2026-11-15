# -*- coding: utf-8 -*-
"""Script tự động chạy từng bài tập và tạo ảnh chụp màn hình terminal Windows PowerShell.

Tạo ra:
  - baitap06/screenshot_bai1.png
  - baitap06/screenshot_bai2.png
  - baitap06/screenshot_bai3.png
  - baitap06/screenshot_bai4.png
  - baitap06/screenshot_bai5.png
  - baitap06/screenshot_bai6.png
"""

import os
import subprocess
import sys
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_BOLD_PATH = "C:/Windows/Fonts/consolab.ttf"
FONT_SIZE = 15
LINE_HEIGHT = 21
PADDING_LEFT = 20
PADDING_TOP = 46
PADDING_BOTTOM = 20
BG_COLOR = (12, 12, 12)
HEADER_COLOR = (31, 31, 31)
TEXT_COLOR = (220, 220, 220)
PROMPT_COLOR = (90, 200, 255)
COMMAND_COLOR = (255, 215, 0)
TITLE_COLOR = (200, 200, 200)

font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
font_bold = ImageFont.truetype(FONT_BOLD_PATH, FONT_SIZE)
font_title = ImageFont.truetype(FONT_PATH, 12)


def render_terminal_screenshot(command_str, output_lines, save_path):
    """Vẽ một cửa sổ terminal chứa dòng lệnh đã chạy và toàn bộ kết quả in ra."""
    all_lines = [("prompt", "PS C:\\Users\\nhatm\\Machine_Learning_VLU\\06. SVM> " + command_str)]
    for line in output_lines:
        all_lines.append(("text", line))

    max_len = max(len(l[1]) for l in all_lines)
    img_width = max(880, int(max_len * 9.2) + PADDING_LEFT * 2)
    img_height = PADDING_TOP + len(all_lines) * LINE_HEIGHT + PADDING_BOTTOM

    img = Image.new("RGB", (img_width, img_height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    # 1. Title bar
    draw.rectangle([0, 0, img_width, 32], fill=HEADER_COLOR)
    draw.text((16, 8), "Windows PowerShell - 06. SVM", font=font_title, fill=TITLE_COLOR)

    # Window controls
    draw.line([img_width - 100, 16, img_width - 90, 16], fill=(180, 180, 180), width=1)
    draw.rectangle([img_width - 68, 11, img_width - 58, 21], outline=(180, 180, 180), width=1)
    draw.line([img_width - 32, 11, img_width - 22, 21], fill=(220, 80, 80), width=2)
    draw.line([img_width - 32, 21, img_width - 22, 11], fill=(220, 80, 80), width=2)

    # 2. Content
    y_pos = PADDING_TOP
    for kind, text in all_lines:
        if kind == "prompt":
            prefix = "PS C:\\Users\\nhatm\\Machine_Learning_VLU\\06. SVM> "
            draw.text((PADDING_LEFT, y_pos), prefix, font=font_bold, fill=PROMPT_COLOR)
            prompt_len = draw.textlength(prefix, font=font_bold)
            draw.text((PADDING_LEFT + prompt_len, y_pos), command_str, font=font_bold, fill=COMMAND_COLOR)
        else:
            text_color = TEXT_COLOR
            if "===" in text or "---" in text:
                text_color = (130, 210, 130)
            elif "BAI TAP" in text or "CAU TRA LOI" in text or "BANG " in text:
                text_color = (255, 200, 80)
            elif "[+]" in text or "CO!" in text:
                text_color = (100, 240, 120)
            elif "->" in text or "=>" in text:
                text_color = (120, 220, 255)
            elif "w1 =" in text or "Cap tham so" in text:
                text_color = (240, 240, 240)
            draw.text((PADDING_LEFT, y_pos), text, font=font, fill=text_color)
        y_pos += LINE_HEIGHT

    img.save(save_path)
    print(f"[+] Da tao anh chup: {save_path}")


def chay_va_chup(cmd, file_ten):
    res = subprocess.run(
        cmd,
        cwd=os.getcwd(),
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    lines = res.stdout.strip().split("\n")
    if res.stderr.strip():
        lines.extend(["STDERR:"] + res.stderr.strip().split("\n"))
    
    save_path = os.path.join("baitap06", file_ten)
    render_terminal_screenshot(cmd, lines, save_path)


if __name__ == "__main__":
    danh_sach_bai = [
        ("python baitap06/bai1.py", "screenshot_bai1.png"),
        ("python baitap06/bai2.py", "screenshot_bai2.png"),
        ("python baitap06/bai3.py", "screenshot_bai3.png"),
        ("python baitap06/bai4.py", "screenshot_bai4.png"),
        ("python baitap06/bai5.py", "screenshot_bai5.png"),
        ("python baitap06/bai6.py", "screenshot_bai6.png"),
    ]

    for cmd, file_ten in danh_sach_bai:
        print(f"[*] Dang chay: {cmd} ...")
        chay_va_chup(cmd, file_ten)

    print("\n[V] Hoan thanh tao toan bo anh chup man hinh ket qua!")
