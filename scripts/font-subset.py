#!/usr/bin/env python3
"""
字体子集化脚本：将 LXGW WenKai 和 Maple Mono 字体子集化 + 转换为 woff2
"""
import subprocess
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS_DIR = os.path.join(BASE_DIR, "src", "assets", "fonts")
OUTPUT_DIR = FONTS_DIR  # 直接覆盖源文件位置，保持引用路径不变
CHARS_FILE = os.path.join(os.path.dirname(__file__), "used-chars.txt")

# 读取文章中使用的字符
with open(CHARS_FILE, "r", encoding="utf-8") as f:
    article_chars = f.read().strip()

# 扩充字符集：加入所有 ASCII + 常用标点 + 全角符号
extra_chars = (
    # ASCII
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "0123456789"
    r'!@#$%^&*()_+-=[]{}|;:,.<>?/~ '
    "\"'`"
    # 中文标点
    "，。！？；：""''（）【】《》…—～·、"
    # 数学符号
    "×÷±≈≠≤≥∞∑∏∫∂√∆∇∈∉⊂⊃∪∩∧∨¬⇒⇔∀∃"
    # 希腊字母
    "αβγδεζηθικλμνξπρστυφχψω"
    # 箭头
    "←→↑↓↔⇐⇒⇑⇓⇕"
    # 日文假名（blog 中可能出现）
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
    "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
    "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ"
    "ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ"
    "ゃゅょっャュョッ"
)
all_chars = "".join(sorted(set(article_chars + extra_chars)))
print(f"Subsetting fonts with {len(all_chars)} unique characters...")

fonts = [
    {
        "name": "LXGWWenKai-Regular",
        "input": os.path.join(FONTS_DIR, "LXGWWenKai-Regular.ttf"),
        "output": os.path.join(FONTS_DIR, "LXGWWenKai-Regular-subset.woff2"),
    },
    {
        "name": "MapleMono-CN-Regular",
        "input": os.path.join(FONTS_DIR, "MapleMono-CN-Regular.ttf"),
        "output": os.path.join(FONTS_DIR, "MapleMono-CN-Regular-subset.woff2"),
    },
]

for font in fonts:
    if not os.path.exists(font["input"]):
        print(f"  SKIP {font['name']}: source file not found")
        continue

    cmd = [
        "pyftsubset",
        font["input"],
        f'--text={all_chars}',
        f'--output-file={font["output"]}',
        "--flavor=woff2",
        "--layout-features=*",
        "--no-hinting",
        "--desubroutinize",
    ]
    print(f"  Running: {' '.join(cmd[:2])} ...")
    subprocess.run(cmd, check=True)

    in_size = os.path.getsize(font["input"]) / 1024
    out_size = os.path.getsize(font["output"]) / 1024
    print(f"  {font['name']}: {in_size:.0f} KB -> {out_size:.0f} KB ({(1-out_size/in_size)*100:.0f}% reduction)")

print("\nDone! Update fonts.css to reference the .woff2 subset files.")
print('Example: src: url("../assets/fonts/LXGWWenKai-Regular-subset.woff2") format("woff2");')
