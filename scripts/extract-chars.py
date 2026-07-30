import os
import sys

posts_dir = r"d:\Nodejs\Souyersblog_astro\astro-blog-shokax-main\src\posts"
config_dir = r"d:\Nodejs\Souyersblog_astro\astro-blog-shokax-main\src"
components_dir = r"d:\Nodejs\Souyersblog_astro\astro-blog-shokax-main\src\components"
layouts_dir = r"d:\Nodejs\Souyersblog_astro\astro-blog-shokax-main\src\layouts"

all_chars = set()

# 基础 ASCII
base_chars = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "0123456789"
    r'!@#$%^&*()_+-=[]{}|;:,<>.?/~ '
    "\"'"
    "\n\r\t"
)
# 常用中文标点和数学符号
cn_punct = "，。！？；：""''（）【】《》…—～·、×÷±≈≠≤≥∞∑∏∫∂√∆∇∈∉⊂⊃∪∩∧∨¬⇒⇔∀∃αβγδεζηθικλμνξπρστυφχψω←→↑↓↔⇐⇒⇑⇓⇕"
all_chars.update(cn_punct)

for d in [posts_dir, components_dir, layouts_dir, config_dir]:
    for root, dirs, files in os.walk(d):
        for f in files:
            if f.endswith((".mdx", ".md", ".astro", ".svelte", ".ts", ".json")):
                try:
                    with open(os.path.join(root, f), "r", encoding="utf-8") as fh:
                        all_chars.update(fh.read())
                except:
                    pass

all_chars.update(base_chars)
cjk = sorted(c for c in all_chars if "\u4e00" <= c <= "\u9fff")
print(f"CJK chars: {len(cjk)}")
with open("scripts/used-chars.txt", "w", encoding="utf-8") as f:
    f.write("".join(cjk))

print("Written to scripts/used-chars.txt")
