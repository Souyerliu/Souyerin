"""为中文字体创建子集：保留 used-chars.txt 中的所有字符 + ASCII + 标点 + 常用符号 + Live2D 看板娘文本"""
import json
import os

from fontTools.subset import Subsetter
from fontTools.ttLib import TTFont

FONTS_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "assets", "fonts")
CHARS_FILE = os.path.join(os.path.dirname(__file__), "used-chars.txt")
LIVE2D_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "live2d-models")

def extract_chars_from_json(filepath):
    """递归提取 JSON 文件中所有字符串包含的字符"""
    chars = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return chars

    def walk(obj):
        if isinstance(obj, str):
            for c in obj:
                if c.strip():
                    chars.add(c)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)

    walk(data)
    return chars


def extract_chars_from_live2d():
    """提取 Live2D 看板娘目录下所有 JSON 文件中的字符"""
    chars = set()
    if not os.path.isdir(LIVE2D_DIR):
        return chars

    for fname in os.listdir(LIVE2D_DIR):
        if fname.endswith(".json"):
            fpath = os.path.join(LIVE2D_DIR, fname)
            file_chars = extract_chars_from_json(fpath)
            chars.update(file_chars)

    return chars


def build_charset():
    """收集所有需要保留的字符"""
    chars = set()

    # 1. ASCII printable
    for c in range(32, 127):
        chars.add(chr(c))

    # 2. 日语五十音：平假名 + 片假名
    # 平假名 U+3041–U+3096, 浊点半浊点 U+3099–U+309C, 繰返し記号 U+309D–U+309F
    for c in range(0x3041, 0x3097):
        chars.add(chr(c))
    for c in range(0x3099, 0x30A0):
        chars.add(chr(c))
    # 片假名 U+30A1–U+30FA, 繰返し記号 U+30FD–U+30FF
    for c in range(0x30A1, 0x30FB):
        chars.add(chr(c))
    for c in range(0x30FD, 0x3100):
        chars.add(chr(c))
    # 半角片假名 U+FF66–U+FF9F
    for c in range(0xFF66, 0xFFA0):
        chars.add(chr(c))

    # 3. 常用标点和符号
    extra = (
        "　、。〃々「」『』【】〒〓〔〕〖〗！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～"
        "¡¢£¤¥¦§¨©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"
        "–—―‖‗†‡•…‰′″‹›‼‾⁄⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔ℃№℡℉"
        "←↑→↓↔↕↖↗↘↙↚↛↜↝↞↟↠↡↢↣↤↥↦↧↨↩↪↫↬↭↮↯↰↱↲↳↴↵↶↷↸↹↺↻↼↽↾↿⇀⇁⇂⇃⇄⇅⇆⇇⇈⇉⇊⇋⇌⇍⇎⇏"
        "∀∁∂∃∄∅∆∇∈∉∊∋∌∍∎∏∐∑−∓∔∕∖∗∘∙√∛∜∝∞∟∠∡∢∣∥∦∧∨∩∪∫∬∭∮∯∰∱∲∳∴∵∶∷∸∹∺∻∼∽∾∿≀≁≂≃≄≅≆≇≈≉≊≋≌≍≎≏≐≑≒≓≔≕≖≗≘≙≚≛≜≝≞≟≠≡≢≣≤≥≦≧≨≩≪≫≬≭≮≯≰≱≲≳≴≵≶≷≸≹≺≻≼≽≾≿"
        "─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿"
        "▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟■□▢▣▤▥▦▧▨▩▪▫▬▭▮▯▰▱▲△▴▵▶▷▸▹►▻▼▽▾▿◀◁◂◃◄◅◆◇◈◉◌◍◎●◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮◯◰◱◲◳◴◵◶◷◸◹◺◻◼◽◾◿"
        "☆★✠✡✢✣✤✥✦✧✩✪✫✬✭✮✯✰✱✲✳✴✵✶✷✸✹✺✻✼✽✾✿❀❁❂❃❄❅❆❇❈❉❊❋"
        "🀀🀁🀂🀃🀄🀅🀆🀇🀈🀉🀊🀋🀌🀍🀎🀏🀐🀑🀒🀓🀔🀕🀖🀗🀘🀙🀚🀛🀜🀝🀞🀟🀠🀡🀢🀣🀤🀥🀦🀧🀨🀩🀪🀫"
        "🥰😊😭😂❤️💪🙏🔥👍💕😘😍"
        "·ãéñöü"
    )
    for c in extra:
        chars.add(c)

    # 4. 读取 used-chars.txt 中的中文字符
    if os.path.exists(CHARS_FILE):
        with open(CHARS_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        for c in text:
            if c.strip():
                chars.add(c)

    # 5. 提取 Live2D 看板娘 JSON 中的字符（waifu-tips.json / model_list.json 等）
    live2d_chars = extract_chars_from_live2d()
    chars.update(live2d_chars)

    return "".join(sorted(chars))


def main():
    os.chdir(FONTS_DIR)
    charset = build_charset()
    print(f"Total unique chars to keep: {len(charset)}")

    fonts_to_subset = [
        ("LXGWWenKai-Regular.ttf", "LXGWWenKai-Regular-subset.woff2"),
        ("MapleMono-CN-Regular.ttf", "MapleMono-CN-Regular-subset.woff2"),
    ]

    unicodes = [ord(c) for c in charset]

    for src, dst in fonts_to_subset:
        if not os.path.exists(src):
            print(f"SKIP: {src} not found")
            continue
        print(f"\nSubsetting {src} -> {dst} ...")

        font = TTFont(src)
        subsetter = Subsetter()
        subsetter.populate(unicodes=unicodes)
        subsetter.subset(font)

        font.flavor = "woff2"
        font.save(dst)
        font.close()

        src_size = os.path.getsize(src) / (1024 * 1024)
        dst_size = os.path.getsize(dst) / (1024 * 1024)
        print(f"  OK: {src_size:.1f}MB -> {dst_size:.2f}MB ({dst_size*1024:.0f}KB)")


if __name__ == "__main__":
    main()
