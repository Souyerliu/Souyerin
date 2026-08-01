"""为中文字体创建子集：保留 used-chars.txt 中的所有字符 + ASCII + 标点 + 常用符号"""
import os
import subprocess
import sys

FONTS_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "assets", "fonts")
CHARS_FILE = os.path.join(os.path.dirname(__file__), "used-chars.txt")

def build_charset():
    """收集所有需要保留的字符"""
    chars = set()

    # 1. ASCII printable
    for c in range(32, 127):
        chars.add(chr(c))

    # 2. 常用标点和符号
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

    # 3. 读取 used-chars.txt 中的中文字符
    if os.path.exists(CHARS_FILE):
        with open(CHARS_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        for c in text:
            if c.strip():
                chars.add(c)

    return "".join(sorted(chars))


def main():
    os.chdir(FONTS_DIR)
    charset = build_charset()
    print(f"Total unique chars to keep: {len(charset)}")

    fonts_to_subset = [
        ("LXGWWenKai-Regular.ttf", "LXGWWenKai-Regular-subset.woff2"),
        ("MapleMono-CN-Regular.ttf", "MapleMono-CN-Regular-subset.woff2"),
    ]

    # Convert chars to hex codepoints
    hex_codes = ",".join(f"{ord(c):04X}" for c in charset)
    
    for src, dst in fonts_to_subset:
        if not os.path.exists(src):
            print(f"SKIP: {src} not found")
            continue
        print(f"\nSubsetting {src} -> {dst} ...")
        cmd = [
            sys.executable, "-m", "fontTools.subset",
            f"--unicodes={hex_codes}",
            "--output-file=" + dst,
            "--flavor=woff2",
            src,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            src_size = os.path.getsize(src) / (1024 * 1024)
            dst_size = os.path.getsize(dst) / (1024 * 1024)
            print(f"  OK: {src_size:.1f}MB -> {dst_size:.2f}MB ({dst_size*1024:.0f}KB)")
        else:
            print(f"  FAILED: {result.stderr[-500:]}")


if __name__ == "__main__":
    main()
