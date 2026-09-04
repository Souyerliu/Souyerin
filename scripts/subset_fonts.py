"""Generate the checked-in WOFF2 subsets used by the Astro site.

The subset combines all text-bearing source files with the shared dictionary
used by the original Astro project. An optional scripts/used-chars.txt file
can still provide characters from other external data sources.
"""

from pathlib import Path

from fontTools.subset import Subsetter
from fontTools.ttLib import TTFont


ROOT = Path(__file__).resolve().parents[1]
FONTS_DIR = ROOT / "src" / "assets" / "fonts"
SHARED_DICTIONARY_FILE = (
    ROOT.parent
    / "Souyersblog_astro"
    / "astro-blog-shokax-main"
    / "scripts"
    / "dictionary.txt"
)
EXTRA_CHARS_FILE = Path(__file__).with_name("used-chars.txt")
SOURCE_EXTENSIONS = {
    ".astro",
    ".css",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".mdx",
    ".svelte",
    ".ts",
    ".tsx",
    ".txt",
    ".yml",
    ".yaml",
}


def add_text(chars: set[str], text: str) -> None:
    chars.update(char for char in text if not char.isspace())


def collect_source_chars() -> set[str]:
    chars: set[str] = set(chr(codepoint) for codepoint in range(32, 127))

    if not SHARED_DICTIONARY_FILE.is_file():
        raise FileNotFoundError(
            f"Shared font dictionary not found: {SHARED_DICTIONARY_FILE}"
        )
    add_text(chars, SHARED_DICTIONARY_FILE.read_text(encoding="utf-8"))

    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        if "node_modules" in path.parts or "dist" in path.parts:
            continue
        try:
            add_text(chars, path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue

    if EXTRA_CHARS_FILE.exists():
        add_text(chars, EXTRA_CHARS_FILE.read_text(encoding="utf-8"))

    return chars


def subset_font(source_name: str, output_name: str, unicodes: list[int]) -> None:
    source = FONTS_DIR / source_name
    output = FONTS_DIR / output_name
    if not source.exists():
        print(f"SKIP: {source_name} not found")
        return

    font = TTFont(source)
    subsetter = Subsetter()
    subsetter.populate(unicodes=unicodes)
    subsetter.subset(font)
    font.flavor = "woff2"
    font.save(output)
    font.close()

    source_size = source.stat().st_size / (1024 * 1024)
    output_size = output.stat().st_size / (1024 * 1024)
    print(f"OK: {source_name} {source_size:.1f}MB -> {output_name} {output_size:.2f}MB")


def main() -> None:
    chars = collect_source_chars()
    unicodes = sorted(ord(char) for char in chars)
    print(f"Using shared dictionary: {SHARED_DICTIONARY_FILE}")
    print(f"Keeping {len(unicodes)} unique characters")
    subset_font("LXGWWenKai-Regular.ttf", "LXGWWenKai-Regular-subset.woff2", unicodes)
    subset_font("MapleMono-CN-Regular.ttf", "MapleMono-CN-Regular-subset.woff2", unicodes)


if __name__ == "__main__":
    main()
