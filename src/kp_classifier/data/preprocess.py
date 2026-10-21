import re
import unicodedata

PHONE_TOKEN = "<phone>"
ID_TOKEN = "<id>"

CYRILLIC_TO_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e", "ж": "ž",
    "з": "z", "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj", "м": "m", "н": "n",
    "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "ћ": "ć", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "č", "џ": "dž", "ш": "š",
}
CYRILLIC_TO_LATIN.update({k.upper(): v.capitalize() for k, v in CYRILLIC_TO_LATIN.items()})

DIACRITICS = str.maketrans({"č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "dj"})

CYRILLIC_X_BETWEEN_DIGITS_RE = re.compile(r"(?<=\d)\s?[хХ]\s?(?=\d)")

PHONE_RE = re.compile(
    r"(?<![\w.,])"
    r"(?:(?:\+|00)381[\s/\-.]?(?:\(0\)\s?)?|0)"
    r"[1-9]\d"
    r"(?:[\s/\-.]?\d){6,7}"
    r"(?!\d)"
)

BRAND_ARTICLE_RULES = [
    (
        re.compile(r"\b(?:bosch|gws|gss|gex|pex|psm|gpo|grl|gsr|gsb)\b", re.IGNORECASE),
        re.compile(r"06(?:0[1-7]|1[1-9])\d{5,6}"),
    ),
    (
        re.compile(r"\beta\b", re.IGNORECASE),
        re.compile(r"0\d{3}\s?9\d{4}"),
    ),
]

ID_RE = re.compile(
    r"\b(?:šifra|sifra)\s*(?:oglasa)?\s*[:#.]?\s*\d{3,}\b",
    re.IGNORECASE,
)

NON_WORD_RE = re.compile(r"[^\w<>]+")
WHITESPACE_RE = re.compile(r"\s+")


def transliterate_cyrillic(text):
    text = CYRILLIC_X_BETWEEN_DIGITS_RE.sub("x", text)
    return "".join(CYRILLIC_TO_LATIN.get(c, c) for c in text)


def is_brand_article(match, text):
    candidate = match.group(0)
    return any(
        brand_re.search(text) and article_re.fullmatch(candidate)
        for brand_re, article_re in BRAND_ARTICLE_RULES
    )


def replace_phones(text):
    def repl(match):
        if is_brand_article(match, text):
            return match.group(0)
        return f" {PHONE_TOKEN} "

    return PHONE_RE.sub(repl, text)


def replace_ids(text):
    return ID_RE.sub(f" {ID_TOKEN} ", text)


def remove_diacritics(text):
    text = text.translate(DIACRITICS)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = transliterate_cyrillic(text)
    text = replace_phones(text)
    text = replace_ids(text)
    text = text.lower()
    text = remove_diacritics(text)
    text = NON_WORD_RE.sub(" ", text)
    text = text.replace("_", " ")
    return WHITESPACE_RE.sub(" ", text).strip()