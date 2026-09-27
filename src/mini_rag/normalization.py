import re
import unicodedata

_ZWNJ = "\u200c"

# Character variants -> one standard form
_TRANSLATION = {
    0x064A: "\u06cc",  # Arabic yeh -> Persian yeh
    0x0649: "\u06cc",  # alef maksura -> Persian yeh
    0x0643: "\u06a9",  # Arabic kaf -> Persian keh
    0x0623: "\u0627",  # alef with hamza above -> alef
    0x0625: "\u0627",  # alef with hamza below -> alef
}
for _start in (0x06F0, 0x0660):  # Persian and Arabic-Indic digits -> 0-9
    for _i in range(10):
        _TRANSLATION[_start + _i] = str(_i)

# Characters that carry no meaning for search and are deleted
_REMOVE = re.compile(
    "["
    "\u064b-\u065f"  # Arabic diacritics
    "\u0670"  # superscript alef
    "\u0640"  # tatweel
    "\u200b"  # zero-width space
    "\u200e\u200f"  # LTR / RTL marks
    "\u202a-\u202e"  # bidi embedding controls
    "\u2066-\u2069"  # bidi isolates
    "\ufeff"  # BOM
    "\u00ad"  # soft hyphen
    "]"
)

_ZWNJ_REPEATED = re.compile(_ZWNJ + "{2,}")
_ZWNJ_AT_EDGE = re.compile(
    r"(?:(?<=\s)|^)" + _ZWNJ + r"+|" + _ZWNJ + r"+(?=\s|$)", re.MULTILINE
)
_SPACES = re.compile(r"[^\S\n]+")  # any whitespace except newline
_SPACE_AROUND_NEWLINE = re.compile(r" ?\n ?")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text):
    """Normalize Persian text for search.

    Use the SAME function for documents and for user queries.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_TRANSLATION)
    text = _REMOVE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZWNJ_REPEATED.sub(_ZWNJ, text)
    text = _ZWNJ_AT_EDGE.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()