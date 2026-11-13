"""Draw desktop screens into `<dest>/screens/`: what a remote desktop mostly shows.

The public datasets are photos and camera video. A streamed desktop is mostly something
else: text a few pixels tall, one-pixel lines, flat colour, icons, and a photo here and
there. An upscaler trained only on photos has never seen a glyph, so it cannot be judged on
one either. This script draws screens the way an operating system does - real fonts,
hinted and anti-aliased, sometimes with ClearType's colour fringes - in the layouts people
share: code editors (real Python source), documents, terminals, spreadsheets, chat, file
explorers and settings pages, light and dark, at display scales from 100% to 200%.

Two kinds of picture, because a stream shows two kinds of motion:

- **page**: one application filling a tall canvas - a long document, a file of code, a
  sheet - so a window sliding down it is a scroll;
- **desktop**: a whole screen - wallpaper, taskbar, two or three windows - which mostly
  stands still.

Each split gets its own pictures and its own photos (DIV2K train for `train`, DIV2K valid for
`test`, when DIV2K is on disk), so nothing in a test screen was seen in training. Fonts are
looked up in the usual folders (`--fonts` adds one); with none found, Pillow's own font is
used. Text comes from Python's standard library: its docstrings for prose, its source for
code.

    uv run datasets/screens.py                       # 300 train, 60 test
    uv run datasets/screens.py --train 600 --test 100
    uv run datasets/screens.py --fonts "C:/Windows/Fonts"

Writes `screens/train/*.png`, `screens/test/*.png`, the `train.txt` / `test.txt` lists
`download.py` writes for every dataset, and `screens.json`: how each picture was drawn.
Every picture has its own seed, so the same arguments draw the same screens.
"""

import argparse
import ast
import json
import math
import os
import re
import sysconfig
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ.get("DATASETS_DIR", Path(__file__).resolve().parent))
FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/mnt/c/Windows/Fonts"), Path("/usr/share/fonts"),
             Path("/usr/local/share/fonts"), Path.home() / ".fonts", Path("/System/Library/Fonts"),
             Path("/Library/Fonts")]

# (regular, bold, kind) font file names without extension, matched case-insensitively
FONTS = [
    ("segoeui", "segoeuib", "sans"), ("arial", "arialbd", "sans"), ("calibri", "calibrib", "sans"),
    ("tahoma", "tahomabd", "sans"), ("verdana", "verdanab", "sans"), ("trebuc", "trebucbd", "sans"),
    ("corbel", "corbelb", "sans"), ("LiberationSans-Regular", "LiberationSans-Bold", "sans"),
    ("DejaVuSans", "DejaVuSans-Bold", "sans"), ("NotoSans-Regular", "NotoSans-Bold", "sans"),
    ("Roboto-Regular", "Roboto-Bold", "sans"), ("Ubuntu-R", "Ubuntu-B", "sans"),
    ("times", "timesbd", "serif"), ("georgia", "georgiab", "serif"), ("cambria", "cambriab", "serif"),
    ("constan", "constanb", "serif"), ("LiberationSerif-Regular", "LiberationSerif-Bold", "serif"),
    ("DejaVuSerif", "DejaVuSerif-Bold", "serif"),
    ("consola", "consolab", "mono"), ("CascadiaMono", None, "mono"), ("CascadiaCode", None, "mono"),
    ("cour", "courbd", "mono"), ("lucon", None, "mono"), ("LiberationMono-Regular", "LiberationMono-Bold", "mono"),
    ("DejaVuSansMono", "DejaVuSansMono-Bold", "mono"),
]
ICON_FONTS = ("SegoeIcons", "segmdl2")   # Windows' UI icons, used for toolbar and taskbar glyphs
ICONS = [0xE700, 0xE710, 0xE711, 0xE713, 0xE715, 0xE716, 0xE721, 0xE722, 0xE72C, 0xE72D, 0xE734,
         0xE74D, 0xE74E, 0xE768, 0xE769, 0xE77B, 0xE77F, 0xE787, 0xE80F, 0xE895, 0xE896, 0xE898,
         0xE8A5, 0xE8B7, 0xE8BD, 0xE8C6, 0xE8C8, 0xE8D4, 0xE8EC, 0xE8F1, 0xE909, 0xE91B, 0xE946,
         0xE70F, 0xE7C3, 0xE7E8, 0xE7B8]

# words with accents, mixed into prose now and then: not every screen is in English
ACCENTED = ("árvíztűrő tükörfúrógép beállítások mentés megnyitás Größe Straße übermäßig Schließen "
            "Datei façade naïve café déjà señor mañana niño Configuración ação coração łódź żółć "
            "příliš žluťoučký kůň Ünnepek felhasználó jelszó").split()

LIGHT = {"bg": (255, 255, 255), "panel": (243, 243, 243), "text": (30, 30, 30), "muted": (105, 105, 105),
         "line": (222, 222, 222), "title": (238, 238, 238), "hover": (229, 241, 251)}
DARK = {"bg": (30, 30, 30), "panel": (37, 37, 38), "text": (212, 212, 212), "muted": (140, 140, 140),
        "line": (62, 62, 62), "title": (45, 45, 46), "hover": (42, 45, 46)}
ACCENTS = [(0, 120, 215), (0, 99, 177), (16, 124, 16), (196, 43, 28), (136, 23, 152), (202, 80, 16),
           (0, 153, 188), (194, 57, 179), (45, 125, 154), (98, 100, 167)]

# Python source colours, VS Code's Dark+ and Light+
SYNTAX = {
    "dark": {"comment": (106, 153, 85), "string": (206, 145, 120), "number": (181, 206, 168),
             "keyword": (86, 156, 214), "call": (220, 220, 170), "text": (212, 212, 212)},
    "light": {"comment": (0, 128, 0), "string": (163, 21, 21), "number": (9, 134, 88),
              "keyword": (0, 0, 255), "call": (121, 94, 38), "text": (0, 0, 0)},
}
TOKEN = re.compile(r"(?P<comment>#.*$)|(?P<string>\"[^\"]*\"?|'[^']*'?)|(?P<number>\b\d+(?:\.\d+)?\b)"
                   r"|(?P<keyword>\b(?:def|class|return|if|elif|else|for|while|in|not|and|or|import|from|as"
                   r"|with|try|except|finally|raise|yield|lambda|pass|break|continue|None|True|False|self"
                   r"|is|async|await|global|del|assert)\b)|(?P<call>[A-Za-z_]\w*(?=\())|(?P<text>\w+|\s+|.)")

# the LCD filter FreeType applies to subpixel text, so a fringe is not a full-strength colour
LCD_FILTER = np.array([8, 77, 86, 77, 8], np.float32) / 256


# ---------------------------------------------------------------- what is drawn with

def find_fonts(extra=()):
    """{kind: [(regular path, bold path or None), ...]} and the icon font, from the font folders."""
    files = {}
    for folder in [*map(Path, extra), *FONT_DIRS]:
        if folder.is_dir():
            for path in folder.rglob("*"):
                if path.suffix.lower() in (".ttf", ".ttc", ".otf"):
                    files.setdefault(path.stem.lower(), path)
    found = {"sans": [], "serif": [], "mono": []}
    for regular, bold, kind in FONTS:
        if regular.lower() in files:
            found[kind].append((str(files[regular.lower()]), str(files[bold.lower()]) if bold and bold.lower() in files else None))
    icons = next((str(files[name.lower()]) for name in ICON_FONTS if name.lower() in files), None)
    return found, icons


def load_corpus():
    """Sentences from the standard library's docstrings, and its source files as lines."""
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    sentences, sources, names = [], [], []
    for path in sorted(stdlib.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [line.expandtabs(4) for line in text.splitlines()]
        if len(lines) > 80:
            sources.append(lines)
        names.append(path.name)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    sentences += [s.strip().rstrip(".") + "." for s in " ".join(doc.split()).split(". ")
                                  if len(s.split()) > 3]
    return {"sentences": sentences, "sources": sources, "names": names}


@lru_cache(maxsize=512)
def font_at(path, size):
    if path is None:
        return ImageFont.load_default(size)
    return ImageFont.truetype(path, size)


# ---------------------------------------------------------------- one screen

class Screen:
    """A canvas and everything one picture is drawn with: its theme, scale, fonts, text."""

    def __init__(self, rng, size, fonts, icons, corpus, photos):
        self.rng, self.corpus, self.photos = rng, corpus, photos
        self.image = Image.new("RGB", size, (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)
        self.scale = float(rng.choice([1.0, 1.25, 1.5, 2.0]))
        self.dark = bool(rng.random() < 0.4)
        self.theme = dict(DARK if self.dark else LIGHT)
        shift = int(rng.integers(-6, 7))     # no two screens quite the same grey
        self.theme = {k: tuple(int(np.clip(c + shift, 0, 255)) for c in v) for k, v in self.theme.items()}
        self.accent = tuple(int(c) for c in ACCENTS[int(rng.integers(len(ACCENTS)))])
        self.subpixel = bool(rng.random() < 0.4)    # ClearType, as many Windows apps still draw text
        self.faces = {}
        for kind, options in fonts.items():
            self.faces[kind] = options[int(rng.integers(len(options)))] if options else (None, None)
        self.icons = icons
        self.record = {"scale": self.scale, "theme": "dark" if self.dark else "light", "subpixel": self.subpixel,
                       "fonts": {k: Path(v[0]).stem if v[0] else "default" for k, v in self.faces.items()}}

    def px(self, value):
        """A size in logical pixels, at this screen's display scale."""
        return max(1, int(round(value * self.scale)))

    def font(self, kind="sans", size=13, bold=False):
        regular, heavy = self.faces[kind]
        return font_at(heavy if bold and heavy else regular, self.px(size))

    # -- text

    def text(self, xy, string, kind="sans", size=13, bold=False, fill=None, clip=None):
        """Draw a run of text with its top-left at `xy`; returns its width in pixels."""
        font = self.font(kind, size, bold)
        fill = fill or self.theme["text"]
        if clip is not None:   # cut the run where it would leave its box
            while string and xy[0] + font.getlength(string) > clip:
                string = string[:-1]
        if not string.strip():
            return font.getlength(string)
        if self.subpixel:
            self._subpixel_text(xy, string, kind, size, bold, fill)
        else:
            self.draw.text(xy, string, font=font, fill=fill)
        return font.getlength(string)

    def _subpixel_text(self, xy, string, kind, size, bold, fill):
        """ClearType-style text: drawn three times wider, each third of a pixel one of R, G
        and B, then softened with the LCD filter."""
        regular, heavy = self.faces[kind]
        path = heavy if bold and heavy else regular
        big = font_at(path, self.px(size) * 3)
        left, top, right, bottom = big.getbbox(string)
        width, height = int(right) + 6, int(bottom) + 3
        width, height = width + (-width) % 3, height + (-height) % 3
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).text((0, 0), string, font=big, fill=255)
        cover = np.asarray(mask, np.float32).reshape(height // 3, 3, width).mean(axis=1) / 255
        cover = sum(w * np.roll(np.pad(cover, ((0, 0), (2, 2))), shift, axis=1)[:, 2:-2]
                    for w, shift in zip(LCD_FILTER, range(-2, 3)))
        cover = cover.reshape(height // 3, width // 3, 3)
        x, y = int(round(xy[0])), int(round(xy[1]))
        h, w = cover.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, self.image.width), min(y + h, self.image.height)
        if x1 <= x0 or y1 <= y0:
            return
        cover = cover[y0 - y:y1 - y, x0 - x:x1 - x]
        region = np.asarray(self.image.crop((x0, y0, x1, y1)), np.float32)
        region = region * (1 - cover) + np.asarray(fill, np.float32) * cover
        self.image.paste(Image.fromarray(region.round().astype(np.uint8)), (x0, y0))

    def words(self, count):
        """`count` words of prose, an accented one now and then."""
        out = []
        while len(out) < count:
            sentence = self.corpus["sentences"][int(self.rng.integers(len(self.corpus["sentences"])))]
            out += sentence.split()
            if self.rng.random() < 0.08:
                out.insert(int(self.rng.integers(len(out) + 1)), ACCENTED[int(self.rng.integers(len(ACCENTED)))])
        return out[:count]

    def phrase(self, low=1, high=4):
        return " ".join(w.strip(".,:;()") for w in self.words(int(self.rng.integers(low, high + 1)))).capitalize()

    def paragraph(self, box, kind="sans", size=14, fill=None, spacing=1.45, link_share=0.03):
        """Words wrapped into `box`; returns the y where it ended."""
        x0, y, x1, y1 = box
        font = self.font(kind, size)
        line_height = self.px(size) * spacing
        space = font.getlength(" ")
        x = x0
        for word in self.words(int(self.rng.integers(25, 120))):
            width = font.getlength(word)
            if x + width > x1 and x > x0:
                x, y = x0, y + line_height
            if y + line_height > y1:
                break
            if self.rng.random() < link_share:   # a link: accent colour, underlined
                self.text((x, y), word, kind, size, fill=self.accent)
                base = y + self.px(size) * 1.15
                self.draw.line([(x, base), (x + width, base)], fill=self.accent, width=max(1, self.px(1)))
            else:
                self.text((x, y), word, kind, size, fill=fill)
            x += width + space
        return y + line_height

    # -- shapes

    def icon(self, xy, size=16, fill=None):
        """A UI icon: a glyph of Windows' icon font, or a simple shape without one."""
        fill = fill or self.theme["text"]
        if self.icons:
            glyph = chr(ICONS[int(self.rng.integers(len(ICONS)))])
            self.draw.text(xy, glyph, font=font_at(self.icons, self.px(size)), fill=fill)
            return
        x, y, s = xy[0], xy[1], self.px(size)
        if self.rng.random() < 0.5:
            self.draw.rounded_rectangle([x, y, x + s, y + s], radius=self.px(3), outline=fill, width=self.px(1.5))
        else:
            self.draw.ellipse([x, y, x + s, y + s], outline=fill, width=self.px(1.5))

    def photo(self, box):
        """One of the split's photos, cropped into `box`, or a gradient without any."""
        x0, y0, x1, y1 = (int(v) for v in box)
        width, height = x1 - x0, y1 - y0
        if width < 8 or height < 8:
            return
        if self.photos:
            picture = Image.open(self.photos[int(self.rng.integers(len(self.photos)))]).convert("RGB")
            ratio = max(width / picture.width, height / picture.height)
            picture = picture.resize((max(width, int(picture.width * ratio)), max(height, int(picture.height * ratio))),
                                     Image.LANCZOS)
            left = int(self.rng.integers(0, picture.width - width + 1))
            top = int(self.rng.integers(0, picture.height - height + 1))
            self.image.paste(picture.crop((left, top, left + width, top + height)), (x0, y0))
        else:
            a, b = (np.asarray(self.rng.integers(0, 256, 3), np.float32) for _ in range(2))
            t = np.linspace(0, 1, height)[:, None, None]
            self.image.paste(Image.fromarray(np.broadcast_to(a + (b - a) * t, (height, width, 3)).astype(np.uint8)), (x0, y0))

    def scrollbar(self, box):
        x0, y0, x1, y1 = box
        width = self.px(12)
        self.draw.rectangle([x1 - width, y0, x1, y1], fill=self.theme["panel"])
        length = (y1 - y0) * float(self.rng.uniform(0.1, 0.5))
        top = y0 + (y1 - y0 - length) * float(self.rng.random())
        self.draw.rounded_rectangle([x1 - width + self.px(3), top, x1 - self.px(3), top + length],
                                    radius=self.px(3), fill=self.theme["muted"])

    # -- windows and applications

    def window(self, box, app):
        """A window: title bar, then `app` in the rest of it."""
        x0, y0, x1, y1 = (int(v) for v in box)
        bar = self.px(32)
        self.draw.rectangle([x0, y0, x1, y1], fill=self.theme["bg"], outline=self.theme["line"])
        self.draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y0 + bar], fill=self.theme["title"])
        self.icon((x0 + self.px(10), y0 + self.px(8)), 16, self.accent)
        self.text((x0 + self.px(36), y0 + self.px(8)), self.phrase(2, 5), size=12, clip=x1 - self.px(150))
        cx = x1 - self.px(23)
        for symbol in ("close", "max", "min"):   # the three caption buttons, drawn as Windows does
            c, r, w = (cx, y0 + bar // 2), self.px(5), max(1, self.px(1))
            colour = self.theme["text"]
            if symbol == "close":
                self.draw.line([(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)], fill=colour, width=w)
                self.draw.line([(c[0] - r, c[1] + r), (c[0] + r, c[1] - r)], fill=colour, width=w)
            elif symbol == "max":
                self.draw.rectangle([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=colour, width=w)
            else:
                self.draw.line([(c[0] - r, c[1]), (c[0] + r, c[1])], fill=colour, width=w)
            cx -= self.px(46)
        APPS[app](self, (x0 + 1, y0 + bar + 1, x1 - 1, y1 - 1))

    def taskbar(self, height_px):
        width, height = self.image.size
        top = height - height_px
        self.draw.rectangle([0, top, width, height], fill=(32, 32, 32) if self.dark else (238, 240, 244))
        x = width // 2 - self.px(24) * 5
        for _ in range(int(self.rng.integers(6, 11))):
            self.icon((x, top + (height_px - self.px(24)) // 2), 24, self.accent if self.rng.random() < 0.3 else None)
            x += self.px(44)
        clock = f"{int(self.rng.integers(0, 24)):02d}:{int(self.rng.integers(0, 60)):02d}"
        self.text((width - self.px(90), top + self.px(6)), clock, size=12)
        self.text((width - self.px(90), top + self.px(24)), f"2026. {int(self.rng.integers(1, 13)):02d}. "
                  f"{int(self.rng.integers(1, 29)):02d}.", size=11)


# ---------------------------------------------------------------- applications, each filling a box

def document(s, box):
    """A web page or a word processor: headings, paragraphs, lists, a photo, a table."""
    x0, y0, x1, y1 = box
    s.draw.rectangle(box, fill=s.theme["bg"])
    column = min(x1 - x0 - s.px(80), s.px(int(s.rng.integers(620, 920))))
    left = x0 + (x1 - x0 - column) // 2
    kind = "serif" if s.rng.random() < 0.35 else "sans"
    size = int(s.rng.integers(13, 17))
    y = y0 + s.px(30)
    while y < y1 - s.px(40):
        block = s.rng.choice(["heading", "paragraph", "list", "photo", "table"], p=[0.15, 0.5, 0.15, 0.1, 0.1])
        if block == "heading":
            hsize = int(s.rng.integers(19, 30))
            s.text((left, y), s.phrase(2, 6), "sans", hsize, bold=True, clip=left + column)
            y += s.px(hsize) * 1.8
        elif block == "paragraph":
            y = s.paragraph((left, y, left + column, y1), kind, size) + s.px(size)
        elif block == "list":
            for _ in range(int(s.rng.integers(2, 6))):
                middle = y + s.px(size) * 0.62   # the middle of a lower-case letter
                s.draw.ellipse([left + s.px(8), middle - s.px(2.5), left + s.px(13), middle + s.px(2.5)],
                               fill=s.theme["text"])
                y = s.paragraph((left + s.px(26), y, left + column, min(y1, y + s.px(size) * 3)), kind, size) + s.px(4)
            y += s.px(size)
        elif block == "photo":
            height = s.px(int(s.rng.integers(160, 380)))
            s.photo((left, y, left + column, min(y + height, y1)))
            y += height + s.px(size) * 1.5
        else:
            y = table(s, (left, y, left + column, y1), size) + s.px(size)


def table(s, box, size=13, rows=None):
    """A bordered table with a header row; returns the y where it ended."""
    x0, y, x1, y1 = box
    columns = int(s.rng.integers(3, 6))
    edges = np.linspace(x0, x1, columns + 1)
    row_height = s.px(size) * 2
    for row in range(rows or int(s.rng.integers(3, 9))):
        if y + row_height > y1:
            break
        if row == 0:
            s.draw.rectangle([x0, y, x1, y + row_height], fill=s.theme["panel"])
        for c in range(columns):
            cell = s.phrase(1, 3) if row == 0 or s.rng.random() < 0.5 else f"{s.rng.uniform(0, 10000):,.2f}"
            s.text((edges[c] + s.px(8), y + s.px(size) * 0.45), cell, size=size, bold=row == 0, clip=edges[c + 1] - s.px(4))
        s.draw.line([(x0, y + row_height), (x1, y + row_height)], fill=s.theme["line"], width=1)
        y += row_height
    return y


def code(s, box):
    """A code editor: file tree, tabs, line numbers, highlighted Python."""
    x0, y0, x1, y1 = box
    palette = SYNTAX["dark" if s.dark else "light"]
    s.draw.rectangle(box, fill=s.theme["bg"])
    tree = s.px(int(s.rng.integers(180, 260))) if s.rng.random() < 0.7 else 0
    if tree:
        s.draw.rectangle([x0, y0, x0 + tree, y1], fill=s.theme["panel"])
        y = y0 + s.px(10)
        while y < y1 - s.px(22):
            depth = int(s.rng.integers(0, 4))
            name = s.corpus["names"][int(s.rng.integers(len(s.corpus["names"])))]
            s.text((x0 + s.px(14 + 12 * depth), y), name, size=12, clip=x0 + tree - s.px(6))
            y += s.px(22)
    tabs = s.px(34)
    s.draw.rectangle([x0 + tree, y0, x1, y0 + tabs], fill=s.theme["panel"])
    x = x0 + tree
    for index in range(int(s.rng.integers(1, 5))):
        name = s.corpus["names"][int(s.rng.integers(len(s.corpus["names"])))]
        width = s.px(150)
        if index == 0:
            s.draw.rectangle([x, y0, x + width, y0 + tabs], fill=s.theme["bg"])
            s.draw.line([(x, y0), (x + width, y0)], fill=s.accent, width=s.px(2))
        s.text((x + s.px(12), y0 + s.px(9)), name, size=12, clip=x + width - s.px(8))
        x += width
    size = int(s.rng.integers(12, 16))
    font = s.font("mono", size)
    advance = font.getlength("M")
    line_height = s.px(size) * float(s.rng.uniform(1.3, 1.5))
    lines = s.corpus["sources"][int(s.rng.integers(len(s.corpus["sources"])))]
    first = int(s.rng.integers(0, max(1, len(lines) - 200)))
    gutter = x0 + tree + advance * 5
    left = gutter + s.px(16)
    current = int(s.rng.integers(3, 30))
    y = y0 + tabs + s.px(6)
    for index, line in enumerate(lines[first:], start=first + 1):
        if y + line_height > y1:
            break
        if index - first == current:
            s.draw.rectangle([x0 + tree, y, x1, y + line_height], fill=s.theme["hover"])
        s.text((gutter - font.getlength(str(index)), y), str(index), "mono", size, fill=s.theme["muted"])
        column = 0
        for match in TOKEN.finditer(line):
            token = match.group()
            if token.strip():
                s.text((left + column * advance, y), token, "mono", size, fill=palette[match.lastgroup], clip=x1 - s.px(14))
            column += len(token)
        y += line_height
    s.scrollbar((x0, y0 + tabs, x1, y1))


def terminal(s, box):
    """A terminal: prompts, commands and their output, in a monospace font."""
    x0, y0, x1, y1 = box
    background = [(12, 12, 12), (1, 36, 86), (40, 42, 54), (0, 0, 0)][int(s.rng.integers(4))]
    s.draw.rectangle(box, fill=background)
    size = int(s.rng.integers(12, 16))
    line_height = s.px(size) * 1.3
    y, x = y0 + s.px(8), x0 + s.px(10)
    plain, green, blue, yellow = (204, 204, 204), (22, 198, 12), (59, 120, 255), (229, 229, 16)
    while y + line_height < y1:
        roll = s.rng.random()
        if roll < 0.2:   # a prompt and a command
            width = s.text((x, y), "user@desktop", "mono", size, fill=green)
            width += s.text((x + width, y), ":", "mono", size, fill=plain)
            width += s.text((x + width, y), "~/project", "mono", size, fill=blue)
            command = s.rng.choice(["ls -la", "git log --oneline", "python -m pytest -q", "uv sync", "cat README.md",
                                    "ffmpeg -i in.mp4 -vf scale=960:540 out.mp4", "nvidia-smi"])
            s.text((x + width, y), f"$ {command}", "mono", size, fill=plain)
        elif roll < 0.5:   # a listing
            name = s.corpus["names"][int(s.rng.integers(len(s.corpus["names"])))]
            s.text((x, y), f"-rw-r--r-- 1 user user {int(s.rng.integers(100, 99999)):>6} Oct {int(s.rng.integers(1, 29)):>2} "
                           f"{int(s.rng.integers(0, 24)):02d}:{int(s.rng.integers(0, 60)):02d} {name}", "mono", size, fill=plain,
                   clip=x1 - s.px(8))
        elif roll < 0.65:   # a commit
            digest = "".join(s.rng.choice(list("0123456789abcdef"), 7))
            width = s.text((x, y), digest, "mono", size, fill=yellow)
            s.text((x + width + s.px(8), y), s.phrase(3, 8), "mono", size, fill=plain, clip=x1 - s.px(8))
        else:   # output, or a line of code
            lines = s.corpus["sources"][int(s.rng.integers(len(s.corpus["sources"])))]
            s.text((x, y), lines[int(s.rng.integers(len(lines)))], "mono", size, fill=plain, clip=x1 - s.px(8))
        y += line_height


def spreadsheet(s, box):
    """A spreadsheet: ribbon, formula bar, a grid of numbers and words, a selected cell."""
    x0, y0, x1, y1 = box
    s.draw.rectangle(box, fill=s.theme["bg"])
    ribbon = s.px(92)
    s.draw.rectangle([x0, y0, x1, y0 + ribbon], fill=s.theme["panel"])
    x = x0 + s.px(12)
    while x < x1 - s.px(60):
        s.icon((x, y0 + s.px(14)), 20, s.accent if s.rng.random() < 0.3 else None)
        s.text((x - s.px(4), y0 + s.px(44)), s.phrase(1, 1), size=11, clip=x + s.px(54))
        x += s.px(int(s.rng.integers(52, 80)))
    bar = y0 + ribbon + s.px(6)
    s.draw.rectangle([x0 + s.px(80), bar, x1 - s.px(8), bar + s.px(24)], fill=s.theme["bg"], outline=s.theme["line"])
    s.text((x0 + s.px(10), bar + s.px(4)), f"{chr(65 + int(s.rng.integers(8)))}{int(s.rng.integers(1, 40))}", size=12)
    s.text((x0 + s.px(88), bar + s.px(4)), f"=SUM({chr(65 + int(s.rng.integers(5)))}2:{chr(65 + int(s.rng.integers(5)))}"
                                           f"{int(s.rng.integers(10, 99))})", size=12)
    top = bar + s.px(32)
    size = int(s.rng.integers(11, 14))
    row_height, header = s.px(size) * 1.75, s.px(40)
    widths = [s.px(int(s.rng.integers(64, 130))) for _ in range(40)]
    grid = (218, 220, 224) if not s.dark else (70, 70, 70)
    s.draw.rectangle([x0, top, x1, top + row_height], fill=s.theme["panel"])
    s.draw.rectangle([x0, top, x0 + header, y1], fill=s.theme["panel"])
    x, edges = x0 + header, [x0 + header]
    for c, width in enumerate(widths):
        if x > x1:
            break
        s.text((x + width / 2 - s.px(4), top + s.px(4)), chr(65 + c % 26), size=size, fill=s.theme["muted"])
        x += width
        edges.append(x)
        s.draw.line([(x, top), (x, y1)], fill=grid, width=1)
    y, row = top + row_height, 1
    selected = (int(s.rng.integers(1, min(8, len(edges) - 1))), int(s.rng.integers(1, 20)))
    while y + row_height < y1:
        s.text((x0 + s.px(10), y + s.px(4)), str(row), size=size, fill=s.theme["muted"])
        s.draw.line([(x0, y + row_height), (x1, y + row_height)], fill=grid, width=1)
        for c in range(len(edges) - 1):
            if s.rng.random() < 0.35:
                continue
            value = s.rng.random()
            if row == 1:
                cell, bold = s.phrase(1, 2), True
            elif value < 0.6:
                cell, bold = f"{s.rng.uniform(-500, 50000):,.2f}", False
            elif value < 0.8:
                cell, bold = s.phrase(1, 2), False
            else:
                cell, bold = f"2026-{int(s.rng.integers(1, 13)):02d}-{int(s.rng.integers(1, 29)):02d}", False
            s.text((edges[c] + s.px(5), y + s.px(4)), cell, size=size, bold=bold, clip=edges[c + 1] - s.px(3))
        if row == selected[1]:
            c = selected[0]
            s.draw.rectangle([edges[c], y, edges[c + 1], y + row_height], outline=s.accent, width=s.px(2))
        y += row_height
        row += 1


def chat(s, box):
    """A chat or mail client: a list of conversations, message bubbles, an input box."""
    x0, y0, x1, y1 = box
    s.draw.rectangle(box, fill=s.theme["bg"])
    side = s.px(int(s.rng.integers(260, 340)))
    s.draw.rectangle([x0, y0, x0 + side, y1], fill=s.theme["panel"])
    y = y0 + s.px(10)
    while y < y1 - s.px(64):
        avatar = tuple(int(c) for c in s.rng.integers(60, 220, 3))
        d = s.px(40)
        s.draw.ellipse([x0 + s.px(12), y + s.px(8), x0 + s.px(12) + d, y + s.px(8) + d], fill=avatar)
        initials = "".join(w[0] for w in s.phrase(2, 2).split()[:2]).upper()
        s.text((x0 + s.px(12) + d * 0.25, y + s.px(18)), initials, size=13, bold=True, fill=(255, 255, 255))
        s.text((x0 + s.px(62), y + s.px(10)), s.phrase(2, 3), size=13, bold=True, clip=x0 + side - s.px(50))
        s.text((x0 + s.px(62), y + s.px(30)), s.phrase(4, 9), size=12, fill=s.theme["muted"], clip=x0 + side - s.px(10))
        s.text((x0 + side - s.px(44), y + s.px(10)), f"{int(s.rng.integers(0, 24)):02d}:{int(s.rng.integers(0, 60)):02d}",
               size=11, fill=s.theme["muted"])
        y += s.px(64)
    y = y0 + s.px(20)
    size = int(s.rng.integers(13, 16))
    while y < y1 - s.px(120):
        mine = s.rng.random() < 0.45
        width = s.px(int(s.rng.integers(200, 520)))
        left = x1 - width - s.px(30) if mine else x0 + side + s.px(30)
        lines = int(s.rng.integers(1, 5))
        height = s.px(size) * 1.45 * lines + s.px(20)
        s.draw.rounded_rectangle([left, y, left + width, y + height], radius=s.px(12),
                                 fill=s.accent if mine else s.theme["panel"])
        s.paragraph((left + s.px(12), y + s.px(10), left + width - s.px(12), y + height - s.px(6)), "sans", size,
                    fill=(255, 255, 255) if mine else None, link_share=0)
        y += height + s.px(14)
    s.draw.rounded_rectangle([x0 + side + s.px(20), y1 - s.px(56), x1 - s.px(20), y1 - s.px(16)], radius=s.px(8),
                             fill=s.theme["bg"], outline=s.theme["line"])
    s.text((x0 + side + s.px(36), y1 - s.px(46)), s.phrase(2, 4), size=13, fill=s.theme["muted"])


def explorer(s, box):
    """A file explorer: address bar, navigation tree, a list of files with one selected."""
    x0, y0, x1, y1 = box
    s.draw.rectangle(box, fill=s.theme["bg"])
    bar = s.px(44)
    s.draw.rectangle([x0, y0, x1, y0 + bar], fill=s.theme["panel"])
    for i in range(4):
        s.icon((x0 + s.px(12 + 34 * i), y0 + s.px(12)), 16)
    s.draw.rounded_rectangle([x0 + s.px(150), y0 + s.px(8), x1 - s.px(260), y0 + bar - s.px(8)], radius=s.px(4),
                             fill=s.theme["bg"], outline=s.theme["line"])
    path = " > ".join(["This PC", "Documents", *[w.capitalize() for w in s.phrase(2, 3).split()]])
    s.text((x0 + s.px(162), y0 + s.px(14)), path, size=12, clip=x1 - s.px(270))
    nav = s.px(220)
    y = y0 + bar + s.px(10)
    while y < y1 - s.px(26):
        s.icon((x0 + s.px(14), y + s.px(2)), 16, s.accent if s.rng.random() < 0.4 else None)
        s.text((x0 + s.px(40), y), s.phrase(1, 2), size=12, clip=x0 + nav - s.px(6))
        y += s.px(28)
    s.draw.line([(x0 + nav, y0 + bar), (x0 + nav, y1)], fill=s.theme["line"])
    columns = [x0 + nav + s.px(16), x0 + nav + s.px(420), x0 + nav + s.px(600), x0 + nav + s.px(760)]
    y = y0 + bar + s.px(8)
    for label, cx in zip(("Name", "Date modified", "Type", "Size"), columns):
        s.text((cx, y), label, size=12, fill=s.theme["muted"], clip=x1)
    y += s.px(30)
    selected = int(s.rng.integers(0, 12))
    for row in range(200):
        if y + s.px(28) > y1:
            break
        if row == selected:
            s.draw.rectangle([x0 + nav + 2, y - s.px(4), x1 - s.px(16), y + s.px(22)], fill=s.theme["hover"])
        s.icon((columns[0], y), 16, s.accent if s.rng.random() < 0.5 else None)
        name = s.corpus["names"][int(s.rng.integers(len(s.corpus["names"])))]
        s.text((columns[0] + s.px(26), y), name, size=12, clip=columns[1] - s.px(8))
        s.text((columns[1], y), f"2026. {int(s.rng.integers(1, 13)):02d}. {int(s.rng.integers(1, 29)):02d}. "
                                f"{int(s.rng.integers(0, 24)):02d}:{int(s.rng.integers(0, 60)):02d}", size=12,
               fill=s.theme["muted"], clip=columns[2] - s.px(8))
        s.text((columns[2], y), s.rng.choice(["Python File", "Text Document", "PNG File", "File folder"]), size=12,
               fill=s.theme["muted"], clip=columns[3] - s.px(8))
        s.text((columns[3], y), f"{int(s.rng.integers(1, 9999)):,} KB", size=12, fill=s.theme["muted"], clip=x1 - s.px(16))
        y += s.px(28)
    s.scrollbar((x0, y0 + bar, x1, y1))


def settings(s, box):
    """A settings page: navigation, then labelled rows with toggles, drop-downs and buttons."""
    x0, y0, x1, y1 = box
    s.draw.rectangle(box, fill=s.theme["panel"])
    nav = s.px(260)
    y = y0 + s.px(60)
    for index in range(30):
        if y > y1 - s.px(40):
            break
        if index == 2:
            s.draw.rounded_rectangle([x0 + s.px(8), y - s.px(6), x0 + nav - s.px(8), y + s.px(26)], radius=s.px(4),
                                     fill=s.theme["hover"])
        s.icon((x0 + s.px(18), y), 16)
        s.text((x0 + s.px(48), y), s.phrase(1, 2), size=13, clip=x0 + nav - s.px(10))
        y += s.px(40)
    left = x0 + nav + s.px(24)
    s.text((left, y0 + s.px(20)), s.phrase(1, 3), size=26, bold=True, clip=x1 - s.px(20))
    y = y0 + s.px(80)
    while y < y1 - s.px(80):
        s.draw.rounded_rectangle([left, y, x1 - s.px(24), y + s.px(68)], radius=s.px(6), fill=s.theme["bg"])
        s.icon((left + s.px(18), y + s.px(24)), 20)
        s.text((left + s.px(56), y + s.px(14)), s.phrase(2, 5), size=14, clip=x1 - s.px(260))
        s.text((left + s.px(56), y + s.px(36)), s.phrase(4, 10), size=12, fill=s.theme["muted"], clip=x1 - s.px(260))
        control = s.rng.choice(["toggle", "dropdown", "button", "check"])
        cx, cy = x1 - s.px(200), y + s.px(20)
        if control == "toggle":
            on = s.rng.random() < 0.5
            s.draw.rounded_rectangle([cx + s.px(100), cy, cx + s.px(140), cy + s.px(20)], radius=s.px(10),
                                     fill=s.accent if on else None, outline=s.accent if on else s.theme["muted"], width=s.px(1))
            knob = cx + s.px(124 if on else 104)
            s.draw.ellipse([knob, cy + s.px(4), knob + s.px(12), cy + s.px(16)], fill=(255, 255, 255) if on else s.theme["muted"])
        elif control == "dropdown":
            s.draw.rounded_rectangle([cx, cy - s.px(4), cx + s.px(170), cy + s.px(28)], radius=s.px(4),
                                     fill=s.theme["panel"], outline=s.theme["line"])
            s.text((cx + s.px(10), cy + s.px(2)), s.phrase(1, 2), size=13, clip=cx + s.px(140))
        elif control == "button":
            s.draw.rounded_rectangle([cx + s.px(40), cy - s.px(4), cx + s.px(170), cy + s.px(28)], radius=s.px(4),
                                     fill=s.accent)
            s.text((cx + s.px(56), cy + s.px(2)), s.phrase(1, 2), size=13, fill=(255, 255, 255), clip=cx + s.px(166))
        else:
            s.draw.rounded_rectangle([cx + s.px(150), cy, cx + s.px(170), cy + s.px(20)], radius=s.px(3),
                                     fill=s.accent, outline=s.accent)
            s.draw.line([(cx + s.px(154), cy + s.px(10)), (cx + s.px(159), cy + s.px(15)), (cx + s.px(166), cy + s.px(5))],
                        fill=(255, 255, 255), width=s.px(2))
        y += s.px(76)


APPS = {"document": document, "code": code, "terminal": terminal, "spreadsheet": spreadsheet,
        "chat": chat, "explorer": explorer, "settings": settings}
APP_WEIGHTS = {"document": 0.24, "code": 0.22, "terminal": 0.08, "spreadsheet": 0.14, "chat": 0.12,
               "explorer": 0.10, "settings": 0.10}


def pick_app(rng):
    return str(rng.choice(list(APP_WEIGHTS), p=list(APP_WEIGHTS.values())))


# ---------------------------------------------------------------- the two kinds of picture

def draw_screen(seed, fonts, icons, corpus, photos):
    """One picture and how it was drawn, from its own seed."""
    rng = np.random.default_rng(seed)
    if rng.random() < 0.6:
        # page: one application filling a tall canvas, to be scrolled through
        width = int(rng.choice([1920, 2560]))
        height = int(rng.integers(1800, 3000))
        screen = Screen(rng, (width, height), fonts, icons, corpus, photos)
        app = pick_app(rng)
        APPS[app](screen, (0, 0, width - 1, height - 1))
        screen.record.update(kind="page", apps=[app])
        return screen.image, screen.record

    # desktop: wallpaper, a taskbar, and two or three windows over each other
    width, height = [(1920, 1080), (2560, 1440)][int(rng.integers(2))]
    screen = Screen(rng, (width, height), fonts, icons, corpus, photos)
    screen.photo((0, 0, width, height))
    taskbar = screen.px(48)
    apps = []
    for index in range(int(rng.integers(1, 4))):
        app = pick_app(rng)
        if index == 0 and rng.random() < 0.5:   # often the front window is maximised
            box = (0, 0, width - 1, height - taskbar - 1)
        else:
            w = int(width * rng.uniform(0.45, 0.8))
            h = int((height - taskbar) * rng.uniform(0.5, 0.9))
            x = int(rng.integers(0, width - w))
            y = int(rng.integers(0, height - taskbar - h))
            box = (x, y, x + w, y + h)
        screen.window(box, app)
        apps.append(app)
    screen.taskbar(taskbar)
    screen.record.update(kind="desktop", apps=apps)
    return screen.image, screen.record


# ---------------------------------------------------------------- writing a split

_WORKER = {}


def _start(fonts, icons, photos):
    _WORKER.update(fonts=fonts, icons=icons, photos=photos, corpus=load_corpus())


def _one(job):
    seed, path = job
    image, record = draw_screen(seed, _WORKER["fonts"], _WORKER["icons"], _WORKER["corpus"], _WORKER["photos"])
    image.save(path, optimize=False, compress_level=6)
    return {"file": path.relative_to(path.parents[1]).as_posix(), "size": list(image.size), **record}


def photos_for(split, root):
    """The DIV2K photos a split's screens may show: train ones for train, valid for test."""
    listing = root / "div2k" / f"{split}.txt"
    if not listing.exists():
        return []
    return [str(root / "div2k" / line) for line in listing.read_text(encoding="utf-8").splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", type=int, default=300, help="screens for training")
    parser.add_argument("--test", type=int, default=60, help="screens for validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dest", type=Path, default=ROOT / "raw", help="the raw datasets folder")
    parser.add_argument("--fonts", action="append", default=[], help="another folder to look for fonts in")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    fonts, icons = find_fonts(args.fonts)
    print("fonts:", {kind: [Path(regular).stem for regular, _ in found] for kind, found in fonts.items()},
          "icons:", Path(icons).stem if icons else None)
    if not any(fonts.values()):
        print("no fonts found - drawing with Pillow's own font")
        fonts = {"sans": [(None, None)], "serif": [(None, None)], "mono": [(None, None)]}

    folder = args.dest / "screens"
    records = []
    for split, count in (("train", args.train), ("test", args.test)):
        out = folder / split
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("*.png"):
            old.unlink()
        photos = photos_for(split, args.dest)
        # a different seed range per split, so no screen is drawn twice
        jobs = [(args.seed * 1_000_000 + (0 if split == "train" else 500_000) + i, out / f"{i:05d}.png") for i in range(count)]
        with ProcessPoolExecutor(args.workers, initializer=_start, initargs=(fonts, icons, photos)) as pool:
            done = list(pool.map(_one, jobs, chunksize=4))
        (folder / f"{split}.txt").write_text("".join(f"{r['file']}\n" for r in done), encoding="utf-8")
        records += [{**r, "split": split} for r in done]
        print(f"{split}: {len(done)} screens, {len(photos)} photos to draw from")
    (folder / "screens.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"written to {folder}")


if __name__ == "__main__":
    main()
