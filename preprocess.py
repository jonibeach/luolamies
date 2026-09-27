from collections.abc import Callable
from mwparserfromhell.nodes import Node
from mwparserfromhell.wikicode import Wikicode
import bz2, re
from concurrent.futures import ProcessPoolExecutor
from itertools import batched
from pathlib import Path

import mwparserfromhell as mw
import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

REMOVE_TAGS = ("ref", "table", "noinclude")
UNWRAP_TEMPLATES = re.compile(r"^(k-\w+|lang(-\w+)?|ipa)$", re.IGNORECASE)
REMOVE_SECTIONS = {
    "lähteet",
    "viitteet",
    "aiheesta muualla",
    "katso myös",
    "kirjallisuutta",
}
LINKS = re.compile(r"^(Luokka|Tiedosto|Kuva|File|Image|Category):", re.IGNORECASE)
BLANK_LINES = re.compile(r"\n{3,}")
SPACES = re.compile(r"[^\S\n]+")
EMPTY_BRACKETS = re.compile(r"\(\s*\)")


def pages(input):
    for _, page in etree.iterparse(input, tag="{*}page"):
        is_article = page.findtext("{*}ns") == "0"
        is_redirect = page.find("{*}redirect") is not None
        if not is_article or is_redirect:
            continue

        id = page.findtext("{*}id")
        title = page.findtext("{*}title")
        text = page.findtext("{*}revision/{*}text")

        yield {"id": id, "title": title, "text": text}

        page.clear()
        while page.getprevious() is not None:
            del page.getparent()[0]


def tr(f: Callable):
    try:
        f()
    except ValueError:
        pass


def try_remove(code: Wikicode, node: Node | Wikicode):
    tr(lambda: code.remove(node))


def format(page):
    code = mw.parse(page["text"])

    for section in code.get_sections(include_lead=False, include_headings=True):
        headings = section.filter_headings()
        if headings and any(
            h.title.strip_code().strip().lower() in REMOVE_SECTIONS for h in headings
        ):
            try_remove(code, section)

    for t in code.filter_templates(recursive=True):
        if not UNWRAP_TEMPLATES.match(str(t.name).strip()):
            continue
        positional = [p.value for p in t.params if not p.showkey]
        if positional:
            tr(lambda: code.replace(t, positional[-1]))

    for n in code.filter(recursive=True):
        is_tag = isinstance(n, mw.nodes.Tag) and n.tag.lower() in REMOVE_TAGS
        is_link = isinstance(n, mw.nodes.Wikilink) and LINKS.match(str(n.title))

        if is_tag or is_link:
            try_remove(code, n)

    text = code.strip_code(normalize=True, collapse=True)
    text = SPACES.sub(" ", text)
    text = EMPTY_BRACKETS.sub("", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    page["text"] = BLANK_LINES.sub("\n\n", text).strip()

    return page


SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("text", pa.string()),
    ]
)

FIFTYK = int(5e4)


def preprocess():
    outfile = Path("./data/wiki.parquet")
    if outfile.exists():
        return outfile

    with (
        bz2.open("./data/wiki.bz2", "rb") as input,
        ProcessPoolExecutor(max_workers=8) as e,
        pq.ParquetWriter(outfile, SCHEMA, compression="zstd") as writer,
    ):
        formatted = batched(e.map(format, pages(input), buffersize=FIFTYK), FIFTYK)

        for b in formatted:
            table = pa.Table.from_pylist(b, schema=SCHEMA)
            writer.write(table, row_group_size=FIFTYK / 10)

    return outfile
