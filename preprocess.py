import bz2
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass


import mwparserfromhell as mw
from lxml import etree

REMOVE_TAGS = ("ref", "table", "noinclude")


@dataclass
class Page:
    id: str
    title: str
    text: str


def pages(input):
    for _, page in etree.iterparse(input, tag="{*}page"):
        is_article = page.findtext("{*}ns") == "0"
        is_redirect = page.find("{*}redirect") is not None
        if not is_article or is_redirect:
            continue

        id = page.findtext("{*}id")
        title = page.findtext("{*}title")
        text = page.findtext("{*}revision/{*}text")

        yield Page(id, title, text)

        page.clear()
        while page.getprevious() is not None:
            del page.getparent()[0]


def format(page: Page):
    code = mw.parse(page.text)
    for tag in code.filter_tags(recursive=True):
        if tag.tag.lower() in REMOVE_TAGS:
            try:
                code.remove(tag)
            except ValueError:
                pass
    text = code.strip_code(normalize=True, collapse=True)
    page.text = text
    return page


def preprocess():
    outfile = Path("./data/wiki.jsonl")
    if outfile.exists():
        return

    with (
        bz2.open("./data/wiki.bz2", "rb") as input,
        open(outfile, "w") as out,
        ProcessPoolExecutor(max_workers=8) as e,
    ):
        formatted = e.map(format, pages(input), buffersize=int(5e4))
        out.writelines(json.dumps(asdict(page)) + "\n" for page in formatted)
