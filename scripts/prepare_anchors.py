"""Reconstruct the audit's 3,000 FineWeb-Edu text prefixes and check their fingerprint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

DATASET = "HuggingFaceFW/fineweb-edu"
OFFSETS = (215700, 332100, 1200800, 1431100, 1518200, 1561900, 2305500, 2480800,
           2693000, 2719900, 2761700, 3033800, 3191200, 3274300, 3274700, 3876400,
           3991300, 4288900, 4472900, 4641400, 4934200, 5013500, 5410800, 5648400,
           5706400, 5733400, 6216400, 6622800, 6811300, 6921900)
EXPECTED = "e2ea65d24052d7a884683920132bb93447ad0e2b78070edae8fd794c29bf9bce"
DEFAULT = Path(__file__).resolve().parents[1] / "raw/anchors.jsonl"


def verify(path):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED:
        raise ValueError(f"Expected {EXPECTED}, found {actual} in {path}. Check the dataset revision and rows.")


def write_window(handle, offset, rows):
    if [r["row_idx"] for r in rows] != list(range(offset, offset + 100)):
        raise ValueError(f"Incomplete or reordered sample at offset {offset}")
    for record in rows:
        row = record["row"]
        selected = dict(id=row["id"], offset=offset, row_idx=record["row_idx"], text=row["text"][:600])
        handle.write(json.dumps(selected, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only or args.output.exists():
        verify(args.output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as output:
                for offset in OFFSETS:
                    query = urlencode(dict(dataset=DATASET, config="sample-10BT", split="train",
                                           offset=offset, length=100))
                    with urlopen("https://datasets-server.huggingface.co/rows?" + query, timeout=90) as response:
                        write_window(output, offset, json.load(response)["rows"])
            verify(temporary)
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"{EXPECTED}  {args.output}")


if __name__ == "__main__":
    main()
