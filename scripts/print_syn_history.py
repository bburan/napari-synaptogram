"""Print the modification history of a .syn file.

Usage
-----
    python print_syn_history.py path/to/file.syn
"""

import argparse
import json
import sys

import tifffile


def print_history(path: str) -> int:
    with tifffile.TiffFile(path) as fh:
        metadata = json.loads(fh.pages[0].description)

    history = metadata.get("history", [])
    if not history:
        print(f"No history recorded in {path}")
        return 0

    print(
        f"History for {path} ({len(history)} entr{'y' if len(history) == 1 else 'ies'}):"
    )
    for i, entry in enumerate(history, start=1):
        user = entry.get("user", "?")
        host = entry.get("host", "?")
        modified = entry.get("modified", "?")
        print(f"  {i}. {modified}  {user}@{host}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a .syn file")
    args = parser.parse_args()
    return print_history(args.path)


if __name__ == "__main__":
    sys.exit(main())
