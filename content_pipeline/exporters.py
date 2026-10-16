"""Minimal scheduler CSV export plus machine-readable final-content metadata."""

from __future__ import annotations

import csv
from pathlib import Path

from .models import FinalPiece


def export_csv(finals: list[FinalPiece], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for final in finals:
        image = final.image or ""
        for index, tweet in enumerate(final.draft.tweets):
            rows.append({"Post Text": tweet.text, "Image": image if index == 0 else ""})
    try:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Post Text", "Image"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write {output}. Close the file in another program or choose a new --output directory."
        ) from exc
    return len(rows)

