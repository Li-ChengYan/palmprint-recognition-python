"""PolyU subset parsing and deterministic selection."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SelectionRecord:
    """One selected image record for a benchmark run."""

    path: Path
    subset: str
    class_no: int
    sample_no: int
    name: str

_POLYU_PATTERN = re.compile(r"^P_([A-Z])_(\d+)_(\d+)\.bmp$")


def _normalize_subset_codes(subset_codes: str | Iterable[str]) -> list[str]:
    if isinstance(subset_codes, str):
        stripped = subset_codes.strip().upper()
        if len(stripped) <= 1:
            return [stripped] if stripped else []
        return list(stripped)

    normalized = [str(code).strip().upper() for code in subset_codes]
    return [code for code in normalized if code]


def select_polyu_subset(
    img_path: str | Path,
    subset_codes: str | Iterable[str],
    num_classes: int,
    images_per_class: int,
) -> list[SelectionRecord]:
    """Deterministically select a balanced PolyU subset."""

    normalized_codes = _normalize_subset_codes(subset_codes)
    if not normalized_codes:
        raise ValueError("subset_codes must contain at least one subset code")

    root = Path(img_path)
    records: list[SelectionRecord] = []

    for subset_code in normalized_codes:
        for file_path in root.glob(f"P_{subset_code}_*.bmp"):
            match = _POLYU_PATTERN.match(file_path.name)
            if not match:
                continue

            records.append(
                SelectionRecord(
                    path=file_path,
                    subset=match.group(1),
                    class_no=int(match.group(2)),
                    sample_no=int(match.group(3)),
                    name=file_path.name,
                )
            )

    if not records:
        raise ValueError(f"No valid PolyU filenames were found under {root} for subset codes {normalized_codes}.")

    records.sort(key=lambda record: (record.class_no, record.subset, record.sample_no, record.name))

    selected: list[SelectionRecord] = []
    class_ids = sorted({record.class_no for record in records})
    chosen_classes = 0

    for class_id in class_ids:
        class_records = [record for record in records if record.class_no == class_id]
        if len(class_records) < images_per_class:
            continue

        selected.extend(class_records[:images_per_class])
        chosen_classes += 1
        if chosen_classes == num_classes:
            break

    if chosen_classes < num_classes:
        raise ValueError(
            f"Only {chosen_classes} classes have at least {images_per_class} images under {root}."
        )

    return selected
