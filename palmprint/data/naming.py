"""Filename-based class-number extraction helpers."""

from __future__ import annotations


def extract_class_no_from_name(img_name: str, dataset_type: str) -> int:
    """Extract the class number from one filename."""

    parts = img_name.split("_")

    if dataset_type == "PolyU":
        return int(parts[2])
    if dataset_type == "IITD":
        return int(parts[0])
    if dataset_type == "Tongji":
        return int(parts[0])
    if dataset_type == "PolyU_CF":
        return int(parts[0])
    if dataset_type == "REST":
        class_no = int(parts[0][1:])
        return -class_no if parts[1] == "l" else class_no
    if dataset_type == "Zhou_1295":
        return int(parts[0])
    if dataset_type == "MPDv2":
        return int(parts[0])

    raise ValueError(
        "Unsupported datasetType "
        f"{dataset_type!r}. Supported values: PolyU, IITD, Tongji, PolyU_CF, REST, Zhou_1295, MPDv2."
    )
