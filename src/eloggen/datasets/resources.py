#!/usr/bin/env python3
"""Install or verify resources described by the ElogGen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence
from urllib.parse import quote
from urllib.request import urlopen

from .paths import MANIFEST_PATH

PROJECT_ROOT = Path(
    os.environ.get("ELOGGEN_ROOT", Path(__file__).resolve().parents[3])
).expanduser().resolve()
DEFAULT_MANIFEST = MANIFEST_PATH
BASE_URL_ENV = {
    "task-data": "ELOGGEN_TASK_DATA_BASE_URL",
    "openarm-assets": "ELOGGEN_OPENARM_ASSET_BASE_URL",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, item: dict[str, object]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(item["size"])
        and sha256(path) == item["sha256"]
    )


def source_path(source_root: Path, item: dict[str, object]) -> Path:
    return source_root / str(item["root"]) / str(item["path"])


def resource_url(item: dict[str, object], overrides: dict[str, str]) -> str | None:
    base = overrides.get(str(item["group"])) or os.environ.get(
        BASE_URL_ENV[str(item["group"])], ""
    )
    if base:
        encoded = "/".join(quote(part) for part in str(item["path"]).split("/"))
        return f"{base.rstrip('/')}/{encoded}"
    direct = item.get("url")
    return str(direct) if direct else None


def install_one(
    item: dict[str, object],
    destination: Path,
    source_root: Path | None,
    url_overrides: dict[str, str],
    check_only: bool,
) -> bool:
    if verified(destination, item):
        print(f"[OK] {item['id']}: {destination}")
        return True
    if destination.exists():
        print(f"[FAIL] checksum or size mismatch: {destination}", file=sys.stderr)
        return False
    if check_only:
        print(f"[MISSING] {item['id']}: {destination}", file=sys.stderr)
        return False

    local_source = source_path(source_root, item) if source_root else None
    url = resource_url(item, url_overrides)
    if local_source and local_source.is_file():
        method = f"copy {local_source}"
    elif url:
        method = f"download {url}"
    else:
        env_name = BASE_URL_ENV[str(item["group"])]
        print(
            f"[MISSING] {item['id']}: no source configured; set {env_name} "
            "or use --source-root",
            file=sys.stderr,
        )
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        print(f"[FETCH] {item['id']}: {method}")
        if local_source and local_source.is_file():
            shutil.copyfile(local_source, temporary)
        else:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if not verified(temporary, item):
            raise RuntimeError(f"downloaded resource failed verification: {item['id']}")
        temporary.replace(destination)
        print(f"[INSTALLED] {destination}")
        return True
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--group", choices=("all", "task-data", "openarm-assets"), default="all"
    )
    parser.add_argument("--task-data-base-url")
    parser.add_argument("--openarm-assets-base-url")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    dataset_root_value = args.dataset_root or os.environ.get("ELOGGEN_DATASET_ROOT")
    dataset_root = Path(dataset_root_value).expanduser().resolve() if dataset_root_value else None
    roots = {"project": args.project_root.expanduser().resolve(), "dataset": dataset_root}
    source_root = args.source_root.expanduser().resolve() if args.source_root else None
    url_overrides = {
        "task-data": args.task_data_base_url or "",
        "openarm-assets": args.openarm_assets_base_url or "",
    }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    resources = manifest["resources"]
    selected = [item for item in resources if args.group == "all" or item["group"] == args.group]

    ok = True
    for item in selected:
        root = roots.get(str(item["root"]))
        if root is None:
            print(
                f"[FAIL] {item['id']} needs --dataset-root or ELOGGEN_DATASET_ROOT",
                file=sys.stderr,
            )
            ok = False
            continue
        ok = install_one(
            item,
            root / str(item["path"]),
            source_root,
            url_overrides,
            args.check,
        ) and ok
    if ok:
        print(f"ElogGen resources ready: {len(selected)} verified")
        return 0
    print("Resource preparation incomplete.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
