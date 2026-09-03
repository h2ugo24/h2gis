"""
Regenerate the vendored snapshot of CF standard names used by config.yaml.

The full CF standard name table is ~4.5 MB of XML and changes independently of
this repo, so the test suite checks against a snapshot of just the names
``config.yaml`` actually uses, together with their canonical units. A name that
is not in the snapshot fails the compliance test, which is the intended prompt:
verify it against the published table, then run this to record it.

Needs the network. Run it after adding or changing a ``standard_name``:

    uv run --with cf-units python scripts/refresh_cf_standard_names.py

Writes ``tests/fixtures/cf_standard_names.json``. Not ``tests/data/`` — the
repo's .gitignore excludes every ``data/``, so a snapshot there would never be
committed and CI would fail on a file it cannot see.
"""

from __future__ import annotations

import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

TABLE_URL = (
    "https://cfconventions.org/Data/cf-standard-names/current/src/"
    "cf-standard-name-table.xml"
)
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "fixtures" / "cf_standard_names.json"

#: Names the pipeline puts on coordinates rather than data variables, so they
#: never appear in config and would otherwise drop out of the snapshot.
COORD_NAMES = ("longitude", "latitude", "depth", "time")


def main() -> None:
    print(f"fetching {TABLE_URL}")
    with urllib.request.urlopen(TABLE_URL) as response:
        root = ET.fromstring(response.read())

    version = root.findtext("version_number") or "unknown"
    canonical = {
        entry.get("id"): (entry.findtext("canonical_units") or "").strip()
        for entry in root.iter("entry")
    }
    aliases = {entry.get("id") for entry in root.iter("alias")}
    print(f"table version {version}: {len(canonical)} entries, {len(aliases)} aliases")

    attrs = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    used = {
        str(info["standard_name"]).split(" ")[0]
        for info in attrs["variable_attrs"].values()
        if info.get("standard_name")
    }
    used.update(COORD_NAMES)

    missing = sorted(n for n in used if n not in canonical)
    deprecated = sorted(n for n in used if n in aliases)
    if missing:
        raise SystemExit(f"not in the CF table: {missing}")
    if deprecated:
        raise SystemExit(f"deprecated aliases, use the current name: {deprecated}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "_source": TABLE_URL,
                "_table_version": version,
                "_regenerate_with": "scripts/refresh_cf_standard_names.py",
                "canonical_units": {name: canonical[name] for name in sorted(used)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(used)} names to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
