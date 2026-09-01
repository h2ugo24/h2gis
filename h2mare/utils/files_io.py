"""
Input/Output Help functions
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import time
from pathlib import Path

from loguru import logger

# ========================== IO ==========================================


def _force_remove(func, path, exc_info):
    """
    Error handler for shutil.rmtree. Tries to make the file writable and retries.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError as e:
        logger.debug(f"_force_remove: could not remove {path}: {e}")


def prune_empty_dirs(root: Path) -> int:
    """
    Remove empty directories beneath *root*, deepest first, so a chain of
    nested empty folders (e.g. ``eddies/nrt`` or ``CMEMS_2nd_productivity/mnkc``)
    collapses in one pass. *root* itself is kept; directories containing any
    file are untouched.

    Returns:
        Number of directories removed.
    """
    if not root.exists():
        return 0
    removed = 0
    subdirs = sorted(
        (p for p in root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for d in subdirs:
        try:
            d.rmdir()  # succeeds only when empty
            removed += 1
        except OSError:
            continue
    return removed


def safe_rmtree(path: Path, retries=10, delay=0.5) -> None:
    """
    Remove a directory tree with retries (prevent Windows file locks).

    Args:
        path: Directory to remove
        retries: Number of retries. Defaults to 10.
        delay: Delay between retries. Defaults to 0.5s.
    """
    last_err = None

    for i in range(retries):
        try:
            if not path.exists():
                return

            shutil.rmtree(path, onerror=_force_remove)
            return

        except (PermissionError, OSError) as e:
            last_err = e
            time.sleep(delay * (i + 1))

    raise RuntimeError(
        f"Failed to remove {path} after {retries} attempts"
    ) from last_err


def filter_raw_files(paths: list[Path], var_config) -> list[Path]:
    """
    Keep only the raw files a variable's ``raw_include`` regex admits.

    A download directory can hold files the pipeline must not read. AVISO ships
    META3.2 eddy trajectories as long/short/untracked variants side by side, and
    only the long ones belong in the store — the untracked files do not even
    carry a ``track`` variable, so converting one fails deep inside the
    processor rather than being skipped.

    Returns *paths* unchanged when the variable sets no ``raw_include``.
    """
    pattern = getattr(var_config, "raw_include", None)
    if not pattern:
        return paths

    regex = re.compile(pattern)
    kept = [p for p in paths if regex.search(p.name)]
    dropped = len(paths) - len(kept)
    if dropped:
        logger.debug(
            f"raw_include={pattern!r} excluded {dropped} raw file(s), kept {len(kept)}"
        )
    return kept


def safe_move_files(
    paths: Path | list[Path], dest_dir: Path, retries=10, delay=0.5
) -> None:
    """
    Move a list of files paths with retries.

    Args:
        paths: File path or List of files paths to move.
        dest_dir: Directory to move file.
        retries: Number of retries. Defaults to 10.
        delay: Delay between retries. Defaults to 0.5s.
    """
    paths = [paths] if isinstance(paths, Path) else paths
    for path in paths:
        dest_path = dest_dir / path.name

        # A file already at its destination must be left alone. The retry loop
        # below unlinks dest_path before moving, so without this a same-path
        # move deletes the source outright rather than failing harmlessly.
        if path.resolve() == dest_path.resolve():
            logger.debug(f"Already at destination, not moving: {path}")
            continue

        last_err = None

        for i in range(retries):
            try:
                # Avoid errors if file aready exits in dest_dir
                if dest_path.exists():
                    dest_path.unlink()

                logger.debug(
                    f"Moving {path} -> {dest_path} (exists={dest_path.exists()})"
                )
                shutil.move(path, dest_path)
                break

            except (PermissionError, OSError) as e:
                last_err = e
                time.sleep(delay * (i + 1))
        else:
            raise RuntimeError(
                f"Failed to move {path} after {retries} attempts"
            ) from last_err
