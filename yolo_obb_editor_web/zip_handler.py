"""Upload and download ZIP handling for YOLO OBB Editor web."""

import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path


def extract_dataset_zip(uploaded_file, old_temp_dir: str | None = None) -> tuple[Path, str]:
    """Extract uploaded ZIP to a temp directory.

    If the ZIP has a single root folder, uses that as the dataset root.
    Cleans up old temp dir if provided.

    Returns (dataset_path, temp_root_dir) where temp_root_dir is the
    mkdtemp path to use for cleanup later.
    """
    if old_temp_dir:
        try:
            shutil.rmtree(old_temp_dir, ignore_errors=True)
        except Exception:
            pass

    temp_dir = tempfile.mkdtemp(prefix="yolo_obb_")

    with zipfile.ZipFile(BytesIO(uploaded_file.read()), "r") as zf:
        zf.extractall(temp_dir)

    temp_path = Path(temp_dir)

    # Detect single root folder (e.g., zip contains "dataset/images/..." )
    entries = [e for e in temp_path.iterdir() if not e.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0], temp_dir

    return temp_path, temp_dir


def create_labels_zip(labels_dir: Path) -> bytes:
    """Create a ZIP with all .txt label files from labels_dir (recursive)."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        label_root = labels_dir
        for txt_file in sorted(labels_dir.rglob("*.txt")):
            arcname = txt_file.relative_to(label_root)
            zf.write(txt_file, arcname)
    return buf.getvalue()
