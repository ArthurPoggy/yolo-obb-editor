"""Core data module for YOLO OBB Editor (web version).

Adapted from yolo_obb_editor_android/main.py - uses PIL instead of Kivy.
"""

from pathlib import Path
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EditorData:
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.images_dir, self.labels_dir = self._resolve_dirs()
        self.image_paths: list[Path] = []
        self.idx = 0

        self.pil_image: Image.Image | None = None
        self.w = 0
        self.h = 0

        self.boxes: list[list] = []  # [[class_id, [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]], ...]
        self.original_txt = ""
        self.changed = False

    def _resolve_dirs(self):
        imgs = self.dataset_path / "images"
        lbls = self.dataset_path / "labels"
        if not imgs.exists():
            imgs = self.dataset_path
        if not lbls.exists():
            lbls = self.dataset_path
        return imgs, lbls

    def load_images(self):
        paths = []
        if self.images_dir.exists():
            for f in self.images_dir.iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    paths.append(f)
            for sub in ["train", "val", "test"]:
                d = self.images_dir / sub
                if d.exists():
                    for f in d.iterdir():
                        if f.suffix.lower() in IMG_EXTS:
                            paths.append(f)

        self.image_paths = sorted(set(paths), key=lambda p: str(p).lower())

    def find_label(self, img_path: Path) -> Path | None:
        stem = img_path.stem + ".txt"
        for p in [
            self.labels_dir / stem,
            self.labels_dir / "train" / stem,
            self.labels_dir / "val" / stem,
            self.labels_dir / "test" / stem,
        ]:
            if p.exists():
                return p
        return None

    def load(self):
        if not self.image_paths:
            self.pil_image = None
            self.w = 0
            self.h = 0
            self.boxes = []
            self.original_txt = ""
            self.changed = False
            return

        path = self.image_paths[self.idx]
        try:
            self.pil_image = Image.open(path)
            self.w, self.h = self.pil_image.size
        except Exception:
            self.pil_image = None
            self.w = 0
            self.h = 0
            self.boxes = []
            self.original_txt = ""
            self.changed = False
            return

        self.boxes = []
        self.original_txt = ""

        lp = self.find_label(path)
        if lp:
            txt = lp.read_text(encoding="utf-8")
            self.original_txt = txt
            self._parse_boxes(txt)

        self.changed = False

    def _parse_boxes(self, txt: str):
        self.boxes = []
        for line in txt.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 9:
                try:
                    cls = int(float(parts[0]))
                    pts = []
                    for i in range(4):
                        x = float(parts[1 + i * 2]) * self.w
                        y = float(parts[2 + i * 2]) * self.h
                        pts.append([int(x), int(y)])
                    self.boxes.append([cls, pts])
                except Exception:
                    pass

    def save(self):
        if not self.image_paths or self.pil_image is None:
            return

        path = self.labels_dir / (self.image_paths[self.idx].stem + ".txt")
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for cls, pts in self.boxes:
            coords = []
            for x, y in pts:
                coords.append(f"{x / self.w:.6f}")
                coords.append(f"{y / self.h:.6f}")
            lines.append(f"{cls} " + " ".join(coords))

        path.write_text("\n".join(lines), encoding="utf-8")
        self.original_txt = "\n".join(lines)
        self.changed = False

    def undo(self):
        self._parse_boxes(self.original_txt)
        self.changed = False
