#!/usr/bin/env python3
import math
import os
import json
from pathlib import Path

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.graphics import Color, Line, Ellipse, Rectangle
from kivy.core.image import Image as CoreImage
from kivy.metrics import dp
from kivy.utils import platform

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COL_BOX = (0.0, 0.86, 0.0, 1.0)
COL_BOX_SEL = (1.0, 1.0, 0.0, 1.0)
COL_VERTEX = (0.0, 0.0, 0.86, 1.0)
COL_PREVIEW = (1.0, 0.55, 0.0, 1.0)


def dist_point_to_segment(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq == 0:
        return math.hypot(apx, apy)
    t = (apx * abx + apy * aby) / ab_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return math.hypot(px - proj_x, py - proj_y)


def point_in_poly(p, pts):
    x, y = p
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            denom = (y2 - y1)
            if denom == 0:
                continue
            x_int = (x2 - x1) * (y - y1) / denom + x1
            if x < x_int:
                inside = not inside
    return inside


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


class EditorData:
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.images_dir, self.labels_dir = self._resolve_dirs()
        self.image_paths = []
        self.idx = 0
        self.progress_file = self.labels_dir / ".progress"

        self.texture = None
        self.w = 0
        self.h = 0

        self.boxes = []
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

    def set_dataset_path(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.images_dir, self.labels_dir = self._resolve_dirs()
        self.image_paths = []
        self.idx = 0
        self.progress_file = self.labels_dir / ".progress"
        self.texture = None
        self.w = 0
        self.h = 0
        self.boxes = []
        self.original_txt = ""
        self.changed = False
        self.load_images()
        if self.image_paths:
            self.load()

    def save_progress(self):
        try:
            if self.image_paths:
                img_name = self.image_paths[self.idx].name
                self.progress_file.parent.mkdir(parents=True, exist_ok=True)
                self.progress_file.write_text(f"{self.idx}\n{img_name}", encoding="utf-8")
        except Exception:
            pass

    def load_progress(self):
        if not self.progress_file.exists():
            return 0
        try:
            content = self.progress_file.read_text(encoding="utf-8").strip().split("\n")
            saved_idx = int(content[0])
            saved_name = content[1] if len(content) > 1 else ""
            if 0 <= saved_idx < len(self.image_paths):
                if self.image_paths[saved_idx].name == saved_name:
                    return saved_idx
            for i, p in enumerate(self.image_paths):
                if p.name == saved_name:
                    return i
            return saved_idx if 0 <= saved_idx < len(self.image_paths) else 0
        except Exception:
            return 0

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
        if self.image_paths:
            self.idx = self.load_progress()

    def find_label(self, img_path: Path):
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
            self.texture = None
            self.w = 0
            self.h = 0
            self.boxes = []
            self.original_txt = ""
            self.changed = False
            return

        path = self.image_paths[self.idx]
        try:
            core = CoreImage(str(path))
        except Exception:
            self.texture = None
            self.w = 0
            self.h = 0
            self.boxes = []
            self.original_txt = ""
            self.changed = False
            return

        self.texture = core.texture
        self.w, self.h = self.texture.size
        self.boxes = []
        self.original_txt = ""

        lp = self.find_label(path)
        if lp:
            txt = lp.read_text(encoding="utf-8")
            self.original_txt = txt
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

        self.changed = False

    def save(self):
        if not self.image_paths or self.texture is None:
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
        self.save_progress()

    def undo(self):
        self.boxes = []
        for line in self.original_txt.strip().split("\n"):
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
        self.changed = False


class OBBEditorWidget(Widget):
    def __init__(self, data: EditorData, state_cb=None, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.state_cb = state_cb or (lambda: None)

        self.sel_box = None
        self.dragging = False
        self.drag_type = None
        self.drag_start = None
        self.drag_box_start = None

        self.creating = False
        self.create_p1 = None
        self.create_p2 = None

        self.active_touch = None
        self.active_vert = None
        self.active_edge = None

        self.touch_radius = dp(24)
        self.edge_radius = dp(16)
        self.vertex_radius = dp(6)

        self.view_scale = 1.0
        self.view_offset = [0.0, 0.0]
        self.min_scale = 1.0
        self.max_scale = 6.0

        self.drag_start_disp = None
        self._pan_start_offset = None

        self._touches = {}
        self._gesture_active = False
        self._gesture_start_dist = 0.0
        self._gesture_start_scale = 1.0
        self._gesture_start_mid = (0.0, 0.0)
        self._gesture_start_img = None

        self._img_pos = (0.0, 0.0)
        self._img_size = (1.0, 1.0)
        self._scale = 1.0

        self.bind(pos=self._redraw, size=self._redraw)

    def reset_state(self):
        self.sel_box = None
        self.dragging = False
        self.drag_type = None
        self.drag_start = None
        self.drag_box_start = None
        self.creating = False
        self.create_p1 = None
        self.create_p2 = None
        self.active_touch = None
        self.active_vert = None
        self.active_edge = None
        self.drag_start_disp = None
        self._pan_start_offset = None
        self._touches = {}
        self._gesture_active = False
        self._gesture_start_dist = 0.0
        self._gesture_start_scale = self.view_scale
        self._gesture_start_mid = (0.0, 0.0)
        self._gesture_start_img = None
        self._redraw()
        self.state_cb()

    def reset_view(self):
        self.view_scale = 1.0
        self.view_offset = [0.0, 0.0]
        self._redraw()

    def zoom_by(self, factor, center=None):
        if self.data.texture is None:
            return
        if center is None:
            center = self.center
        img_pt = self._disp_to_img(*center)
        if img_pt is None:
            img_pt = (self.data.w / 2.0, self.data.h / 2.0)
        new_scale = clamp(self.view_scale * factor, self.min_scale, self.max_scale)
        self._set_view_for_focus(center, img_pt, new_scale)
        self._redraw()

    def _base_rect(self):
        if self.data.w <= 0 or self.data.h <= 0:
            x = self.x
            y = self.y
            w = max(1.0, self.width)
            h = max(1.0, self.height)
            return (x, y), (w, h), 1.0, (x + w / 2.0, y + h / 2.0)
        scale = min(self.width / self.data.w, self.height / self.data.h)
        disp_w = self.data.w * scale
        disp_h = self.data.h * scale
        x = self.x + (self.width - disp_w) / 2.0
        y = self.y + (self.height - disp_h) / 2.0
        center = (x + disp_w / 2.0, y + disp_h / 2.0)
        return (x, y), (disp_w, disp_h), scale, center

    def _clamp_view(self):
        _, (bw, bh), _, base_center = self._base_rect()
        disp_w = bw * self.view_scale
        disp_h = bh * self.view_scale
        cx = base_center[0] + self.view_offset[0]
        cy = base_center[1] + self.view_offset[1]

        if disp_w <= self.width:
            cx = self.x + self.width / 2.0
        else:
            min_cx = self.x + self.width - disp_w / 2.0
            max_cx = self.x + disp_w / 2.0
            cx = clamp(cx, min_cx, max_cx)

        if disp_h <= self.height:
            cy = self.y + self.height / 2.0
        else:
            min_cy = self.y + self.height - disp_h / 2.0
            max_cy = self.y + disp_h / 2.0
            cy = clamp(cy, min_cy, max_cy)

        self.view_offset = [cx - base_center[0], cy - base_center[1]]

    def _apply_view(self):
        _, (bw, bh), base_scale, base_center = self._base_rect()
        self._clamp_view()
        disp_w = bw * self.view_scale
        disp_h = bh * self.view_scale
        cx = base_center[0] + self.view_offset[0]
        cy = base_center[1] + self.view_offset[1]
        img_x = cx - disp_w / 2.0
        img_y = cy - disp_h / 2.0
        self._img_pos = (img_x, img_y)
        self._img_size = (disp_w, disp_h)
        self._scale = base_scale * self.view_scale if base_scale > 0 else 1.0

    def _set_view_for_focus(self, disp_pt, img_pt, new_scale):
        if self.data.w <= 0 or self.data.h <= 0:
            return
        _, (bw, bh), base_scale, base_center = self._base_rect()
        self.view_scale = clamp(new_scale, self.min_scale, self.max_scale)

        scale = base_scale * self.view_scale
        ix, iy = img_pt
        img_x = disp_pt[0] - ix * scale
        img_y = disp_pt[1] - (self.data.h - 1 - iy) * scale

        disp_w = bw * self.view_scale
        disp_h = bh * self.view_scale
        cx = img_x + disp_w / 2.0
        cy = img_y + disp_h / 2.0
        self.view_offset = [cx - base_center[0], cy - base_center[1]]
        self._clamp_view()

    def _start_gesture(self):
        if len(self._touches) < 2:
            return
        touches = list(self._touches.values())[:2]
        p1 = touches[0].pos
        p2 = touches[1].pos

        self._gesture_active = True
        self.dragging = False
        self.drag_type = None
        self.active_vert = None
        self.active_edge = None
        self.active_touch = None
        self.drag_start = None
        self.drag_box_start = None
        self.drag_start_disp = None
        self._pan_start_offset = None

        self._gesture_start_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        self._gesture_start_mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        self._gesture_start_scale = self.view_scale

        img_pt = self._disp_to_img(*self._gesture_start_mid)
        if img_pt is None:
            img_pt = (self.data.w / 2.0, self.data.h / 2.0)
        self._gesture_start_img = img_pt

    def _update_gesture(self):
        if len(self._touches) < 2:
            return
        touches = list(self._touches.values())[:2]
        p1 = touches[0].pos
        p2 = touches[1].pos
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if self._gesture_start_dist <= 0.0:
            return
        mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        new_scale = self._gesture_start_scale * (dist / self._gesture_start_dist)
        self._set_view_for_focus(mid, self._gesture_start_img, new_scale)

    def _img_to_disp(self, ix, iy):
        x0, y0 = self._img_pos
        dx = x0 + ix * self._scale
        dy = y0 + (self.data.h - 1 - iy) * self._scale
        return dx, dy

    def _disp_to_img(self, dx, dy):
        x0, y0 = self._img_pos
        w, h = self._img_size
        if dx < x0 or dx > x0 + w or dy < y0 or dy > y0 + h:
            return None
        if self._scale == 0:
            return None
        ix = (dx - x0) / self._scale
        iy = (self.data.h - 1) - (dy - y0) / self._scale
        return ix, iy

    def _clamp_pt(self, x, y):
        if self.data.w <= 0 or self.data.h <= 0:
            return 0, 0
        x = clamp(int(round(x)), 0, self.data.w - 1)
        y = clamp(int(round(y)), 0, self.data.h - 1)
        return x, y

    def _hit_test(self, ix, iy):
        if not self.data.boxes:
            return None
        vert_thresh = self.touch_radius / self._scale
        edge_thresh = self.edge_radius / self._scale

        for bi, (cls, pts) in enumerate(self.data.boxes):
            for vi, (vx, vy) in enumerate(pts):
                if abs(ix - vx) <= vert_thresh and abs(iy - vy) <= vert_thresh:
                    return ("vertex", bi, vi)

        for bi, (cls, pts) in enumerate(self.data.boxes):
            for ei in range(4):
                p1 = pts[ei]
                p2 = pts[(ei + 1) % 4]
                d = dist_point_to_segment((ix, iy), p1, p2)
                if d <= edge_thresh:
                    return ("edge", bi, ei)

        for bi, (cls, pts) in enumerate(self.data.boxes):
            if point_in_poly((ix, iy), pts):
                return ("box", bi, None)

        return None

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        if self.data.texture is None:
            return super().on_touch_down(touch)

        if touch.is_mouse_scrolling:
            if touch.button == "scrollup":
                self.zoom_by(1.15, touch.pos)
            elif touch.button == "scrolldown":
                self.zoom_by(0.87, touch.pos)
            return True

        self._touches[touch.uid] = touch
        touch.grab(self)

        if len(self._touches) >= 2:
            self._start_gesture()
            return True

        img_pt = self._disp_to_img(*touch.pos)
        if img_pt is None:
            if not self.creating:
                self.sel_box = None
                self.dragging = True
                self.drag_type = "pan"
                self.drag_start_disp = touch.pos
                self._pan_start_offset = list(self.view_offset)
                self._redraw()
                self.state_cb()
            return True

        ix, iy = img_pt
        self.active_touch = touch

        if self.creating:
            self.create_p1 = (ix, iy)
            self.create_p2 = (ix, iy)
            self.dragging = True
            self.drag_type = "create"
            self._redraw()
            return True

        hit = self._hit_test(ix, iy)
        if hit:
            htype, bi, hi = hit
            self.sel_box = bi
            self.dragging = True
            self.drag_type = htype
            self.drag_start = (ix, iy)
            self.drag_box_start = [p.copy() for p in self.data.boxes[bi][1]]
            if htype == "vertex":
                self.active_vert = (bi, hi)
            elif htype == "edge":
                self.active_edge = (bi, hi)
        else:
            self.sel_box = None
            self.dragging = True
            self.drag_type = "pan"
            self.drag_start_disp = touch.pos
            self._pan_start_offset = list(self.view_offset)

        self._redraw()
        self.state_cb()
        return True

    def on_touch_move(self, touch):
        if self._gesture_active:
            self._touches[touch.uid] = touch
            self._update_gesture()
            self._redraw()
            return True

        if touch.grab_current is not self:
            return super().on_touch_move(touch)
        if not self.dragging:
            return True

        if self.drag_type == "pan":
            dx = touch.pos[0] - self.drag_start_disp[0]
            dy = touch.pos[1] - self.drag_start_disp[1]
            self.view_offset = [self._pan_start_offset[0] + dx, self._pan_start_offset[1] + dy]
            self._clamp_view()
            self._redraw()
            return True

        img_pt = self._disp_to_img(*touch.pos)
        if img_pt is None:
            return True

        ix, iy = img_pt
        if self.drag_type == "create":
            self.create_p2 = (ix, iy)
            self._redraw()
            return True

        if self.sel_box is None:
            return True

        dx = ix - self.drag_start[0]
        dy = iy - self.drag_start[1]

        if self.drag_type == "vertex":
            bi, vi = self.active_vert
            nx, ny = self._clamp_pt(ix, iy)
            self.data.boxes[bi][1][vi] = [nx, ny]
            self.data.changed = True

        elif self.drag_type == "edge":
            bi, ei = self.active_edge
            v1 = ei
            v2 = (ei + 1) % 4
            p1 = self.drag_box_start[v1]
            p2 = self.drag_box_start[v2]
            x1, y1 = self._clamp_pt(p1[0] + dx, p1[1] + dy)
            x2, y2 = self._clamp_pt(p2[0] + dx, p2[1] + dy)
            self.data.boxes[bi][1][v1] = [x1, y1]
            self.data.boxes[bi][1][v2] = [x2, y2]
            self.data.changed = True

        elif self.drag_type == "box":
            bi = self.sel_box
            for i in range(4):
                p = self.drag_box_start[i]
                nx, ny = self._clamp_pt(p[0] + dx, p[1] + dy)
                self.data.boxes[bi][1][i] = [nx, ny]
            self.data.changed = True

        self._redraw()
        self.state_cb()
        return True

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return super().on_touch_up(touch)

        self._touches.pop(touch.uid, None)

        if self._gesture_active:
            if len(self._touches) < 2:
                self._gesture_active = False
            touch.ungrab(self)
            return True

        if self.drag_type == "pan":
            self.dragging = False
            self.drag_type = None
            self.drag_start_disp = None
            self._pan_start_offset = None
            touch.ungrab(self)
            self._redraw()
            self.state_cb()
            return True

        if self.drag_type == "create" and self.create_p1 and self.create_p2:
            x1, y1 = self.create_p1
            x2, y2 = self.create_p2
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                pts = [
                    [int(min(x1, x2)), int(min(y1, y2))],
                    [int(max(x1, x2)), int(min(y1, y2))],
                    [int(max(x1, x2)), int(max(y1, y2))],
                    [int(min(x1, x2)), int(max(y1, y2))],
                ]
                self.data.boxes.append([0, pts])
                self.sel_box = len(self.data.boxes) - 1
                self.data.changed = True

            self.creating = False
            self.create_p1 = None
            self.create_p2 = None

        self.dragging = False
        self.drag_type = None
        self.active_vert = None
        self.active_edge = None
        touch.ungrab(self)
        self._redraw()
        self.state_cb()
        return True

    def _redraw(self, *args):
        self.canvas.clear()
        with self.canvas:
            Color(0.08, 0.08, 0.08, 1.0)
            Rectangle(pos=self.pos, size=self.size)

            if self.data.texture is None:
                return

            self._apply_view()
            img_pos = self._img_pos
            img_size = self._img_size

            Color(1.0, 1.0, 1.0, 1.0)
            Rectangle(texture=self.data.texture, pos=img_pos, size=img_size)

            if self.creating and self.create_p1 and self.create_p2:
                x1, y1 = self._img_to_disp(*self.create_p1)
                x2, y2 = self._img_to_disp(*self.create_p2)
                Color(*COL_PREVIEW)
                Line(points=[x1, y1, x2, y1, x2, y2, x1, y2, x1, y1], width=dp(2))

            for bi, (cls, pts) in enumerate(self.data.boxes):
                is_sel = (bi == self.sel_box)
                col = COL_BOX_SEL if is_sel else COL_BOX
                width = dp(2) if is_sel else dp(1)

                Color(*col)
                for ei in range(4):
                    p1 = pts[ei]
                    p2 = pts[(ei + 1) % 4]
                    x1, y1 = self._img_to_disp(*p1)
                    x2, y2 = self._img_to_disp(*p2)
                    Line(points=[x1, y1, x2, y2], width=width)

                for vi, (vx, vy) in enumerate(pts):
                    x, y = self._img_to_disp(vx, vy)
                    r = self.vertex_radius + (dp(2) if is_sel else 0)
                    Color(*COL_VERTEX)
                    Ellipse(pos=(x - r, y - r), size=(r * 2, r * 2))
                    Color(1.0, 1.0, 1.0, 1.0)
                    Line(circle=(x, y, r), width=dp(1))


class MainView(BoxLayout):
    def __init__(self, data: EditorData, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.data = data

        self.status_label = Label(size_hint_y=None, height=dp(40), text="")
        self.add_widget(self.status_label)

        self.editor = OBBEditorWidget(self.data, state_cb=self.refresh)
        self.add_widget(self.editor)

        controls = BoxLayout(orientation="vertical", size_hint_y=None, height=dp(156))

        row1 = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(6), padding=(dp(6), dp(6), dp(6), 0))
        row2 = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(6), padding=(dp(6), 0, dp(6), 0))
        row3 = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(6), padding=(dp(6), 0, dp(6), dp(6)))

        row1.add_widget(self._btn("Prev", self.do_prev))
        row1.add_widget(self._btn("Next", self.do_next))
        row1.add_widget(self._btn("New", self.do_new))
        row1.add_widget(self._btn("Del", self.do_delete))

        self.class_label = Label(text="Class -", size_hint_x=0.6)
        row2.add_widget(self._btn("Undo", self.do_undo))
        row2.add_widget(self._btn("Save", self.do_save))
        row2.add_widget(self._btn("Class -", lambda: self.change_class(-1)))
        row2.add_widget(self.class_label)
        row2.add_widget(self._btn("Class +", lambda: self.change_class(1)))

        row3.add_widget(self._btn("Zoom -", self.do_zoom_out))
        row3.add_widget(self._btn("Zoom +", self.do_zoom_in))
        row3.add_widget(self._btn("Reset View", self.do_reset_view))
        row3.add_widget(self._btn("Dataset", self.open_dataset_picker))

        controls.add_widget(row1)
        controls.add_widget(row2)
        controls.add_widget(row3)
        self.add_widget(controls)

        self.refresh()

    def _btn(self, text, cb):
        b = Button(text=text)
        b.bind(on_release=lambda *_: cb())
        return b

    def refresh(self):
        if not self.data.image_paths:
            ds = str(self.data.images_dir)
            self.status_label.text = f"No images found. Dataset: {ds}"
            self.class_label.text = "Class -"
            return

        name = self.data.image_paths[self.data.idx].name
        status = "UNSAVED" if self.data.changed else "saved"
        extra = " | CREATE MODE" if self.editor.creating else ""
        ds_name = self.data.dataset_path.name if self.data.dataset_path else "-"
        self.status_label.text = (
            f"[{self.data.idx + 1}/{len(self.data.image_paths)}] {name} | "
            f"Boxes: {len(self.data.boxes)} | {status}{extra} | DS: {ds_name}"
        )

        if self.editor.sel_box is not None and self.editor.sel_box < len(self.data.boxes):
            cls = self.data.boxes[self.editor.sel_box][0]
            self.class_label.text = f"Class {cls}"
        else:
            self.class_label.text = "Class -"

    def _reload_after_nav(self):
        self.data.load()
        self.editor.reset_state()
        self.refresh()

    def do_prev(self):
        if not self.data.image_paths:
            return
        if self.data.changed:
            self.data.save()
        if self.data.idx > 0:
            self.data.idx -= 1
            self.data.save_progress()
            self._reload_after_nav()

    def do_next(self):
        if not self.data.image_paths:
            return
        if self.data.changed:
            self.data.save()
        if self.data.idx < len(self.data.image_paths) - 1:
            self.data.idx += 1
            self.data.save_progress()
            self._reload_after_nav()

    def do_new(self):
        if not self.data.image_paths:
            return
        self.editor.creating = True
        self.editor.create_p1 = None
        self.editor.create_p2 = None
        self.editor._redraw()
        self.refresh()

    def do_delete(self):
        if self.editor.sel_box is None:
            return
        if self.editor.sel_box < len(self.data.boxes):
            del self.data.boxes[self.editor.sel_box]
            self.editor.sel_box = None
            self.data.changed = True
            self.editor._redraw()
            self.refresh()

    def do_save(self):
        self.data.save()
        self.refresh()

    def do_undo(self):
        self.data.undo()
        self.editor.reset_state()
        self.refresh()

    def do_zoom_in(self):
        self.editor.zoom_by(1.25)

    def do_zoom_out(self):
        self.editor.zoom_by(0.8)

    def do_reset_view(self):
        self.editor.reset_view()

    def open_dataset_picker(self):
        start_path = str(self.data.dataset_path)
        chooser = FileChooserListView(path=start_path, dirselect=True)

        layout = BoxLayout(orientation="vertical")
        layout.add_widget(chooser)

        btn_row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(6), padding=(dp(6), dp(6), dp(6), dp(6)))
        popup = Popup(title="Select dataset folder", content=layout, size_hint=(0.95, 0.95))

        def select_path(*_):
            if chooser.selection:
                path = chooser.selection[0]
            else:
                path = chooser.path
            if path:
                self.set_dataset(path)
            popup.dismiss()

        btn_row.add_widget(self._btn("Cancel", popup.dismiss))
        btn_row.add_widget(self._btn("Select", select_path))
        layout.add_widget(btn_row)
        popup.open()

    def set_dataset(self, path):
        self.data.set_dataset_path(Path(path))
        self.editor.reset_state()
        self.editor.reset_view()
        self.refresh()
        app = App.get_running_app()
        if app:
            app.save_dataset_path(path)

    def change_class(self, delta):
        if self.editor.sel_box is None:
            return
        if self.editor.sel_box >= len(self.data.boxes):
            return
        cls = self.data.boxes[self.editor.sel_box][0]
        cls = max(0, cls + delta)
        self.data.boxes[self.editor.sel_box][0] = cls
        self.data.changed = True
        self.refresh()


class OBBApp(App):
    def build(self):
        dataset_path = self._default_dataset_path()
        self.data = EditorData(dataset_path)
        self.data.load_images()
        if self.data.image_paths:
            self.data.load()

        root = MainView(self.data)
        return root

    def on_stop(self):
        if hasattr(self, "data"):
            self.data.save_progress()

    def _load_saved_dataset(self):
        try:
            path = Path(self.user_data_dir) / "settings.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                ds = data.get("dataset_path")
                if ds:
                    return Path(ds)
        except Exception:
            return None
        return None

    def save_dataset_path(self, dataset_path):
        try:
            path = Path(self.user_data_dir) / "settings.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"dataset_path": str(dataset_path)}
            path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    def _default_dataset_path(self):
        env = os.environ.get("OBB_DATASET")
        if env:
            return Path(env)

        saved = self._load_saved_dataset()
        if saved:
            return saved

        if platform == "android":
            try:
                from android.storage import primary_external_storage_path
                root = Path(primary_external_storage_path())
            except Exception:
                root = Path("/sdcard")
            path = root / "OCR-PLACAS" / "dataset"
            return path

        return Path(__file__).resolve().parent / "dataset"


if __name__ == "__main__":
    OBBApp().run()
