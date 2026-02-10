"""Conversion between YOLO OBB boxes and FabricJS Path objects.

Each OBB is rendered as a FabricJS Path with SVG commands (M, L, z).
Class IDs are encoded via stroke color.
"""

import json

# Distinct colors for class IDs (stroke color on canvas)
CLASS_COLORS = [
    "#00DD00",  # 0 - green
    "#DD0000",  # 1 - red
    "#0066DD",  # 2 - blue
    "#DDAA00",  # 3 - yellow/gold
    "#DD00DD",  # 4 - magenta
    "#00DDDD",  # 5 - cyan
    "#FF6600",  # 6 - orange
    "#8800FF",  # 7 - purple
    "#668800",  # 8 - olive
    "#FF0088",  # 9 - pink
]

# Reverse map: hex color -> class_id
_COLOR_TO_CLASS = {c: i for i, c in enumerate(CLASS_COLORS)}


def class_color(class_id: int) -> str:
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def color_to_class(hex_color: str) -> int:
    return _COLOR_TO_CLASS.get(hex_color, 0)


def boxes_to_fabric_paths(
    boxes: list[list],
    img_w: int,
    img_h: int,
    canvas_w: int,
    canvas_h: int,
) -> list[dict]:
    """Convert YOLO OBB boxes (pixel coords) to FabricJS Path JSON objects.

    Each box: [class_id, [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]]
    Returns list of FabricJS object dicts suitable for st_canvas initial_drawing.
    """
    if img_w <= 0 or img_h <= 0:
        return []

    scale_x = canvas_w / img_w
    scale_y = canvas_h / img_h

    paths = []
    for cls, pts in boxes:
        # Scale pixel coords to canvas coords
        scaled = [[x * scale_x, y * scale_y] for x, y in pts]

        # Bounding box of scaled points -> left, top
        xs = [p[0] for p in scaled]
        ys = [p[1] for p in scaled]
        left = min(xs)
        top = min(ys)

        # Path commands relative to left, top
        rel = [[x - left, y - top] for x, y in scaled]
        svg_path = (
            f"M {rel[0][0]:.2f} {rel[0][1]:.2f} "
            f"L {rel[1][0]:.2f} {rel[1][1]:.2f} "
            f"L {rel[2][0]:.2f} {rel[2][1]:.2f} "
            f"L {rel[3][0]:.2f} {rel[3][1]:.2f} z"
        )

        path_obj = {
            "type": "path",
            "path": [
                ["M", rel[0][0], rel[0][1]],
                ["L", rel[1][0], rel[1][1]],
                ["L", rel[2][0], rel[2][1]],
                ["L", rel[3][0], rel[3][1]],
                ["z"],
            ],
            "left": left,
            "top": top,
            "fill": "",
            "stroke": class_color(cls),
            "strokeWidth": 2,
            "scaleX": 1,
            "scaleY": 1,
            "angle": 0,
            "originX": "left",
            "originY": "top",
        }
        paths.append(path_obj)

    return paths


def fabric_paths_to_boxes(
    json_data: dict | None,
    img_w: int,
    img_h: int,
    canvas_w: int,
    canvas_h: int,
) -> list[list]:
    """Convert FabricJS canvas JSON back to YOLO OBB boxes (pixel coords).

    Parses objects from json_data, applies left/top + scaleX/Y transforms,
    then descales from canvas coords back to image pixel coords.
    Only accepts paths with exactly 4 vertices.
    """
    if json_data is None or img_w <= 0 or img_h <= 0:
        return []

    scale_x = canvas_w / img_w
    scale_y = canvas_h / img_h

    objects = json_data.get("objects", [])
    boxes = []

    for obj in objects:
        if obj.get("type") != "path":
            continue

        path_cmds = obj.get("path", [])
        stroke = obj.get("stroke", "#00DD00")
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        sx = obj.get("scaleX", 1)
        sy = obj.get("scaleY", 1)

        # Extract vertex coords from path commands (M, L only)
        vertices = []
        for cmd in path_cmds:
            if len(cmd) >= 3 and cmd[0] in ("M", "L"):
                # Absolute canvas coord = left + relative * scale
                cx = left + cmd[1] * sx
                cy = top + cmd[2] * sy
                # Convert canvas coord to image pixel coord
                ix = cx / scale_x
                iy = cy / scale_y
                vertices.append([int(round(ix)), int(round(iy))])

        # Only accept quadrilaterals
        if len(vertices) != 4:
            continue

        cls = color_to_class(stroke)
        boxes.append([cls, vertices])

    return boxes


def make_initial_drawing(paths: list[dict]) -> dict:
    """Wrap path objects into the initial_drawing dict for st_canvas."""
    return {
        "version": "4.4.0",
        "objects": paths,
    }
