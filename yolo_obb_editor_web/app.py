"""YOLO OBB Editor - Web version (Streamlit)."""

import copy

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

from editor_data import EditorData
from canvas_bridge import (
    boxes_to_fabric_paths,
    fabric_paths_to_boxes,
    make_initial_drawing,
    class_color,
    CLASS_COLORS,
)
from zip_handler import extract_dataset_zip, create_labels_zip

CANVAS_W = 800
CANVAS_H = 600


def init_session_state():
    defaults = {
        "editor_data": None,
        "idx": 0,
        "boxes": [],
        "original_txt": "",
        "changed": False,
        "canvas_key": 0,
        "drawing_mode": "transform",
        "temp_dir": None,
        "current_class": 0,
        "sel_box_idx": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_current_image() -> Image.Image | None:
    ed = st.session_state.editor_data
    if ed is None or not ed.image_paths:
        return None
    ed.load()
    st.session_state.boxes = copy.deepcopy(ed.boxes)
    st.session_state.original_txt = ed.original_txt
    st.session_state.changed = False
    st.session_state.sel_box_idx = None
    return ed.pil_image


def fit_image_to_canvas(img: Image.Image) -> Image.Image:
    """Resize image to fit canvas while maintaining aspect ratio."""
    w, h = img.size
    scale = min(CANVAS_W / w, CANVAS_H / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def do_save():
    ed = st.session_state.editor_data
    if ed is None or not ed.image_paths:
        return
    ed.boxes = st.session_state.boxes
    ed.changed = True
    ed.save()
    st.session_state.changed = False


def do_undo():
    ed = st.session_state.editor_data
    if ed is None:
        return
    ed.undo()
    st.session_state.boxes = copy.deepcopy(ed.boxes)
    st.session_state.changed = False
    st.session_state.sel_box_idx = None
    st.session_state.canvas_key += 1


def do_navigate(delta: int):
    ed = st.session_state.editor_data
    if ed is None or not ed.image_paths:
        return
    if st.session_state.changed:
        do_save()
    new_idx = ed.idx + delta
    if 0 <= new_idx < len(ed.image_paths):
        ed.idx = new_idx
        st.session_state.idx = new_idx
        load_current_image()
        st.session_state.canvas_key += 1


def do_delete_selected():
    idx = st.session_state.sel_box_idx
    if idx is not None and 0 <= idx < len(st.session_state.boxes):
        del st.session_state.boxes[idx]
        st.session_state.sel_box_idx = None
        st.session_state.changed = True
        st.session_state.canvas_key += 1


def do_change_class(delta: int):
    idx = st.session_state.sel_box_idx
    if idx is not None and 0 <= idx < len(st.session_state.boxes):
        cls = st.session_state.boxes[idx][0]
        cls = max(0, cls + delta)
        st.session_state.boxes[idx][0] = cls
        st.session_state.changed = True
        st.session_state.canvas_key += 1


def main():
    st.set_page_config(page_title="YOLO OBB Editor", layout="wide")
    init_session_state()

    ed = st.session_state.editor_data

    # --- Sidebar ---
    with st.sidebar:
        st.header("YOLO OBB Editor")

        uploaded = st.file_uploader("Upload dataset ZIP", type=["zip"])
        if uploaded is not None:
            # Only process if it's a new file
            upload_id = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.get("_last_upload_id") != upload_id:
                st.session_state._last_upload_id = upload_id
                with st.spinner("Extracting ZIP..."):
                    dataset_path, temp_root = extract_dataset_zip(
                        uploaded, st.session_state.temp_dir
                    )
                    st.session_state.temp_dir = temp_root
                    ed = EditorData(dataset_path)
                    ed.load_images()
                    st.session_state.editor_data = ed
                    if ed.image_paths:
                        ed.idx = 0
                        st.session_state.idx = 0
                        load_current_image()
                        st.session_state.canvas_key += 1
                    st.rerun()

        st.divider()

        if ed and ed.image_paths:
            img_name = ed.image_paths[ed.idx].name
            status = "UNSAVED" if st.session_state.changed else "saved"
            st.markdown(f"**Image:** {ed.idx + 1}/{len(ed.image_paths)}")
            st.markdown(f"**File:** {img_name}")
            st.markdown(f"**Boxes:** {len(st.session_state.boxes)} | **Status:** {status}")

            sel = st.session_state.sel_box_idx
            if sel is not None and sel < len(st.session_state.boxes):
                cls = st.session_state.boxes[sel][0]
                color = class_color(cls)
                st.markdown(f"**Selected box:** #{sel} | Class **{cls}** <span style='color:{color}'>&#9632;</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Selected box:** none")
        else:
            st.info("Upload a ZIP with images/ and labels/ folders.")

        st.divider()

        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Prev", use_container_width=True):
                do_navigate(-1)
                st.rerun()
        with col2:
            if st.button("Next", use_container_width=True):
                do_navigate(1)
                st.rerun()

        # Drawing mode
        mode = st.radio(
            "Mode",
            ["Transform", "Draw Polygon"],
            horizontal=True,
            index=0 if st.session_state.drawing_mode == "transform" else 1,
        )
        st.session_state.drawing_mode = "transform" if mode == "Transform" else "polygon"

        # New box class
        st.session_state.current_class = st.number_input(
            "New box class ID", min_value=0, max_value=len(CLASS_COLORS) - 1,
            value=st.session_state.current_class,
        )

        st.divider()

        # Download labels
        if ed and ed.labels_dir and ed.labels_dir.exists():
            zip_bytes = create_labels_zip(ed.labels_dir)
            if zip_bytes:
                st.download_button(
                    "Download Labels ZIP",
                    data=zip_bytes,
                    file_name="labels.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

        # Class legend
        st.divider()
        st.caption("Class colors")
        legend = " | ".join(
            f"<span style='color:{c}'>&#9632;</span> {i}"
            for i, c in enumerate(CLASS_COLORS)
        )
        st.markdown(legend, unsafe_allow_html=True)

    # --- Main area ---
    if ed is None or not ed.image_paths:
        st.title("YOLO OBB Editor")
        st.write("Upload a dataset ZIP file in the sidebar to get started.")
        st.write("The ZIP should contain `images/` and `labels/` folders with YOLO OBB format labels.")
        return

    # Action buttons
    btn_cols = st.columns([1, 1, 1, 1, 1, 1, 4])
    with btn_cols[0]:
        if st.button("Save"):
            do_save()
            st.rerun()
    with btn_cols[1]:
        if st.button("Undo"):
            do_undo()
            st.rerun()
    with btn_cols[2]:
        if st.button("Delete"):
            do_delete_selected()
            st.rerun()
    with btn_cols[3]:
        if st.button("Class -"):
            do_change_class(-1)
            st.rerun()
    with btn_cols[4]:
        if st.button("Class +"):
            do_change_class(1)
            st.rerun()
    with btn_cols[5]:
        # Jump to image
        jump = st.number_input(
            "Go to #", min_value=1, max_value=len(ed.image_paths),
            value=ed.idx + 1, label_visibility="collapsed",
        )
        if jump - 1 != ed.idx:
            if st.session_state.changed:
                do_save()
            ed.idx = jump - 1
            st.session_state.idx = ed.idx
            load_current_image()
            st.session_state.canvas_key += 1
            st.rerun()

    # Prepare canvas image and paths
    pil_img = ed.pil_image
    if pil_img is None:
        st.error("Could not load image.")
        return

    fitted = fit_image_to_canvas(pil_img)
    cw, ch = fitted.size

    paths = boxes_to_fabric_paths(
        st.session_state.boxes, ed.w, ed.h, cw, ch
    )
    initial = make_initial_drawing(paths)

    # Determine canvas drawing mode
    if st.session_state.drawing_mode == "polygon":
        canvas_mode = "polygon"
        stroke_color = class_color(st.session_state.current_class)
    else:
        canvas_mode = "transform"
        stroke_color = class_color(st.session_state.current_class)

    # Canvas
    result = st_canvas(
        fill_color="",
        stroke_width=2,
        stroke_color=stroke_color,
        background_image=fitted,
        initial_drawing=initial,
        drawing_mode=canvas_mode,
        height=ch,
        width=cw,
        key=f"canvas_{st.session_state.canvas_key}",
        display_toolbar=True,
    )

    # Process canvas result
    if result is not None and result.json_data is not None:
        new_boxes = fabric_paths_to_boxes(
            result.json_data, ed.w, ed.h, cw, ch
        )
        # Check if boxes changed from canvas interaction
        if new_boxes != st.session_state.boxes:
            # Detect new boxes added via polygon mode
            old_count = len(st.session_state.boxes)
            if len(new_boxes) > old_count:
                # Assign current_class to newly drawn boxes
                for i in range(old_count, len(new_boxes)):
                    new_boxes[i][0] = st.session_state.current_class

            st.session_state.boxes = new_boxes
            st.session_state.changed = True

        # Detect selected object for sel_box_idx
        objects = result.json_data.get("objects", [])
        # Find the active (selected) object index - FabricJS doesn't directly
        # expose selection in json_data, so we track based on path count
        # The user clicks a box -> transform mode shows handles on it
        # We use a simple heuristic: if there's only one path that changed, that's selected
        if len(objects) > 0:
            # Default: select last box if we just added one
            if len(new_boxes) > 0 and len(new_boxes) > len(st.session_state.get("_prev_boxes", [])):
                st.session_state.sel_box_idx = len(new_boxes) - 1
        st.session_state._prev_boxes = copy.deepcopy(st.session_state.boxes)


if __name__ == "__main__":
    main()
