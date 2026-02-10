# YOLO OBB Editor (Android)

This folder contains a Kivy-based port of yolo_obb_editor.py intended for Android APK.

What changed
- OpenCV HighGUI replaced by Kivy Canvas (touch friendly)
- Buttons replace keyboard shortcuts
- Labels saved in the same YOLO OBB format
- Pinch-zoom and pan
- Dataset folder picker

Dataset path (default)
- Android: /sdcard/OCR-PLACAS/dataset
- Desktop: <this folder>/dataset
- You can override on desktop with OBB_DATASET env var

Expected dataset structure
- dataset/images (or images/train|val|test)
- dataset/labels (or labels/train|val|test)

Controls
- Tap to select a box
- Drag a vertex, edge, or the box to move
- Tap New then drag to create a new box
- Pinch to zoom, two-finger drag to pan
- Drag empty area to pan
- Zoom +/- and Reset View buttons
- Dataset button to choose folder

Build (Linux or WSL)
1. cd yolo_obb_editor_android
2. python3 -m venv .venv
3. source .venv/bin/activate
4. pip install buildozer
5. sudo apt-get install -y git zip unzip openjdk-17-jdk
6. buildozer -v android debug

Install
- buildozer android deploy run
- Or: adb install -r bin/*.apk
GitHub Actions (cloud build)
1. Push this repo to GitHub
2. Go to Actions tab
3. Run workflow "Build Android APK"
4. Download artifact "yolo-obb-editor-apk"

Notes
- First build can take 10-30 minutes
- APK will be in the artifact zip

