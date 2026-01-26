import sys
import cv2
import numpy as np
import json
import os
import math
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (https://github.com/AI-Inspact-Lab/PCB-AOI-LABELING-TOOLS/tree/main
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QGroupBox, QFileDialog, QMessageBox, QScrollArea,
    QSizePolicy, QListWidget, QListWidgetItem, QSplitter
)
from PySide6.QtCore import Qt, QPoint, QRect, Signal
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QKeySequence, QAction
)

DEFAULT_IMAGE_PATH = r"./example_image/1-2-SPRCK64AA_06718_2_1.jpg"
OUT_DIR = "labels"

BLUR_KSIZE = 5
MIN_COMPONENT_AREA = 50

ROI_LINE_WIDTH_DRAGGING = 1
ROI_LINE_WIDTH_CONFIRMED = 1
MEASUREMENT_LINE_WIDTH = 1
MEASUREMENT_MARKER_SIZE = 2
MEASUREMENT_TEXT_SIZE = 8
MEASUREMENT_TEXT_OFFSET = 25

os.makedirs(OUT_DIR, exist_ok=True)


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img


def rotate_image_and_get_M(img, angle_deg, border_value=0):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderValue=border_value
    )
    M_inv = cv2.invertAffineTransform(M)
    return rotated, M, M_inv


def affine_point(M, x, y):
    xx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    yy = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return float(xx), float(yy)


def _threshold_and_clean(gray_roi, threshold_value, invert):
    if BLUR_KSIZE and BLUR_KSIZE >= 3:
        gray = cv2.GaussianBlur(gray_roi, (BLUR_KSIZE, BLUR_KSIZE), 0)
    else:
        gray = gray_roi.copy()

    th_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, threshold_value, 255, th_type)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cleaned = np.zeros_like(bw)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_COMPONENT_AREA:
            cleaned[labels == i] = 255
    return cleaned


def analyze_lead_notch(gray_roi, threshold_value, invert=False):
    bw = _threshold_and_clean(gray_roi, threshold_value, invert)

    widths, lefts, rights = [], [], []
    h, _w = bw.shape

    for y in range(h):
        xs = np.where(bw[y] > 0)[0]
        if xs.size < 2:
            widths.append(np.nan); lefts.append(np.nan); rights.append(np.nan)
            continue
        left = int(xs.min()); right = int(xs.max())
        widths.append(right - left + 1)
        lefts.append(left)
        rights.append(right)

    widths = np.array(widths, dtype=np.float32)
    lefts = np.array(lefts, dtype=np.float32)
    rights = np.array(rights, dtype=np.float32)

    valid = ~np.isnan(widths)
    if valid.sum() < max(10, 0.1 * h):
        return None

    widths_v = widths[valid]
    ref_w = float(np.nanpercentile(widths_v, 90))
    min_w = float(np.nanpercentile(widths_v, 10))

    notch_pct = 0.0
    if ref_w > 1e-6:
        notch_pct = max(0.0, (ref_w - min_w) / ref_w * 100.0)

    y_min = int(np.nanargmin(widths))
    l_min = int(lefts[y_min]) if not np.isnan(lefts[y_min]) else None
    r_min = int(rights[y_min]) if not np.isnan(rights[y_min]) else None

    return {
        "ref_width_px": ref_w,
        "min_width_px": min_w,
        "notch_ratio_percent": notch_pct,
        "y_min": y_min,
        "min_left_x": l_min,
        "min_right_x": r_min,
        "binary_thresholded": bw,
        "invert": invert
    }


def analyze_lead_notch_with_angle(gray_roi, threshold_value, invert, angle_deg):
    bw = _threshold_and_clean(gray_roi, threshold_value, invert)

    rot_bw, M, M_inv = rotate_image_and_get_M(bw, -angle_deg, border_value=0)

    h, w = rot_bw.shape
    widths, lefts, rights = [], [], []
    for y in range(h):
        xs = np.where(rot_bw[y] > 0)[0]
        if xs.size < 2:
            widths.append(np.nan); lefts.append(np.nan); rights.append(np.nan)
            continue
        left = int(xs.min())
        right = int(xs.max())
        widths.append(right - left + 1)
        lefts.append(left)
        rights.append(right)

    widths = np.array(widths, dtype=np.float32)
    lefts = np.array(lefts, dtype=np.float32)
    rights = np.array(rights, dtype=np.float32)

    valid = ~np.isnan(widths)
    
    if valid.sum() < max(10, 0.1 * h):
        return None

    widths_v = widths[valid]
    ref_w = float(np.nanpercentile(widths_v, 90))
    min_w = float(np.nanpercentile(widths_v, 10))

    notch_pct = 0.0
    if ref_w > 1e-6:
        notch_pct = max(0.0, (ref_w - min_w) / ref_w * 100.0)

    y_min = int(np.nanargmin(widths))
    l_min = int(lefts[y_min]) if not np.isnan(lefts[y_min]) else None
    r_min = int(rights[y_min]) if not np.isnan(rights[y_min]) else None
    if l_min is None or r_min is None:
        return None

    xA, yA = affine_point(M_inv, l_min, y_min)
    xB, yB = affine_point(M_inv, r_min, y_min)

    return {
        "ref_width_px": ref_w,
        "min_width_px": min_w,
        "notch_ratio_percent": notch_pct,
        "binary_thresholded": bw,
        "binary_rotated": rot_bw,
        "invert": invert,
        "angle_deg": float(angle_deg),
        "measure_line_roi": ((xA, yA), (xB, yB)),
    }


class ImageLabel(QLabel):
    roi_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setMouseTracking(True)

        self.start_point = None
        self.end_point = None
        self.dragging = False

        self.setMinimumSize(10, 10)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.NoFocus)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        # 방향 지정 모드: ROI 드래그 시작 금지
        if self.parent_window.direction_mode:
            self.parent_window.start_direction_drag(event.position().toPoint())
            self.roi_changed.emit()
            event.accept()
            return

        self.start_point = event.position().toPoint()
        self.end_point = self.start_point
        self.dragging = True
        self.roi_changed.emit()
        event.accept()

    def mouseMoveEvent(self, event):
        # 방향선 드래그 중이면 우선 처리
        if self.parent_window.direction_dragging:
            self.parent_window.update_direction_drag(event.position().toPoint())
            self.roi_changed.emit()
            event.accept()
            return

        if self.dragging:
            self.end_point = event.position().toPoint()
            self.roi_changed.emit()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        # 방향선 드래그 종료 우선
        if self.parent_window.direction_dragging:
            self.parent_window.end_direction_drag(event.position().toPoint())
            self.roi_changed.emit()
            event.accept()
            return

        if self.dragging:
            self.end_point = event.position().toPoint()
            self.dragging = False
            self.parent_window.finalize_roi()
            event.accept()


class LeadNotchAnalyzer(QMainWindow):
    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lead Notch Analysis Tool - Professional Edition")
        self.setGeometry(100, 100, 1500, 880)

        self.image_paths = []
        self.image_index = -1
        self.current_image_path = None

        self.original_image = None
        self.roi = None
        self.analysis = None
        self.threshold_value = 100
        self.mode = "reflect"

        self.zoom = 1.0
        self.zoom_step = 0.5
        self.zoom_min = 0.1
        self.zoom_max = 8.0

        self.direction_mode = False
        self.direction_dragging = False
        self.dir_pt1_img = None
        self.dir_pt2_img = None
        self.measure_angle_deg = 0.0

        self.init_ui()
        self.statusBar().showMessage("Ready", 1500)
        self.setup_menu()
        self.setup_actions_and_shortcuts()

        if Path(DEFAULT_IMAGE_PATH).exists():
            self.set_image_list([DEFAULT_IMAGE_PATH], start_index=0)
        else:
            QMessageBox.information(self, "Info", "기본 이미지 경로가 없습니다. 상단 메뉴에서 이미지를 열어주세요.")

        self.raise_()
        self.activateWindow()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout()
        central_widget.setLayout(root_layout)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(220)
        self.list_widget.setMaximumWidth(340)
        self.list_widget.itemSelectionChanged.connect(self.on_list_selection_changed)
        splitter.addWidget(self.list_widget)

        image_container = QWidget()
        image_layout = QVBoxLayout()
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_container.setLayout(image_layout)

        self.image_label = ImageLabel(self)
        self.image_label.setStyleSheet("border: 2px solid #555; background-color: #2b2b2b;")
        self.image_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.image_label.roi_changed.connect(self.update_display)
        self.image_label.setScaledContents(False)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setMinimumHeight(600)

        image_layout.addWidget(self.scroll_area)
        splitter.addWidget(image_container)

        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 950, 320])

    def setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("파일(&F)")
        view_menu = menubar.addMenu("보기(&V)")
        mode_menu = menubar.addMenu("모드(&M)")

        self.act_open_images = QAction("이미지 열기...(&O)", self)
        self.act_open_images.setShortcut(QKeySequence.Open)
        self.act_open_images.triggered.connect(self.open_images_dialog)
        file_menu.addAction(self.act_open_images)

        self.act_open_folder = QAction("폴더 열기...(&D)", self)
        self.act_open_folder.triggered.connect(self.open_folder_dialog)
        file_menu.addAction(self.act_open_folder)

        file_menu.addSeparator()

        self.act_prev = QAction("이전 이미지(&P)", self)
        self.act_prev.setShortcut(QKeySequence("A"))
        self.act_prev.triggered.connect(self.prev_image)
        file_menu.addAction(self.act_prev)

        self.act_next = QAction("다음 이미지(&N)", self)
        self.act_next.setShortcut(QKeySequence("D"))
        self.act_next.triggered.connect(self.next_image)
        file_menu.addAction(self.act_next)

        file_menu.addSeparator()

        self.act_save = QAction("결과 저장(&S)", self)
        self.act_save.setShortcut(QKeySequence.Save)
        self.act_save.triggered.connect(self.save_results)
        file_menu.addAction(self.act_save)

        self.act_reset = QAction("ROI/방향 초기화(&R)", self)
        self.act_reset.setShortcut(QKeySequence("R"))
        self.act_reset.triggered.connect(self.reset_roi)
        file_menu.addAction(self.act_reset)

        file_menu.addSeparator()

        self.act_quit = QAction("종료(&Q)", self)
        self.act_quit.setShortcut(QKeySequence("Q"))
        self.act_quit.triggered.connect(self.close)
        file_menu.addAction(self.act_quit)

        self.act_zoom_in = QAction("확대(+)", self)
        self.act_zoom_in.setShortcut(QKeySequence("+"))
        self.act_zoom_in.triggered.connect(self.zoom_in)
        view_menu.addAction(self.act_zoom_in)

        self.act_zoom_out = QAction("축소(-)", self)
        self.act_zoom_out.setShortcut(QKeySequence("-"))
        self.act_zoom_out.triggered.connect(self.zoom_out)
        view_menu.addAction(self.act_zoom_out)

        self.act_direction = QAction("측정 방향 지정(T)", self)
        self.act_direction.setShortcut(QKeySequence("T"))
        self.act_direction.triggered.connect(self.toggle_direction_mode)
        view_menu.addAction(self.act_direction)

        self.act_mode_reflect = QAction("반사 모드 (1)", self)
        self.act_mode_reflect.setCheckable(True)
        self.act_mode_reflect.setChecked(True)
        self.act_mode_reflect.setShortcut(QKeySequence("1"))
        self.act_mode_reflect.triggered.connect(lambda: self.set_mode("reflect"))
        mode_menu.addAction(self.act_mode_reflect)

        self.act_mode_transmit = QAction("투과 모드 (2)", self)
        self.act_mode_transmit.setCheckable(True)
        self.act_mode_transmit.setChecked(False)
        self.act_mode_transmit.setShortcut(QKeySequence("2"))
        self.act_mode_transmit.triggered.connect(lambda: self.set_mode("transmit"))
        mode_menu.addAction(self.act_mode_transmit)

        self.mode_actions = [self.act_mode_reflect, self.act_mode_transmit]

    def setup_actions_and_shortcuts(self):
        all_actions = [
            self.act_open_images, self.act_open_folder, self.act_prev, self.act_next,
            self.act_save, self.act_reset, self.act_quit,
            self.act_zoom_in, self.act_zoom_out, self.act_direction,
            self.act_mode_reflect, self.act_mode_transmit
        ]
        for a in all_actions:
            a.setShortcutContext(Qt.ApplicationShortcut)
            self.addAction(a)

        self.act_th_up = QAction(self)
        self.act_th_up.setShortcut(QKeySequence("Alt+Up"))
        self.act_th_up.setShortcutContext(Qt.ApplicationShortcut)
        self.act_th_up.triggered.connect(self.increase_threshold)
        self.addAction(self.act_th_up)

        self.act_th_down = QAction(self)
        self.act_th_down.setShortcut(QKeySequence("Alt+Down"))
        self.act_th_down.setShortcutContext(Qt.ApplicationShortcut)
        self.act_th_down.triggered.connect(self.decrease_threshold)
        self.addAction(self.act_th_down)

        self.act_esc = QAction(self)
        self.act_esc.setShortcut(QKeySequence(Qt.Key_Escape))
        self.act_esc.setShortcutContext(Qt.ApplicationShortcut)
        self.act_esc.triggered.connect(self.reset_roi)
        self.addAction(self.act_esc)

        self.act_zoom_in_alt = QAction(self)
        self.act_zoom_in_alt.setShortcut(QKeySequence("="))
        self.act_zoom_in_alt.setShortcutContext(Qt.ApplicationShortcut)
        self.act_zoom_in_alt.triggered.connect(self.zoom_in)
        self.addAction(self.act_zoom_in_alt)

    def create_control_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(360)
        panel.setStyleSheet("""
            QWidget { background-color: #f5f5f5; font-family: 'Segoe UI', Arial, sans-serif; }
            QGroupBox {
                font-weight: bold; font-size: 13px;
                border: 2px solid #cccccc; border-radius: 6px;
                margin-top: 12px; padding-top: 12px;
                background-color: white;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton {
                background-color: #0078d4; color: white; border: none;
                padding: 10px; border-radius: 4px; font-size: 13px; font-weight: bold; min-height: 34px;
            }
            QPushButton:hover { background-color: #106ebe; }
            QPushButton:pressed { background-color: #005a9e; }
            QLabel { font-size: 12px; color: #333; }
        """)

        layout = QVBoxLayout()
        panel.setLayout(layout)

        title = QLabel("Control")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 8px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        dir_group = QGroupBox("Direction (Drag to set)")
        dir_layout = QVBoxLayout()
        self.dir_label = QLabel("Direction: 기본(수평) / T로 방향 지정(드래그)")
        dir_layout.addWidget(self.dir_label)

        btn_dir = QPushButton("방향 지정 모드 (T)")
        btn_dir.setFocusPolicy(Qt.NoFocus)
        btn_dir.clicked.connect(self.toggle_direction_mode)
        dir_layout.addWidget(btn_dir)

        btn_dir_clear = QPushButton("방향 초기화")
        btn_dir_clear.setFocusPolicy(Qt.NoFocus)
        btn_dir_clear.setStyleSheet("QPushButton { background-color: #666; } QPushButton:hover { background-color: #555; }")
        btn_dir_clear.clicked.connect(self.clear_direction_only)
        dir_layout.addWidget(btn_dir_clear)

        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout()

        self.mode_label = QLabel("Current: 반사 모드 (하얀쪽이 대상)")
        self.mode_label.setStyleSheet("font-size: 12px; padding: 6px; color: #333;")
        mode_layout.addWidget(self.mode_label)

        self.btn_reflect = QPushButton("반사 모드 (1)")
        self.btn_reflect.setCheckable(True)
        self.btn_reflect.setChecked(True)
        self.btn_reflect.setFocusPolicy(Qt.NoFocus)
        self.btn_reflect.clicked.connect(lambda: self.set_mode("reflect"))
        mode_layout.addWidget(self.btn_reflect)

        self.btn_transmit = QPushButton("투과 모드 (2)")
        self.btn_transmit.setCheckable(True)
        self.btn_transmit.setChecked(False)
        self.btn_transmit.setFocusPolicy(Qt.NoFocus)
        self.btn_transmit.clicked.connect(lambda: self.set_mode("transmit"))
        mode_layout.addWidget(self.btn_transmit)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        threshold_group = QGroupBox("Threshold")
        threshold_layout = QVBoxLayout()

        self.threshold_label = QLabel(f"Threshold: {self.threshold_value}")
        self.threshold_label.setStyleSheet("font-size: 13px; color: #0078d4; font-weight: bold; padding: 4px;")
        threshold_layout.addWidget(self.threshold_label)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(self.threshold_value)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)

        hint = QLabel("Alt+↑ / Alt+↓")
        hint.setStyleSheet("font-size: 10px; color: #666; padding: 2px;")
        threshold_layout.addWidget(hint)

        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)

        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.ref_width_label = QLabel("REF Width: -")
        self.min_width_label = QLabel("MIN Width: -")
        self.notch_ratio_label = QLabel("Notch Ratio: -")
        results_layout.addWidget(self.ref_width_label)
        results_layout.addWidget(self.min_width_label)
        self.notch_ratio_label.setStyleSheet("font-size: 15px; padding: 8px; font-weight: bold;")
        results_layout.addWidget(self.notch_ratio_label)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(8)

        zoom_in_btn = QPushButton("➕ Zoom In (+)")
        zoom_in_btn.setFocusPolicy(Qt.NoFocus)
        zoom_in_btn.clicked.connect(self.zoom_in)
        actions_layout.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("➖ Zoom Out (-)")
        zoom_out_btn.setFocusPolicy(Qt.NoFocus)
        zoom_out_btn.clicked.connect(self.zoom_out)
        actions_layout.addWidget(zoom_out_btn)

        reset_btn = QPushButton("🔄 Reset ROI/방향 (R/ESC)")
        reset_btn.setFocusPolicy(Qt.NoFocus)
        reset_btn.clicked.connect(self.reset_roi)
        actions_layout.addWidget(reset_btn)

        save_btn = QPushButton("💾 Save (Ctrl+S)")
        save_btn.setFocusPolicy(Qt.NoFocus)
        save_btn.setStyleSheet("""
            QPushButton { background-color: #107c10; }
            QPushButton:hover { background-color: #0e6b0e; }
            QPushButton:pressed { background-color: #0c5a0c; }
        """)
        save_btn.clicked.connect(self.save_results)
        actions_layout.addWidget(save_btn)

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        layout.addStretch()
        return panel

    # ---------- direction: drag-based ----------
    def toggle_direction_mode(self):
        self.direction_mode = not self.direction_mode
        self.direction_dragging = False
        if self.direction_mode:
            self.statusBar().showMessage("방향 지정 모드 ON: 이미지에서 드래그로 방향선을 그리세요 (T로 취소)", 6000)
        else:
            self.statusBar().showMessage("방향 지정 모드 OFF", 2000)
        self.update_dir_label()
        self.update_display()

    def clear_direction_only(self):
        self.direction_mode = False
        self.direction_dragging = False
        self.dir_pt1_img = None
        self.dir_pt2_img = None
        self.measure_angle_deg = 0.0
        self.statusBar().showMessage("방향 초기화됨", 2000)
        if self.roi is not None:
            self.analyze_roi()
        self.update_dir_label()
        self.update_display()

    def update_dir_label(self):
        if self.direction_mode:
            self.dir_label.setText("Direction: 지정 모드 ON (드래그로 선 그리기)")
        else:
            if self.dir_pt1_img is not None and self.dir_pt2_img is not None:
                self.dir_label.setText(f"Direction: 사용자 지정 ({self.measure_angle_deg:.1f}°)")
            else:
                self.dir_label.setText("Direction: 기본(수평) / T로 방향 지정(드래그)")

    def _label_to_img(self, pt_label: QPoint):
        if self.original_image is None:
            return None
        x = float(pt_label.x() / self.zoom)
        y = float(pt_label.y() / self.zoom)
        h, w = self.original_image.shape[:2]
        x = max(0.0, min(x, w - 1.0))
        y = max(0.0, min(y, h - 1.0))
        return (x, y)

    def start_direction_drag(self, pt_label: QPoint):
        p = self._label_to_img(pt_label)
        if p is None:
            return
        self.direction_dragging = True
        self.dir_pt1_img = p
        self.dir_pt2_img = p

    def update_direction_drag(self, pt_label: QPoint):
        if not self.direction_dragging:
            return
        p = self._label_to_img(pt_label)
        if p is None:
            return
        self.dir_pt2_img = p

    def end_direction_drag(self, pt_label: QPoint):
        if not self.direction_dragging:
            return
        p = self._label_to_img(pt_label)
        self.direction_dragging = False
        self.direction_mode = False

        if p is None:
            self.dir_pt1_img = None
            self.dir_pt2_img = None
            self.measure_angle_deg = 0.0
            self.update_dir_label()
            self.update_display()
            return

        self.dir_pt2_img = p
        x1, y1 = self.dir_pt1_img
        x2, y2 = self.dir_pt2_img
        dx, dy = (x2 - x1), (y2 - y1)

        if (dx * dx + dy * dy) < 25.0:  # 5px
            self.dir_pt1_img = None
            self.dir_pt2_img = None
            self.measure_angle_deg = 0.0
            self.statusBar().showMessage("방향선이 너무 짧아서 무시됨", 2000)
        else:
            self.measure_angle_deg = math.degrees(math.atan2(dy, dx))
            self.statusBar().showMessage(f"방향 설정 완료: {self.measure_angle_deg:.1f}°", 3500)

        self.update_dir_label()
        if self.roi is not None:
            self.analyze_roi()
        self.update_display()

    # ---------- list ----------
    def refresh_list(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
    
        for i, p in enumerate(self.image_paths):
            name = os.path.basename(p)
            item = QListWidgetItem(name)
            item.setToolTip(p)
            item.setData(Qt.UserRole, i)
    
            # ✅ 라벨링된 파일이면 회색 표시
            if self.is_labeled(p):
                item.setForeground(QColor(140, 140, 140))  # 글자 회색
            else:
                item.setForeground(QColor(20, 20, 20))
    
            self.list_widget.addItem(item)
    
        if 0 <= self.image_index < self.list_widget.count():
            self.list_widget.setCurrentRow(self.image_index)
    
        self.list_widget.blockSignals(False)


    def on_list_selection_changed(self):
        items = self.list_widget.selectedItems()
        if not items:
            return
        idx = items[0].data(Qt.UserRole)
        if isinstance(idx, int) and idx != self.image_index:
            self.load_image_by_index(idx)
            
    def label_path_for_image(self, img_path: str) -> str:
        stem = Path(img_path).stem
        return str(Path(OUT_DIR) / f"{stem}.json")

    def is_labeled(self, img_path: str) -> bool:
        return Path(self.label_path_for_image(img_path)).exists()

    # ---------- open ----------
    def open_images_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "이미지 열기", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*)"
        )
        if not paths:
            return
        self.set_image_list(paths, start_index=0)

    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "폴더 열기", "")
        if not folder:
            return

        p = Path(folder)
        files = []
        for ext in sorted(self.SUPPORTED_EXTS):
            files.extend(p.glob(f"*{ext}"))
            files.extend(p.glob(f"*{ext.upper()}"))

        files = sorted({str(f) for f in files})
        if not files:
            QMessageBox.information(self, "Info", "해당 폴더에서 이미지 파일을 찾지 못했습니다.")
            return

        self.set_image_list(files, start_index=0)

    def set_image_list(self, paths, start_index=0):
        cleaned = []
        for x in paths:
            try:
                px = str(Path(x))
            except Exception:
                continue
            if Path(px).exists():
                cleaned.append(px)

        if not cleaned:
            QMessageBox.warning(self, "Warning", "유효한 이미지 경로가 없습니다.")
            return

        self.image_paths = cleaned
        self.image_index = max(0, min(start_index, len(self.image_paths) - 1))
        self.refresh_list()
        self.load_image_by_index(self.image_index)

    def load_image_by_index(self, idx: int):
        if not self.image_paths:
            return
        idx = max(0, min(idx, len(self.image_paths) - 1))
        path = self.image_paths[idx]

        img = imread_unicode(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{path}")
            return

        self.image_index = idx
        self.current_image_path = path

        self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.zoom = 3.0 ## 기본 줌 배율

        # 요구사항: 다음 이미지면 방향 유지 X
        self.direction_mode = False
        self.direction_dragging = False
        self.dir_pt1_img = None
        self.dir_pt2_img = None
        self.measure_angle_deg = 0.0
        self.update_dir_label()

        self.reset_roi(silent=True)
        self.update_display()
        self.update_window_title()
        self.refresh_list()

    def update_window_title(self):
        if not self.image_paths or self.image_index < 0:
            self.setWindowTitle("Lead Notch Analysis Tool - Professional Edition")
            return
        name = os.path.basename(self.image_paths[self.image_index])
        self.setWindowTitle(f"Lead Notch Analysis Tool - {name}  ({self.image_index+1}/{len(self.image_paths)})")

    def next_image(self):
        if not self.image_paths:
            return
        if self.image_index + 1 >= len(self.image_paths):
            return
        self.load_image_by_index(self.image_index + 1)

    def prev_image(self):
        if not self.image_paths:
            return
        if self.image_index - 1 < 0:
            return
        self.load_image_by_index(self.image_index - 1)

    # ---------- mode ----------
    def set_mode(self, mode: str):
        if mode not in ("reflect", "transmit"):
            return
        self.mode = mode

        self.btn_reflect.blockSignals(True)
        self.btn_transmit.blockSignals(True)
        self.btn_reflect.setChecked(mode == "reflect")
        self.btn_transmit.setChecked(mode == "transmit")
        self.btn_reflect.blockSignals(False)
        self.btn_transmit.blockSignals(False)

        self.act_mode_reflect.blockSignals(True)
        self.act_mode_transmit.blockSignals(True)
        self.act_mode_reflect.setChecked(mode == "reflect")
        self.act_mode_transmit.setChecked(mode == "transmit")
        self.act_mode_reflect.blockSignals(False)
        self.act_mode_transmit.blockSignals(False)

        if mode == "reflect":
            self.mode_label.setText("Current: 반사 모드 (하얀쪽이 대상)")
        else:
            self.mode_label.setText("Current: 투과 모드 (검은쪽이 대상)")

        if self.roi is not None:
            self.analyze_roi()
        self.update_display()

    # ---------- threshold ----------
    def increase_threshold(self):
        new_value = min(255, self.threshold_value + 5)
        self.threshold_slider.setValue(new_value)

    def decrease_threshold(self):
        new_value = max(0, self.threshold_value - 5)
        self.threshold_slider.setValue(new_value)

    def on_threshold_changed(self, value):
        self.threshold_value = value
        self.threshold_label.setText(f"Threshold: {value}")
        if self.roi is not None:
            self.analyze_roi()
        self.update_display()

    # ---------- zoom ----------
    def zoom_in(self):
        self.zoom = min(self.zoom_max, self.zoom + self.zoom_step)
        self.update_display()

    def zoom_out(self):
        self.zoom = max(self.zoom_min, self.zoom - self.zoom_step)
        self.update_display()

    # ---------- roi ----------
    def finalize_roi(self):
        if self.image_label.start_point is None or self.image_label.end_point is None or self.original_image is None:
            return

        img_h, img_w = self.original_image.shape[:2]

        x1 = int(self.image_label.start_point.x() / self.zoom)
        y1 = int(self.image_label.start_point.y() / self.zoom)
        x2 = int(self.image_label.end_point.x() / self.zoom)
        y2 = int(self.image_label.end_point.y() / self.zoom)

        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(0, min(y2, img_h - 1))

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            QMessageBox.warning(self, "Warning", "ROI is too small!")
            self.image_label.start_point = None
            self.image_label.end_point = None
            return

        self.roi = (x1, y1, x2, y2)
        self.analyze_roi()
        self.update_display()

        self.image_label.start_point = None
        self.image_label.end_point = None

    # ---------- analysis ----------
    def analyze_roi(self):
        if self.roi is None or self.original_image is None:
            return

        x1, y1, x2, y2 = self.roi
        gray_roi = cv2.cvtColor(self.original_image[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        invert = (self.mode == "transmit")

        if self.dir_pt1_img is not None and self.dir_pt2_img is not None:
            
            anlge_for_width = self.measure_angle_deg + 90
            self.analysis = analyze_lead_notch_with_angle(
                gray_roi, self.threshold_value, invert=invert, angle_deg=anlge_for_width
            )
        else:
            self.analysis = analyze_lead_notch(gray_roi, self.threshold_value, invert=invert)

        if self.analysis is None:
            QMessageBox.warning(self, "Warning", "Lead separation failed. Try adjusting threshold or ROI.")
            self.ref_width_label.setText("REF Width: Failed")
            self.min_width_label.setText("MIN Width: Failed")
            self.notch_ratio_label.setText("Notch Ratio: Failed")
        else:
            ref_w = self.analysis["ref_width_px"]
            min_w = self.analysis["min_width_px"]
            pct = self.analysis["notch_ratio_percent"]
            self.ref_width_label.setText(f"REF Width: {ref_w:.2f} px")
            self.min_width_label.setText(f"MIN Width: {min_w:.2f} px")
            self.notch_ratio_label.setText(f"Notch Ratio: {pct:.2f}%")

    # ---------- display ----------
    def update_display(self):
        if self.original_image is None:
            return

        display = self.original_image.copy()
        img_h, img_w = display.shape[:2]

        bytes_per_line = 3 * img_w
        q_image = QImage(display.data, img_w, img_h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # 방향선: 보라 점선 + 마커(채우기 없음) -> 내용 안가림
        if self.dir_pt1_img is not None:
            x1, y1 = self.dir_pt1_img
            painter.setPen(QPen(QColor(180, 0, 255), 1, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(int(x1) - 3, int(y1) - 3, 6, 6)

        if self.dir_pt1_img is not None and self.dir_pt2_img is not None:
            x1, y1 = self.dir_pt1_img
            x2, y2 = self.dir_pt2_img
            painter.setPen(QPen(QColor(180, 0, 255), 1, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
            painter.drawEllipse(int(x2) - 3, int(y2) - 3, 6, 6)

        # ROI 드래깅
        if self.image_label.dragging and self.image_label.start_point and self.image_label.end_point:
            x1 = int(self.image_label.start_point.x() / self.zoom)
            y1 = int(self.image_label.start_point.y() / self.zoom)
            x2 = int(self.image_label.end_point.x() / self.zoom)
            y2 = int(self.image_label.end_point.y() / self.zoom)
            painter.setPen(QPen(QColor(255, 255, 0), ROI_LINE_WIDTH_DRAGGING, Qt.DashLine))
            painter.drawRect(QRect(QPoint(x1, y1), QPoint(x2, y2)))

        # ROI 확정
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            painter.setPen(QPen(QColor(0, 255, 255), ROI_LINE_WIDTH_CONFIRMED))
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

        # 측정선
        if self.analysis is not None and self.roi is not None:
            rx1, ry1, rx2, ry2 = self.roi

            if "measure_line_roi" in self.analysis:
                (ax, ay), (bx, by) = self.analysis["measure_line_roi"]
                A = (int(rx1 + ax), int(ry1 + ay))
                B = (int(rx1 + bx), int(ry1 + by))

                painter.setPen(QPen(QColor(255, 0, 0), MEASUREMENT_LINE_WIDTH))
                painter.drawLine(A[0], A[1], B[0], B[1])

                painter.setBrush(QColor(255, 0, 0))
                painter.drawEllipse(A[0] - MEASUREMENT_MARKER_SIZE, A[1] - MEASUREMENT_MARKER_SIZE,
                                   MEASUREMENT_MARKER_SIZE * 2, MEASUREMENT_MARKER_SIZE * 2)
                painter.drawEllipse(B[0] - MEASUREMENT_MARKER_SIZE, B[1] - MEASUREMENT_MARKER_SIZE,
                                   MEASUREMENT_MARKER_SIZE * 2, MEASUREMENT_MARKER_SIZE * 2)

                min_w = self.analysis["min_width_px"]
                mid_x = (A[0] + B[0]) // 2
                mid_y = (A[1] + B[1]) // 2

                text = f"{min_w:.1f}px"
                painter.setFont(QFont("Arial", MEASUREMENT_TEXT_SIZE, QFont.Bold))
                text_rect = painter.fontMetrics().boundingRect(text)

                bg_rect = QRect(
                    mid_x - text_rect.width() // 2 - 5,
                    mid_y - MEASUREMENT_TEXT_OFFSET - text_rect.height(),
                    text_rect.width() + 10,
                    text_rect.height() + 10
                )
                painter.fillRect(bg_rect, QColor(255, 0, 0, 200))
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(bg_rect.x() + 5, bg_rect.y() + text_rect.height() + 2, text)

            else:
                y_min = self.analysis["y_min"]
                lmin = self.analysis["min_left_x"]
                rmin = self.analysis["min_right_x"]
                if lmin is not None and rmin is not None:
                    yy = ry1 + y_min
                    painter.setPen(QPen(QColor(255, 0, 0), MEASUREMENT_LINE_WIDTH))
                    painter.drawLine(rx1 + lmin, yy, rx1 + rmin, yy)

                    painter.setBrush(QColor(255, 0, 0))
                    painter.drawEllipse(rx1 + lmin - MEASUREMENT_MARKER_SIZE, yy - MEASUREMENT_MARKER_SIZE,
                                        MEASUREMENT_MARKER_SIZE * 2, MEASUREMENT_MARKER_SIZE * 2)
                    painter.drawEllipse(rx1 + rmin - MEASUREMENT_MARKER_SIZE, yy - MEASUREMENT_MARKER_SIZE,
                                        MEASUREMENT_MARKER_SIZE * 2, MEASUREMENT_MARKER_SIZE * 2)

                    min_w = self.analysis["min_width_px"]
                    mid_x = rx1 + (lmin + rmin) // 2
                    text = f"{min_w:.1f}px"
                    painter.setFont(QFont("Arial", MEASUREMENT_TEXT_SIZE, QFont.Bold))
                    text_rect = painter.fontMetrics().boundingRect(text)

                    text_y = yy - MEASUREMENT_TEXT_OFFSET
                    bg_rect = QRect(
                        mid_x - text_rect.width() // 2 - 5,
                        text_y - text_rect.height() - 5,
                        text_rect.width() + 10,
                        text_rect.height() + 10
                    )
                    painter.fillRect(bg_rect, QColor(255, 0, 0, 200))
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(mid_x - text_rect.width() // 2, text_y, text)

        painter.end()

        target_w = max(1, int(img_w * self.zoom))
        target_h = max(1, int(img_h * self.zoom))
        scaled_pixmap = pixmap.scaled(target_w, target_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())

    # ---------- reset/save ----------
    def reset_roi(self, silent=False):
        self.roi = None
        self.analysis = None
        self.ref_width_label.setText("REF Width: -")
        self.min_width_label.setText("MIN Width: -")
        self.notch_ratio_label.setText("Notch Ratio: -")

        self.direction_mode = False
        self.direction_dragging = False
        self.dir_pt1_img = None
        self.dir_pt2_img = None
        self.measure_angle_deg = 0.0
        self.update_dir_label()

        if not silent:
            self.update_display()

    def save_results(self):
        if self.roi is None or self.analysis is None or self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please select ROI and analyze first!")
            return

        img_path = self.current_image_path or DEFAULT_IMAGE_PATH
        out_path = self.label_path_for_image(img_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # timestamp는 기록용으로만 유지


        x1, y1, x2, y2 = self.roi
        img_h, img_w = self.original_image.shape[:2]
        img_path = self.current_image_path or DEFAULT_IMAGE_PATH

        dir_data = None
        if self.dir_pt1_img is not None and self.dir_pt2_img is not None:
            dir_data = {
                "p1": {"x": self.dir_pt1_img[0], "y": self.dir_pt1_img[1]},
                "p2": {"x": self.dir_pt2_img[0], "y": self.dir_pt2_img[1]},
                "angle_deg": self.measure_angle_deg
            }

        data = {
            "image_path": img_path,
            "image_size": {"w": img_w, "h": img_h},
            "roi": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "threshold_value": self.threshold_value,
            "mode": self.mode,
            "threshold_inverted": (self.mode == "transmit"),
            "zoom": self.zoom,
            "direction": dir_data,
            "ref_width_px_p90": self.analysis["ref_width_px"],
            "min_width_px_p10": self.analysis["min_width_px"],
            "notch_ratio_percent": self.analysis["notch_ratio_percent"],
            "timestamp": ts
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        #QMessageBox.information(self, "Success", f"Results saved to:\n{out_path}")
        self.refresh_list()
        if self.image_paths and self.image_index < len(self.image_paths) - 1:
            self.next_image()


    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, 'Quit',
            'Are you sure you want to quit?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = LeadNotchAnalyzer()
    window.show()

    window.raise_()
    window.activateWindow()

    sys.exit(app.exec())
