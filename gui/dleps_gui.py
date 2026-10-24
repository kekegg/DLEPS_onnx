#!/usr/bin/env python
########################################################
# All rights reserved.
# Author: XIE Zhengwei @ Beijing Gigaceuticals Tech Co., Ltd
#                      @ Peking University International Cancer Institute
# Contact: xiezhengwei@gmail.com
#
#
########################################################

"""PySide6 desktop client for DLEPS (ONNX build).

The option set mirrors the DLEPS tab of the AIDock client
(``AIDock_client/index.html`` + ``render.js``):

    Reverse the UP / Down genes ......... ifReverse
    Set Mean ........................... ifSetMean
    Use absolute value of ES score ..... ifAbs
    Gene Correlation Cutoff ............ geneCorr
    Save / Use onehot array ............ ifSaveUseOneHot
    If use the transformer model ........ modelType   (not shipped here -- disabled)

On top of that it offers the disease presets of ``data/diseases.json``, which
are the local replacement for the AIDock server's own disease handling.

Everything runs locally: the DLEPS graph is executed by ONNX Runtime, no
TensorFlow, no API key, nothing leaves the machine.

    python gui/dleps_gui.py
"""

import csv
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import (QObject, QThread, Qt, QTimer, Signal, Slot)
from PySide6.QtGui import QAction, QFont, QIcon
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QRadioButton, QScrollArea, QSpinBox,
                               QSplitter, QStackedWidget, QStatusBar, QTableWidget,
                               QTableWidgetItem, QTabWidget, QTextBrowser,
                               QVBoxLayout, QWidget)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from dleps_score import (EFFECTIVE_THRESHOLD, DlepsScorer, check_genes,  # noqa: E402
                         load_diseases, read_signature, signature_direction)

SETTINGS_FILE = Path(os.path.expanduser('~/.dleps_onnx_gui.json'))

#: Rows of the result table.
COL_SMILES, COL_SCORE, COL_EFFECTIVE, COL_STD = range(4)


# --------------------------------------------------------------------------
# stdout capture -> log pane
# --------------------------------------------------------------------------

class _StreamRelay(QObject):
    """Forward ``sys.stdout`` writes to the GUI log pane."""

    message = Signal(str)

    def write(self, text):
        if text:
            self.message.emit(text)
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


# --------------------------------------------------------------------------
# scoring worker
# --------------------------------------------------------------------------

class ScoreWorker(QObject):
    """Runs the DLEPS scoring off the GUI thread."""

    done = Signal(list, dict)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, options):
        super().__init__()
        self.options = options

    @Slot()
    def run(self):
        opts = self.options
        try:
            self.progress.emit('正在加载 ONNX 模型 …')
            scorer = DlepsScorer()
            self.progress.emit('正在打分 …')

            smiles = opts['smiles']
            if opts['source'] == 'disease':
                results = scorer.score(
                    smiles, disease_name=opts['disease'],
                    threshold=opts['threshold'], n_repeats=opts['n_repeats'],
                    reverse=opts['reverse'],
                    setmean=opts['setmean'], gene_corr=opts['gene_corr'],
                    abs_score=opts['abs_score'],
                    save_onehot=opts['save_onehot'],
                    load_onehot=opts['load_onehot'])
            else:
                up_file = opts['up_file']
                down_file = opts['down_file']
                results = scorer.score(
                    smiles, threshold=opts['threshold'],
                    n_repeats=opts['n_repeats'],
                    up_file=up_file, down_file=down_file,
                    reverse=opts['reverse'],
                    setmean=opts['setmean'], gene_corr=opts['gene_corr'],
                    abs_score=opts['abs_score'],
                    save_onehot=opts['save_onehot'],
                    load_onehot=opts['load_onehot'])

            meta = {
                'n': len(results),
                'n_effective': sum(1 for r in results if r['effective']),
                'threshold': opts['threshold'],
                'direction': 'reverse（逆转疾病状态）' if opts['reverse'] else 'mimic（模拟表型 / 棕色脂肪）',
                'disease': opts.get('disease') or '自定义签名',
                'expr_std': results[0]['expr_std'] if results else 0.0,
            }
            self.done.emit(results, meta)
        except Exception:
            self.failed.emit(traceback.format_exc())


# --------------------------------------------------------------------------
# main window
# --------------------------------------------------------------------------

class DlepsWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('DLEPS — 本地 ONNX 药效预测')
        self.resize(1400, 900)

        self._thread = None
        self._worker = None
        self._relay = _StreamRelay()
        self._relay.message.connect(self.append_log)

        self._build_ui()
        self.load_settings()
        self._populate_diseases()
        self.append_log('就绪。模型：%s' % (REPO_ROOT / 'data' / 'onnx'))

    # -- construction ------------------------------------------------------

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([560, 840])
        self.setCentralWidget(splitter)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage('就绪')

        menu = self.menuBar().addMenu('文件')
        act_export = QAction('导出结果 CSV…', self)
        act_export.triggered.connect(self.export_csv)
        menu.addAction(act_export)
        act_save = QAction('保存当前设置', self)
        act_save.triggered.connect(self.save_settings)
        menu.addAction(act_save)
        act_load = QAction('读取已保存设置', self)
        act_load.triggered.connect(self.load_settings)
        menu.addAction(act_load)

    def _build_left(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(520)
        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setSpacing(10)

        col.addWidget(self._build_disease_box())
        col.addWidget(self._build_option_box())
        col.addWidget(self._build_input_box())

        self.run_btn = QPushButton('运行 DLEPS')
        self.run_btn.setMinimumHeight(42)
        self.run_btn.setStyleSheet('QPushButton { font-size: 15px; font-weight: 600; }')
        self.run_btn.clicked.connect(self.run_scoring)
        col.addWidget(self.run_btn)

        col.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _build_disease_box(self):
        box = QGroupBox('疾病预设（本地签名，无需 API key）')
        lay = QVBoxLayout(box)

        self.disease_combo = QComboBox()
        self.disease_combo.currentIndexChanged.connect(self._on_disease_changed)
        lay.addWidget(self.disease_combo)

        self.disease_info = QLabel()
        self.disease_info.setWordWrap(True)
        self.disease_info.setStyleSheet('color: #444;')
        lay.addWidget(self.disease_info)
        return box

    def _build_option_box(self):
        box = QGroupBox('DLEPS 选项')
        grid = QGridLayout(box)

        self.chk_reverse = QCheckBox('Reverse the UP / Down genes')
        self.chk_reverse.setToolTip('勾选：逆转疾病状态；不勾选：模拟该表型（如诱导棕色脂肪）')
        self.chk_reverse.setChecked(True)
        self.chk_reverse.toggled.connect(self._update_direction_hint)
        grid.addWidget(self.chk_reverse, 0, 0)

        self.chk_setmean = QCheckBox('Set Mean')
        self.chk_setmean.setToolTip('用当前输入分子的 RNA 水平集合均值做中心化（输出不平衡时用）')
        grid.addWidget(self.chk_setmean, 0, 1)

        self.chk_abs = QCheckBox('Use absolute value of ES score')
        self.chk_abs.setToolTip('不考虑 ES 分数的正负号')
        grid.addWidget(self.chk_abs, 1, 0)

        self.chk_onehot = QCheckBox('Save / Use onehot array')
        self.chk_onehot.setToolTip('保存输入 SMILES 的 one-hot 数组，重复运行时直接复用以提速')
        self.chk_onehot.setChecked(True)
        grid.addWidget(self.chk_onehot, 1, 1)

        self.chk_transformer = QCheckBox('If use the transformer model')
        self.chk_transformer.setToolTip('AG-BT 指纹的 transformer 模型：权重未随本仓库发布')
        self.chk_transformer.setEnabled(False)
        grid.addWidget(self.chk_transformer, 2, 0)

        lbl_corr = QLabel('Gene Correlation Cutoff')
        lbl_corr.setToolTip('0 – 0.8；0 = 使用全部基因，0.68376 = 只用预测得好的基因')
        self.spin_gene_corr = QDoubleSpinBox()
        self.spin_gene_corr.setRange(0.0, 0.8)
        self.spin_gene_corr.setSingleStep(0.05)
        self.spin_gene_corr.setDecimals(5)
        self.spin_gene_corr.setValue(0.0)
        grid.addWidget(self._pair(lbl_corr, self.spin_gene_corr), 2, 1)

        lbl_rep = QLabel('重复次数 n_repeats')
        lbl_rep.setToolTip('模型头是随机采样（eps ~ N(0, 0.01)）。多次运行取表达量平均可消除抖动，筛选决策建议 3–5')
        self.spin_repeats = QSpinBox()
        self.spin_repeats.setRange(1, 50)
        self.spin_repeats.setValue(3)
        grid.addWidget(self._pair(lbl_rep, self.spin_repeats), 3, 0)

        lbl_thr = QLabel('有效阈值 (cs_score >)')
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(-10.0, 10.0)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setDecimals(3)
        self.spin_threshold.setValue(EFFECTIVE_THRESHOLD)
        grid.addWidget(self._pair(lbl_thr, self.spin_threshold), 3, 1)

        self.direction_hint = QLabel()
        self.direction_hint.setWordWrap(True)
        self.direction_hint.setStyleSheet('color: #1B4FA0;')
        grid.addWidget(self.direction_hint, 4, 0, 1, 2)
        return box

    @staticmethod
    def _pair(label, widget):
        """Put a label and its editor on one row so they cannot drift apart."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(label)
        h.addStretch(1)
        h.addWidget(widget)
        return row

    def _build_input_box(self):
        box = QGroupBox('输入')
        outer = QVBoxLayout(box)

        # -- SMILES: always visible, text or CSV file
        lay_s = QVBoxLayout()
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel('SMILES'))
        self.edit_smiles_file = QLineEdit()
        self.edit_smiles_file.setPlaceholderText('留空则使用下方文本框')
        h.addWidget(self.edit_smiles_file, 1)
        btn = QPushButton('浏览 CSV…')
        btn.clicked.connect(lambda: self._pick_file(self.edit_smiles_file))
        h.addWidget(btn)
        lay_s.addWidget(row)

        self.txt_smiles = QPlainTextEdit()
        self.txt_smiles.setPlaceholderText('CC(=O)Oc1ccccc1C(=O)O\nN[C@@H](Cc1ccc(O)cc1)C(=O)O')
        self.txt_smiles.setMinimumHeight(96)
        lay_s.addWidget(self.txt_smiles)
        outer.addLayout(lay_s)

        # -- signature source
        self.source_combo = QComboBox()
        self.source_combo.addItem('基因来源：疾病预设', 'disease')
        self.source_combo.addItem('基因来源：文本输入', 'text')
        self.source_combo.addItem('基因来源：文件输入', 'files')
        self.source_combo.currentIndexChanged.connect(
            lambda: self.stack.setCurrentIndex(self.source_combo.currentIndex()))
        outer.addWidget(self.source_combo)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack)

        # -- page 0: disease preset
        page0 = QWidget()
        lay0 = QVBoxLayout(page0)
        lay0.addWidget(QLabel('使用「疾病预设」中选定的签名打分；SMILES 取自上方。'))
        lay0.addStretch(1)
        self.stack.addWidget(page0)

        # -- page 1: text
        page1 = QWidget()
        lay1 = QVBoxLayout(page1)
        g = QGridLayout()
        g.addWidget(QLabel('Up genes (ENTREZGENE_ACC)'), 0, 0)
        self.txt_up = QPlainTextEdit()
        self.txt_up.setPlaceholderText('2289\n111\n6515')
        g.addWidget(self.txt_up, 1, 0)
        g.addWidget(QLabel('Down genes (ENTREZGENE_ACC)'), 0, 1)
        self.txt_down = QPlainTextEdit()
        self.txt_down.setPlaceholderText('133060\n1149\n7350')
        g.addWidget(self.txt_down, 1, 1)
        lay1.addLayout(g)

        btn_check = QPushButton('检查基因（看哪些在 12328 基因表内）')
        btn_check.clicked.connect(self.check_genes_now)
        lay1.addWidget(btn_check)

        self.gene_report = QTextBrowser()
        self.gene_report.setMaximumHeight(160)
        lay1.addWidget(self.gene_report)
        self.stack.addWidget(page1)

        # -- page 2: files
        page2 = QWidget()
        lay2 = QVBoxLayout(page2)
        self.edit_up_file = self._file_row(lay2, 'Up 基因文件')
        self.edit_down_file = self._file_row(lay2, 'Down 基因文件')
        lay2.addStretch(1)
        self.stack.addWidget(page2)
        return box

    def _file_row(self, layout, label):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel(label))
        edit = QLineEdit()
        h.addWidget(edit, 1)
        btn = QPushButton('浏览…')
        btn.clicked.connect(lambda: self._pick_file(edit))
        h.addWidget(btn)
        layout.addWidget(row)
        return edit

    def _pick_file(self, edit):
        path, _ = QFileDialog.getOpenFileName(self, '选择文件', '', '所有文件 (*);;CSV (*.csv);;文本 (*.txt)')
        if path:
            edit.setText(path)

    def _build_right(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['SMILES', 'cs_score', '有效', 'expr_std'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in (1, 2, 3):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        lay.addWidget(self.table, 3)

        self.chart = QLabel('结果图（打分后显示）')
        self.chart.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chart.setMinimumHeight(220)
        self.chart.setStyleSheet('background: #FAFAFA; border: 1px solid #DDD;')
        lay.addWidget(self.chart, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setStyleSheet('font-family: Menlo, monospace; font-size: 11px;')
        lay.addWidget(self.log, 1)
        return page

    # -- population --------------------------------------------------------

    def _populate_diseases(self):
        try:
            diseases = load_diseases()
        except Exception:
            diseases = {}
        for key, spec in diseases.items():
            direction = 'reverse' if signature_direction(spec) else 'mimic'
            self.disease_combo.addItem('%s  [%s]' % (key, direction), key)
        self._on_disease_changed()

    def _on_disease_changed(self):
        key = self.disease_combo.currentData()
        if not key:
            return
        try:
            diseases = load_diseases()
        except Exception:
            return
        spec = diseases.get(key)
        if not spec:
            return
        direction = 'reverse' if signature_direction(spec) else 'mimic'
        self.disease_info.setText('%s\n签名：%s%s' % (
            spec.get('label', ''),
            spec.get('up', ''),
            (' / ' + spec['down']) if spec.get('down') else '（无下调集）'))
        self.disease_info.setToolTip(spec.get('note', ''))
        # Adopt the declared direction; the checkbox stays editable so the
        # user can override it.
        self.chk_reverse.blockSignals(True)
        self.chk_reverse.setChecked(signature_direction(spec))
        self.chk_reverse.blockSignals(False)
        self._update_direction_hint(direction)

        data_dir = REPO_ROOT / 'data'
        up = data_dir / spec.get('up', '')
        down = data_dir / spec.get('down', '') if spec.get('down') else None
        self.edit_up_file.setText(str(up) if up.exists() else '')
        self.edit_down_file.setText(str(down) if down and down.exists() else '')
        self._update_direction_hint()

    def _update_direction_hint(self, declared=None):
        if self.chk_reverse.isChecked():
            text = '当前方向：<b>reverse</b> —— 正分表示化合物<b>逆转</b>该疾病状态。'
        else:
            text = '当前方向：<b>mimic</b> —— 正分表示化合物<b>模拟</b>该表型（如诱导棕色脂肪）。'
        if declared:
            text += '<br>疾病预设声明的方向：<b>%s</b>%s' % (
                declared, '（已被手动覆盖）' if
                (declared == 'reverse') != self.chk_reverse.isChecked() else '')
        self.direction_hint.setText(text)

    # -- actions -----------------------------------------------------------

    def append_log(self, text):
        self.log.appendPlainText(text.rstrip())
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _collect_smiles(self):
        """Return the SMILES list: the CSV file if one is given, else the text box."""
        path = self.edit_smiles_file.text().strip()
        if path:
            if not Path(path).exists():
                raise ValueError('SMILES 文件不存在：%s' % path)
            raw = Path(path).read_text(encoding='utf-8', errors='replace')
        else:
            raw = self.txt_smiles.toPlainText()

        lines = [ln for ln in raw.splitlines()
                 if ln.strip() and not ln.strip().startswith('#')]
        if not lines:
            return []

        # A CSV may carry the SMILES in a named column ("string SMILES",
        # "SMILES", ...); a plain list is one SMILES per line.
        first = [c.strip() for c in lines[0].split(',')]
        header_idx = None
        if len(first) > 1:
            for i, cell in enumerate(first):
                if 'smiles' in cell.lower():
                    header_idx = i
                    break
        if header_idx is not None:
            lines = lines[1:]
            out = []
            for ln in lines:
                cells = [c.strip() for c in ln.split(',')]
                if len(cells) > header_idx and cells[header_idx]:
                    out.append(cells[header_idx])
            return out

        out = []
        for ln in lines:
            out.append(ln.split(',')[0].strip())
        return out

    def _collect_gene_text(self):
        up = [l.strip() for l in self.txt_up.toPlainText().split() if l.strip()]
        down = [l.strip() for l in self.txt_down.toPlainText().split() if l.strip()]
        return up, down

    def check_genes_now(self):
        up, down = self._collect_gene_text()
        if not up and not down:
            self.append_log('没有输入基因 ID')
            return
        up_ok, up_bad = check_genes(up)
        down_ok, down_bad = check_genes(down)
        self.gene_report.setHtml(
            '<table width="100%" cellspacing="0">'
            '<tr><th align="left">Up genes（在 12328 表内）</th>'
            '<th align="left">NOT in the list</th>'
            '<th align="left">Down genes（在 12328 表内）</th>'
            '<th align="left">NOT in the list</th></tr>'
            '<tr><td valign="top">' + self._fmt_genes(up_ok) + '</td>'
            '<td valign="top" style="color:#ad1515">' + self._fmt_genes(up_bad) + '</td>'
            '<td valign="top">' + self._fmt_genes(down_ok) + '</td>'
            '<td valign="top" style="color:#ad1515">' + self._fmt_genes(down_bad) +
            '</td></tr></table>')
        self.append_log('基因检查：Up %d/%d 命中，Down %d/%d 命中'
                        % (len(up_ok), len(up), len(down_ok), len(down)))

    @staticmethod
    def _fmt_genes(genes):
        if not genes:
            return '<i>—</i>'
        return ', '.join(str(g) for g in genes[:400]) + (
            '<br>…共 %d 个' % len(genes) if len(genes) > 400 else '')

    def _write_text_signatures(self):
        """Materialise the text-mode gene lists into temporary signature files."""
        import tempfile
        up, down = self._collect_gene_text()
        if not up:
            raise ValueError('请填写 Up genes')
        paths = []
        for genes in (up, down):
            if not genes:
                paths.append(None)
                continue
            tmp = tempfile.NamedTemporaryFile('w', suffix='_genes.txt',
                                               prefix='dleps_gui_', delete=False,
                                               encoding='utf-8')
            tmp.write('\n'.join(str(g) for g in genes) + '\n')
            tmp.close()
            paths.append(tmp.name)
        return paths[0], paths[1]

    def run_scoring(self):
        if self._thread is not None:
            QMessageBox.information(self, '正在运行', '上一次计算尚未结束。')
            return
        source = self.source_combo.currentData()
        try:
            smiles = self._collect_smiles()
        except Exception as e:
            QMessageBox.warning(self, '输入有误', str(e))
            return
        if not smiles:
            QMessageBox.warning(self, '输入有误', '没有可打分的 SMILES。')
            return

        up_file = down_file = None
        tmp_files = []
        try:
            if source == 'text':
                up_file, down_file = self._write_text_signatures()
                tmp_files = [p for p in (up_file, down_file) if p]
                if down_file is None:
                    raise ValueError('Down genes 不能为空（文本模式）')
            elif source == 'files':
                up_file = self.edit_up_file.text().strip()
                down_file = self.edit_down_file.text().strip()
                if not up_file or not down_file:
                    raise ValueError('请同时选择 Up / Down 基因文件')
                if not Path(up_file).exists() or not Path(down_file).exists():
                    raise ValueError('基因文件不存在')
        except Exception as e:
            QMessageBox.warning(self, '输入有误', str(e))
            for p in tmp_files:
                try:
                    os.unlink(p)
                except OSError:
                    pass
            return

        onehot_dir = Path(os.path.expanduser('~/.cache/dleps/onehot'))
        onehot_dir.mkdir(parents=True, exist_ok=True)
        # Key the one-hot cache on the SMILES content, not on a file name: the
        # upstream DLEPS reuses the cache whenever the input *path* matches,
        # which silently scores a changed molecule set against a stale
        # encoding.  The content hash makes the cache safe.
        key = hashlib.sha256('\n'.join(smiles).encode('utf-8')).hexdigest()[:16]
        onehot_stem = onehot_dir / ('gui_onehot_%s' % key)
        onehot_npy = Path(str(onehot_stem) + '.npy')
        use_onehot = self.chk_onehot.isChecked()
        # np.save() appends '.npy' itself, so the stem must not carry it.
        save_onehot = str(onehot_stem) if use_onehot else None
        load_onehot = str(onehot_npy) if (use_onehot and onehot_npy.exists()) else None

        opts = {
            'source': source,
            'disease': self.disease_combo.currentData() if source == 'disease' else None,
            'smiles': smiles,
            'up_file': up_file,
            'down_file': down_file,
            'reverse': self.chk_reverse.isChecked(),
            'setmean': self.chk_setmean.isChecked(),
            'abs_score': self.chk_abs.isChecked(),
            'gene_corr': self.spin_gene_corr.value(),
            'n_repeats': self.spin_repeats.value(),
            'threshold': self.spin_threshold.value(),
            'save_onehot': save_onehot,
            'load_onehot': load_onehot,
        }

        self.run_btn.setEnabled(False)
        self.status.showMessage('正在计算 …')
        self.append_log('── 运行 DLEPS：%d 个分子，来源=%s，方向=%s，n_repeats=%d'
                        % (len(smiles), source,
                           'reverse' if opts['reverse'] else 'mimic', opts['n_repeats']))

        self._thread = QThread(self)
        self._worker = ScoreWorker(opts)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.progress.connect(self.status.showMessage)
        self._thread.start()

        # keep the temporary signature files alive until the run finished
        self._tmp_files = tmp_files

    @Slot(list, dict)
    def _on_done(self, results, meta):
        self._cleanup_thread()
        self.run_btn.setEnabled(True)
        self._show_results(results)
        self.status.showMessage('完成：%d 个分子，%d 个超过阈值 %.2f'
                                % (meta['n'], meta['n_effective'], meta['threshold']))
        self.append_log('完成：%d/%d 超过阈值 %.2f（方向：%s，签名：%s，jitter=%.4f）'
                        % (meta['n_effective'], meta['n'], meta['threshold'],
                           meta['direction'], meta['disease'], meta['expr_std']))

    @Slot(str)
    def _on_failed(self, message):
        self._cleanup_thread()
        self.run_btn.setEnabled(True)
        self.status.showMessage('出错了')
        self.append_log(message)
        QMessageBox.critical(self, '运行失败', message.strip().splitlines()[-1]
                              if message.strip() else '未知错误')

    def _cleanup_thread(self):
        for p in getattr(self, '_tmp_files', []) or []:
            try:
                os.unlink(p)
            except OSError:
                pass
        self._tmp_files = []
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
            self._worker = None

    # -- results -----------------------------------------------------------

    def _show_results(self, results):
        rows = sorted(results, key=lambda r: r['cs_score'], reverse=True)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            item = QTableWidgetItem(r['smiles'])
            item.setToolTip(r['smiles'])
            self.table.setItem(i, COL_SMILES, item)
            score = QTableWidgetItem('%.4f' % r['cs_score'])
            score.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if r['effective']:
                score.setForeground(Qt.GlobalColor.darkGreen)
                score.setToolTip('超过阈值')
            self.table.setItem(i, COL_SCORE, score)
            eff = QTableWidgetItem('✓' if r['effective'] else '')
            eff.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, COL_EFFECTIVE, eff)
            std = QTableWidgetItem('%.4f' % r['expr_std'])
            std.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(i, COL_STD, std)
        self.table.resizeColumnsToContents()
        self._draw_chart(rows)

    def _draw_chart(self, rows):
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception as e:
            self.chart.setText('matplotlib 不可用（%s）' % e)
            return
        try:
            fig = Figure(figsize=(6.4, 2.4), dpi=100)
            # Recompute the layout on every draw, so the axis labels stay
            # visible however the widget is squeezed.
            fig.set_layout_engine('tight')
            ax = fig.add_subplot(111)
            scores = [r['cs_score'] for r in rows]
            ax.hist(scores, bins=min(40, max(5, len(scores))), color='#4C8BF5',
                    edgecolor='white')
            ax.axvline(0.0, color='#888', linewidth=1)
            ax.set_xlabel('cs_score')
            ax.set_ylabel('molecules')
            ax.set_title('%d molecules, %d above threshold'
                         % (len(rows), sum(1 for r in rows if r['effective'])),
                         fontsize=9)
            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumHeight(200)
            container = QWidget()
            lay = QVBoxLayout(container)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(canvas)
            old = self.chart.parent()
            self.chart.setVisible(False)
            if old is not None:
                old.layout().replaceWidget(self.chart, container)
            self.chart = container
        except Exception as e:
            self.chart.setText('绘图失败：%s' % e)

    def export_csv(self):
        if self.table.rowCount() == 0:
            QMessageBox.information(self, '导出', '还没有结果。')
            return
        path, _ = QFileDialog.getSaveFileName(self, '导出结果', 'dleps_scores.csv',
                                              'CSV (*.csv)')
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['SMILES', 'cs_score', 'effective', 'expr_std'])
            for i in range(self.table.rowCount()):
                w.writerow([self.table.item(i, c).text()
                            for c in (COL_SMILES, COL_SCORE, COL_EFFECTIVE, COL_STD)])
        self.append_log('已导出 %s' % path)
        self.status.showMessage('已导出 %s' % path)

    # -- settings ----------------------------------------------------------

    def _settings_dict(self):
        return {
            'disease': self.disease_combo.currentData(),
            'source': self.source_combo.currentData(),
            'reverse': self.chk_reverse.isChecked(),
            'setmean': self.chk_setmean.isChecked(),
            'abs': self.chk_abs.isChecked(),
            'onehot': self.chk_onehot.isChecked(),
            'gene_corr': self.spin_gene_corr.value(),
            'n_repeats': self.spin_repeats.value(),
            'threshold': self.spin_threshold.value(),
            'smiles': self.txt_smiles.toPlainText(),
            'up': self.txt_up.toPlainText(),
            'down': self.txt_down.toPlainText(),
            'smiles_file': self.edit_smiles_file.text(),
            'up_file': self.edit_up_file.text(),
            'down_file': self.edit_down_file.text(),
        }

    def save_settings(self):
        SETTINGS_FILE.write_text(json.dumps(self._settings_dict(), indent=2,
                                            ensure_ascii=False), encoding='utf-8')
        self.append_log('设置已保存到 %s' % SETTINGS_FILE)
        self.status.showMessage('设置已保存')

    def load_settings(self):
        if not SETTINGS_FILE.exists():
            return
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding='utf-8'))
        except Exception as e:
            self.append_log('读取设置失败：%s' % e)
            return
        idx = self.disease_combo.findData(data.get('disease'))
        if idx >= 0:
            self.disease_combo.setCurrentIndex(idx)
        for i in range(self.source_combo.count()):
            if self.source_combo.itemData(i) == data.get('source'):
                self.source_combo.setCurrentIndex(i)
        self.chk_reverse.setChecked(bool(data.get('reverse', True)))
        self.chk_setmean.setChecked(bool(data.get('setmean', False)))
        self.chk_abs.setChecked(bool(data.get('abs', False)))
        self.chk_onehot.setChecked(bool(data.get('onehot', True)))
        self.spin_gene_corr.setValue(float(data.get('gene_corr', 0.0)))
        self.spin_repeats.setValue(int(data.get('n_repeats', 3)))
        self.spin_threshold.setValue(float(data.get('threshold', EFFECTIVE_THRESHOLD)))
        self.txt_smiles.setPlainText(data.get('smiles', ''))
        self.txt_up.setPlainText(data.get('up', ''))
        self.txt_down.setPlainText(data.get('down', ''))
        self.edit_smiles_file.setText(data.get('smiles_file', ''))
        self.edit_up_file.setText(data.get('up_file', ''))
        self.edit_down_file.setText(data.get('down_file', ''))
        self.append_log('已读取设置 %s' % SETTINGS_FILE)

    def closeEvent(self, event):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)


def main(argv=None):
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName('DLEPS')
    app.setFont(QFont('Helvetica Neue', 12))

    window = DlepsWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
