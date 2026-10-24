"""End-to-end GUI test: run a real scoring through the PySide6 window.

Runs the window on the offscreen platform, picks a disease, feeds SMILES,
presses Run, waits for the worker thread and saves a screenshot.  The run is
repeated so the one-hot cache round trip is exercised as well.

Standalone::

    python gui/test_gui_e2e.py

or with pytest::

    pytest gui/test_gui_e2e.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

import dleps_gui

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CCO",
    "O=C(O)c1ccccc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "c1ccccc1",
]

OUT = Path("/tmp/dleps_onnx/gui_shot.png")


def wait_for_run(win, app, timeout_ms=120000):
    """Run the event loop until the worker thread is cleaned up."""
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(50)

    def poll():
        if win._thread is None:
            loop.quit()

    timer.timeout.connect(poll)
    timer.start()
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    timer.stop()
    app.processEvents()


def table_rows(win):
    return [[win.table.item(i, c).text() for c in range(4)]
            for i in range(win.table.rowCount())]


def main():
    # Keep the run headless: a modal error box would block forever.
    QMessageBox.critical = staticmethod(lambda *a, **k: print('ERROR:', a[-1], flush=True))
    QMessageBox.warning = staticmethod(lambda *a, **k: print('WARN:', a[-1], flush=True))

    app = QApplication([])
    win = dleps_gui.DlepsWindow()
    win.resize(1400, 900)
    win.show()

    idx = win.disease_combo.findData("Gout")
    assert idx >= 0, "Gout not in the disease list"
    win.disease_combo.setCurrentIndex(idx)
    win.source_combo.setCurrentIndex(0)          # signature source: disease preset
    win.txt_smiles.setPlainText("\n".join(SMILES))
    win.spin_repeats.setValue(3)
    app.processEvents()

    print('--- run 1 (cold: computes and caches the one-hot) ---', flush=True)
    win.run_scoring()
    wait_for_run(win, app)
    rows1 = table_rows(win)
    print('rows:', len(rows1), flush=True)
    for r in rows1[:4]:
        print('   ', r, flush=True)
    assert len(rows1) == len(SMILES), 'expected %d rows, got %d' % (len(SMILES), len(rows1))

    print('--- run 2 (warm: reuses the one-hot cache) ---', flush=True)
    win.run_scoring()
    wait_for_run(win, app)
    rows2 = table_rows(win)
    assert rows1 == rows2, 'cached run differs:\n%s\n%s' % (rows1, rows2)
    print('cached run identical: True', flush=True)

    print('--- text mode: custom gene lists + "check the genes" ---', flush=True)
    win.source_combo.setCurrentIndex(1)          # signature source: text
    win.txt_up.setPlainText('3934 4053 718 999999999')      # last one is not a gene
    win.txt_down.setPlainText('731 27232 888888888')
    win.check_genes_now()
    report = win.gene_report.toPlainText()
    print('gene report:', report.replace('\n', ' ')[:200], flush=True)
    assert '999999999' in report and '888888888' in report, \
        'the missing genes should be listed'
    assert '3934' in report, 'the known genes should be listed'

    win.run_scoring()
    wait_for_run(win, app)
    rows3 = table_rows(win)
    print('text-mode rows:', len(rows3), flush=True)
    for r in rows3[:3]:
        print('   ', r, flush=True)
    assert len(rows3) == len(SMILES), 'text mode returned %d rows' % len(rows3)

    app.processEvents()
    win.grab().save(str(OUT))
    print('screenshot ->', OUT, OUT.exists(), flush=True)

    print('--- log tail ---', flush=True)
    for line in win.log.toPlainText().strip().splitlines()[-6:]:
        print('   ', line, flush=True)
    return 0


def test_gui_end_to_end():
    """pytest entry point."""
    assert main() == 0


if __name__ == '__main__':
    sys.exit(main())
