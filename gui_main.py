import sys

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import *

import gui_control

if __name__ == "__main__":
    QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    MainWindow = gui_control.MyWindow()

    MainWindow.show()
    sys.exit(app.exec())