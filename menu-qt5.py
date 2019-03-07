import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic

# load the qt gui produced by qt-designer
Ui_MainWindow, QtBaseClass = uic.loadUiType('menu.ui')

class MyApp(QMainWindow):
    # initialize GUI-program connect
    def __init__(self):
        super(MyApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # define pushbutton action
        self.ui.calcButton.clicked.connect(self.Index)

    def Index(self):
        
        # d-spacings <=8
        d1 = float(self.ui.d1box.toPlainText())
        d2 = float(self.ui.d2box.toPlainText())
        d3 = float(self.ui.d3box.toPlainText())
        d4 = float(self.ui.d4box.toPlainText())
        d5 = float(self.ui.d5box.toPlainText())
        d6 = float(self.ui.d6box.toPlainText())
        d7 = float(self.ui.d7box.toPlainText())
        d8 = float(self.ui.d8box.toPlainText())
        
        # input ANN params
        num_d = int(self.ui.num_d.toPlainText())
        epochs = int(self.ui.epochs.toPlainText())
        initLR = float(self.ui.initLR.toPlainText())
        LRdecay = float(self.ui.LRdecay.toPlainText())
        modelPath = self.ui.modelPath.toPlainText()
        # operation = self.ui.operation.currentText()
        # use_q = self.ui.use_q.currentText()
        scalerPath = self.ui.scalerPath.toPlainText()
        
        # run the ANN ?
        #
        
        # output result (just testingat this point)
        lattice_a = d1 + d2
        lattice_b = d3 - d4
        lattice_gam = d5+d6+d7+d8
        a_string = str(lattice_a)
        self.ui.a_box.setText(a_string)
        b_string = str(lattice_b)
        self.ui.b_box.setText(b_string)
        gam_string = str(lattice_gam)
        self.ui.gam_box.setText(gam_string)
        error_string = scalerPath
        self.ui.error_box.setText(error_string)

# main
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())


