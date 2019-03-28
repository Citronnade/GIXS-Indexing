import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import uic

import torch
import numpy as np
from sklearn.externals import joblib

import index
import deep_d



# load the qt gui produced by qt-designer
Ui_MainWindow, QtBaseClass = uic.loadUiType('menu.ui')

class MyApp(QMainWindow):
    # initialize GUI-program connect
    def __init__(self):
        super(MyApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # define pushbutton action
        self.ui.calcButton.clicked.connect(self.go)
        self.ui.modelPathButton.clicked.connect(self.Getfile(self.ui.modelPath, "save", self))
        self.ui.scalerButton.clicked.connect(self.Getfile( self.ui.scalerPath, "save", self))


    def go(self):
        try:
            operation = self.ui.operation.currentText() # get combobox type
            self.ui.error_toast.clear()
            if operation == "train":
                self.train()
            elif operation == "index":
                self.Index()
            elif operation == "evaluate":
                pass
            else:
                raise Exception
        except Exception as e:
            self.ui.error_toast.setText("Error occurred: {}".format(repr(e)))
            print(e)
    class Getfile:
        # puts the file path into a textedit once clicking a button
        def __init__(self, textbox, dialog_type, app):
            self.dialog_type = dialog_type
            self.textbox = textbox
            self.app = app
        def __call__(self):
            if self.dialog_type == "open":
                # no warning when clicking existing file
                method = QFileDialog.getOpenFileName
            else:
                # warning when clicking existing file
                method = QFileDialog.getSaveFileName
            filename, _ = method(self.app, "Path", "", "All files (*)")
            self.textbox.setText(filename)
            return filename

    def train(self):
        num_d = int(self.ui.num_d.toPlainText())
        epochs = int(self.ui.epochs.toPlainText())
        initLR = float(self.ui.initLR.toPlainText())
        LRdecay = float(self.ui.LRdecay.toPlainText())
        batch_size = int(self.ui.BatchSize.toPlainText())
        modelPath = self.ui.modelPath.text()
        if not modelPath:
            self.ui.error_toast.setText("Please enter a path to save the model path file in.")
            return

        # operation = self.ui.operation.currentText()
        use_q = self.ui.use_q.currentText() == "yes"
        scalerPath = self.ui.scalerPath.text()
        if not scalerPath:
            self.ui.error_toast.setText("Please enter a path to save the scaler file in.")
            return
        deep_d.train_model(num_epochs=epochs, path=modelPath, gamma_scheduler=LRdecay, batch_size=batch_size, use_qs=use_q, lr=initLR, num_spacings=num_d, scaler_path=scalerPath)

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
        batch_size = float(self.ui.BatchSize.toPlainText())
        #modelPath = self.ui.modelPath.toPlainText()
        
        # operation = self.ui.operation.currentText()
        #use_q = self.ui.use_q.currentText()
        scalerPath = self.ui.scalerPath.text()
        
        # run the ANN ?
        try:
            scaler = joblib.load(scalerPath)
        except FileNotFoundError as e:
            print(e)
            print(e.filename)
            self.ui.error_toast.setText("Scaler file not found: {}".format(e.filename))
            return
        model = deep_d.SimpleNet()
        path = self.ui.modelPath.text()
        try:
            model.load_state_dict(torch.load(path))
        except FileNotFoundError as e:
            self.ui.error_toast.setText("Model path file not found: {}".format(e.filename))
            return
        result, percent_error = index.index(np.array([d1, d2, d3, d4, d5, d6, d7, d8]).reshape(1,-1), model, scaler=scaler)
        # results i an array of shape (,3)
        print(result)
        # output result 
        lattice_a = result[0]
        lattice_b = result[1]
        lattice_gam = result[2]
        a_string = str(lattice_a)
        self.ui.a_box.setText(a_string)
        b_string = str(lattice_b)
        self.ui.b_box.setText(b_string)
        gam_string = str(lattice_gam)
        self.ui.gam_box.setText(gam_string)
        error_string = str(percent_error)
        self.ui.error_box.setText(error_string)
# main
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())


