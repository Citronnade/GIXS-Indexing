import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import uic

import torch
import numpy as np
from sklearn.externals import joblib
import traceback

import index
import deep_d
import evaluate


# load the qt gui produced by qt-designer
Ui_MainWindow, QtBaseClass = uic.loadUiType('menu.ui')

def format_decimal(s):
    return "{0:.6g}".format(s)


class MyApp(QMainWindow):
    # initialize GUI-program connect
    def __init__(self):
        super(MyApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # define pushbutton action
        self.ui.calcButton.clicked.connect(self.go)
        self.ui.modelPathButton.clicked.connect(self.Getfile(self.ui.modelPath, self))
        self.ui.scalerButton.clicked.connect(self.Getfile( self.ui.scalerPath, self))
        self.ui.operation.currentIndexChanged.connect(self.set_operation)
        self.ui.num_d.textChanged.connect(self.set_num_ds)
        self.ANN_params = [self.ui.num_d, self.ui.epochs, self.ui.initLR, self.ui.LRdecay, self.ui.BatchSize]
        self.d_boxes = [self.ui.d1box, self.ui.d2box, self.ui.d3box, self.ui.d4box, self.ui.d5box, self.ui.d6box, self.ui.d7box, self.ui.d8box]
        self.ANN_outputs = [self.ui.a_box, self.ui.b_box, self.ui.gam_box, self.ui.error_box]
        self.set_operation(0) # 0 is train, 1 is index, 2 is evaluate

    def emphasize(self, widget):
        widget.setStyleSheet(
            """
            QTextEdit {
            border-style: outset; 
            border-width: 2px; 
            border-color: green}""")

    def deemphasize(self, widget):
        widget.setStyleSheet("QTextEdit {}")

    def set_num_ds(self):
        num = self.ui.num_d.toPlainText()
        try:
            num = int(num)
        except ValueError:
            self.ui.error_toast.setText("Please enter a positive integer for # of d-spacings.")
            return
        if num < 1 or num > 8:
            self.ui.error_toast.setText("Only 1-8 d-spacings supported right now.")
            return
        self.ui.error_toast.clear()
        for x in self.d_boxes: # reset hiding
            x.show()
        if self.op == 1: #only hide when indexing
            for x in self.d_boxes[num:]: # hide the ones we don't want
                x.hide()
    def set_operation(self, op):
        self.op = op
        if op == 0: # train
            for x in self.ANN_params:
                x.setReadOnly(False)
                self.emphasize(x)
                #x.setDisabled(False)
            for x in self.d_boxes:
                x.setReadOnly(True)
                self.deemphasize(x)
                #x.setDisabled(True)
            for x in self.ANN_outputs:
                x.setReadOnly(True)
                self.deemphasize(x)
                #x.setDisabled(True)
            self.ui.num_d.setReadOnly(False)
            self.emphasize(self.ui.num_d)
        elif op == 1: # index
            for x in self.ANN_params:
                x.setReadOnly(True)
                self.deemphasize(x)
                #x.setDisabled(True)
            for x in self.d_boxes:
                x.setReadOnly(False)
                self.emphasize(x)
                #x.setDisabled(False)
            for x in self.ANN_outputs:
                x.setReadOnly(True)
                self.deemphasize(x)
                #x.setDisabled(True)
            self.ui.num_d.setReadOnly(False)
            self.emphasize(self.ui.num_d)
        elif op == 2: # evaluate
            for x in self.ANN_params:
                x.setReadOnly(True)
                self.deemphasize(x)
                #x.setDisabled(True)
            for x in self.d_boxes:
                x.setReadOnly(True)
                #x.setDisabled(True)
                self.deemphasize(x)
            for x in self.ANN_outputs:
                x.setReadOnly(False)
                self.emphasize(x)
                #x.setDisabled(False)

            self.ui.num_d.setReadOnly(False)
            self.emphasize(self.ui.num_d)
        self.set_num_ds()
    def go(self):
        try:
            operation = self.ui.operation.currentText() # get combobox type
            self.ui.error_toast.clear()
            if operation == "train":
                self.train()
            elif operation == "index":
                self.Index()
            elif operation == "evaluate":
                self.evaluate()
            else:
                raise Exception
        except Exception as e:
            self.ui.error_toast.setText("Error occurred: {}".format(repr(e)))
            traceback.print_exc()
    class Getfile:
        # puts the file path into a textedit once clicking a button
        def __init__(self, textbox, app):
            self.textbox = textbox
            self.app = app
        def __call__(self):
            if self.app.op == 1 or self.app.op == 2: #indexing and evaluation
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

    def evaluate(self):
        a = float(self.ui.a_box.toPlainText())
        b = float(self.ui.b_box.toPlainText())
        gamma = float(self.ui.gam_box.toPlainText())
        num_d = int(self.ui.num_d.toPlainText())
        scalerPath = self.ui.scalerPath.text()
        model_path = self.ui.modelPath.text()
        try:
            scaler = joblib.load(scalerPath)
        except FileNotFoundError as e:
            print(e)
            print(e.filename)
            self.ui.error_toast.setText("Scaler file not found: {}".format(e.filename))
            return
        result, ds = evaluate.evaluate(model_path, a, b, gamma, scaler=scaler, num_spacings=num_d)
        self.ui.a_box.setText(str(format_decimal(result.x[0])))
        self.ui.b_box.setText(str(format_decimal(result.x[1])))
        self.ui.gam_box.setText(str(format_decimal(np.degrees(result.x[2]))))
        self.ui.error_box.setText(format_decimal(100 * np.abs(1 - np.linalg.norm((result.x - np.array([a,b,gamma])) / np.array([a,b,gamma])))))

        print(ds)
        for i,d in enumerate(self.d_boxes[:num_d]):
            d.setText(format_decimal(ds[i]))

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
        # operation = self.ui.operation.currentText()
        #use_q = self.ui.use_q.currentText()
        scalerPath = self.ui.scalerPath.text()
        num_d = int(self.ui.num_d.toPlainText())
        # run the ANN ?
        try:
            scaler = joblib.load(scalerPath)
        except FileNotFoundError as e:
            print(e)
            print(e.filename)
            self.ui.error_toast.setText("Scaler file not found: {}".format(e.filename))
            return
        model = deep_d.SimpleNet(num_spacings=num_d)
        path = self.ui.modelPath.text()
        try:
            model.load_state_dict(torch.load(path))
        except FileNotFoundError as e:
            self.ui.error_toast.setText("Model path file not found: {}".format(e.filename))
            return
        result, percent_error = index.index(np.array([d1, d2, d3, d4, d5, d6, d7, d8][:num_d]).reshape(1,-1), model, scaler=scaler)
        # results i an array of shape (,3)
        print(result)
        # output result 
        lattice_a = result[0]
        lattice_b = result[1]
        lattice_gam = np.degrees(result[2])
        a_string = format_decimal(lattice_a)
        self.ui.a_box.setText(a_string)
        b_string = format_decimal(lattice_b)
        self.ui.b_box.setText(b_string)
        gam_string = format_decimal(lattice_gam)
        self.ui.gam_box.setText(gam_string)
        error_string = format_decimal(percent_error)
        self.ui.error_box.setText(error_string)
# main
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())


