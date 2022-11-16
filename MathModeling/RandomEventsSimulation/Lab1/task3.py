from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import ComplexDependentEvent as CDE


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(650, 700)
        MainWindow.setStyleSheet("background-color: rgb(246, 246, 246);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("background-color: rgb(230, 255, 220);")
        self.centralwidget.setObjectName("centralwidget")
        self.lable_title = QtWidgets.QLabel(self.centralwidget)
        self.lable_title.setGeometry(QtCore.QRect(0, 5, 600, 50))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lable_title.setFont(font)
        self.lable_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lable_title.setObjectName("lable_title")
        self.btn_calculate = QtWidgets.QPushButton(self.centralwidget)
        self.btn_calculate.setGeometry(QtCore.QRect(180, 630, 300, 60))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_calculate.setFont(font)
        self.btn_calculate.setStyleSheet("background-color: rgb(197, 255, 176);")
        self.btn_calculate.setObjectName("btn_calculate")
        self.lable_task = QtWidgets.QLabel(self.centralwidget)
        self.lable_task.setGeometry(QtCore.QRect(18, 60, 600, 80))
        self.lable_task.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lable_task.setWordWrap(True)
        self.lable_task.setObjectName("lable_task")
        self.line_edit_Pa = QtWidgets.QLineEdit(self.centralwidget)
        self.line_edit_Pa.setGeometry(QtCore.QRect(15, 200, 250, 60))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_edit_Pa.setFont(font)
        self.line_edit_Pa.setStyleSheet("background-color: rgb(197, 255, 176);")
        self.line_edit_Pa.setText("")
        self.line_edit_Pa.setObjectName("line_edit_Pa")
        self.label_Pa = QtWidgets.QLabel(self.centralwidget)
        self.label_Pa.setGeometry(QtCore.QRect(10, 160, 60, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_Pa.setFont(font)
        self.label_Pa.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Pa.setObjectName("label_Pa")
        self.line_edit_Pba = QtWidgets.QLineEdit(self.centralwidget)
        self.line_edit_Pba.setGeometry(QtCore.QRect(380, 200, 250, 60))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_edit_Pba.setFont(font)
        self.line_edit_Pba.setStyleSheet("background-color: rgb(197, 255, 176);")
        self.line_edit_Pba.setText("")
        self.line_edit_Pba.setObjectName("line_edit_Pba")
        self.label_Pba = QtWidgets.QLabel(self.centralwidget)
        self.label_Pba.setGeometry(QtCore.QRect(375, 160, 100, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_Pba.setFont(font)
        self.label_Pba.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Pba.setObjectName("label_Pba")
        self.label_result_title = QtWidgets.QLabel(self.centralwidget)
        self.label_result_title.setGeometry(QtCore.QRect(10, 280, 120, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_result_title.setFont(font)
        self.label_result_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_result_title.setObjectName("label_result_title")
        self.plain_text_edit_result = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plain_text_edit_result.setGeometry(QtCore.QRect(15, 320, 620, 60))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.plain_text_edit_result.setFont(font)
        self.plain_text_edit_result.setStyleSheet("background-color: rgb(197, 255, 176);")
        self.plain_text_edit_result.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.plain_text_edit_result.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.plain_text_edit_result.setPlainText("")
        self.plain_text_edit_result.setObjectName("plain_text_edit_result")
        self.plain_text_edit_p_10_6 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plain_text_edit_p_10_6.setGeometry(QtCore.QRect(15, 440, 620, 60))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.plain_text_edit_p_10_6.setFont(font)
        self.plain_text_edit_p_10_6.setStyleSheet("background-color: rgb(197, 255, 176);")
        self.plain_text_edit_p_10_6.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.plain_text_edit_p_10_6.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.plain_text_edit_p_10_6.setPlainText("")
        self.plain_text_edit_p_10_6.setObjectName("plain_text_edit_p_10_6")
        self.label_p_10_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_p_10_6.setGeometry(QtCore.QRect(10, 400, 130, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_p_10_6.setFont(font)
        self.label_p_10_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_p_10_6.setObjectName("label_p_10_6")

        self.plain_text_edit_teor = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plain_text_edit_teor.setGeometry(QtCore.QRect(15, 560, 620, 60))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.plain_text_edit_teor.setFont(font)
        self.plain_text_edit_teor.setStyleSheet("background-color: rgb(197, 255, 176);")
        self.plain_text_edit_teor.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.plain_text_edit_teor.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.plain_text_edit_teor.setPlainText("")
        self.plain_text_edit_teor.setObjectName("plain_text_edit_teor")
        self.label_teor = QtWidgets.QLabel(self.centralwidget)
        self.label_teor.setGeometry(QtCore.QRect(10, 520, 550, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_teor.setFont(font)
        self.label_teor.setAlignment(QtCore.Qt.AlignCenter)
        self.label_teor.setObjectName("label_teor")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.add_functions()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Задание 2"))
        self.lable_title.setText(_translate("MainWindow", "Имитация сложного события, \nсостоящего из зависимых"))
        self.btn_calculate.setText(_translate("MainWindow", "Сгенерировать"))
        self.lable_task.setText(_translate("MainWindow", "    На вход генератора подается вероятность P(A) и условная вероятность\n"
"P(B|A). В результате работы генератор должен вернуть индикатор (число 0, 1,\n"
"2 или 3) одного из четырех событий AB, ~AB, A~B, ~A~B с соответствующими\n"
"вероятностями P(AB), P(~AB), P(A~B), P(~A~B)."))
        self.label_Pa.setText(_translate("MainWindow", " P(A)"))
        self.label_Pba.setText(_translate("MainWindow", " P(B|A)"))
        self.label_result_title.setText(_translate("MainWindow", "Результат"))
        self.label_p_10_6.setText(_translate("MainWindow", "p при 10\u2076"))
        self.label_teor.setText(_translate("MainWindow", "Теоретические значения для случаев 0, 1, 2, 3"))

    def add_functions(self):
        self.btn_calculate.clicked.connect(lambda: self.calculate_result())

    def calculate_result(self):
        Pa = self.line_edit_Pa.text()
        Pba = self.line_edit_Pba.text()

        try:
            Pa = float(Pa)
            if not 0 <= Pa <= 1:
                raise ValueError
        except ValueError:
            self.line_edit_Pa.setText('')
            error = QMessageBox()
            error.setWindowTitle('Неверный ввод P(A)')
            error.setText('P(A) в [0, 1]')
            error.setIcon(QMessageBox.Information)
            error.exec_()
            return
        
        self.plain_text_edit_p_10_6.setPlainText('')
        self.plain_text_edit_result.setPlainText('')
        self.plain_text_edit_teor.setPlainText('')

        try:
            Pba = float(Pba)
            if not 0 <= Pba <= 1:
                raise ValueError
        except ValueError:
            self.line_edit_Pba.setText('')
            error = QMessageBox()
            error.setWindowTitle('Неверный ввод P(B|A)')
            error.setText('P(B|A) в [0, 1]')
            error.setIcon(QMessageBox.Information)
            error.exec_()
            return
        
        result, p_10_6, teor = CDE.simulation_complex_dependent_event(Pa, Pba)
        self.plain_text_edit_p_10_6.setPlainText('  ' + str(p_10_6))
        self.plain_text_edit_result.setPlainText('  ' + str(result) + '  (' + str(p_10_6[result]) + ')' )
        self.plain_text_edit_teor.setPlainText('  ' + str(teor))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


