from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import CompleteGroupEvent as CGE


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(650, 620)
        MainWindow.setStyleSheet("background-color: rgb(246, 246, 246);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("background-color: rgb(230, 255, 220);")
        self.centralwidget.setObjectName("centralwidget")
        self.lable_title = QtWidgets.QLabel(self.centralwidget)
        self.lable_title.setGeometry(QtCore.QRect(0, 3, 600, 55))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lable_title.setFont(font)
        self.lable_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lable_title.setObjectName("lable_title")
        self.btn_calculate = QtWidgets.QPushButton(self.centralwidget)
        self.btn_calculate.setGeometry(QtCore.QRect(180, 540, 300, 60))
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
        self.line_edit_p = QtWidgets.QLineEdit(self.centralwidget)
        self.line_edit_p.setGeometry(QtCore.QRect(15, 200, 620, 60))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_edit_p.setFont(font)
        self.line_edit_p.setStyleSheet("background-color: rgb(197, 255, 176);")
        self.line_edit_p.setText("")
        self.line_edit_p.setObjectName("line_edit_p")
        self.label_p = QtWidgets.QLabel(self.centralwidget)
        self.label_p.setGeometry(QtCore.QRect(10, 160, 440, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_p.setFont(font)
        self.label_p.setAlignment(QtCore.Qt.AlignCenter)
        self.label_p.setObjectName("label_p")
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
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.add_functions()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Задание 2"))
        self.lable_title.setText(_translate("MainWindow", "Имитация событий,\nобразующих полную группу"))
        self.btn_calculate.setText(_translate("MainWindow", "Сгенерировать"))
        self.lable_task.setText(_translate("MainWindow", "    На вход генератора подается список, содержащий вероятности k случайных\n"
"независимых событий, образующих полную группу. В результате своей работы\n"
"генератор должен с заданными вероятностями вернуть индикатор (0, 1,..., k-1)\n"
"произошедшего на данном испытании события."))
        self.label_p.setText(_translate("MainWindow", " Список вероятностей p (через пробел)"))
        self.label_result_title.setText(_translate("MainWindow", "Результат"))
        self.label_p_10_6.setText(_translate("MainWindow", "p при 10\u2076"))

    def add_functions(self):
        self.btn_calculate.clicked.connect(lambda: self.calculate_result())

    def calculate_result(self):
        p = self.line_edit_p.text()

        try:
            p = [ float(pi) for pi in p.split() ]
            for i in range(len(p)):
                if not 0 <= p[i] <= 1:
                    raise ValueError
            if p == []:
                raise ValueError
            if not 1 - 0.1**10 <= sum(p) <= 1 + 0.1**10:
                self.line_edit_p.setText('')
                error = QMessageBox()
                error.setWindowTitle('Неверный ввод')
                error.setText('События не образуют полную группу:\n   sum(pi) != 1')
                error.setIcon(QMessageBox.Information)
                error.exec_()
                return
        except ValueError:
            self.line_edit_p.setText('')
            error = QMessageBox()
            error.setWindowTitle('Неверный ввод')
            error.setText('p - массив чисел в [0, 1]')
            error.setIcon(QMessageBox.Information)
            error.exec_()
            return
        
        self.plain_text_edit_p_10_6.setPlainText('')
        self.plain_text_edit_result.setPlainText('')
        
        result, p_10_6 = CGE.simulation_complete_group_events(p)
        self.plain_text_edit_p_10_6.setPlainText('  ' + str(p_10_6))
        self.plain_text_edit_result.setPlainText('  ' + str(result))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


