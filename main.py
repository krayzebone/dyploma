import sys
import tensorflow
from PyQt6.QtWidgets import QApplication
from GUI.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

#python -m PyInstaller --onefile --add-data "C:/Users/marci/Desktop/Praca_dyplomowa/Projekt_sieci_neuronowej/moj_program/resources/images;resources/images" main.py
