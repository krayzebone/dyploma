from PyQt6.QtWidgets import QTabWidget
from GUI.tabs.main_tab import MainTab
from GUI.tabs.beam_tab import BeamTab
from GUI.tabs.T_section_tab import SectionTab
from GUI.tabs.rect_tab import RectSectionTab

class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.main_tab = MainTab()
        self.beam_tab = BeamTab()
        self.T_section_tab = SectionTab()
        self.optimize_rect_section_tab = RectSectionTab()

        self.addTab(self.main_tab, "1. Wstęp")
        self.addTab(self.beam_tab, "2. Belka")
        self.addTab(self.T_section_tab, "3. Przekrój")
        self.addTab(self.optimize_rect_section_tab, "4. Płyta pomostowa")

        # Connect
        self.main_tab.next_button.clicked.connect(self.go_to_beam_tab)
        self.beam_tab.next_button.clicked.connect(self.go_to_section_tab)

        # Window Setup
        self.setWindowTitle("Beam Calculation - Two Tabs")
        self.setMinimumSize(640, 480)

    def go_to_beam_tab(self):
        self.setCurrentIndex(1)  # switch to tab 3 (0-based index)

    def go_to_section_tab(self):
        moment_value = self.beam_tab.moment_max
        self.section_tab.display_moment(moment_value)
        self.setCurrentIndex(2)  # switch to tab 3 (0-based index)
    
