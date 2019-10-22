print("Beginning of the code")
from configparser import ConfigParser
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.animation as animation
import pims
from skimage.color import rgb2gray
import os.path
from threading import RLock

verrou = RLock()
from PyQt5.QtWidgets import (QApplication, QFileDialog)
from PyQt5.uic import loadUiType
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from PyQt5 import QtWidgets

# To maintain the tips on editing, run pyuic5 mainMenu.ui > mainMenu.py in terminal
Ui_MainWindow, QMainWindow = loadUiType('mainMenu.ui')
from mainMenu import (Ui_MainWindow)  # This is used only to have the tips on editing.
from dragRectangle import DraggableRectangle
from solver import Solver
from my_utils import create_dirs, export_results, plot_results

config = ConfigParser()
config.read('settings.ini')


# TODO Export the error too !
class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.figure = None
        self.boxes_dict = []  # List of boxes to analyse
        self.plots_dict = {}  # List of plots to plot
        self.output_name = []
        self.solver_list = []
        self.basename = None
        self.orignalVideoLen = 0
        self.timer = 0

        self.actionOpen.triggered.connect(self.browse_file)
        self.actionExport_results.triggered.connect(self.export_results)
        self.actionSubstract.triggered.connect(self.substract)
        # self.actionSubstract.setDisabled(True)
        self.actionAdd_box.triggered.connect(self.addDraggableRectangle)
        self.actionViolin.triggered.connect(self.plotSelection)
        self.actionPos.triggered.connect(self.plotSelection)
        self.actiony_shift.triggered.connect(self.plotSelection)
        self.actionx_shift.triggered.connect(self.plotSelection)
        self.actionViolin_all_on_one.triggered.connect(self.plotSelection)
        self.actionViolin_chop.triggered.connect(self.plotSelection)
        # self.actionStart_analysis.triggered.connect(self.startAnalysis)
        # self.actionShow_results.triggered.connect(self.showResults)
        self.actionx_shift.setChecked(bool(config.getboolean('section_b', 'actionx_shift')))
        self.actiony_shift.setChecked(bool(config.getboolean('section_b', 'actiony_shift')))
        self.actionPos.setChecked(bool(config.getboolean('section_b', 'actionpos')))
        self.actionViolin.setChecked(bool(config.getboolean('section_b', 'actionviolin')))
        self.actionViolin_all_on_one.setChecked(bool(config.getboolean('section_b', 'actionviolin_all_on_one')))
        self.actionViolin_chop.setChecked(bool(config.getboolean('section_b', 'actionviolin_chop')))
        self.checkBox_track.setChecked(bool(config.getboolean('section_a', 'checkBox_track')))

        self.lineEdit_pix_size.setText(config.get('section_a', 'pix_size'))
        self.lineEdit_magn.setText(config.get('section_a', 'magn'))
        self.lineEdit_sub_pix.setText(config.get('section_a', 'sub_pix'))
        self.lineEdit_fps.setText(config.get('section_a', 'fps'))
        self.lineEdit_start_frame.setText(config.get('section_a', 'start_frame'))
        self.lineEdit_start_frame.editingFinished.connect(self.startFrame)
        self.lineEdit_stop_frame.setText(config.get('section_a', 'stop_frame'))
        self.lineEdit_stop_frame.editingFinished.connect(self.stopFrame)
        self.lineEdit_w.setText(config.get('section_a', 'w'))
        self.lineEdit_h.setText(config.get('section_a', 'h'))
        self.checkBox_substract.stateChanged.connect(self.substract)
        self.lineEdit_substract_lvl.editingFinished.connect(self.substract)
        for i in plt.colormaps():
            self.comboBox_substract_col.addItem(i)
        self.comboBox_substract_col.setCurrentText(config.get('section_b', 'substract_col'))
        self.lineEdit_substract_lvl.setText(config.get('section_b', 'substract_lvl'))
        self.comboBox_substract_col.currentIndexChanged.connect(self.substract)
        self.lineEdit_chop_sec.setText(config.get('section_b', 'chop_sec'))

        self.actionAdd_box = QtWidgets.QAction()
        self.actionAdd_box.setObjectName('actionAdd_box')
        self.menubar.addAction(self.actionAdd_box)
        self.actionAdd_box.setText('Add analyse box')
        self.actionAdd_box.triggered.connect(self.addDraggableRectangle)
        self.actionAdd_box.setShortcut("A")

        self.actionStart_solver = QtWidgets.QAction()
        self.actionStart_solver.setObjectName('actionStart_solver')
        self.menubar.addAction(self.actionStart_solver)
        self.actionStart_solver.setText('Start Analysis')
        self.actionStart_solver.triggered.connect(self.startAnalysis)
        self.actionStart_solver.setShortcut("S")

        self.actionShow_results = QtWidgets.QAction()
        self.actionShow_results.setObjectName('actionShow_results')
        self.menubar.addAction(self.actionShow_results)
        self.actionShow_results.setText('Show plots')
        self.actionShow_results.triggered.connect(self.showResults)
        self.actionShow_results.setShortcut("V")

        self.actionStop_solver = QtWidgets.QAction()
        self.actionStop_solver.setObjectName('actionStop_solver')
        self.menubar.addAction(self.actionStop_solver)
        self.actionStop_solver.setText('Stop Analysis')
        self.actionStop_solver.triggered.connect(self.stopAnalysis)

        self.fileName = ""
        self.cell_n = ""
        self.polyg_size = 40
        self.videodata = None

        self.cursor = None
        self.plotSelection()  # Set options to the bools wanted even if the user didn't change anything

    def startFrame(self, update=True):
        if self.orignalVideoLen != float('inf'):
            if int(self.lineEdit_start_frame.text()) >= self.orignalVideoLen:
                self.lineEdit_start_frame.setText(str(self.orignalVideoLen - 1))
            if self.fileName != "":
                try:
                    if update:
                        self.imshow.set_data(rgb2gray(self.videodata.get_frame(int(self.lineEdit_start_frame.text()))))
                        self.figure.canvas.draw()
                        self.figure.canvas.flush_events()
                    return self.videodata.get_frame(int(self.lineEdit_start_frame.text()))
                except:
                    print("Tried to change the 1st frame to show. FAILED")
                    return 0
        else:
            return 0

    def stopFrame(self):
        if self.orignalVideoLen != float('inf'):
            if int(self.lineEdit_stop_frame.text()) >= self.orignalVideoLen:
                self.lineEdit_stop_frame.setText(str(self.orignalVideoLen - 1))
            if self.fileName != "":
                try:
                    return self.videodata.get_frame(int(self.lineEdit_stop_frame.text()))
                except:
                    print("Tried to load the last frame. FAILED")
                    return 0
            else:
                return 0

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        print("Drop event")
        if e.mimeData().hasUrls:
            for url in e.mimeData().urls():
                fname = str(url.toLocalFile())
            self.fileName = fname
            # print(self.fileName+" is the filename")
            self.loadAndShowFile()

    def mouse_event(self, e):
        self.cursor = (e.xdata, e.ydata)
        # print(e)

    def plotSelection(self):
        for action in self.menuView_plot.actions():
            self.plots_dict[action.text()] = action.isChecked()
            print("menuView is " + action.text() + " " + str(action.isChecked()))
        self.lineEdit_chop_sec.setEnabled(self.actionViolin_chop.isChecked())
        self.label_chop_sec.setEnabled(self.actionViolin_chop.isChecked())

    def browse_file(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                       "All Files (*);;mp4 movie (*.mp4)")
        self.goodFile = 1
        self.loadAndShowFile()

    def removeFile(self):
        self.views.clear()
        self.boxes.clear()
        for box in self.boxes_dict:
            box.disconnect()
        self.boxes_dict.clear()
        self.basename = None
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

    def loadAndShowFile(self):  # TODO change name to refer at what this method does
        # print("showFile()")
        try:
            self.removeFile()
        except:
            print("Nothing to clear")

        try:
            self.videodata = pims.ImageSequence(self.fileName)

        except:
            self.videodata = pims.Video(self.fileName)

        shape = np.shape(self.videodata.get_frame(0))
        try:
            self.orignalVideoLen = self.videodata._len  # Gives the right with some python environnements or get inf
        except:
            print("Cant get video length")
        print("Shape of videodata[0] : " + str(shape) + " x " + str(self.orignalVideoLen) + " frames. Obj type " + str(
            type(self.videodata)))
        self.figure = Figure()
        sub = self.figure.add_subplot(111)
        try:
            self.imshow = sub.imshow(rgb2gray(self.videodata.get_frame(int(self.lineEdit_start_frame.text()))),
                                     cmap='gray')
        except:
            self.imshow = sub.imshow(rgb2gray(self.videodata.get_frame(0)), cmap='gray')
        self.basename = os.path.basename(self.fileName)
        self.views.addItem(self.basename)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('motion_notify_event', self.mouse_event)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.addToolBar(self.toolbar)
        self.stopFrame()  # Check new boundary

    def substract(self):  # TODO edit it to work with boundaries we set
        if self.checkBox_substract.isChecked():
            print("substract")
            try:
                start_frame = int(self.lineEdit_start_frame.text())
                stop_frame = int(self.lineEdit_stop_frame.text())
                n_frames = stop_frame - start_frame
                first_frame = rgb2gray(self.videodata.get_frame(start_frame))
                print(type(first_frame))
                cumulative_frame = np.zeros(np.shape(first_frame))
                print(type(cumulative_frame))
                for i in range(stop_frame, start_frame, -int(n_frames / int(self.lineEdit_substract_lvl.text()))):
                    print(i)
                    cumulative_frame += rgb2gray(self.videodata.get_frame(i)) - first_frame

                self.imshow.set_data(rgb2gray(cumulative_frame))
                print("upto set_cmap")
                self.imshow.set_cmap(self.comboBox_substract_col.currentText())
                self.figure.canvas.draw()
            except:
                print("Wasn't able to substract")

        else:
            try:
                print("un-substract")
                self.startFrame()
                self.imshow.set_cmap('gray')
                self.figure.canvas.draw()
            except:
                print("Unable to un-substract")

    def addDraggableRectangle(self):
        # print("\naddDraggableRectangle()")
        w = int(self.lineEdit_w.text())
        h = int(self.lineEdit_h.text())
        if self.cursor[0] is not None and self.cursor[1] is not None:
            x0 = self.cursor[0] - w / 2
            y0 = self.cursor[1] - h / 2
        else:
            x0 = 15
            y0 = 15
        length = len(self.boxes_dict)
        print("Adding box " + str(length) + " to figure")

        ax = self.figure.add_subplot(111)
        rect = patches.Rectangle(xy=(x0, y0), width=w, height=h, linewidth=1, edgecolor='r', facecolor='b',
                                 fill=False)
        ax.add_patch(rect)
        text = ax.text(x=x0, y=y0, s=str(length))
        dr = DraggableRectangle(rect, rectangle_number=length, text=text)
        dr.connect()
        self.boxes_dict.append(dr)
        self.boxes.addItem(str(length))

    def startAnalysis(self):
        self.solver_list.clear()
        config.set('section_a', 'pix_size', self.lineEdit_pix_size.text())
        config.set('section_a', 'magn', self.lineEdit_magn.text())
        config.set('section_a', 'sub_pix', self.lineEdit_sub_pix.text())
        config.set('section_a', 'fps', self.lineEdit_fps.text())
        config.set('section_a', 'start_frame', self.lineEdit_start_frame.text())
        config.set('section_a', 'stop_frame', self.lineEdit_stop_frame.text())
        config.set('section_a', 'w', self.lineEdit_w.text())
        config.set('section_a', 'h', self.lineEdit_h.text())
        config.set('section_b', 'substract_col', self.comboBox_substract_col.currentText())
        config.set('section_b', 'substract_lvl', self.lineEdit_substract_lvl.text())
        config.set('section_b', 'chop_sec', self.lineEdit_chop_sec.text())
        config.set('section_b', 'actionx_shift', str(self.actionx_shift.isChecked()))
        config.set('section_b', 'actiony_shift', str(self.actiony_shift.isChecked()))
        config.set('section_b', 'actionpos', str(self.actionPos.isChecked()))
        config.set('section_b', 'actionviolin', str(self.actionViolin.isChecked()))
        config.set('section_b', 'actionviolin_all_on_one', str(self.actionViolin.isChecked()))
        config.set('section_b', 'actionviolin_chop', str(self.actionViolin_chop.isChecked()))
        config.set('section_a', 'checkBox_track', str(self.checkBox_track.isChecked()))
        with open('settings.ini', 'w') as configfile:
            config.write(configfile)
        print('Parameters saved')

        self.output_name = create_dirs(self.fileName, "")
        solver = Solver(videodata=self.videodata, fps=float(self.lineEdit_fps.text()),
                        box_dict=self.boxes_dict, solver_number=0,
                        my_upsample_factor=int(self.lineEdit_sub_pix.text()),
                        stop_frame=int(self.lineEdit_stop_frame.text()),
                        start_frame=int(self.lineEdit_start_frame.text()),
                        res=float(self.lineEdit_pix_size.text()),
                        track=self.checkBox_track.isChecked())
        self.solver_list.append(solver)
        self.solver_list[0].progressChanged.connect(self.updateProgress)
        self.solver_list[0].start()
        self.figure.savefig(self.output_name + "boxes_selection.png")
        # animation.FuncAnimation(self.figure, self.updateProgress, interval=1000)
        # self.timer = animation.FuncAnimation(self.figure, self.updateProgress, interval=10)
        plt.show()
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateProgress)
        self.timer.start(100)
        print('timer started etc')

    def stopAnalysis(self):
        print("STOPPING ANALYSIS")
        self.timer.stop()
        for solver in self.solver_list:
            solver.stop()

    def updateProgress(self):
        for s in self.solver_list:
            if s.progress == 100:
                self.timer.stop()
            for j in range(len(self.boxes_dict)):
                item = self.boxes.item(j)
                item.setText(str(j) + " " + str(s.progress) + "%: f#" + str(
                    s.actual_i - int(self.lineEdit_start_frame.text())) + "/"
                             + str(int(self.lineEdit_stop_frame.text()) - int(self.lineEdit_start_frame.text())))
            self.imshow.set_data(rgb2gray(s.frame_n))
            for r in self.boxes_dict:
                r.update_from_solver()
            self.figure.canvas.draw()
            # plt.show(block=False)
            # self.figure.canvas.flush_events()

    def showResults(self):
        # print(self.solver_list)
        for solver in self.solver_list:
            plot_results(shift_x=solver.shift_x, shift_x_y_error=solver.shift_x_y_error, shift_y=solver.shift_y,
                         fps=solver.fps, res=solver.res,
                         output_name=self.output_name, plots_dict=self.plots_dict, boxes_dict=self.boxes_dict,
                         chop_sec=float(self.lineEdit_chop_sec.text()), start_frame=solver.start_frame)
        print("Plots showed")

    def export_results(self):
        for solver in self.solver_list:
            for j in range(len(solver.box_dict)):
                export_results(shift_x=solver.shift_x[j], shift_y=solver.shift_y[j], fps=solver.fps, res=solver.res,
                               w=solver.box_dict[j].rect._width, h=solver.box_dict[j].rect._height,
                               z_std=solver.z_std[j],
                               dz_rms=solver.z_rms[j],
                               v=solver.v_rms[j],
                               output_name=self.output_name + str(j))
        print("Files exported")


if __name__ == '__main__':
    print("Interpreter location is : " + os.path.dirname(sys.executable))
    sys.stdout.flush()
    app = QApplication(sys.argv)
    menu = Main()
    menu.show()
    sys.exit(app.exec_())
