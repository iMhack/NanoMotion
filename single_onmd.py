import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
import pims
from PyQt5.QtGui import QPainter

from skimage.color import rgb2gray
from skimage.feature import register_translation
import pandas as pd

import seaborn as sns

import os.path

from threading import Thread, RLock

verrou = RLock()
from PyQt5.QtWidgets import (QApplication, QPushButton, QGridLayout, QDialog, QLineEdit,

                             QFileDialog, QWidget, QAction, QProgressBar, QListWidgetItem)
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt

from PyQt5.uic import loadUiType

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)


# To maintain the tips on editing, run pyuic5 mainMenu.ui > mainMenu.py in terminal
Ui_MainWindow, QMainWindow = loadUiType('mainMenu.ui')
from mainMenu import (Ui_MainWindow)  # This is used only to have the tips on editing.

from dragRectangle import DraggableRectangle
from solver import Solver

class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.fig_dict = {}  # List of figures
        self.views.itemClicked.connect(self.changefig)
        fig = Figure()
        self.addmpl(fig)
        self.boxes_dict = []  # List of boxes to analyse
        self.plots_dict = {}  # List of plots to plot
        self.output_name = []
        self.solver_list = []

        self.goodFile = 0
        self.orignalVideoLen = None

        self.actionOpen.triggered.connect(self.browse_file)
        self.actionExport_results.triggered.connect(self.export_results)
        self.actionSubstract.triggered.connect(self.substract)
        self.actionSub_frameset.triggered.connect(self.subframeset)
        self.actionAdd_box.triggered.connect(self.addDraggableRectangle)
        self.actionViolin.triggered.connect(self.plotSelection)
        self.actionPos.triggered.connect(self.plotSelection)
        self.actiony_shift.triggered.connect(self.plotSelection)
        self.actionx_shift.triggered.connect(self.plotSelection)
        self.actionStart_analysis.triggered.connect(self.startAnalysis)
        self.actionShow_results.triggered.connect(self.showResults)
        self.popup = None

        self.fileName = ""
        self.magnitude = 1
        self.pixelSize = 1.2
        self.res = 1.2
        self.cell_n = ""
        self.polyg_size = 40
        self.fps = 25
        self.videodata = None
        self.w = None
        self.h = None
        self.startFrame = 0
        self.stopFrame = None

        self.shift_x = []
        self.shift_y = []
        self.z_std = []
        self.z_rms = []
        self.v_rms = []

        self.test = None

        self.QProgress = None

        self.plotSelection()  # Set options to the bools wanted even if the user didn't change anything

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
            self.showFile()

    def plotSelection(self):
        for action in self.menuView_plot.actions():
            self.plots_dict[action.text()] = action.isChecked()
            print("menuView is " + action.text() + " " + str(action.isChecked()))

    def changefig(self, item):
        text = item.text()
        self.rmmpl()  # Remove plot to show
        self.addmpl(self.fig_dict[text])  # Add plot to show

    def addfig(self, name, fig):  # Add fig item in the list
        self.fig_dict[name] = fig
        self.views.addItem(name)

    def addmpl(self, fig):  # Add plot to show
        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas,
                                         self, coordinates=True)
        self.addToolBar(self.toolbar)

    def rmmpl(self, ):  # Remove plot to show
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

    def browse_file(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                       "All Files (*);;mp4 movie (*.mp4)")
        self.goodFile = 1
        self.showFile()

    def showFile(self):
        print("\nshowFile")
        # self.videodata = pims.Video(self.fileName)
        self.videodata = pims.ImageIOReader(self.fileName)
        # self.videodata = pims.MoviePyReader(self.fileName)
        print("\nShape of videodata[1] : ")
        print(type(self.videodata))
        shape = np.shape(self.videodata.get_frame(0))
        fig = Figure()
        sub = fig.add_subplot(111)
        sub.imshow(rgb2gray(self.videodata.get_frame(0)), cmap=plt.cm.gray)
        self.addfig('Raw video', fig)
        self.orignalVideoLen = self.videodata._len  # Gives the right with some python environnements or get inf
        print("File lenght is " + str(self.orignalVideoLen))
        self.changefig(self.views.item(0))

        print(shape)
        self.w = shape[0]
        self.h = shape[1]

    def substract(self):
        if self.stopFrame == None: self.stopFrame = self.orignalVideoLen
        fig = Figure()
        sub = fig.add_subplot(111)
        subset = rgb2gray(self.videodata[self.stopFrame - 1]) - rgb2gray(self.videodata[self.startFrame])
        sub.imshow(subset, cmap=plt.cm.gray)
        self.addfig('Subtracted video', fig)

    def subframeset(self):  # To be done
        print("\nsubframeset")
        self.popup = MyPopup()
        self.popup.setGeometry(100, 100, 100, 100)
        self.popup.show()

    def addDraggableRectangle(self):
        print("\naddDraggableRectangle()")
        print("Imported file " + self.fileName)
        w = self.polyg_size
        h = self.polyg_size
        x0 = 15
        y0 = 15

        self.test = Figure()
        for x in self.fig_dict:
            ax = self.fig_dict[x].add_subplot(111)
            rect_1 = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect_1)
            dr = DraggableRectangle(rect_1)
            dr.connect()
            self.boxes_dict.append(dr)
            self.boxes.addItem(str(len(self.boxes_dict)))

    def startAnalysis(self):
        self.shift_x.clear()  # So it can be run, modify boxes, rerun.
        self.shift_y.clear()
        self.output_name.clear()
        self.z_rms.clear()
        self.z_std.clear()
        self.v_rms.clear()
        self.solver_list.clear()
        for i in range(len(self.boxes_dict)):
            self.output_name.append(create_dirs(self.fileName, str(i)))
            print("\nAnalysing " + self.output_name[i])
            x_rect = int(self.boxes_dict[i].x_rect)
            y_rect = int(self.boxes_dict[i].y_rect)
            print("\nSelected coordinates for polygon are: ", x_rect, y_rect)
            solver = Solver(videodata=self.videodata, fps=self.fps, res=self.res, w=self.w, h=self.h, x_rect=x_rect,
                            y_rect=y_rect, solver_number = i)
            self.solver_list.append(solver)
            self.solver_list[i].progressChanged.connect(self.updateProgress)
            self.solver_list[i].start()


    def updateProgress(self, solver_number, progress):
        item = self.boxes.item(int(solver_number))
        item.setText(str(solver_number)+" "+str(progress)+"%")


    def showResults(self):
        print(self.solver_list)
        for i in range(len(self.boxes_dict)):
            self.shift_x.append(self.solver_list[i].shift_x)
            self.shift_y.append(self.solver_list[i].shift_y)
            self.z_std.append(self.solver_list[i].get_z_std())
            z_rms, v_rms = self.solver_list[i].get_z_delta_rms()
            self.z_rms.append(z_rms)
            self.v_rms.append(v_rms)
            plot_results(self.shift_x[i], self.shift_y[i], self.fps, self.res, self.output_name[i], self.plots_dict)
        print("Plots showed")

    def export_results(self):
        for i in range(len(self.boxes_dict)):
            export_results(self.shift_x[i], self.shift_y[i], self.fps, self.res, self.w, self.h, self.z_std,
                           self.z_rms[i], self.v_rms[i], self.output_name[i])
        print("Files exported")

class MyPopup(QWidget):
    def __init__(self):
        QWidget.__init__(self)

    def paintEvent(self, e):
        dc = QPainter(self)
        dc.drawLine(0, 0, 100, 100)
        dc.drawLine(100, 0, 0, 100)


def plot_results(shift_x, shift_y, fps, res, output_name, plots_dict):
    if (plots_dict["x(t)_shift"]):
        plt.figure(num=output_name + 'x(t), um(s)')
        plt.plot([frame / fps for frame in range(len(shift_x))], [x * res for x in shift_x], "-")
        plt.grid()
        plt.title("x(t)")
        plt.xlabel("t, s")
        plt.ylabel("x, um")
        plt.savefig(output_name + "_x(t).png")
    if (plots_dict["Violin"]):
        plt.figure(num=output_name + 'violin of shift_x*shift_y')
        sns.violinplot(data=np.sqrt(np.multiply(np.square(shift_x), np.square(shift_y))))
        plt.title("violin")
        plt.savefig(output_name + "_violin.png")

    if (plots_dict["y(t)_shift"]):
        plt.figure(num=output_name + 'y(t), um(s)')
        plt.plot([frame / fps for frame in range(len(shift_y))], [x * res for x in shift_y], "-")
        plt.grid()
        plt.title("y(t)")
        plt.xlabel("t, s")
        plt.ylabel("y, um")
        plt.savefig(output_name + "_y(t).png")

    if (plots_dict["pos(t)"]):
        plt.figure(num='y(x), um(um)')
        plt.plot([x * res for x in shift_x], [y * res for y in shift_y], "-")
        plt.grid()
        plt.title("y(x)")
        plt.xlabel("x, um")
        plt.ylabel("y, um")
        plt.savefig(output_name + "_y(x).png")
    plt.show()


def export_results(shift_x, shift_y, fps, res, w, h, z_std, dz_rms, v, output_name):
    df = pd.DataFrame({"t, s": [frame / fps for frame in range(len(shift_x))], "x, px": shift_x, "y, px": shift_y,

                       "x, um": [x * res for x in shift_x], "y, um": [y * res for y in shift_y]})

    df = pd.concat([df, pd.DataFrame(

        {"z std, um": [z_std], "total z, um": [dz_rms], "v, um/s": [v], "window, px": [str(w) + " x " + str(h)],
         "window, um": [str(w * res) + " x " + str(h * res)],

         "um per px": [res]})], axis=1)

    df = df[
        ["t, s", "x, px", "y, px", "x, um", "y, um", "z std, um", "total z, um", "v, um/s", "window, px", "window, um",
         "um per px"]]

    writer = pd.ExcelWriter(

        os.path.join(output_name + "_output.xlsx"))

    df.to_excel(excel_writer=writer, sheet_name="Sheet 1", index=False)

    writer.save()


def create_dirs(file, cell_name):  # check if /results/filename/ directories exists and create them, if not

    videofile_dir = os.path.dirname(os.path.abspath(file))

    if not os.path.isdir(os.path.join(videofile_dir, "results")):
        os.makedirs(os.path.join(videofile_dir, "results"))

    output_dir = os.path.join(videofile_dir, "results", os.path.basename(file)[:-4])

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_name = os.path.join(output_dir, os.path.basename(file)[:-4] + "_cell_" + cell_name)

    return output_name


if __name__ == '__main__':
    print("Interpreter location is : " + os.path.dirname(sys.executable))
    sys.stdout.flush()
    app = QApplication(sys.argv)
    menu = Main()
    menu.show()
    sys.exit(app.exec_())
