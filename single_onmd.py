print("Beginning of the code")
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.text as mpltext
import pims
from PyQt5.QtGui import QPainter

from skimage.color import rgb2gray
from skimage.feature import register_translation
import pandas as pd

import seaborn as sns

import os.path

from threading import RLock

verrou = RLock()
from PyQt5.QtWidgets import (QApplication, QPushButton, QGridLayout, QDialog, QLineEdit,

                             QFileDialog, QWidget, QAction, QProgressBar, QListWidgetItem)
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt5 import QtGui

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
        self.actionSubstract.setDisabled(True)
        self.actionAdd_box.triggered.connect(self.addDraggableRectangle)
        self.actionViolin.triggered.connect(self.plotSelection)
        self.actionPos.triggered.connect(self.plotSelection)
        self.actiony_shift.triggered.connect(self.plotSelection)
        self.actionx_shift.triggered.connect(self.plotSelection)
        self.actionStart_analysis.triggered.connect(self.startAnalysis)
        self.actionShow_results.triggered.connect(self.showResults)

        self.fileName = ""
        self.cell_n = ""
        self.polyg_size = 40
        self.videodata = None

        self.cursor = None
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

    def mouse_event(self, e):
        self.cursor = (e.xdata, e.ydata)
        #print(e)

    def plotSelection(self):
        for action in self.menuView_plot.actions():
            self.plots_dict[action.text()] = action.isChecked()
            #print("menuView is " + action.text() + " " + str(action.isChecked()))

    def changefig(self, item):
        text = item.text()
        self.rmmpl()  # Remove plot to show
        self.addmpl(self.fig_dict[text])  # Add plot to show

    def addfig(self, name, fig):  # Add fig item in the list
        self.fig_dict[name] = fig
        self.views.addItem(name)

    def addmpl(self, fig):  # Add plot to show
        self.canvas = FigureCanvas(fig)
        self.canvas.mpl_connect('motion_notify_event', self.mouse_event)
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

    def showFile(self):  # TODO change name to refer at what this method does
        #print("showFile()")
        # self.videodata = pims.Video(self.fileName)
        self.videodata = pims.ImageIOReader(self.fileName)
        # self.videodata = pims.MoviePyReader(self.fileName)
        shape = np.shape(self.videodata.get_frame(0))
        self.orignalVideoLen = self.videodata._len  # Gives the right with some python environnements or get inf
        print("Shape of videodata[1] : "+str(shape)+" x "+ str(self.orignalVideoLen)+" frames. Obj type " + str(type(self.videodata)))
        #print(type(self.videodata))
        fig = Figure()
        sub = fig.add_subplot(111)
        sub.imshow(rgb2gray(self.videodata.get_frame(0)), cmap=plt.cm.gray)
        self.addfig('Raw video', fig)
        self.changefig(self.views.item(0))

    def substract(self): # TODO edit it to work with boundaries we set
        if self.stopFrame == None: self.stopFrame = self.orignalVideoLen
        fig = Figure()
        sub = fig.add_subplot(111)
        subset = rgb2gray(self.videodata[self.stopFrame - 1]) - rgb2gray(self.videodata[self.startFrame])
        sub.imshow(subset, cmap=plt.cm.gray)
        self.addfig('Subtracted video', fig)

    def addDraggableRectangle(self):
        #print("\naddDraggableRectangle()")
        w = int(self.lineEdit_w.text())
        h = int(self.lineEdit_h.text())
        if self.cursor[0] is not None and self.cursor[1] is not None:
            x0 = self.cursor[0] - w / 2
            y0 = self.cursor[1] - h / 2
        else:
            x0 = 15
            y0 = 15
        for x in self.fig_dict:
            length = len(self.boxes_dict)
            print("Adding box "+str(length)+" to figure")
            ax = self.fig_dict[x].add_subplot(111)
            rect = patches.Rectangle(xy=(x0, y0), width=w, height=h, linewidth=1, edgecolor='r', facecolor='b', fill=False)
            ax.add_patch(rect)
            text = ax.text(x=x0, y=y0, s=str(length))
            #ax.text(x0, y0, str(length))
            dr = DraggableRectangle(rect, rectangle_number=length, text=text)
            dr.connect()
            self.boxes_dict.append(dr)
            self.boxes.addItem(str(length))

    def startAnalysis(self):
        self.output_name.clear()
        self.solver_list.clear()
        for i in range(len(self.boxes_dict)):
            solver = Solver(videodata=self.videodata, fps=float(self.lineEdit_fps.text()),
                            w=self.boxes_dict[i].rect._width,
                            h=self.boxes_dict[i].rect._height, x_rect=int(self.boxes_dict[i].x_rect),
                            y_rect=int(self.boxes_dict[i].y_rect),
                            my_upsample_factor=int(self.lineEdit_sub_pix.text()),
                            stop_frame=int(self.lineEdit_stop_frame.text()),
                            start_frame=int(self.lineEdit_start_frame.text()),
                            res=float(self.lineEdit_pix_size.text()),
                            output_name=create_dirs(self.fileName, str(i)), solver_number=i)
            self.solver_list.append(solver)
            self.solver_list[i].progressChanged.connect(self.updateProgress)
            self.solver_list[i].start()
        self.fig_dict['Raw video'].savefig(create_dirs(self.fileName, "boxes_selection")+".png")

    def updateProgress(self, solver_number, progress):
        item = self.boxes.item(int(solver_number))
        item.setText(str(solver_number) + " " + str(progress) + "%")

    def showResults(self):
        print(self.solver_list)
        for solver in self.solver_list:
            plot_results(solver.shift_x, solver.shift_y, solver.fps, solver.res, solver.output_name, self.plots_dict)
        print("Plots showed")

    def export_results(self):
        for solver in self.solver_list:
            export_results(solver.shift_x, solver.shift_y, solver.fps, solver.res, solver.w, solver.h, solver.z_std,
                           solver.z_rms, solver.v_rms, solver.output_name)
        print("Files exported")


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
        shift_length=np.sqrt(np.square(shift_x)+np.square(shift_y))
        sns.violinplot(data=shift_length)
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
    df = pd.concat([df, pd.DataFrame({"z std, um": [z_std], "total z, um": [dz_rms], "v, um/s": [v], "window, px":
        [str(w) + " x " + str(h)], "window, um": [str(w * res) + " x " + str(h * res)],"um per px": [res]})], axis=1)
    df = df[["t, s", "x, px", "y, px", "x, um", "y, um", "z std, um", "total z, um", "v, um/s", "window, px",
             "window, um", "um per px"]]
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
