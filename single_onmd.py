print("Beginning of the code")
from configparser import ConfigParser
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
import pims
from skimage.color import rgb2gray
import os.path
from threading import RLock

verrou = RLock()
from PyQt5.QtWidgets import (QApplication, QFileDialog)
from PyQt5.uic import loadUiType
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

# To maintain the tips on editing, run pyuic5 mainMenu.ui > mainMenu.py in terminal
Ui_MainWindow, QMainWindow = loadUiType('mainMenu.ui')
from mainMenu import (Ui_MainWindow)  # This is used only to have the tips on editing.
from dragRectangle import DraggableRectangle
from solver import Solver
from my_utils import create_dirs, export_results, plot_results
config = ConfigParser()
config.read('settings.ini')

# TODO Be able to analyse a video be sequences of lenght l seconds. Then plot theses sequences side by side on same fig
# TODO Substract many frames from each other.
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
        self.basename = None
        self.orignalVideoLen = None

        self.actionOpen.triggered.connect(self.browse_file)
        self.actionExport_results.triggered.connect(self.export_results)
        self.actionSubstract.triggered.connect(self.substract)
        #self.actionSubstract.setDisabled(True)
        self.actionAdd_box.triggered.connect(self.addDraggableRectangle)
        self.actionViolin.triggered.connect(self.plotSelection)
        self.actionPos.triggered.connect(self.plotSelection)
        self.actiony_shift.triggered.connect(self.plotSelection)
        self.actionx_shift.triggered.connect(self.plotSelection)
        self.actionStart_analysis.triggered.connect(self.startAnalysis)
        self.actionShow_results.triggered.connect(self.showResults)
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

        self.fileName = ""
        self.cell_n = ""
        self.polyg_size = 40
        self.videodata = None

        self.cursor = None
        self.plotSelection()  # Set options to the bools wanted even if the user didn't change anything
    def startFrame(self):
        if int(self.lineEdit_start_frame.text())>=self.orignalVideoLen:
            self.lineEdit_start_frame.setText(str(self.orignalVideoLen-1))
        if self.fileName != "" :
            try:
                self.test.set_data(self.videodata.get_frame(int(self.lineEdit_start_frame.text())))
                self.fig_dict[self.basename].canvas.draw()
                self.fig_dict[self.basename].canvas.flush_events()
            except:
                print("Tried to change the 1st frame to show. FAILED")

    def stopFrame(self):
        if int(self.lineEdit_stop_frame.text())>=self.orignalVideoLen:
            self.lineEdit_stop_frame.setText(str(self.orignalVideoLen-1))

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
            # print("menuView is " + action.text() + " " + str(action.isChecked()))

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
        self.loadAndShowFile()

    def loadAndShowFile(self):  # TODO change name to refer at what this method does
        # print("showFile()")
        self.views.clear()
        self.boxes.clear()
        self.fig_dict.clear()
        for box in self.boxes_dict:
            box.disconnect()
        self.boxes_dict.clear()
        self.basename = None
        try:
            self.videodata = pims.Video(self.fileName)
        except:
            self.videodata = pims.ImageSequence(self.fileName)  # So one can drop a set of images

        shape = np.shape(self.videodata.get_frame(0))
        try:
            self.orignalVideoLen = self.videodata._len  # Gives the right with some python environnements or get inf
        except:
            print("Cant get video length")
        print("Shape of videodata[1] : " + str(shape) + " x " + str(self.orignalVideoLen) + " frames. Obj type " + str(
            type(self.videodata)))
        fig = Figure()
        sub = fig.add_subplot(111)
        try:
            self.test = sub.imshow(self.videodata.get_frame(int(self.lineEdit_start_frame.text())))
        except:
            self.test = sub.imshow(self.videodata.get_frame(0))
        self.basename=os.path.basename(self.fileName)
        self.addfig(self.basename, fig)
        self.changefig(self.views.item(0))
        self.stopFrame()  #Check new boundary

    def substract(self):  # TODO edit it to work with boundaries we set
        print("substract")
        #self.views.addItem("Substracted")

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
        for f in self.fig_dict:
            ax = self.fig_dict[f].add_subplot(111)
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
        with open('settings.ini', 'w') as configfile:
            config.write(configfile)
        print('Parameters saved')

        self.output_name = create_dirs(self.fileName, "")
        solver = Solver(videodata=self.videodata, fps=float(self.lineEdit_fps.text()),
                        box_dict=self.boxes_dict, solver_number=0,
                        my_upsample_factor=int(self.lineEdit_sub_pix.text()),
                        stop_frame=int(self.lineEdit_stop_frame.text()),
                        start_frame=int(self.lineEdit_start_frame.text()),
                        res=float(self.lineEdit_pix_size.text()))
        self.solver_list.append(solver)
        self.solver_list[0].progressChanged.connect(self.updateProgress)
        self.solver_list[0].start()
        self.fig_dict[self.basename].savefig(self.output_name + "boxes_selection.png")

    def updateProgress(self, progress, frame, image):
        for j in range(len(self.boxes_dict)):
            item = self.boxes.item(j)
            item.setText(str(j) + " " + str(progress) + "%: f#" + str(frame-int(self.lineEdit_start_frame.text())) + "/"
                         + str(int(self.lineEdit_stop_frame.text()) - int(self.lineEdit_start_frame.text())))
        self.test.set_data(image)
        self.fig_dict[self.basename].canvas.draw()
        #self.fig_dict[self.basename[0]].canvas.flush_events()

    def showResults(self):
        # print(self.solver_list)
        for solver in self.solver_list:
            for j in range(len(solver.box_dict)):
                plot_results(shift_x=solver.shift_x[j], shift_y=solver.shift_y[j], fps=solver.fps, res=solver.res,
                             output_name=(self.output_name + str(j)), plots_dict=self.plots_dict, solver_number=j)
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
