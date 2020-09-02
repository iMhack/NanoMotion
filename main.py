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
import json

from PyQt5.QtWidgets import (QApplication, QFileDialog)
from PyQt5.uic import loadUiType
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from PyQt5 import QtWidgets

from dragRectangle import DraggableRectangle
from solver import Solver
from utils import create_dirs, export_results, plot_results

print("Beginning of the code.")

# To keep the tips when editing, run pyuic5 mainMenu.ui > mainMenu.py in terminal
Ui_MainWindow, QMainWindow = loadUiType('mainMenu.ui')


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.figure = None
        self.saved_boxes = {}
        self.boxes_dict = []  # list of boxes to analyse
        self.plots_dict = {}  # list of plots to plot
        self.output_name = []
        self.solver_list = []
        self.basename = None
        self.orignalVideoLen = 0
        self.timer = 0

        self.actionOpen.triggered.connect(self.browseFiles)
        self.actionExport_results.triggered.connect(self.exportResults)
        self.actionSubstract.triggered.connect(self.substract)
        # self.actionSubstract.setDisabled(True)
        self.actionAdd_box.triggered.connect(self.addDraggableRectangle)
        self.view_position_x.triggered.connect(self.plotSelection)
        self.view_position_y.triggered.connect(self.plotSelection)
        self.view_position.triggered.connect(self.plotSelection)
        self.view_phase.triggered.connect(self.plotSelection)
        self.view_violin.triggered.connect(self.plotSelection)
        self.view_viollin_all_on_one.triggered.connect(self.plotSelection)
        self.view_violin_chop.triggered.connect(self.plotSelection)

        self.actionAdd_box = QtWidgets.QAction()
        self.actionAdd_box.setObjectName('actionAdd_box')
        self.menubar.addAction(self.actionAdd_box)
        self.actionAdd_box.setText('Add analysis box')
        self.actionAdd_box.triggered.connect(self.addDraggableRectangle)
        self.actionAdd_box.setShortcut("A")

        self.actionRemove_box = QtWidgets.QAction()
        self.actionRemove_box.setObjectName('actionRemove_box')
        self.menubar.addAction(self.actionRemove_box)
        self.actionRemove_box.setText('Remove analysis box')
        self.actionRemove_box.triggered.connect(self.removeDraggableRectangle)
        self.actionRemove_box.setShortcut("R")

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

        self.json_data = {}

        self.loadParameters()

        self.fileName = ""
        self.videodata = None

        self.cursor = None
        self.plotSelection()  # set options to the booleans wanted even if the user didn't change anything

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
                except Exception:
                    print("Failed to show the first frame.")  # TODO: print appropriate text
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
                except Exception:
                    print("Failed to load the last frame.")
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
        print("Event dropped.")
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
            self.plots_dict[action.objectName()] = action.isChecked()
            print("Menu option '%s' ('%s') is set to %s." % (action.objectName(), action.text(), action.isChecked()))
        self.lineEdit_chop_sec.setEnabled(self.view_violin_chop.isChecked())
        self.label_chop_sec.setEnabled(self.view_violin_chop.isChecked())

    def browseFiles(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                       "All Files (*);;mp4 movie (*.mp4)")
        self.goodFile = 1
        self.loadAndShowFile()

    def unloadFile(self):
        self.views.clear()  # clear the image
        self.boxes.clear()  # clear the list of boxes on the right side

        for box in self.boxes_dict:  # remove the boxes
            box.disconnect()
        self.boxes_dict.clear()

        for solver in self.solver_list:  # remove the arrows
            solver.clear_annotations()

        self.basename = None

        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

    def loadAndShowFile(self):
        try:
            self.videodata = pims.ImageSequence(self.fileName)
        except Exception:
            try:
                self.videodata = pims.Video(self.fileName)
            except Exception:
                print("Failed to load file/folder.")
                return

        print(self.fileName)
        if self.fileName in self.json_data["boxes"]:
            self.saved_boxes = self.json_data["boxes"][self.fileName]
            print("Loaded previously saved boxes.")
        else:
            self.saved_boxes = {}

        try:
            self.unloadFile()
        except AttributeError:
            print("Nothing to clear.")

        print(self.videodata)

        shape = np.shape(self.videodata.get_frame(0))
        try:
            self.orignalVideoLen = len(self.videodata)  # try to get the video length (can vary depending on the Python environment)
        except Exception:
            print("Can't get video length.")

        print("Shape of videodata[0]: %s x %d frames. Object type: %s." % (shape, self.orignalVideoLen, type(self.videodata)))

        self.figure = Figure()
        sub = self.figure.add_subplot(111)
        try:
            self.imshow = sub.imshow(rgb2gray(self.videodata.get_frame(int(self.lineEdit_start_frame.text()))),
                                     cmap='gray')
        except Exception:
            self.imshow = sub.imshow(rgb2gray(self.videodata.get_frame(0)), cmap='gray')

        self.basename = os.path.basename(self.fileName)
        self.views.addItem(self.basename)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('motion_notify_event', self.mouse_event)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.addToolBar(self.toolbar)
        self.stopFrame()  # check new boundaries

        for box in self.saved_boxes.values():
            self._addRectangle(box["number"], box["x0"], box["y0"], box["width"], box["height"])

    def substract(self):  # TODO: edit it to work with boundaries we set
        if self.checkBox_substract.isChecked():
            print("Enabled substract.")
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
                self.imshow.set_cmap(self.comboBox_substract_col.currentText())
                self.figure.canvas.draw()
            except Exception:
                print("Unable to enable substract.")
        else:
            try:
                print("Disabled substract.")
                self.startFrame()
                self.imshow.set_cmap('gray')
                self.figure.canvas.draw()
            except Exception:
                print("Unable to disable substract.")

    def addDraggableRectangle(self):
        if self.cursor is None:  # no file opened, return gracefully
            return

        width = int(self.lineEdit_w.text())
        height = int(self.lineEdit_h.text())

        if self.cursor[0] is not None and self.cursor[1] is not None:
            x0 = self.cursor[0] - width / 2
            y0 = self.cursor[1] - height / 2
        else:
            x0 = width / 2 + 15
            y0 = height / 2 + 15

        number = len(self.boxes_dict)

        self._addRectangle(number, x0, y0, width, height)

        self.saved_boxes[str(number)] = {
            "number": number,
            "x0": x0,
            "y0": y0,
            "width": width,
            "height": height
        }

    def _addRectangle(self, number, x0, y0, width, height):
        print("Adding box %d to figure." % (number))

        ax = self.figure.add_subplot(111)

        rect = patches.Rectangle(xy=(x0, y0), width=width, height=height, linewidth=1, edgecolor='r', facecolor='b', fill=False)
        ax.add_patch(rect)

        text = ax.text(x=x0, y=y0, s=str(number))
        dr = DraggableRectangle(rect, rectangle_number=number, text=text)
        dr.connect()

        self.boxes_dict.append(dr)
        self.boxes.addItem(str(number))

    def removeDraggableRectangle(self):
        length = len(self.boxes_dict)
        if length <= 0:  # no box present, return gracefully
            return

        current = self.boxes.currentRow()
        if current == -1:  # no box selected (-1), delete the last one (length - 1)
            current = length - 1

        print("Removing box %d from figure." % (current))

        rectangle = self.boxes_dict[current]
        rectangle.disconnect()
        rectangle.text.remove()

        self.boxes_dict.pop(current)
        self.boxes.takeItem(current)

        self.figure.axes[0].patches[current].remove()
        self.figure.texts[current].remove()

        self.figure.canvas.draw()

        if str(current) in self.saved_boxes:
            self.saved_boxes.pop(str(current))

    def loadParameters(self):
        with open('settings.json', 'r') as json_file:
            self.json_data = json.load(json_file)

            self.lineEdit_pix_size.setText(str(self.json_data["parameters"]["pixel_size"]))
            self.lineEdit_magn.setText(str(self.json_data["parameters"]["magnification"]))
            self.lineEdit_sub_pix.setText(str(self.json_data["parameters"]["sub_pixel"]))
            self.lineEdit_fps.setText(str(self.json_data["parameters"]["fps"]))
            self.lineEdit_start_frame.setText(str(self.json_data["parameters"]["start_frame"]))
            self.lineEdit_stop_frame.setText(str(self.json_data["parameters"]["stop_frame"]))
            # TODO: is it better when disabled?
            # self.lineEdit_start_frame.editingFinished.connect(self.startFrame)
            # self.lineEdit_stop_frame.editingFinished.connect(self.stopFrame)
            self.lineEdit_w.setText(str(self.json_data["parameters"]["box_width"]))
            self.lineEdit_h.setText(str(self.json_data["parameters"]["box_height"]))
            self.checkBox_track.setChecked(self.json_data["parameters"]["tracking"])
            self.checkBox_compare_first.setChecked(self.json_data["parameters"]["compare_to_first"])
            self.lineEdit_chop_sec.setText(str(self.json_data["parameters"]["chop_sec"]))

            self.comboBox_substract_col.setCurrentText(self.json_data["extra"]["substract_type"])
            for i in plt.colormaps():
                self.comboBox_substract_col.addItem(i)

            self.lineEdit_substract_lvl.setText(str(self.json_data["extra"]["substract_level"]))

            self.comboBox_substract_col.currentIndexChanged.connect(self.substract)
            # TODO: save/load substract checkbox
            self.checkBox_substract.stateChanged.connect(self.substract)
            self.lineEdit_substract_lvl.editingFinished.connect(self.substract)

            self.view_position.setChecked(self.json_data["actions"]["position"])
            self.view_position_x.setChecked(self.json_data["actions"]["position_x"])
            self.view_position_y.setChecked(self.json_data["actions"]["position_y"])
            self.view_phase.setChecked(self.json_data["actions"]["phase"])
            self.view_violin.setChecked(self.json_data["actions"]["violin"])
            self.view_violin_chop.setChecked(self.json_data["actions"]["violin_chop"])
            self.view_viollin_all_on_one.setChecked(self.json_data["actions"]["violin_all_on_one"])

            print("Parameters loaded.")

    def saveParameters(self):
        # Ensure moved boxes are saved with the updated coordinates
        for j in range(len(self.saved_boxes)):
            self.saved_boxes[str(j)]["x0"] = self.boxes_dict[j].x_rect
            self.saved_boxes[str(j)]["y0"] = self.boxes_dict[j].y_rect

        self.json_data["boxes"][self.fileName] = self.saved_boxes

        self.json_data = {
                     "parameters": {
                         "pixel_size": float(self.lineEdit_pix_size.text()),
                         "magnification": int(self.lineEdit_magn.text()),
                         "sub_pixel": int(self.lineEdit_sub_pix.text()),
                         "fps": int(self.lineEdit_fps.text()),
                         "start_frame": int(self.lineEdit_start_frame.text()),
                         "stop_frame": int(self.lineEdit_stop_frame.text()),
                         "box_width": int(self.lineEdit_w.text()),
                         "box_height": int(self.lineEdit_h.text()),
                         "tracking": self.checkBox_track.isChecked(),
                         "compare_to_first": self.checkBox_compare_first.isChecked(),
                         "chop_sec": int(self.lineEdit_chop_sec.text())
                     },
                     "extra": {
                         "substract_type": self.comboBox_substract_col.currentText(),
                         "substract_level": int(self.lineEdit_substract_lvl.text())
                     },
                     "actions": {
                         "position": self.view_position.isChecked(),
                         "position_x": self.view_position_x.isChecked(),
                         "position_y": self.view_position_y.isChecked(),
                         "phase": self.view_phase.isChecked(),
                         "violin": self.view_violin.isChecked(),
                         "violin_chop": self.view_violin_chop.isChecked(),
                         "violin_all_on_one": self.view_viollin_all_on_one.isChecked()
                     },
                     "boxes": self.json_data["boxes"]
        }

        with open('settings.json', 'w') as json_file:
            json.dump(self.json_data, json_file, indent=4)

            print("Parameters saved.")

    def startAnalysis(self):
        self.stopAnalysis()  # ensure no analysis is already running

        if self.videodata is None:  # no video loaded, return gracefully
            return

        for solver in self.solver_list:  # remove the arrows
            solver.clear_annotations()

        self.solver_list.clear()
        self.saveParameters()
        self.output_name = create_dirs(self.fileName, "")
        print("Tracking: %s." % (self.checkBox_track.isChecked()))
        solver = Solver(videodata=self.videodata, fps=float(self.lineEdit_fps.text()),
                        box_dict=self.boxes_dict, solver_number=0,
                        upsample_factor=int(self.lineEdit_sub_pix.text()),
                        stop_frame=int(self.lineEdit_stop_frame.text()),
                        start_frame=int(self.lineEdit_start_frame.text()),
                        res=float(self.lineEdit_pix_size.text()),
                        track=self.checkBox_track.isChecked(),
                        compare_first=self.checkBox_compare_first.isChecked(),
                        figure=self.figure
                        )
        self.solver_list.append(solver)
        self.solver_list[0].progressChanged.connect(self.updateProgress)
        self.solver_list[0].start()
        self.figure.savefig(self.output_name + "boxes_selection.png")
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateProgress)
        self.timer.start(100)
        print("Started timer.")

    def stopAnalysis(self):
        try:
            print("Analysis stopped.")
            self.timer.stop()
            for solver in self.solver_list:
                solver.stop()
        except Exception:  # no timer launched, return gracefully
            pass

    def updateProgress(self):
        for s in self.solver_list:
            if s.progress == 100:
                self.timer.stop()

            for j in range(len(self.boxes_dict)):
                item = self.boxes.item(j)
                item.setText("%d - %d%% (frame %d/%d)"
                             % (j, s.progress, s.current_i - int(self.lineEdit_start_frame.text()), int(self.lineEdit_stop_frame.text()) - int(self.lineEdit_start_frame.text())))

            if s.progress == 100 or self.checkBox_live_preview.isChecked():
                self.imshow.set_data(s.frame_n)
                for r in self.boxes_dict:
                    r.update_from_solver()
                self.figure.canvas.draw()
                self.figure.canvas.flush_events()

    def showResults(self):
        self.saveParameters()

        # print(self.solver_list)
        for solver in self.solver_list:
            plot_results(shift_x=solver.shift_x, shift_y=solver.shift_y, shift_x_y_error=solver.shift_x_y_error,
                         box_shift=solver.box_shift, fps=solver.fps, res=solver.res,
                         output_name=self.output_name, plots_dict=self.plots_dict, boxes_dict=self.boxes_dict,
                         chop_sec=float(self.lineEdit_chop_sec.text()), start_frame=solver.start_frame, shift_p=solver.shift_p)

        print("Plots shown.")

    def exportResults(self):
        for solver in self.solver_list:
            for j in range(len(solver.box_dict)):
                export_results(shift_x=solver.shift_x[j], shift_y=solver.shift_y[j], box_shift=solver.box_shift[j],
                               fps=solver.fps, res=solver.res,
                               w=solver.box_dict[j].rect._width, h=solver.box_dict[j].rect._height,
                               z_std=solver.z_std[j],
                               dz_rms=solver.z_rms[j],
                               v=solver.v_rms[j],
                               output_name=self.output_name + str(j))
        print("Files exported.")


if __name__ == '__main__':
    print("Python interpreter: %s." % (os.path.dirname(sys.executable)))
    sys.stdout.flush()
    app = QApplication(sys.argv)
    menu = Main()
    menu.show()
    sys.exit(app.exec_())
