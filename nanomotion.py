import argparse
import hashlib
import json
import os
import os.path
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import pims
import utils
from dragRectangle import DraggableRectangle
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.uic import loadUiType
from solver import Solver

dirname = os.path.dirname(__file__)

# To keep the tips when editing, run pyuic5 mainMenu.ui -o mainMenu.py in the terminal
Ui_MainWindow, QMainWindow = loadUiType(os.path.join(dirname, "mainMenu.ui"))


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.figure = None
        self.opened_plots = []
        self.saved_boxes = {}
        self.boxes_dict = []  # list of boxes to analyse
        self.plots_dict = {}  # list of plots to plot
        self.output_basepath = None
        self.solver = None
        self.basename = None
        self.originalVideoLength = 0
        self.timer = 0

        self.actionOpen.triggered.connect(self.browseFiles)
        self.actionExport_results.triggered.connect(self.exportResults)
        self.actionSubstract.triggered.connect(self.substract)
        self.actionSubstract.setDisabled(True)  # TODO: improve substraction
        self.actionAdd_box.triggered.connect(self.addDraggableRectangle)

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
        self.actionStart_solver.setText('Start analysis')
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
        self.actionStop_solver.setText('Stop analysis')
        self.actionStop_solver.triggered.connect(self.stopAnalysis)

        self.actionReset_boxes = QtWidgets.QAction()
        self.actionReset_boxes.setObjectName('actionReset_boxes')
        self.menubar.addAction(self.actionReset_boxes)
        self.actionReset_boxes.setText('Reset boxes')
        self.actionReset_boxes.triggered.connect(self.reset_boxes)

        self.json_data = {}

        self.fileName = None
        self.id = None
        self.videodata = None

        self.loadParameters()

        if args.open is not None:
            self.fileName = args.open

        self.cursor = None

        self.loadAndShowFile()

        if args.autostart:
            self.startAnalysis()

    def startFrame(self, update=True):
        if self.originalVideoLength != float('inf'):
            if int(self.lineEdit_start_frame.text()) >= self.originalVideoLength:
                self.lineEdit_start_frame.setText(str(self.originalVideoLength - 1))

            if self.fileName != "":
                try:
                    if update:
                        self.imshow.set_data(skimage.color.rgb2gray(
                            self.videodata.get_frame(int(self.lineEdit_start_frame.text()))))
                        self.figure.canvas.draw()
                        self.figure.canvas.flush_events()
                    return self.videodata.get_frame(int(self.lineEdit_start_frame.text()))
                except Exception:
                    print("Failed to show the first frame.")
                    return 0
        else:
            return 0

    def stopFrame(self):
        if self.originalVideoLength != float('inf'):
            if int(self.lineEdit_stop_frame.text()) >= self.originalVideoLength:
                self.lineEdit_stop_frame.setText(str(self.originalVideoLength - 1))

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
        print("File dropped in the window.")

        if e.mimeData().hasUrls:
            for url in e.mimeData().urls():
                fname = str(url.toLocalFile())
            self.fileName = fname
            self.loadAndShowFile()

    def mouse_event(self, e):
        self.cursor = (e.xdata, e.ydata)

    def setPlotOptions(self):
        for action in self.menuView_plot.actions():
            self.plots_dict[action.objectName()] = action.isChecked()
            print("Menu option '%s' ('%s') is set to %s." %
                  (action.objectName(), action.text(), action.isChecked()))

        self.lineEdit_chop_sec.setEnabled(self.view_violin_chop.isChecked())
        self.label_chop_sec.setEnabled(self.view_violin_chop.isChecked())

    def browseFiles(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                       "All Files (*);mp4 movie (*.mp4)")
        self.loadAndShowFile()

    def unloadFile(self):
        self.views.clear()  # clear the image
        self.boxes.clear()  # clear the list of boxes on the right side

        for box in self.boxes_dict:  # remove the boxes
            box.disconnect()
        self.boxes_dict.clear()

        if self.solver is not None:
            self.solver.clear_annotations()

        self.basename = None

        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

    def loadAndShowFile(self):
        if self.fileName is None:
            return

        try:
            if os.path.isfile(self.fileName):
                self.videodata = pims.Video(self.fileName)

                with open(self.fileName, "rb") as input:
                    self.id = hashlib.blake2b(input.read()).hexdigest()
            else:
                self.videodata = pims.ImageSequence(self.fileName)

                self.id = self.fileName
        except Exception:
            print("Failed to load file/folder.")
            return

        print("Loaded file: '%s'." % (self.fileName))
        if self.id in self.json_data["boxes"]:
            self.saved_boxes = self.json_data["boxes"][self.id]

            print("Loaded previously saved boxes (with blake2b hash).")
        elif self.fileName in self.json_data["boxes"]:  # fallback to filename before giving up
            self.saved_boxes = self.json_data["boxes"].pop(self.fileName)  # remove previous id

            self.json_data["boxes"][self.id] = self.saved_boxes  # set new id

            print("Loaded previously saved boxes (with filename).")
        else:
            self.saved_boxes = {}

        try:
            self.unloadFile()
        except AttributeError:
            print("Nothing to clear.")

        shape = np.shape(self.videodata.get_frame(0))
        try:
            # try to get the video length (can vary depending on the Python environment)
            self.originalVideoLength = len(self.videodata)
        except Exception:
            print("Can't get video length.")

        print("Shape of videodata[0]: %s x %d frames. Object type: %s." %
              (shape, self.originalVideoLength, type(self.videodata)))

        self.figure = Figure()
        sub = self.figure.add_subplot(111)
        try:
            self.imshow = sub.imshow(skimage.color.rgb2gray(self.videodata.get_frame(int(self.lineEdit_start_frame.text()))),
                                     cmap='gray')
        except Exception:
            self.imshow = sub.imshow(skimage.color.rgb2gray(
                self.videodata.get_frame(0)), cmap='gray')

        self.views.addItem(self.fileName)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('motion_notify_event', self.mouse_event)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.addToolBar(self.toolbar)
        self.stopFrame()  # check new boundaries

        for box in self.saved_boxes.values():
            self._addRectangle(box["number"], box["x0"], box["y0"], box["width"], box["height"])

    def substract(self):
        if self.checkBox_substract.isChecked():
            print("Enabled substract.")
            try:
                start_frame = int(self.lineEdit_start_frame.text())
                stop_frame = int(self.lineEdit_stop_frame.text())
                n_frames = stop_frame - start_frame
                first_frame = skimage.color.rgb2gray(self.videodata.get_frame(start_frame))
                print(type(first_frame))
                cumulative_frame = np.zeros(np.shape(first_frame))
                print(type(cumulative_frame))
                for i in range(stop_frame, start_frame, -int(n_frames / int(self.lineEdit_substract_lvl.text()))):
                    print(i)
                    cumulative_frame += skimage.color.rgb2gray(
                        self.videodata.get_frame(i)) - first_frame

                self.imshow.set_data(skimage.color.rgb2gray(cumulative_frame))
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

        rect = patches.Rectangle(xy=(x0, y0), width=width, height=height,
                                 linewidth=1, edgecolor='r', facecolor='b', fill=False)
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
        with open(os.path.join(dirname, "settings.json"), "r") as json_file:
            self.json_data = json.load(json_file)

            if "last_file" in self.json_data:
                self.fileName = self.json_data["last_file"]

            self.lineEdit_pix_size.setText(str(self.json_data["parameters"]["pixel_size"]))
            self.lineEdit_magn.setText(str(self.json_data["parameters"]["magnification"]))
            self.lineEdit_sub_pix.setText(str(self.json_data["parameters"]["sub_pixel"]))
            self.lineEdit_fps.setText(str(self.json_data["parameters"]["fps"]))
            self.lineEdit_start_frame.setText(str(self.json_data["parameters"]["start_frame"]))
            self.lineEdit_stop_frame.setText(str(self.json_data["parameters"]["stop_frame"]))
            self.lineEdit_w.setText(str(self.json_data["parameters"]["box_width"]))
            self.lineEdit_h.setText(str(self.json_data["parameters"]["box_height"]))
            self.lineEdit_chop_sec.setText(str(self.json_data["parameters"]["chop_sec"]))
            self.checkBox_track.setChecked(self.json_data["parameters"]["tracking"])
            self.checkBox_compare_first.setChecked(
                self.json_data["parameters"]["compare_to_first"])
            self.checkBox_filter.setChecked(self.json_data["parameters"]["filter"])
            self.checkBox_export.setChecked(self.json_data["parameters"]["export"])

            self.comboBox_substract_col.setCurrentText(self.json_data["extra"]["substract_type"])
            for i in plt.colormaps():
                self.comboBox_substract_col.addItem(i)

            self.lineEdit_substract_lvl.setText(str(self.json_data["extra"]["substract_level"]))

            self.comboBox_substract_col.currentIndexChanged.connect(self.substract)
            self.checkBox_substract.stateChanged.connect(self.substract)
            self.lineEdit_substract_lvl.editingFinished.connect(self.substract)

            self.view_position.setChecked(self.json_data["actions"]["position"])
            self.view_position_x.setChecked(self.json_data["actions"]["position_x"])
            self.view_position_y.setChecked(self.json_data["actions"]["position_y"])
            self.view_position_all_on_one.setChecked(
                self.json_data["actions"]["position_all_on_one"])
            self.view_phase.setChecked(self.json_data["actions"]["phase"])
            self.view_violin.setChecked(self.json_data["actions"]["violin"])
            self.view_violin_chop.setChecked(self.json_data["actions"]["violin_chop"])
            self.view_violin_all_on_one.setChecked(self.json_data["actions"]["violin_all_on_one"])
            self.view_step_length.setChecked(self.json_data["actions"]["step_length"])
            self.view_experimental.setChecked(self.json_data["actions"]["experimental"])

            print("Parameters loaded.")

    def saveParameters(self):
        # Ensure moved boxes are saved with the updated coordinates
        for j in range(len(self.saved_boxes)):
            self.saved_boxes[str(j)]["x0"] = self.boxes_dict[j].x_rect
            self.saved_boxes[str(j)]["y0"] = self.boxes_dict[j].y_rect

        self.json_data["boxes"][self.id] = self.saved_boxes

        self.json_data = {
            "last_file": self.fileName,
            "parameters": {
                "pixel_size": float(self.lineEdit_pix_size.text()),
                "magnification": int(self.lineEdit_magn.text()),
                "sub_pixel": int(self.lineEdit_sub_pix.text()),
                "fps": int(self.lineEdit_fps.text()),
                "start_frame": int(self.lineEdit_start_frame.text()),
                "stop_frame": int(self.lineEdit_stop_frame.text()),
                "box_width": int(self.lineEdit_w.text()),
                "box_height": int(self.lineEdit_h.text()),
                "chop_sec": int(self.lineEdit_chop_sec.text()),
                "tracking": self.checkBox_track.isChecked(),
                "compare_to_first": self.checkBox_compare_first.isChecked(),
                "filter": self.checkBox_filter.isChecked(),
                "export": self.checkBox_export.isChecked()
            },
            "extra": {
                "substract_type": self.comboBox_substract_col.currentText(),
                "substract_level": int(self.lineEdit_substract_lvl.text())
            },
            "actions": {
                "position": self.view_position.isChecked(),
                "position_x": self.view_position_x.isChecked(),
                "position_y": self.view_position_y.isChecked(),
                "position_all_on_one": self.view_position_all_on_one.isChecked(),
                "phase": self.view_phase.isChecked(),
                "violin": self.view_violin.isChecked(),
                "violin_chop": self.view_violin_chop.isChecked(),
                "violin_all_on_one": self.view_violin_all_on_one.isChecked(),
                "step_length": self.view_step_length.isChecked(),
                "experimental": self.view_experimental.isChecked()
            },
            "boxes": self.json_data["boxes"]
        }

        with open(os.path.join(dirname, "settings.json"), "w") as json_file:
            json.dump(self.json_data, json_file, indent=4)

            print("Parameters saved.")

    def startAnalysis(self):
        self.stopAnalysis()  # ensure no analysis is already running
        # TODO: return if an analysis is already running instead of restarting a new analysis

        if self.videodata is None:  # no video loaded, return gracefully
            return

        if self.solver is not None:  # remove the arrows
            self.solver.clear_annotations()

        self.setPlotOptions()
        self.saveParameters()

        self.output_basepath = utils.ensure_directory(self.fileName, "results")

        if self.checkBox_export.isChecked():
            write_target = utils.ensure_directory(self.fileName, "exports")
        else:
            write_target = None

        print("Tracking: %s." % (self.checkBox_track.isChecked()))
        self.solver = Solver(videodata=self.videodata,
                             fps=float(self.lineEdit_fps.text()),
                             box_dict=self.boxes_dict,
                             upsample_factor=int(self.lineEdit_sub_pix.text()),
                             stop_frame=int(self.lineEdit_stop_frame.text()),
                             start_frame=int(self.lineEdit_start_frame.text()),
                             res=float(self.lineEdit_pix_size.text()),
                             track=self.checkBox_track.isChecked(),
                             compare_first=self.checkBox_compare_first.isChecked(),
                             filter=self.checkBox_filter.isChecked(),
                             figure=self.figure,
                             write_target=write_target
                             )

        self.figure.savefig("%s_overview.png" % (self.output_basepath))

        self.solver.progressChanged.connect(self.updateProgress)
        self.solver.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.updateProgress)
        self.timer.start(100)
        print("Started timer.")

    def stopAnalysis(self):
        print("Analysis stopped.")

        try:
            self.timer.stop()

            if self.solver is not None:
                self.solver.stop()
        except Exception:  # no timer or solver launched, return gracefully
            pass

    def updateProgress(self):
        if self.solver is not None:
            if self.solver.progress == 100:
                self.timer.stop()

            for j in range(len(self.boxes_dict)):
                item = self.boxes.item(j)
                item.setText("%d - %d%% (frame %d/%d)"
                             % (j,
                                self.solver.progress,
                                self.solver.current_i - int(self.lineEdit_start_frame.text()),
                                int(self.lineEdit_stop_frame.text()) - int(self.lineEdit_start_frame.text())))

            if self.solver.progress == 100 or self.checkBox_live_preview.isChecked():
                self.imshow.set_data(self.solver.frame_n)
                for r in self.boxes_dict:
                    r.update_from_solver()
                self.figure.canvas.draw()
                self.figure.canvas.flush_events()

                if self.solver.progress == 100:
                    if args.show_results:
                        self.showResults()

                    if args.export_results:
                        self.exportResults()

                    if args.quit:
                        app.quit()

    def showResults(self):
        if self.solver is None or self.solver.progress < 100:
            return

        self.setPlotOptions()
        self.saveParameters()  # only save parameters if there are plots to open

        self.opened_plots = utils.plot_results(shift_x=self.solver.shift_x,
                                               shift_y=self.solver.shift_y,
                                               shift_p=self.solver.shift_p,
                                               shift_x_y_error=self.solver.shift_x_y_error,
                                               box_shift=self.solver.box_shift,
                                               fps=self.solver.fps,
                                               res=self.solver.res,
                                               input_path=self.fileName,
                                               output_basepath=self.output_basepath,
                                               plots_dict=self.plots_dict,
                                               boxes_dict=self.boxes_dict,
                                               chop_duration=float(self.lineEdit_chop_sec.text()),
                                               start_frame=self.solver.start_frame)

        print("%d plots shown." % (len(self.opened_plots)))

    def reset_boxes(self):
        self.loadAndShowFile()  # reloading the file resets everything

    def exportResults(self):
        if self.solver is not None:
            utils.export_results(shift_x=self.solver.shift_x,
                                 shift_y=self.solver.shift_y,
                                 shift_p=self.solver.shift_p,
                                 shift_x_y_error=self.solver.shift_x_y_error,
                                 box_shift=self.solver.box_shift,
                                 fps=self.solver.fps,
                                 res=self.solver.res,
                                 output_basepath=self.output_basepath,
                                 boxes_dict=self.boxes_dict,
                                 start_frame=self.solver.start_frame)

        print("Files exported.")


if __name__ == '__main__':
    print("Python interpreter: %s, version: %s." % (os.path.dirname(sys.executable), sys.version))
    sys.stdout.flush()

    parser = argparse.ArgumentParser(description="Nanomotion software.")
    parser.add_argument("-o", "--open", help="File to open.", default=None)
    parser.add_argument("-a", "--autostart", help="Start the analysis.", action="store_true")
    parser.add_argument("-r", "--show_results",
                        help="Show the results after the analysis.", action="store_true")
    parser.add_argument("-x", "--export_results",
                        help="Export  the results after the analysis.", action="store_true")
    parser.add_argument("-q", "--quit", help="Quit after the analysis.", action="store_true")

    args = parser.parse_args()

    app = QApplication(sys.argv)

    menu = Main()
    menu.show()

    sys.exit(app.exec_())
