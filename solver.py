from PyQt5.QtCore import QThread, pyqtSignal
from skimage.color import rgb2gray
from skimage.feature import register_translation
from threading import RLock
import numpy as np
from decimal import *

verrou = RLock()

class Solver(QThread):
    progressChanged = pyqtSignal(int, int, object)

    def __init__(self, videodata, fps, res, box_dict, solver_number, start_frame, stop_frame, my_upsample_factor):
        QThread.__init__(self)
        self.solver_number = solver_number # Stores the ID of the solver
        self.videodata = videodata # Stores an access to the file to get frames
        self.row_min = []
        self.row_max = []
        self.col_min = []
        self.col_max = []
        self.fps = fps # Frames per seconds
        self.res = res # Size of pixel (um / pix)
        self.my_upsample_factor = my_upsample_factor
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.box_dict = box_dict
        self.go_on=True
        print("box_dict in solver")
        print(box_dict)

        self.z_std = []
        self.z_rms = []
        self.v_rms = []

        self.shift_x = [[] for _ in self.box_dict]
        self.shift_y = [[] for _ in self.box_dict]
        self.shift_x_y_error = [[] for _ in self.box_dict]

    def run(self):
        self._crop_coord()
        self._calcul_phase_corr()
        self.get_z_delta_rms()
        self.get_z_std()

    def stop(self):
        self.go_on=False
        print("Solver got the stop() method called")

    def _calcul_phase_corr(self):
        print("self.go_on is set to "+ str(self.go_on))
        # calculate cross correlation for all frames for the selected  polygon crop
        import time_logging
        #print("row min row max col min col max" + str([self.row_min,self.row_max, self.col_min,self.col_max]))
        image_1 = []
        with verrou:
            frame_1 = rgb2gray(self.videodata.get_frame(0))
        for j in range(len(self.box_dict)):
            image_1.append(rgb2gray(
                frame_1[self.row_min[j]:self.row_max[j], self.col_min[j]:self.col_max[j]]))
        #print(image_1.shape)
        lenght=self.stop_frame - self.start_frame
        #t = time_logging.start()
        #t = time_logging.end("Load frame", t)
        progress_pivot = 0 - 5
        for i in range(self.start_frame, self.stop_frame+1):
            if self.go_on:
                with verrou:
                    frame_n = self.videodata.get_frame(i)
                for j in range(len(self.box_dict)):
                    image_n = rgb2gray(
                        frame_n[self.row_min[j]:self.row_max[j], self.col_min[j]:self.col_max[j]])
                    #t = time_logging.end("Load frame", t)
                    shift, error, diffphase = register_translation(image_1[j], image_n, self.my_upsample_factor)
                    #t = time_logging.end("Compute register_translation", t)
                    self.shift_x[j].append(shift[1])
                    self.shift_y[j].append(shift[0])
                    self.shift_x_y_error[j].append(error)
                self.progress = int(((i-self.start_frame) / lenght) * 100)
                #self.progressChanged.emit(self.progress, i, )
                if self.progress>progress_pivot+4:
                    self.progressChanged.emit(self.progress, i, frame_n)
                    print(str(self.progress) + "%: analyse frame " + str(i-self.start_frame) + "/" + str(self.stop_frame-self.start_frame))
                    progress_pivot=self.progress
                    #print(time_logging.text_statistics())
                #print(self.shift_x)
                #print("\n")
            else:
                return
            #print(time_logging.text_statistics())

    def _crop_coord(self):
        for i in range(len(self.box_dict)):
            self.row_min.append(int(self.box_dict[i].y_rect))
            self.row_max.append(int(self.box_dict[i].y_rect) + self.box_dict[i].rect._height)
            self.col_min.append(int(self.box_dict[i].x_rect))
            self.col_max.append(int(self.box_dict[i].x_rect) + self.box_dict[i].rect._width)

    def get_z_std(self):
        self.z_std = self._z_std_calcul()
        #print(self.z_std)
        return self.z_std

    def _z_std_calcul(self):
        z_dec = []
        for j in range(len(self.box_dict)):
            x = [i * self.res for i in self.shift_x[j]]
            y = [i * self.res for i in self.shift_y[j]]
            z = np.sqrt((np.std(x)) ** 2 + (np.std(y)) ** 2)
            z_dec.append(Decimal(str(z)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))  # special Decimal class for the correct rounding

        return z_dec  # rounding, leaves 3 digits after comma

    def get_z_delta_rms(self):
        self.z_rms, self.v_rms = self._z_delta_rms_calcul()
        #print(self.z_std, self.v_rms)
        return self.z_rms, self.v_rms

    def _z_delta_rms_calcul(self):  # To modify
        z_tot_dec = []
        v_dec = []
        for j in range(len(self.box_dict)):
            x = [i * self.res for i in self.shift_x[j]]
            y = [i * self.res for i in self.shift_y[j]]
            dx = []
            dy = []
            dz = []
            dv = []
            for i in range(1, len(x)):
                dx.append(x[i] - x[i - 1])
                dy.append(y[i] - y[i - 1])
                dz.append(np.sqrt(dx[i - 1] ** 2 + dy[i - 1] ** 2))
            z_tot = np.sum(dz)
            v = (self.fps / (len(x) - 1)) * z_tot
            z_tot_dec.append(Decimal(str(z_tot)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))  # special Decimal class for the correct rounding
            v_dec.append(Decimal(str(v)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))  # special Decimal class for the correct rounding
        return z_tot_dec, v_dec  # rounding, leaves 3 digits after comma
