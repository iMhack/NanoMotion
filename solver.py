from PyQt5.QtCore import QThread, pyqtSignal
from skimage.color import rgb2gray
from skimage.feature import register_translation
from threading import RLock
import numpy as np
from decimal import *

verrou = RLock()

class Solver(QThread):
    progressChanged = pyqtSignal(int,int)

    def __init__(self, videodata, fps, res, w, h, x_rect, y_rect, solver_number, start_frame, stop_frame,
                 output_name, my_upsample_factor):
        QThread.__init__(self)
        self.solver_number = solver_number # Stores the ID of the solver
        self.videodata = videodata # Stores an access to the file to get frames
        self.row_min = None
        self.row_max = None
        self.col_min = None
        self.col_max = None
        self.fps = fps # Frames per seconds
        self.res = res # Size of pixel (um / pix)
        self.w = w
        self.h = h
        self.x_rect = x_rect
        self.y_rect = y_rect
        self.my_upsample_factor = my_upsample_factor
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.output_name = output_name

        self.z_std = None
        self.z_rms = None
        self.v_rms = None

        self.shift_x = []
        self.shift_y = []

    def run(self):
        self._crop_coord()
        self._calcul_phase_corr()
        self.get_z_delta_rms()
        self.get_z_std()

    def _calcul_phase_corr(self):
        # calculate cross correlation for all frames for the selected  polygon crop
        import time_logging
        print("row min row max col min col max" + str([self.row_min,self.row_max, self.col_min,self.col_max]))
        with verrou:
            image_1 = rgb2gray(
                self.videodata.get_frame(0)[self.row_min:self.row_max, self.col_min:self.col_max])  # Lock
        progress_pivot = 0
        print(image_1.shape)
        lenght=self.stop_frame - self.start_frame
        t = time_logging.start()
        #t = time_logging.end("Load frame", t)
        for i in range(self.start_frame, self.stop_frame+1):
            self.progress = int(((i-self.start_frame) / lenght) * 100)
            if self.progress>progress_pivot+4:
                self.progressChanged.emit(self.solver_number, self.progress)
                print(str(self.progress) + "%: analyse frame " + str(i-self.start_frame) + "/" + str(self.stop_frame-self.start_frame))
                progress_pivot=self.progress
                print(time_logging.text_statistics())
            with verrou:
                image_2 = rgb2gray(
                    self.videodata.get_frame(i)[self.row_min:self.row_max, self.col_min:self.col_max])  # Lock
            # subpixel precision
            t = time_logging.end("Load frame", t)
            shift, error, diffphase = register_translation(image_1, image_2, self.my_upsample_factor)
            t = time_logging.end("Compute register_translation", t)
            self.shift_x.append(shift[1])
            self.shift_y.append(shift[0])
        print(time_logging.text_statistics())

    def _crop_coord(self):
        self.row_min = self.y_rect
        self.row_max = self.y_rect + self.h
        self.col_min = self.x_rect
        self.col_max = self.x_rect + self.w

    def get_z_std(self):
        if self.z_std == None:
            self.z_std = self._z_std_calcul()
            return self.z_std
        else:
            return self.z_std

    def _z_std_calcul(self):
        x = [i * self.res for i in self.shift_x]
        y = [i * self.res for i in self.shift_y]
        z = np.sqrt((np.std(x)) ** 2 + (np.std(y)) ** 2)
        z_dec = Decimal(str(z))  # special Decimal class for the correct rounding
        return z_dec.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)  # rounding, leaves 3 digits after comma

    def get_z_delta_rms(self):
        if self.z_rms is None or self.v_rms is None:
            self.z_std, self.v_rms = self._z_delta_rms_calcul()
            return self.z_rms, self.v_rms
        else:
            return self.z_rms, self.v_rms

    def _z_delta_rms_calcul(self):  # To modify
        x = [i * self.res for i in self.shift_x]
        y = [i * self.res for i in self.shift_y]
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
        z_tot_dec = Decimal(str(z_tot))  # special Decimal class for the correct rounding
        v_dec = Decimal(str(v))  # special Decimal class for the correct rounding
        return z_tot_dec.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP), v_dec.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)  # rounding, leaves 3 digits after comma
