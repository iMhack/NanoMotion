from PyQt5.QtCore import QThread, pyqtSignal
from skimage.color import rgb2gray
from skimage.registration import phase_cross_correlation
from threading import RLock
import numpy as np

verrou = RLock()


class Solver(QThread):
    progressChanged = pyqtSignal(int, int, object)

    def __init__(self, videodata, fps, res, box_dict, solver_number, start_frame, stop_frame, my_upsample_factor,
                 track, compare_first):
        QThread.__init__(self)
        self.solver_number = solver_number  # stores the ID of the solver
        self.videodata = videodata  # stores an access to the file to get frames

        self.row_min = []
        self.row_max = []
        self.col_min = []
        self.col_max = []
        self.fps = fps  # frames per seconds
        self.res = res  # size of one pixel (um / pixel)
        self.my_upsample_factor = my_upsample_factor
        self.track = track
        self.compare_first = compare_first
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.box_dict = box_dict
        self.go_on = True
        print("box_dict in solver")
        print(box_dict)

        self.z_std = []
        self.z_rms = []
        self.v_rms = []

        self.shift_x = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.shift_y = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.shift_p = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.shift_x_y_error = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.box_shift = [[0 for _ in range(2)] for _ in self.box_dict]

        self.frame_n = self.videodata.get_frame(start_frame)
        self.progress = 0
        self.actual_i = -1

    def run(self):
        try:
            self._crop_coord()
            self._calcul_phase_corr()
            self.get_z_delta_rms()
            self.get_z_std()
        except UserWarning:
            self.stop()

    def stop(self):
        self.go_on = False
        print("Solver thread has been flagged to stop.")

    def _calcul_phase_corr(self):
        print("self.go_on is set to " + str(self.go_on))
        # calculate cross correlation for all frames for the selected  polygon crop
        import time_logging
        # print("row min row max col min col max" + str([self.row_min,self.row_max, self.col_min,self.col_max]))
        image_1 = []
        with verrou:
            frame_1 = rgb2gray(self.videodata.get_frame(0))
        for j in range(len(self.box_dict)):
            image_1.append(rgb2gray(
                frame_1[self.row_min[j]:self.row_max[j], self.col_min[j]:self.col_max[j]]))
            print('Box position started ' + str(
                (self.row_min[j], self.row_max[j], self.col_min[j], self.col_max[j])) + '.')
        # print(image_1.shape)
        lenght = self.stop_frame - self.start_frame
        # t = time_logging.start()
        # t = time_logging.end("Load frame", t)
        progress_pivot = 0 - 5
        for i in range(self.start_frame, self.stop_frame + 1):  # i run on all the frames
            self.actual_i = i
            if self.go_on:  # Condition checked to be able to stop the thread
                with verrou:
                    self.frame_n = rgb2gray(self.videodata.get_frame(i))
                for j in range(len(self.box_dict)):  # j runs on all the boxes
                    image_n = (self.frame_n[self.row_min[j]:self.row_max[j], self.col_min[j]:self.col_max[j]])
                    # t = time_logging.end("Load frame", t)
                    shift, error, diffphase = phase_cross_correlation(image_n, image_1[j], upsample_factor=self.my_upsample_factor)
                    if not self.compare_first:  # We store the actual as the futur old one
                        image_1[j] = image_n
                        if not i == self.start_frame:
                            # print('Before reshift'+str(shift[1])+'and box_shift is '+str(self.box_shift[j][0]))
                            shift[1] = self.shift_x[j][-1] + shift[1]-self.box_shift[j][0]  # This had to be done to have the same output
                            shift[0] = -self.shift_y[j][-1] + shift[0]-self.box_shift[j][1]
                            diffphase = diffphase #Well, i don't know
                            # print('After reshift'+str(shift[1])) # We changed the original frame at each new box mvt.
                            # So it was OK to check the value of shift to see if this box had to be moved again. As
                            # we change the frame every time, it have to be done differently.

                    # t = time_logging.end("Compute register_translation", t)
                    self.shift_x[j][i-self.start_frame] = (shift[1] + self.box_shift[j][0])
                    self.shift_y[j][i-self.start_frame] = (-shift[0] - self.box_shift[j][1])
                    self.shift_p[j][i-self.start_frame] = (diffphase)
                    self.shift_x_y_error[j][i] = (error)
                    if self.track and (abs(shift[1]) >= 1 or abs(shift[0]) >= 1):
                        # print('Moving box ! ' + str(shift[1]) + 'x, ' + str(shift[0]) + 'y !!!')
                        # print('Old box position was ' + str(
                        # (self.row_min[j], self.row_max[j], self.col_min[j], self.col_max[j])) + '.')
                        # Take actual frame as reference.
                        self.box_dict[j].x_rect += int(shift[1])  # 1.2 -> 1-0, 1.8 -> 3-1
                        self.box_dict[j].y_rect += int(shift[0])
                        self._crop_coord(j)  # Uptate the boundaries
                        image_1[j] = self.frame_n[self.row_min[j]:self.row_max[j], self.col_min[j]:self.col_max[j]]
                        # Remember the shift up-to now
                        self.box_shift[j][0] += int(shift[1])
                        self.box_shift[j][1] += int(shift[0])
                        # print('New box position is ' + str(
                        #    (self.row_min[j], self.row_max[j], self.col_min[j], self.col_max[j])) + '.')
                        # print('Ended moving box ! new')
                self.progress = int(((i - self.start_frame) / lenght) * 100)
                # self.progressChanged.emit(self.progress, i, )
                if self.progress > progress_pivot + 4:
                    # self.progressChanged.emit(self.progress, i, self.frame_n)
                    print(str(self.progress) + "%: analyse frame " + str(i - self.start_frame) + "/" + str(
                        self.stop_frame - self.start_frame))
                    progress_pivot = self.progress
                    # print(time_logging.text_statistics())
                # print(self.shift_x)
                # print("\n")
            else:
                return
            # print(time_logging.text_statistics())

    def _crop_coord(self, which=-1):
        if which == -1:
            self.row_min.clear()
            self.row_max.clear()
            self.col_min.clear()
            self.col_max.clear()
            for i in range(len(self.box_dict)):
                self.row_min.append(int(self.box_dict[i].y_rect))
                self.row_max.append(int(self.box_dict[i].y_rect) + self.box_dict[i].rect._height)
                self.col_min.append(int(self.box_dict[i].x_rect))
                self.col_max.append(int(self.box_dict[i].x_rect) + self.box_dict[i].rect._width)
        else:
            # print('_crop_coord of ' + str(which))
            i = which
            self.row_min[i] = int(self.box_dict[i].y_rect)
            self.row_max[i] = int(self.box_dict[i].y_rect) + self.box_dict[i].rect._height
            self.col_min[i] = int(self.box_dict[i].x_rect)
            self.col_max[i] = int(self.box_dict[i].x_rect) + self.box_dict[i].rect._width

    def get_z_std(self):
        self.z_std = self._z_std_calcul()
        # print(self.z_std)
        return self.z_std

    def _z_std_calcul(self):
        z_dec = []
        for j in range(len(self.box_dict)):
            x = [i * self.res for i in self.shift_x[j]]
            y = [i * self.res for i in self.shift_y[j]]
            z = np.sqrt((np.std(x)) ** 2 + (np.std(y)) ** 2)
            z_dec.append(Decimal(str(z)).quantize(Decimal('0.001'),
                                                  rounding=ROUND_HALF_UP))  # special Decimal class for the correct rounding

        return z_dec  # rounding, leaves 3 digits after comma

    def get_z_delta_rms(self):
        self.z_rms, self.v_rms = self._z_delta_rms_calcul()
        # print(self.z_std, self.v_rms)
        return self.z_rms, self.v_rms

    def _z_delta_rms_calcul(self):  # To modify
        z_tot_dec = []
        v_dec = []
        for j in range(len(self.box_dict)):
            x_j = self.shift_x[j]
            y_j = self.shift_y[j]

            if x_j or y_j is None:  # analysis stopped, return gracefully
                raise UserWarning("Analysis stopped.")

            x = [i * self.res for i in x_j]
            y = [i * self.res for i in y_j]
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
            z_tot_dec.append(Decimal(str(z_tot)).quantize(Decimal('0.001'),
                                                          rounding=ROUND_HALF_UP))  # special Decimal class for the correct rounding
            v_dec.append(Decimal(str(v)).quantize(Decimal('0.001'),
                                                  rounding=ROUND_HALF_UP))  # special Decimal class for the correct rounding
        return z_tot_dec, v_dec  # rounding, leaves 3 digits after comma
