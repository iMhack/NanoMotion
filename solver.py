from PyQt5.QtCore import QThread, pyqtSignal
from skimage.color import rgb2gray
from skimage.registration import phase_cross_correlation
from threading import RLock
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import math

lock = RLock()


class Solver(QThread):
    progressChanged = pyqtSignal(int, int, object)

    def __init__(self, videodata, fps, res, box_dict, solver_number, start_frame, stop_frame, upsample_factor,
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
        self.upsample_factor = upsample_factor
        self.track = track
        self.compare_first = compare_first
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.box_dict = box_dict

        self.go_on = True

        self.z_std = []
        self.z_rms = []
        self.v_rms = []

        self.shift_x = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.shift_y = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.shift_p = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.shift_x_y_error = [[None for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.box_shift = [[[None, None] for _ in range(self.start_frame, self.stop_frame + 1)] for _ in self.box_dict]
        self.total_box_shift = [[0, 0] for _ in self.box_dict]
        self.cumulated_shift = [[0, 0] for _ in self.box_dict]
        self.total_value_jump = [[0, 0, 0] for _ in self.box_dict]

        self.frame_n = self.videodata.get_frame(start_frame)
        self.progress = 0
        self.current_i = -1

    def run(self):
        try:
            self._crop_coord()
            self._compute_phase_corr()
            self.get_z_delta_rms()
            self.get_z_std()
        except UserWarning:
            self.stop()

    def stop(self):
        self.go_on = False
        print("Solver thread has been flagged to stop.")

    def _close_to_zero(self, value):
        if value < 0:
            return math.ceil(value)
        else:
            return math.floor(value)

    def _compute_phase_corr(self):  # compute the cross correlation for all frames for the selected polygon crop
        print("self.go_on is set to %s." % (self.go_on))

        images = []  # last image subimages array
        jump = []

        with lock:  # store the first subimages for later comparison
            frame_1 = rgb2gray(self.videodata.get_frame(0))

        for j in range(len(self.box_dict)):  # j iterates over all boxes
            images.append(frame_1[self.row_min[j]:self.row_max[j], self.col_min[j]:self.col_max[j]])
            jump.append(False)
            print("Box %d (%d, %d, %d, %d)." % (j, self.row_min[j], self.row_max[j], self.col_min[j], self.col_max[j]))

            self.shift_x[j][self.start_frame] = 0
            self.shift_y[j][self.start_frame] = 0
            self.shift_p[j][self.start_frame] = 0
            self.shift_x_y_error[j][self.start_frame] = 0
            self.box_shift[j][self.start_frame] = [0, 0]

        length = self.stop_frame - self.start_frame
        progress_pivot = 0 - 5

        for i in range(self.start_frame, self.stop_frame + 1):  # i iterates over all frames
            self.current_i = i
            if self.go_on:  # condition checked to be able to stop the thread
                with lock:
                    self.frame_n = rgb2gray(self.videodata.get_frame(i))

                for j in range(len(self.box_dict)):  # j iterates over all boxes
                    image_n = self.frame_n[self.row_min[j]:self.row_max[j], self.col_min[j]:self.col_max[j]]

                    if i > self.start_frame:
                        shift, error, diffphase = phase_cross_correlation(image_n, images[j], upsample_factor=self.upsample_factor)
                        shift[0], shift[1] = shift[1], shift[0]  # swap (y, x) → (x, y)

                        # TODO: fix tracking
                        if self.compare_first:
                            if jump[j]:  # TODO: improve jump override
                                jump[j] = False

                                self.total_value_jump[j][0] = self.shift_x[j][i - 1] - shift[0]
                                self.total_value_jump[j][1] = self.shift_y[j][i - 1] - shift[1]
                                self.total_value_jump[j][2] = self.shift_p[j][i - 1] - diffphase

                                print("Total jump: (%f, %f, %f)." % (self.total_value_jump[j][0], self.total_value_jump[j][1], self.total_value_jump[j][2]))

                                self.shift_x[j][i] = self.total_value_jump[j][0]
                                self.shift_y[j][i] = self.total_value_jump[j][1]
                                self.shift_p[j][i] = self.total_value_jump[j][2]  # diffphase
                            else:
                                self.shift_x[j][i] = self.total_value_jump[j][0] + shift[0]
                                self.shift_y[j][i] = self.total_value_jump[j][1] + shift[1]
                                self.shift_p[j][i] = self.total_value_jump[j][2] + diffphase
                        else:
                            self.shift_x[j][i] = self.shift_x[j][i - 1] + shift[0]
                            self.shift_y[j][i] = self.shift_y[j][i - 1] + shift[1]
                            self.shift_p[j][i] = self.shift_p[j][i - 1] + diffphase

                        self.shift_x_y_error[j][i] = error

                        if not self.compare_first or abs(shift[0]) >= 0.045:  # TODO: remove 0.045 factor (noise)
                            self.cumulated_shift[j][0] += shift[0]

                        if not self.compare_first or abs(shift[1]) >= 0.045:  # TODO: remove 0.045 factor (noise)
                            self.cumulated_shift[j][1] += shift[1]

                        print("Shift: (%f, %f), cumulated: (%f, %f)." % (shift[0], shift[1], self.cumulated_shift[j][0], self.cumulated_shift[j][1]))

                        to_shift_x = 0
                        to_shift_y = 0
                        if self.track and (abs(self.cumulated_shift[j][0]) >= 1 or abs(self.cumulated_shift[j][1]) >= 1):
                            to_shift_x = self._close_to_zero(self.cumulated_shift[j][0])
                            to_shift_y = self._close_to_zero(self.cumulated_shift[j][1])

                            # print("Cumulated shift: (%f, %f) → to shift: (%f, %f)." % (self.cumulated_shift[j][0], self.cumulated_shift[j][1], to_shift_x, to_shift_y))
                            # print("Shifted at frame %d (~%ds)." % (i, i / self.fps))

                            self.box_dict[j].x_rect += to_shift_x
                            self.box_dict[j].y_rect += to_shift_y
                            self._crop_coord(j)

                            images[j] = self.frame_n[self.row_min[j]:self.row_max[j], self.col_min[j]:self.col_max[j]]  # reframe image for later comparison
                            jump[j] = True
                        elif not self.track:
                            print("Cumulated shift: (%f, %f)." % (self.cumulated_shift[j][0], self.cumulated_shift[j][1]))

                        self.box_shift[j][i] = [to_shift_x, to_shift_y]
                        self.total_box_shift[j][0] += to_shift_x
                        self.total_box_shift[j][1] += to_shift_y

                        # print("Total box shift: (%d, %d)." % (self.total_box_shift[j][0], self.total_box_shift[j][1]))

                        # substract shifted amount from cumulated shift
                        self.cumulated_shift[j][0] -= to_shift_x
                        self.cumulated_shift[j][1] -= to_shift_y

                        if not self.compare_first:  # store the current image to be compared later
                            images[j] = image_n

                self.progress = int(((i - self.start_frame) / length) * 100)
                if self.progress > progress_pivot + 4:
                    print("%d%% (frame %d/%d)." % (self.progress, i - self.start_frame, self.stop_frame - self.start_frame))
                    progress_pivot = self.progress
            else:
                return

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
            i = which
            self.row_min[i] = int(self.box_dict[i].y_rect)
            self.row_max[i] = int(self.box_dict[i].y_rect) + self.box_dict[i].rect._height
            self.col_min[i] = int(self.box_dict[i].x_rect)
            self.col_max[i] = int(self.box_dict[i].x_rect) + self.box_dict[i].rect._width

    def get_z_std(self):
        self.z_std = self._z_std_compute()
        return self.z_std

    def _z_std_compute(self):
        z_dec = []

        for j in range(len(self.box_dict)):
            x = [i * self.res for i in self.shift_x[j]]
            y = [i * self.res for i in self.shift_y[j]]
            z = np.sqrt((np.std(x)) ** 2 + (np.std(y)) ** 2)
            z_dec.append(Decimal(str(z)).quantize(Decimal('0.001'),
                                                  rounding=ROUND_HALF_UP))  # special Decimal class for the correct rounding

        return z_dec  # rounding, leaves 3 digits after comma

    def get_z_delta_rms(self):
        self.z_rms, self.v_rms = self._z_delta_rms_compute()

        return self.z_rms, self.v_rms

    def _z_delta_rms_compute(self):  # TODO: modify
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
