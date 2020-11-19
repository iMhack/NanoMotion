import math
import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import mplcursors
import pandas as pd
import seaborn as sns


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def plot_results(shift_x, shift_y, shift_x_y_error, box_shift, shift_p, fps, res, input_path, output_basepath, plots_dict, boxes_dict, chop=False,
                 chop_duration=0, start_frame=0):
    print("Started plotting results.")
    opened_plots = []
    shift_length_all = []
    position_all = []

    for j in range(len(boxes_dict)):
        my_shift_x = shift_x[j]
        my_shift_y = shift_y[j]
        my_shift_p = shift_p[j]
        my_shift_x_y_error = shift_x_y_error[j]
        my_box_shift = box_shift[j]

        my_shift_x_um = []
        for i in range(len(my_shift_x)):
            my_shift_x_um.append((my_shift_x[i] + my_box_shift[i][0]) * res)

        my_shift_y_um = []
        for i in range(len(my_shift_y)):
            my_shift_y_um.append((my_shift_y[i] + my_box_shift[i][1]) * res)

        my_shift_x_um_error = [e * x for e, x in zip(my_shift_x_y_error, my_shift_x_um)]
        my_shift_y_um_error = [e * y for e, y in zip(my_shift_x_y_error, my_shift_y_um)]

        output_cell_target = "%s_cell_%d" % (output_basepath, j)

        shift_x_step_um = [my_shift_x_um[i + 1] - my_shift_x_um[i] for i in range(len(my_shift_x_um) - 1)]
        shift_y_step_um = [my_shift_y_um[i + 1] - my_shift_y_um[i] for i in range(len(my_shift_y_um) - 1)]

        shift_length_step_um = []
        for i in range(len(shift_x_step_um)):
            shift_length_step_um.append(math.sqrt(math.pow(shift_x_step_um[i], 2) + math.pow(shift_y_step_um[i], 2)))

        ls = "--"
        fmt = "o"
        markersize = 4
        if plots_dict["view_position_x"]:
            figure = plt.figure(num=output_cell_target + "x(t), um(s)")
            plt.title("%s\n\nx(t), #%d" % (input_path, j))
            plt.xlabel("t, s")
            plt.ylabel("x, um")

            plt.grid()
            plt.errorbar([frame / fps for frame in range(len(my_shift_x))], my_shift_x_um, ls=ls, fmt=fmt, markersize=markersize,
                         yerr=my_shift_x_um_error)

            plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

            plt.savefig(output_cell_target + "_x(t).png")
            opened_plots.append(figure)

        if plots_dict["view_position_y"]:
            figure = plt.figure(num=output_cell_target + "y(t), um(s)")
            plt.title("%s\n\ny(t), #%d" % (input_path, j))
            plt.xlabel("t, s")
            plt.ylabel("y, um")

            plt.grid()
            plt.errorbar([frame / fps for frame in range(len(my_shift_y))], my_shift_y_um, ls=ls, fmt=fmt, markersize=markersize,
                         yerr=my_shift_y_um_error)

            plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

            plt.savefig(output_cell_target + "_y(t).png")
            opened_plots.append(figure)

        if plots_dict["view_violin"]:
            figure = plt.figure(num=output_cell_target + "violin of step length")
            plt.title("%s\n\nViolin, #%d" % (input_path, j))
            plt.ylabel("step length, um")

            sns.violinplot(data=shift_length_step_um, inner="stick")

            plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

            plt.savefig(output_cell_target + "_violin.png")
            opened_plots.append(figure)

        if plots_dict["view_violin_chop"]:
            number_of_frame_in_a_chop = math.floor(chop_duration * fps)
            number_of_full_chops = math.floor(len(shift_length_step_um) / number_of_frame_in_a_chop)

            if number_of_full_chops < 1:
                print("WARNING: chop duration would exceed total number of frames.")
            else:
                figure = plt.figure(num=output_cell_target + "violin chopped of step length")
                plt.title("%s\n\nViolin #%d chopped every %d sec" % (input_path, j, chop_duration))
                plt.xlabel("frame range")
                plt.ylabel("step length, um")

                chopped_data = []
                labels = []
                for i in range(number_of_full_chops):
                    chopped_data.append(shift_length_step_um[number_of_frame_in_a_chop * i:number_of_frame_in_a_chop * (i + 1)])

                    labels.append("[%d, %d]" % (start_frame + number_of_frame_in_a_chop * i, start_frame + number_of_frame_in_a_chop * (i + 1) - 1))

                g = sns.violinplot(data=chopped_data, inner="stick")
                g.set_xticklabels(labels, rotation=30)

                axe = plt.gca()
                axe.legend()
                axe.set_ylim([-0.1, 0.5])

                plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

                plt.savefig(output_cell_target + "_violin_chopped.png")
                opened_plots.append(figure)

        if plots_dict["view_position"]:
            figure = plt.figure(num=output_cell_target + "y(x), um(um)")
            plt.title("%s\n\ny(x), #%d" % (input_path, j))
            plt.xlabel("x, um")
            plt.ylabel("y, um")

            plt.grid()
            plt.errorbar(my_shift_x_um, my_shift_y_um, ls=ls, fmt=fmt, markersize=markersize)  # , yerr=my_shift_y_um_error, xerr=my_shift_x_um_error)

            axe = plt.gca()
            axe.set_xlim([-8, 8])
            axe.set_ylim([-8, 8])

            plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

            plt.savefig(output_cell_target + "_y(x).png")
            opened_plots.append(figure)

        if plots_dict["view_phase"]:
            figure = plt.figure(num=output_cell_target + "phase")
            plt.title("%s\n\nPhase, #%d" % (input_path, j))
            plt.xlabel("Frame #")
            plt.ylabel("Phase")

            plt.grid()
            plt.plot(my_shift_p)

            plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

            plt.savefig(output_cell_target + "_p.png")
            opened_plots.append(figure)

        if plots_dict["view_step_length"]:
            figure = plt.figure(num=output_cell_target + "steps")
            plt.title("%s\n\nSteps, #%d" % (input_path, j))
            plt.xlabel("t, s")
            plt.ylabel("length, um")

            plt.grid()
            plt.errorbar([frame / fps for frame in range(len(shift_length_step_um))], shift_length_step_um, ls=None, fmt=fmt, markersize=markersize, alpha=0.5)

            plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

            plt.savefig(output_cell_target + "_p.png")
            opened_plots.append(figure)

        if plots_dict["view_position_all_on_one"]:
            position_all.append([my_shift_x_um, my_shift_y_um])

        if plots_dict["view_violin_all_on_one"]:
            shift_length_all.append(shift_length_step_um)

    if plots_dict["view_position_all_on_one"]:
        figure = plt.figure(num=output_basepath + "_all_y(x).png")
        plt.title("%s\n\nAll cells, y(x), #0 to #%d" % (input_path, j))
        plt.xlabel("x, um")
        plt.ylabel("y, um")

        plt.grid()

        bars = []
        for b in range(0, len(position_all)):
            x_raw = position_all[b][0]
            y_raw = position_all[b][1]

            # x -> y
            # y -> -x
            x = [-e for e in y_raw]
            y = x_raw

            bar = plt.errorbar(x, y, ls="-", fmt=fmt, markersize=0, alpha=0.5, label=("Box #%d" % (b)))
            bars.append(bar)

        axe = plt.gca()
        axe.legend()
        axe.set_xlim([-2, 2])
        axe.set_ylim([-2, 2])

        mplcursors.cursor(bars, highlight=True)

        plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

        plt.savefig("%s%s" % (output_basepath, "_all_y(x).png"))
        opened_plots.append(figure)

    if plots_dict["view_violin_all_on_one"]:
        print("Plotting all (%d) violins containing each %d data points." % np.shape(shift_length_all))

        figure = plt.figure(num=output_cell_target + "Violins (seaborn)")
        plt.title("%s\n\nViolins (seaborn), #0 to #%d" % (input_path, j))
        plt.xlabel("Zone #")
        plt.ylabel("step length, um")

        sns.violinplot(data=shift_length_all, inner="quartiles")

        axe = plt.gca()
        axe.set_ylim([-0.1, 0.5])

        plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

        plt.savefig("%s%s" % (output_basepath, "_violin_all_seaborn.png"))
        opened_plots.append(figure)

    plt.show()

    return opened_plots


def export_results(shift_x, shift_y, box_shift, fps, res, w, h, output_basepath):
    target = "%s_output.xlsx" % (output_basepath)
    print("Exporting results to %s." % (target))

    df = pd.DataFrame({
        "frame": [frame for frame in range(len(shift_x))],
        "t, s": [frame / fps for frame in range(len(shift_x))],
        "x, px": shift_x,
        "y, px": shift_y,
        "box shift x, px": [shift[0] for shift in box_shift],
        "box shift y, px": [shift[1] for shift in box_shift],
        "x, um": [x * res for x in shift_x],
        "y, um": [y * res for y in shift_y]
    })

    df = pd.concat([df, pd.DataFrame({
        "window, px": [str(w) + " x " + str(h)],
        "window, um": [str(w * res) + " x " + str(h * res)],
        "um per px": [res]
    })], axis=1)

    df = df[["frame", "t, s", "x, px", "y, px", "box shift x, px", "box shift y, px", "x, um", "y, um", "window, px", "window, um", "um per px"]]

    writer = pd.ExcelWriter(
        os.path.join(target))
    df.to_excel(excel_writer=writer, sheet_name="Sheet 1", index=False)
    writer.save()


def get_formatted_name(file):
    if "." in file:
        return os.path.basename(file)[:-4]
    else:
        return os.path.basename(file)


def create_results_directory(file, cell_name):
    videofile_dir = os.path.dirname(os.path.abspath(file))
    results_dir = os.path.join(videofile_dir, "results")

    formatted_name = get_formatted_name(file)

    output_dir = os.path.join(results_dir, formatted_name)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    return os.path.join(output_dir, formatted_name)
