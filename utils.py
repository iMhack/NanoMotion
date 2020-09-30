import math
import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_results(shift_x, shift_y, shift_x_y_error, box_shift, shift_p, fps, res, output_basepath, plots_dict, boxes_dict, chop=False,
                 chop_duration=0, start_frame=0):
    print("Started plotting results.")
    shift_length_all = []

    for j in range(len(boxes_dict)):
        my_shift_x = shift_x[j]
        my_shift_y = shift_y[j]
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
        my_shift_p = shift_p[j]

        output_target = "%s%d" % (output_basepath, j)

        shift_x_step_um = [my_shift_x_um[i + 1] - my_shift_x_um[i] for i in range(len(my_shift_x_um) - 1)]
        shift_y_step_um = [my_shift_y_um[i + 1] - my_shift_y_um[i] for i in range(len(my_shift_y_um) - 1)]
        shift_length_step_um = np.sqrt(np.square(shift_x_step_um) + np.square(shift_y_step_um))

        ls = "dotted"
        fmt = "o"
        markersize = 0.9
        if (plots_dict["view_position_x"]):
            plt.figure(num=output_target + "x(t), um(s)")
            plt.title("x(t), #%d" % (j))
            plt.xlabel("t, s")
            plt.ylabel("x, um")

            plt.grid()
            plt.errorbar([frame / fps for frame in range(len(my_shift_x))], my_shift_x_um, ls=ls, fmt=fmt, markersize=markersize,
                         yerr=my_shift_x_um_error)

            plt.savefig(output_target + "_x(t).png")

        if (plots_dict["view_position_y"]):
            plt.figure(num=output_target + "y(t), um(s)")
            plt.title("y(t), #%d" % (j))
            plt.xlabel("t, s")
            plt.ylabel("y, um")

            plt.grid()
            plt.errorbar([frame / fps for frame in range(len(my_shift_y))], my_shift_y_um, ls=ls, fmt=fmt, markersize=markersize,
                         yerr=my_shift_y_um_error)

            plt.savefig(output_target + "_y(t).png")

        if (plots_dict["view_violin"]):
            plt.figure(num=output_target + "violin of step length")
            plt.title("Violin, #%d" % (j))
            plt.ylabel("step length, um")

            sns.violinplot(data=shift_length_step_um, inner="stick")

            plt.savefig(output_target + "_violin.png")

        if (plots_dict["view_violin_chop"]):
            number_of_frame_in_a_chop = math.floor(chop_duration * fps)
            number_of_full_chops = math.floor(len(shift_length_step_um) / number_of_frame_in_a_chop)

            if number_of_full_chops < 1:
                print("WARNING: chop duration would exceed total number of frames.")
            else:
                plt.figure(num=output_target + "violin chopped of step length")
                plt.title("Violin #%d chopped every %d sec" % (j, chop_duration))
                plt.xlabel("frame range")
                plt.ylabel("step length, um")

                chopped_data = []
                labels = []
                for i in range(number_of_full_chops):
                    chopped_data.append(shift_length_step_um[number_of_frame_in_a_chop * i:number_of_frame_in_a_chop * (i + 1)])

                    labels.append("[%d, %d]" % (start_frame + number_of_frame_in_a_chop * i, start_frame + number_of_frame_in_a_chop * (i + 1) - 1))

                g = sns.violinplot(data=chopped_data, inner="stick")
                g.set_xticklabels(labels, rotation=30)

                plt.savefig(output_target + "_violin_chopped.png")

        if (plots_dict["view_viollin_all_on_one"]):
            shift_length_all.append(shift_length_step_um)

        if (plots_dict["view_position"]):
            plt.figure(num=output_target + "y(x), um(um)")
            plt.title("y(x), #%d" % (j))
            plt.xlabel("x, um")
            plt.ylabel("y, um")

            plt.grid()
            plt.errorbar(my_shift_x_um, my_shift_y_um, ls=ls, fmt=fmt, markersize=markersize)  # , yerr=my_shift_y_um_error, xerr=my_shift_x_um_error)

            plt.savefig(output_target + "_y(x).png")

        if (plots_dict["view_phase"]):
            plt.figure(num=output_target + "phase")
            plt.title("Phase, #%d" % (j))
            plt.xlabel("Frame #")
            plt.ylabel("Phase")

            plt.grid()
            plt.plot(my_shift_p)

            plt.savefig(output_target + "_p.png")

    if (plots_dict["view_viollin_all_on_one"]):
        print(np.shape(shift_length_all))

        figure = plt.figure(num=output_target + "Violins")
        plt.title("Violins, #0 to #%d" % (j))
        plt.xlabel("Zone #")
        plt.ylabel("step length, um")

        sns.violinplot(data=shift_length_all, inner="quartiles")

        plt.savefig("%s%s" % (output_basepath, "_violin_all.png"))

    plt.show()


def export_results(shift_x, shift_y, box_shift, fps, res, w, h, z_std, dz_rms, v, output_basepath):
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
        "z std, um": [z_std],
        "total z, um": [dz_rms],
        "v, um/s": [v],
        "window, px": [str(w) + " x " + str(h)],
        "window, um": [str(w * res) + " x " + str(h * res)],
        "um per px": [res]
    })], axis=1)

    df = df[["frame", "t, s", "x, px", "y, px", "box shift x, px", "box shift y, px", "x, um", "y, um", "z std, um",
             "total z, um", "v, um/s", "window, px", "window, um", "um per px"]]

    writer = pd.ExcelWriter(
        os.path.join(target))
    df.to_excel(excel_writer=writer, sheet_name="Sheet 1", index=False)
    writer.save()


def create_results_directory(file, cell_name):
    videofile_dir = os.path.dirname(os.path.abspath(file))
    results_dir = os.path.join(videofile_dir, "results")

    if "." in file:
        formatted_name = os.path.basename(file)[:-4]
    else:
        formatted_name = os.path.basename(file)

    output_dir = os.path.join(results_dir, formatted_name)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    return os.path.join(output_dir, formatted_name + "_cell_")
