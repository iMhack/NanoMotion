import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_results(shift_x, shift_x_y_error, shift_y, shift_p, fps, res, output_name, plots_dict, boxes_dict, chop=False,
                 chop_sec=0, start_frame=0):
    print("Started plotting results.")
    shift_lenght_all = []
    for j in range(len(boxes_dict)):
        my_shift_x_y_error = shift_x_y_error[j]
        my_shift_x = shift_x[j]
        my_shift_x_um = [x * res for x in my_shift_x]
        my_shift_x_um_error = [e * x for e, x in zip(my_shift_x_y_error, my_shift_x_um)]
        my_shift_y = shift_y[j]
        my_shift_y_um = [y * res for y in my_shift_y]
        my_shift_y_um_error = [e * y for e, y in zip(my_shift_x_y_error, my_shift_y_um)]
        my_shift_p = shift_p[j]
        fps = fps
        res = res
        output_name = output_name + str(j)
        plots_dict = plots_dict
        solver_number = j
        shift_x_step_um = [my_shift_x_um[i + 1] - my_shift_x_um[i] for i in range(len(my_shift_x_um) - 1)]
        shift_y_step_um = [my_shift_y_um[i + 1] - my_shift_y_um[i] for i in range(len(my_shift_y_um) - 1)]
        shift_length_step_um = np.sqrt(np.square(shift_x_step_um) + np.square(shift_y_step_um))
        ls = 'dotted'
        fmt = 'o'
        if (plots_dict["view_position_x"]):
            plt.figure(num=output_name + 'x(t), um(s)')
            #            plt.plot([frame / fps for frame in range(len(my_shift_x))], [x * res for x in my_shift_x], "-")
            plt.errorbar([frame / fps for frame in range(len(my_shift_x))], my_shift_x_um, ls=ls, fmt=fmt,
                         yerr=my_shift_x_um_error)
            plt.grid()
            plt.title("x(t), #" + str(solver_number) + "")
            plt.xlabel("t, s")
            plt.ylabel("x, um")
            plt.savefig(output_name + "_x(t).png")

        if (plots_dict["view_position_y"]):
            plt.figure(num=output_name + 'y(t), um(s)')
            plt.errorbar([frame / fps for frame in range(len(my_shift_y))], my_shift_y_um, ls=ls, fmt=fmt,
                         yerr=my_shift_y_um_error)
            plt.grid()
            plt.title("y(t), #" + str(solver_number) + "")
            plt.xlabel("t, s")
            plt.ylabel("y, um")
            plt.savefig(output_name + "_y(t).png")

        if (plots_dict["view_violin"]):
            plt.figure(num=output_name + 'violin of step length')
            plt.ylabel("step length, um")
            sns.violinplot(data=shift_length_step_um, inner="stick")
            plt.title("Violin, #" + str(solver_number) + "")
            plt.savefig(output_name + "_violin.png")

        if (plots_dict["view_violin_chop"]):
            plt.figure(num=output_name + 'violin chopped of step length')
            plt.ylabel("step length, um")
            number_of_frame_in_a_chop = int(chop_sec * fps)
            # print(number_of_frame_in_a_chop)
            # print(len(shift_length_step))
            number_of_full_chops = int(
                len(shift_length_step_um) / number_of_frame_in_a_chop)  # 0 1 2 1 0, [0 1] [2 1], 0 1
            # print(number_of_full_chops)
            chopped_data = [shift_length_step_um[number_of_frame_in_a_chop * i:number_of_frame_in_a_chop * (i + 1)] for
                            i in range(number_of_full_chops)]
            g = sns.violinplot(data=chopped_data, inner="stick")
            labels = ["[" + str(start_frame + number_of_frame_in_a_chop * i) + "," + str(
                start_frame + number_of_frame_in_a_chop * (i + 1) - 1) + "]" for i in range(number_of_full_chops)]
            # print(labels)
            plt.xlabel("frame range")
            g.set_xticklabels(labels, rotation=30)
            plt.title("Violin, #" + str(solver_number) + " chopped every " + str(chop_sec) + " sec")
            plt.savefig(output_name + "_violin_chopped.png")

        if (plots_dict["view_viollin_all_on_one"]):
            shift_lenght_all.append(shift_length_step_um)

        if (plots_dict["view_position"]):
            plt.figure(num=output_name + 'y(x), um(um)')
            plt.errorbar(my_shift_x_um, my_shift_y_um, ls=ls,
                         fmt=fmt)  # , yerr=my_shift_y_um_error, xerr=my_shift_x_um_error)
            plt.grid()
            plt.title("y(x), #" + str(solver_number) + "")
            plt.xlabel("x, um")
            plt.ylabel("y, um")
            plt.savefig(output_name + "_y(x).png")

        if (plots_dict["view_phase"]):
            plt.figure(num=output_name + 'phase')
            plt.grid()
            plt.title("Phase, #{}".format(solver_number))
            plt.xlabel("Frame #")
            plt.ylabel("Phase")
            plt.plot(my_shift_p)
            plt.savefig(output_name + "_p.png")

    print(np.shape(shift_lenght_all))
    if (plots_dict["view_viollin_all_on_one"]):
        plt.figure(num=output_name + 'Violins')
        plt.ylabel("step length, um")
        plt.xlabel("Zone #")
        sns.violinplot(data=shift_lenght_all, inner="stick")
        plt.title("Violin, #0 to #" + str(solver_number) + " ")
        plt.savefig(output_name + "_violin_all.png")

    plt.show()


def export_results(shift_x, shift_y, fps, res, w, h, z_std, dz_rms, v, output_name):
    print("Started exporting results.")
    df = pd.DataFrame({"t, s": [frame / fps for frame in range(len(shift_x))], "x, px": shift_x, "y, px": shift_y,
                       "x, um": [x * res for x in shift_x], "y, um": [y * res for y in shift_y]})
    df = pd.concat([df, pd.DataFrame({"z std, um": [z_std], "total z, um": [dz_rms], "v, um/s": [v], "window, px":
                   [str(w) + " x " + str(h)], "window, um": [str(w * res) + " x " + str(h * res)], "um per px": [res]})], axis=1)
    df = df[["t, s", "x, px", "y, px", "x, um", "y, um", "z std, um", "total z, um", "v, um/s", "window, px",
             "window, um", "um per px"]]
    writer = pd.ExcelWriter(
        os.path.join(output_name + "_output.xlsx"))
    df.to_excel(excel_writer=writer, sheet_name="Sheet 1", index=False)
    writer.save()
    print("Results exported.")


def create_dirs(file, cell_name):  # check if /results/filename/ directories exists and create them, if not
    videofile_dir = os.path.dirname(os.path.abspath(file))
    if not os.path.isdir(os.path.join(videofile_dir, "results")):
        os.makedirs(os.path.join(videofile_dir, "results"))
    output_dir = os.path.join(videofile_dir, "results", os.path.basename(file)[:-4])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_name = os.path.join(output_dir, os.path.basename(file)[:-4] + "_cell_")
    return output_name
