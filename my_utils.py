
def plot_results(shift_x, shift_y, fps, res, output_name, plots_dict, solver_number):
    if (plots_dict["x(t)_shift"]):
        plt.figure(num=output_name + 'x(t), um(s)')
        plt.plot([frame / fps for frame in range(len(shift_x))], [x * res for x in shift_x], "-")
        plt.grid()
        plt.title("x(t), #" + str(solver_number) + "")
        plt.xlabel("t, s")
        plt.ylabel("x, um")
        plt.savefig(output_name + "_x(t).png")
    if (plots_dict["Violin"]):
        plt.figure(num=output_name + 'violin of shift_x*shift_y')
        shift_x_step = [shift_x[i + 1] - shift_x[i] for i in range(len(shift_x) - 1)]
        shift_y_step = [shift_y[i + 1] - shift_y[i] for i in range(len(shift_y) - 1)]
        shift_length_step = np.sqrt(np.square(shift_x_step) + np.square(shift_y_step))
        plt.ylabel("step length, um")
        sns.violinplot(data=shift_length_step)
        plt.title("Violin, #" + str(solver_number) + "")
        plt.savefig(output_name + "_violin.png")

    if (plots_dict["y(t)_shift"]):
        plt.figure(num=output_name + 'y(t), um(s)')
        plt.plot([frame / fps for frame in range(len(shift_y))], [x * res for x in shift_y], "-")
        plt.grid()
        plt.title("y(t), #" + str(solver_number) + "")
        plt.xlabel("t, s")
        plt.ylabel("y, um")
        plt.savefig(output_name + "_y(t).png")

    if (plots_dict["pos(t)"]):
        plt.figure(num=output_name + 'y(x), um(um)')
        plt.plot([x * res for x in shift_x], [y * res for y in shift_y], "-")
        plt.grid()
        plt.title("y(x), #" + str(solver_number) + "")
        plt.xlabel("x, um")
        plt.ylabel("y, um")
        plt.savefig(output_name + "_y(x).png")
    plt.show()


def export_results(shift_x, shift_y, fps, res, w, h, z_std, dz_rms, v, output_name):
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


def create_dirs(file, cell_name):  # check if /results/filename/ directories exists and create them, if not
    videofile_dir = os.path.dirname(os.path.abspath(file))
    if not os.path.isdir(os.path.join(videofile_dir, "results")):
        os.makedirs(os.path.join(videofile_dir, "results"))
    output_dir = os.path.join(videofile_dir, "results", os.path.basename(file)[:-4])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_name = os.path.join(output_dir, os.path.basename(file)[:-4] + "_cell_" + cell_name)
    return output_name