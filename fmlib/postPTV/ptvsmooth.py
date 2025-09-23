"""
Bernhard Roth, ETH Zürich, 08.04.2021, rothbe@ethz.ch
Code to smooth trajectories and calculate velocities and acceleration form .ptv track files.
TODO:
        -write output as tracks
"""
import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
from matplotlib import rc, rcParams
rc('text', usetex=True)
rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
rcParams['text.latex.preamble'] = r'\boldmath'
rcParams["figure.figsize"] = [10,8]
from fmlib.postPTV import trackproc

def test_kernel_widht(path, track_file, n_tracks = 'all', n_min=5):
    tracks = trackproc.load_tracks(path + track_file, n_tracks)
    # declare variables and allocate memory
    sigma = np.arange(1, round(6 / 0.4)) * 0.4 + 1        # define test range of the kernels std. dev.
    #sigma = 5
    #filter_width = np.arange(0, round(2*sigma ))  +1
    a_var = np.zeros(sigma.size)
    v_var = np.zeros(sigma.size)



    for i in range(0, len(sigma)):
        x, v, a, ds,frame = smooth_trajectories(tracks, sigma[i], n_tracks, filter_width='auto',
                                             min_length=12, extrapolate=False, mode='Delete',
                                             print_status=False)
        a_magnitude = np.array([np.sqrt(np.square(a[0][_]) + np.square(a[1][_])) for _ in range(len(a[0]))],
                               dtype=object)   # get magnitude of acceleration vector
        a_var[i] = np.mean(np.array([np.var(a_magnitude[_]) for _ in range(a_magnitude.shape[0])]))

        v_magnitude = np.array([np.sqrt(np.square(v[0][_]) + np.square(v[1][_])) for _ in range(len(v[0]))],
                               dtype=object)  # get magnitude of acceleration vector
        # calculate mean variance
        v_var[i] = np.mean(np.array([np.var(v_magnitude[_]) for _ in range(v_magnitude.shape[0])]))

    # plot results in semi-lograithmic scale
    plt.figure();
    plt.scatter(sigma, v_var, label='Velocity',s=100)
    plt.scatter(sigma, a_var, label='Acceleration',s=100)
    plt.xlabel(r'Sigma of Gaussian Filter Kernel',fontsize=20,labelpad=15)
    plt.ylabel(r'Mean Lagrangian Acceleration Variance',fontsize=20,labelpad=15)
    plt.title(r'Kernel Width Estimation - Mean Var. of ' + str(len(a[0])) + ' trajectories',
              fontsize=20,y=1.02)
    plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    plt.yscale('log')
    plt.legend(fontsize=20)
    plt.grid()
    plt.savefig(path + 'plots_eps/var_smoothing_width.eps')
    plt.savefig(path + 'plots_png/var_smoothing_width.png')
    plt.show()

def make_gaussian(sigma, filter_width, derivative=0):
    # calculate Gaussian filter kernels

    if derivative == 0:
        gaussian = 1 / math.sqrt(2 * np.pi * sigma ** 2) * np.exp(-np.power(np.arange(-filter_width, filter_width + 1), 2)
                                                              / (2 * sigma ** 2))
        # normalize as "c_1 * gaussian + c_2" to correct for truncation errors
        c_2 = 0
        c_1 = 1 / np.sum(gaussian)
        gaussian = c_1 * gaussian + c_2
        return gaussian

    if derivative == 1:
        # gaussian smoothing combined with fist/second derivation
        d_gaussian = -np.arange(-filter_width, filter_width+1) / (sigma ** 3 * math.sqrt(2 * np.pi)) \
                 * np.exp(-np.power(np.arange(-filter_width, filter_width+1), 2) / (2 * sigma ** 2))
        # normalization
        c_1 = -1 / np.sum(np.arange(-filter_width, filter_width + 1) * d_gaussian)
        c_2 = -c_1 / (2 * filter_width + 1) * np.sum(d_gaussian)
        d_gaussian = c_1 * d_gaussian + c_2

        return d_gaussian

    if derivative == 2:
        dd_gaussian = - (sigma ** 2 - np.power(np.arange(-filter_width, filter_width + 1), 2)) / (sigma ** 5 \
                                                                                          * math.sqrt(
                2 * np.pi)) * np.exp(-np.power(np.arange(-filter_width, filter_width + 1), 2) / (2 * sigma ** 2))
        # normalize distribution
        c_1 = 2 / (np.sum(np.power(np.arange(-filter_width, filter_width + 1), 2) * dd_gaussian) - 1 / 2 / filter_width
               * np.sum(dd_gaussian) * np.sum(np.power(np.arange(-filter_width, filter_width + 1), 2)))
        c_2 = -c_1 / (2 * filter_width+1) * np.sum(dd_gaussian)
        dd_gaussian = c_1 * dd_gaussian + c_2
        tmp = np.sum(dd_gaussian)
        return dd_gaussian

def benchmark():
    trajectory_length = 50
    sigma = 6
    filter_width = sigma*2
    noise_amp = 0   # Relative amplification of noise signal

    print('Benchmarking smoothing and derivation kernel...\n')
    # Test extrapolation
    test_trajectory = np.array([[0, math.sin(math.pi * 4 * _ / trajectory_length)
                                 + noise_amp * (np.random.rand()-0.5), _] for _ in range(trajectory_length)])
    test_trajectory = [test_trajectory, test_trajectory]  # add second trajectory to make code work
    # TODO: Fix code to work with only one trajectory
    f, axis = plt.subplots(3)
    x, v, a, frame = smooth_trajectories(test_trajectory, sigma, filter_width=filter_width, extrapolate=False)
    axis[0].plot((test_trajectory[0][:, 2]), (test_trajectory[0][:, 1]), label='Original Trajectory')
    axis[1].plot(range(filter_width, filter_width + len(v[0][1])), v[0][1])
    axis[2].plot(range(filter_width, filter_width + len(v[0][1])), a[0][1])
    x, v, a, frame = smooth_trajectories(test_trajectory, sigma, filter_width=filter_width, fit_polynomial_order=2,
                                         extrapolate=True)
    #axis[0].plot((test_trajectory[0][:, 2]), (test_trajectory[0][:, 1]), label='Extrapolated Trajectory')
    #axis[1].plot(range(len(v[0][1])), v[0][1])
    #axis[2].plot(range(len(a[0][1])), a[0][1])

    # Plot analytical solution
    axis[1].plot(range(len(a[0][1])),
                 math.pi * 4 / trajectory_length
                 * np.cos(math.pi * 4 / trajectory_length * np.arange(-filter_width, len(a[0][1]) - filter_width)))
    axis[2].plot(range(len(a[0][1])),
                 (math.pi * 4 / trajectory_length) ** 2
                 * np.sin(-math.pi * 4 / trajectory_length * np.arange(-filter_width, len(a[0][1]) - filter_width)))

    plt.show()


    # test parabolic trajectory, velocity and acceleration calculation
    test_trajectory = np.array([[0, (_ ** 2) / 2, _] for _ in range(trajectory_length)])
    test_trajectory = [test_trajectory, test_trajectory]  # add second trajectory to make code work
    x, v, a, frame = smooth_trajectories(test_trajectory,sigma, filter_width=filter_width, extrapolate=False)

    f, axis = plt.subplots(2, 3)
    axis[0, 0].plot(test_trajectory[0][:, 2], test_trajectory[0][:, 1],
                    label='Smooth Trajectory', marker='X')
    axis[0, 0].scatter(range(filter_width, trajectory_length - filter_width),
                       x[0][0], label='Smooth Trajectory', marker='X')
    axis[0, 0].axis('equal')
    axis[0, 0].legend()
    axis[0, 1].plot(np.array(v[0][1])-np.array([_ for _ in range(filter_width, trajectory_length - filter_width)]))
    axis[0, 1].set_title('Velocity Difference')
    axis[0, 2].plot(np.array(a[0][1])-np.array([1 for _ in range(filter_width, trajectory_length - filter_width)]))
    axis[0, 2].set_title('Acceleration Difference')

    # test noise signal
    test_trajectory = np.array([[0, np.random.rand()-0.5, _] for _ in range(trajectory_length)])
    test_trajectory = [test_trajectory, test_trajectory]  # add second trajectory to make code work
    # TODO: Fix code to work with only one trajectory
    x, v, a, frame = smooth_trajectories(test_trajectory,sigma, filter_width=filter_width, extrapolate=False)

    axis[1, 0].plot((test_trajectory[0][:, 2]), (test_trajectory[0][:, 1]), label='Extrapolated Trajectory')
    axis[1, 0].scatter(range(filter_width, trajectory_length - filter_width), x[0][0],
                    label='Smooth Trajectory', marker='X')
    axis[1, 1].plot(np.array(v[0][1]))
    axis[1, 1].set_title('Velocity Difference')
    axis[1, 2].plot(np.array(a[0][1]))
    axis[1, 2].set_title('Acceleration Difference')

    plt.show()

    # check gaussian smoothing
    # create gaussian to use as benchmark
    gaussian_test = 1 / math.sqrt(2 * np.pi * 10 ** 2) * np.exp(-np.power(np.arange(-100, 100), 2)
                                                                / (2 * 10 ** 2))
    # normalize
    gaussian = make_gaussian(sigma, filter_width)
    gaussian_test = gaussian_test / np.sum(gaussian)

    # analytical solution for benchmark
    sigma_product = math.sqrt(sigma ** 2 + 100)
    mean_product = 100 * sigma ** 2 + filter_width * 10 ** 2 / (sigma ** 2 + 10 ** 2)
    gaussian_solution = 1 / math.sqrt(2 * np.pi * sigma_product ** 2) * np.exp(-np.power(np.arange(-100, 100), 2)
                                                                               / (2 * sigma_product ** 2))
    # normalize
    gaussian_solution = gaussian_solution / np.sum(gaussian_solution)

    # plt.plot(gaussian_test)
    f, axis = plt.subplots()
    axis_2 = axis.twinx()
    axis.plot(np.arange(-100, 100), gaussian_solution, label='Reference Signal')
    axis_2.plot(np.arange(-100, 100), gaussian_solution - np.convolve(gaussian_test, gaussian, mode='same'),
             label='Error', color='red')
    #plt.axvline(x=0)   # plot center line
    axis_2.legend()
    plt.show()
    print('Done')

def smooth_trajectories(
    tracks,
    sigma,
    n_tracks="all",
    filter_width="auto",
    min_length=1,
    extrapolate=False,
    fit_polynomial_order=2,
    mode="None",
    merge_tracks=None,
    print_status=False,
):
    """
    Function to smooth and derivate trajectory data.

    string file_path:       Path to .piv file
    int sigma:      Standard deviation of gaussian smoothing kernel in pixel.
    int n_tracks:   Number of tracks to evaluate. Default 'all' evaluates all tracks in the file.
    int filter_width:       Half width of filter kernels in pixel. Default 'auto' is 3 * sigma.
    int n_min:      Minimal number of points required in a trajectory
    bool extrapolate:       Flag to activate extrapolation. Allows processing of boundary points.
    int fit_polynomial_order:       Order of fitting polynomial.
    string mode:        Mode determining the behavior of the function, if input trajectory is to short. Options are
                        'None' and 'Delete'
    list merge_tracks:      List of indices of tracks which are to be merged. Default argument None omits merging.
    bool print_status:       Flag to deactivate status output and running time estimation

    return:
    list pos_smooth:        List containing two numpy arrays [x_pos_smooth y_pos_smooth]
    list vel_smooth:        List containing two numpy arrays [u_vel_smooth v_vel_smooth]
    list acc_smooth:        List containing two numpy arrays [ax_acc_smooth ay_acc_smooth]
    """

    # process input parameters
    if filter_width == "auto":
        filter_width = math.ceil(
            2 * sigma
        )  # set filter width to four sigma and make sure it is an integer

    # some flags to activate development functionss
    plotflag = False  # Flag to activate result plotting for develoment and debugging
    benchmark = False  # Flag to activate benchmark of gaussian smoothing test case

    # create gaussian distributions for smoothing and derivation
    gaussian = make_gaussian(sigma, filter_width)
    d_gaussian = make_gaussian(sigma, filter_width, derivative=1)
    dd_gaussian = make_gaussian(sigma, filter_width, derivative=2)

    # initialize output lists
    frame_track = []
    x_track_smooth = []
    y_track_smooth = []
    u_track_smooth = []
    v_track_smooth = []
    ax_track_smooth = []
    ay_track_smooth = []
    ds_track_smooth = []
    orientation_smooth = []

    # smooth all tracks which are long enough
    for i in range(0, len(tracks)):
        if (
            tracks[i] is not None
            and np.size(tracks[i][:, 1]) > (2 * filter_width + 1)
            and np.size(tracks[i][:, 1]) >= min_length
        ):  # check track length
            if plotflag:
                plt.scatter((tracks[i][:, 1]), (tracks[i][:, 0]), label="Raw Data")
            if extrapolate:
                # extrapolate trajectory to solve boundary issues of the convolution
                tracks[i] = trackproc.extrapolate_track(
                    tracks[i], filter_width, order=fit_polynomial_order, mode="forward"
                )
                tracks[i] = trackproc.extrapolate_track(
                    tracks[i], filter_width, order=fit_polynomial_order, mode="backward"
                )

            # store frame/time information
            frame_track.append(tracks[i][filter_width:-filter_width, 6])

            # compute smoothed position
            # 'valid' mode of numpy.convolve only outputs inner points without boundary effects
            x_track_smooth.append(np.convolve(tracks[i][:, 1], gaussian, mode="valid"))
            y_track_smooth.append(np.convolve(tracks[i][:, 0], gaussian, mode="valid"))

            if plotflag:
                plt.plot(
                    (tracks[i][:, 1]), (tracks[i][:, 0]), label="Extrapolated Trajectory"
                )
                plt.scatter(
                    x_track_smooth[len(x_track_smooth) - 1],
                    y_track_smooth[len(y_track_smooth) - 1],
                    label="Smooth Trajectory",
                    marker="X",
                )
                plt.axis("equal")
                plt.legend()
                if i % 10 == 0:
                    plt.show()

            # compute smoothed velocity
            u_track_smooth.append(np.convolve(tracks[i][:, 1], d_gaussian, mode="valid"))
            v_track_smooth.append(np.convolve(tracks[i][:, 0], d_gaussian, mode="valid"))

            # compute smoothed acceleration
            ax_track_smooth.append(
                np.convolve(tracks[i][:, 1], dd_gaussian, mode="valid")
            )
            ay_track_smooth.append(
                np.convolve(tracks[i][:, 0], dd_gaussian, mode="valid")
            )

            ds_track_smooth.append(np.mean(np.sqrt(tracks[i][:, 2] * 4 / np.pi)))
            orientation_smooth.append(
                np.convolve(tracks[i][:, 3], gaussian, mode="valid")
            )

            if plotflag:
                f, axis = plt.subplots(3)
                axis[0].plot(np.convolve(tracks[i][:, 1], gaussian, mode="valid"))
                axis[1].plot(np.convolve(tracks[i][:, 1], d_gaussian, mode="valid"))
                axis[2].plot(np.convolve(tracks[i][:, 1], dd_gaussian, mode="valid"))
                plt.show()

        elif mode == "None":
            tracks[i] = None
            frame_track.append(None)
            x_track_smooth.append(None)
            y_track_smooth.append(None)
            u_track_smooth.append(None)
            v_track_smooth.append(None)
            ax_track_smooth.append(None)
            ay_track_smooth.append(None)
            orientation_smooth.append(None)

    # output data
    return (
        (y_track_smooth, x_track_smooth),
        (v_track_smooth, u_track_smooth),
        (ay_track_smooth, ax_track_smooth),
        ds_track_smooth,
        orientation_smooth,
        frame_track,
    )

    # make it run parallel on shared memory