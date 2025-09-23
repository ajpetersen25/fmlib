"""
Bernhard Roth, 28.04.2021, ETH Zürich
Function library for plotting and binning ptv data, especially saltation and boundary
layer data.
"""

import math
import numpy as np

import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as clr


def get_bins(domain_boundary, n_bins, mode="linear", shift=0):
    """
    Divides given domain into specified bins and returns the values of their boundaries.

    :param domain_boundary: List containing edge coordinates of the binning,
    [z_min, z_max]
    :param n_bins: Number of bins
    :param mode: Mode of bin distribution. 'linear' or 'logarithmic'
    :param shift: Shift for logarithmic binning. The log. dist. converges to the linear
    binning for large shifts.
    :return: Numpy array containing bin edges, [[z_0, z_1], ..., [z_i, z_i+1]]
    """

    bin_bounds = []  # declare list to store bin boundaries

    # write bin boundary for selected binning method
    if mode == "linear":
        bin_start = domain_boundary[0]
        bin_end = domain_boundary[1]
        bin_increment = (bin_end - bin_start) / n_bins

        for i in range(0, n_bins):
            bin_bounds.append(
                [bin_start + i * bin_increment, bin_start + (i + 1) * bin_increment]
            )

    if mode == "logarithmic":
        while domain_boundary[0] + shift <= 0 or domain_boundary[1] + shift <= 0:
            # Find shift to produce none-zero, positive entries to make surethe logarithm
            # is well defined
            shift += 1

        bin_start = math.log(domain_boundary[0] + shift)
        bin_end = math.log(domain_boundary[1] + shift)
        bin_increment = (bin_end - bin_start) / n_bins

        for i in range(0, n_bins):
            bin_bounds.append(
                [
                    math.pow(math.e, bin_start + i * bin_increment) - shift,
                    math.pow(math.e, bin_start + (i + 1) * bin_increment) - shift,
                ]
            )

    return np.array(bin_bounds)


def get_bin_number(value, bin_boundaries):
    """
    :param bin_boundaries: Numpy array containing bin boundaries [[z_0, z_1], ...,
                                                                  [z_i, z_i+1]]
    :param value: Value which is binned
    :return: Number of corresponding bin
    """

    # get bin from boundary array
    for i in range(0, len(bin_boundaries)):
        if bin_boundaries[i, 0] <= value and bin_boundaries[i, 1] >= value:
            return i  # return bin number
    return None  # return None if value is out of range


def get_bin_numbers(values, bin_boundaries):
    """
    :param bin_boundaries: Numpy array containing bin boundaries [[z_0, z_1], ...,
                                                                  [z_i, z_i+1]]
    :param value: Value which is binned
    :return: Number of corresponding bin
    """
    bns = []
    for value in values:
        if bin_boundaries[0, 0] > value or value > bin_boundaries[-1, 1]:
            bns.append(None)
        else:
            # get bin from boundary array
            for i in range(0, len(bin_boundaries)):
                if bin_boundaries[i, 0] <= value and bin_boundaries[i, 1] >= value:
                    bns.append(i)

    return np.array(bns)  # return bin numbers


#        return None     # return None if value is out of range


def grad_hist(
    value_list,
    bins=10,
    density=False,
    label="",
    color="b",
    colorbar=True,
    z_values=[10, 140],
    lw=4,
):
    """
    Create saturation gradation encoded multiple histogram plot.

    :param value_list: List of numpy vectors containing the data, [npvec_1, npvec_2, ...,
                                                                   npvec_i]
    :param bins: Number of bins
    :param density: Boolean to switch between density or count output
    :param label: Label of the plot sequence
    :param color: Base color of the plot sequence
    :param colorbar: Boolean to activate colorbar
    :return: Returns None
    """

    first_plot = True  # flag to assign only labels to the first plots
    # plt.figure()  # create new figure
    figure_number = plt.gcf().number  # get current figure number

    for i in range(0, len(value_list)):
        bin_list_np = np.array(value_list[i])  # convert to numpy array

        if value_list[i] is None:
            warnings.warn(
                "Warning ptvvis.grad_hist: Bin number " + str(i + 1) + " is empty"
            )
            continue  # Jump list entry, if bin is empty
        elif bin_list_np.size <= 1:
            warnings.warn(
                "Warning ptvvis.grad_hist: Bin number "
                + str(i + 1)
                + " has only one entry"
            )
            continue

        plt.figure()  # create dummy figure
        counts, bin_edges, bars = plt.hist(
            value_list[i], bins=bins, density=density, label=""
        )
        plt.cla()  # clear dump figure
        plt.close()  # suppress window

        bin_center = np.array(
            [
                (bin_edges[_ + 1] - bin_edges[_]) / 2 + bin_edges[_]
                for _ in range(len(bin_edges) - 1)
            ]
        )

        # custom plotting
        # set color
        rgb_value = clr.to_rgb(color)
        hsv_value = clr.rgb_to_hsv(rgb_value)  # convert to HSV to manipulate brightness
        hsv_value[2] = (
            i / (len(value_list) - 1) * 0.49
        ) + 0.5  # start at 0.5, where colordifference is visible
        rgb_value = clr.hsv_to_rgb(hsv_value)

        plt.figure(figure_number)  # set plotting to correct figure

        # set colorbar
        # sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.normalize(v_min=z_values[0],
        # v_max=z_values[len(z_values)]))

        if first_plot:
            plt.plot(
                bin_center[np.where(counts > 0)],
                counts[np.where(counts > 0)],
                label=label,
                color=rgb_value,
                lw=lw,
            )
            first_plot = False
        else:
            plt.plot(
                bin_center[np.where(counts > 0)],
                counts[np.where(counts > 0)],
                label="",
                color=rgb_value,
                lw=lw,
            )

    return None


def optimize_logarithmic_bins(
    bin_counts_0, bin_range, n_bins, shift_0=0, plot_flag=False
):
    """
    Function to optimize shift for logarithmic binning. Uses linear interpolation and
    piecewise constant count density estimation to avoid iterating. Returns shift which
    minimizes the variance of the bin counts.

    :param bin_counts_0: List, bin counts of input binning
    :param bin_range: List containing binning range, [z_min, z_max]
    :param n_bins: Number of bins
    :param shift_0: Shift of input binning
    :param plot_flag: Boolean to activate plotting

    :return: Estimated optimal integer shift
    """
    # get boundaries of initial binning
    bin_boundaries_0 = get_bins(bin_range, n_bins, mode="logarithmic", shift=shift_0)
    mean_bin_position_0 = np.round(
        (bin_boundaries_0[:, 1] - bin_boundaries_0[:, 0]) / 2 + bin_boundaries_0[:, 0]
    )

    # estimate counts in a given bin from input bin counts
    def interp_bin_count(bin_boundary, bin_counts_0, bin_boundaries_0):
        count_density = bin_counts_0 / (bin_boundaries_0[:, 1] - bin_boundaries_0[:, 0])
        # estimate new counts from count density using the average and linear
        # interpolation (constant outside of domain)
        bin_count = (
            (
                np.interp(
                    bin_boundary[1],
                    mean_bin_position_0,
                    count_density,
                    count_density[0],
                    count_density[len(count_density) - 1],
                )
                + np.interp(
                    bin_boundary[0],
                    mean_bin_position_0,
                    count_density,
                    count_density[0],
                    count_density[len(count_density) - 1],
                )
            )
            / 2
            * (bin_boundary[1] - bin_boundary[0])
        )
        return bin_count

    # local function to compute variance of bin count
    def number_variance(bin_boundaries, bin_counts_0, bin_boundaries_0):
        bin_counts = [
            interp_bin_count(bin_boundaries[i, :], bin_counts_0, bin_boundaries_0)
            for i in range(0, bin_boundaries.shape[0])
        ]
        return np.var(np.array(bin_counts))  # return variance

    # optimize shift by minimizing
    shift = 0  # start with zero shift
    var_list = [0] * 1000  # list to store variance for minimum-search and plotting
    for i in range(0, 1000):  # scan range of relevant shifts
        bin_boundaries = get_bins(bin_range, n_bins, mode="logarithmic", shift=shift)
        var_list[i] = number_variance(
            bin_boundaries, bin_counts_0, bin_boundaries_0
        )  # store variance for given shift
        shift = i

    # get minimum from bin variance vector
    shift_min = np.argmin(np.array(var_list))
    bin_boundaries = get_bins(bin_range, n_bins, mode="logarithmic", shift=shift_min)
    mean_bin_position = np.round(
        (bin_boundaries[:, 1] - bin_boundaries[:, 0]) / 2 + bin_boundaries[:, 0]
    )
    if plot_flag:
        # plot results
        plt.figure()
        plt.plot(var_list)
        plt.xlabel("Shift")
        plt.ylabel("Count Variance")
        plt.title("Bin Count Variance")
        plt.grid()
        plt.legend()

        plt.figure()
        plt.scatter(
            bin_counts_0,
            mean_bin_position_0,
            label="Original Binning, Shift = " + str(shift_0),
        )
        plt.scatter(
            [
                interp_bin_count(bin_boundaries[i, :], bin_counts_0, bin_boundaries_0)
                for i in range(0, bin_boundaries.shape[0])
            ],
            mean_bin_position,
            label="Optimized, Shift = " + str(shift_min),
        )

        # plot linear- and un-shifted log-binning counts as reference
        bin_boundaries = get_bins(bin_range, n_bins, mode="logarithmic")
        mean_bin_position = np.round(
            (bin_boundaries[:, 1] - bin_boundaries[:, 0]) / 2 + bin_boundaries[:, 0]
        )
        plt.plot(
            [
                interp_bin_count(bin_boundaries[i, :], bin_counts_0, bin_boundaries_0)
                for i in range(0, bin_boundaries.shape[0])
            ],
            mean_bin_position,
            label="Log. Reference",
        )
        bin_boundaries = get_bins(bin_range, n_bins, mode="linear")
        mean_bin_position = np.round(
            (bin_boundaries[:, 1] - bin_boundaries[:, 0]) / 2 + bin_boundaries[:, 0]
        )
        plt.plot(
            [
                interp_bin_count(bin_boundaries[i, :], bin_counts_0, bin_boundaries_0)
                for i in range(0, bin_boundaries.shape[0])
            ],
            mean_bin_position,
            label="Linear Reference",
        )

        plt.xlabel("Counts")
        plt.ylabel("z-Coordinate")
        plt.title("Bin Count")
        plt.grid()
        plt.legend()
        plt.show()

    return shift
