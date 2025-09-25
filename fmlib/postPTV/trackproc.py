
# standard modules
import time

# third party/ custom modules
import numpy as np
import matplotlib.pyplot as plt
import h5py

# custom modules
from fmlib.fio import ptvio

# add I/O-module to communicate with .piv track file


def load_tracks(file, number="all", print_status=True):
    if print_status:
        print("Loading trajectory data into memory...")

    # read track data from .ptv file into memory as list
    track_data = ptvio.ptvio(file)  # initialize IO object
    track_num = track_data.num_tracks()  # get number of tracks

    # load trajectories into local memory
    tracks = []  # initialize tracks as empty list
    tmp_time = time.perf_counter()

    if number == "all":
        with h5py.File(file, mode="r") as f:
            for j, entries in enumerate(f["tracks"]["all"].keys()):
                if (j == 1000 or j % round(track_num / 10) == 0) and j != 0:
                    print(
                        "Estimated time for loading "
                        + str(track_num)
                        + " trajectories: "
                        + str(round((time.perf_counter() - tmp_time) / j * track_num))
                        + "s"
                    )
                tracks.append(f["tracks/all/" + entries][:, :])
                # TODO: Make sure output is sorted by frame

        return tracks

    # read tracks as stored
    for i in range(0, number):
        if i == 1000:
            print(
                "Estimated time for loading "
                + str(number)
                + " trajectories: "
                + str(round((time.perf_counter() - tmp_time) / 1000 * number))
                + "s"
            )
        tracks.append(track_data.read_track(i))

    if print_status:
        print("Done.\n")

    return tracks


def merge_tracks(tracks, merge_matrix, print_status=True):
    # initialize array to store the output
    merged_tracks = []

    timing_flag = True

    merge_index = 0  # index pointing at the current entry of the merge list
    merge_counter = 0  # simple counter to count merged tracks

    # Set number of tacks
    number = len(tracks)

    for i in range(0, number):
        if timing_flag:
            if i == 0:
                start_time = time.perf_counter()
            if i == round(0.025 * number):
                diff_time = (
                    time.perf_counter() - start_time
                )  # Time difference in seconds
                if print_status:
                    print("Estimated time:\t" + str(round(diff_time * 40)) + "s")

        # read joint track, check if track is merged with another tack
        if merge_index < merge_matrix.shape[0] and i == merge_matrix[merge_index, 0]:
            track = tracks[i]
            track_add = tracks[merge_matrix[merge_index, 1]]
            track = np.row_stack((track, track_add))

            if merge_counter % 100 == 0:
                print(
                    "merged tracks: "
                    + str(i)
                    + ", "
                    + str(merge_matrix[merge_index, 1])
                )
            merge_counter += 1

            # search for other tracks to add
            for j in range(merge_index + 1, merge_matrix.shape[0]):
                if (
                    merge_matrix[j, 0] != -1
                    and merge_matrix[j, 0] == merge_matrix[merge_index, 1]
                    and merge_index + 1 < merge_matrix.shape[0]
                ):
                    track_add = tracks[merge_matrix[merge_index, 1]]
                    track = np.row_stack((track, track_add))
                    merge_matrix[
                        j, 0
                    ] = (
                        -1
                    )  # overwrite track index with negative value to avoid double detection

                    merge_counter += 1

            merged_tracks.append(track)
            merge_index += 1

        # skip trajectories flagged with -1
        elif merge_index < merge_matrix.shape[0] and merge_matrix[merge_index, 0] == -1:
            merge_index += 1

        elif (
            i not in merge_matrix[:, 1]
        ):  # If tracks are not merged, write as stored in the file
            merged_tracks.append(tracks[i])

    if print_status:
        print("Done. " + str(merge_counter) + " tracks were merged.\n")

    return merged_tracks


def extrapolate_track(track, num_fitting_points="all", order=2, mode="forward"):
    if num_fitting_points == "all":
        num_fitting_points = np.size(track[:, 0])

    # extrapolate forwards, add points at the end of the track
    if mode == "forward":
        # polynomial fit to position data
        extrap_start = np.size(track[:, 0]) - num_fitting_points
        poly_coefficients_x = np.polyfit(
            track[extrap_start:, 2], track[extrap_start:, 0], order
        )
        poly_coefficients_y = np.polyfit(
            track[extrap_start:, 2], track[extrap_start:, 1], order
        )

        # extrapolate points
        extrapolated_timesteps = track[np.size(track[:, 2]) - 1, 2] + np.arange(
            1, num_fitting_points + 1
        )
        extrapolated_x = np.polyval(poly_coefficients_x, extrapolated_timesteps)
        extrapolated_y = np.polyval(poly_coefficients_y, extrapolated_timesteps)

        # append extrapolated values to the trajectory and return result
        return np.append(
            track,
            np.column_stack((extrapolated_x, extrapolated_y, extrapolated_timesteps)),
            0,
        )

    # extrapolate backwards, add points at the beginning of the track
    if mode == "backward":
        # polynomial fit to data
        poly_coefficients_x = np.polyfit(
            track[:num_fitting_points, 2], track[:num_fitting_points, 0], order
        )
        poly_coefficients_y = np.polyfit(
            track[:num_fitting_points, 2], track[:num_fitting_points, 1], order
        )

        # extrapolate points
        extrapolated_timesteps = track[0, 2] + np.arange(-num_fitting_points, 0)
        extrapolated_x = np.polyval(poly_coefficients_x, extrapolated_timesteps)
        extrapolated_y = np.polyval(poly_coefficients_y, extrapolated_timesteps)

        # append extrapolated values to the trajectory and return result
        return np.append(
            np.column_stack((extrapolated_x, extrapolated_y, extrapolated_timesteps)),
            track,
            0,
        )


def reconnect_trajectories(
    position,
    velocities,
    accelerations,
    frame,
    extrapolation_points=8,
    pos_threshold=2,
    relative_vel_threshold=1,
):
    print("Processing " + str(len(position[0])) + " to repair split trajectories...")
    # define some variables
    plot_flag = False
    reconnect_counter = 0

    # allocate memory
    connection_indices = []

    # Plot all tracks
    if False:  # write plot_flag instead of false vor activation
        for k in range(0, len(position[0])):  # compute for all trajectories
            if position[0][k] is not None:
                plt.scatter(position[0][k], position[1][k])

    for k in range(0, len(position[0])):  # compute for all trajectories
        if position[0][k] is not None:
            # if plot_flag:
            # plt.scatter(position[0][k], position[1][k])
            # extrapolate tracks from data of last point
            end_index = np.shape(position[0][k])[0] - 1  # get end index of the track

            # read position, velocity and acceleration at the last point of the
            # trajectory into numpy arrays and extrapolate
            pos = np.array([position[0][k][end_index], position[1][k][end_index]])
            vel = np.array([velocities[0][k][end_index], velocities[1][k][end_index]])
            acc = np.array(
                [accelerations[0][k][end_index], accelerations[1][k][end_index]]
            )

            # Calculate distance to other trajectories
            metric_min = None  # Initialize default vector
            for i in range(k, len(position[0])):  # compare with all other trajectories

                if frame[i] is not None:  # check if there is a valid frame
                    frame_diff = (
                        frame[i][0] - frame[k][end_index]
                    )  # time difference of both trajectories

                    if (
                        frame_diff > extrapolation_points
                    ):  # break loop if trajectories are too far away in time
                        break
                    elif (
                        position[0][i] is not None
                        and frame_diff <= extrapolation_points
                        and i != k
                        and frame[i][0] > frame[k][end_index] + 1
                    ):  # enforce extrapolation

                        # extrapolate trajectory
                        pos_extrap = pos + vel * frame_diff + acc * frame_diff**2
                        vel_extrap = vel + acc * frame_diff

                        # get position, velocity and frame number at the first
                        # point of the candidate trajectory
                        pos_next = np.array([position[0][i][0], position[1][i][0]])
                        vel_next = np.array([velocities[0][i][0], velocities[1][i][0]])

                        pos_diff = np.linalg.norm(
                            pos_next - pos_extrap
                        )  # get magnitude of vector
                        vel_diff = np.linalg.norm(vel_next - vel_extrap)
                        rel_vel_diff = np.linalg.norm(vel_next - vel_extrap) / (
                            np.linalg.norm(vel_next)
                        )

                        metric = pos_diff
                        # metric = math.sqrt(pos_diff ** 2 + vel_diff) # Metric of Xu et al. for delta_t = 1
                        # store minimum value of the metric
                        if (
                            metric_min is None
                            and abs(rel_vel_diff) < relative_vel_threshold
                        ):
                            metric_min = [metric, i]
                        if metric_min is not None:
                            if (
                                metric_min[0] > metric
                                and abs(rel_vel_diff) < relative_vel_threshold
                            ):
                                metric_min = [metric, i]

            if metric_min is not None and metric_min[0] < pos_threshold:
                reconnect_counter += 1
                if reconnect_counter % 100 == 0:
                    print(
                        "reconnected trajectory "
                        + str(k)
                        + " with "
                        + str(metric_min[1])
                        + ". Distance: "
                        + str(metric_min[0])
                    )

                if plot_flag:
                    plt.scatter(
                        position[0][k],
                        position[1][k],
                        color="b",
                        label="Extrapolated Trajectory",
                    )
                    plt.scatter(
                        position[0][metric_min[1]],
                        position[1][metric_min[1]],
                        color="g",
                        label="Connected Trajectory",
                    )
                    plt.scatter(
                        pos_extrap[0],
                        pos_extrap[1],
                        color="r",
                        label="Extrapolated Value",
                    )
                    plt.axis("equal")
                    plt.legend()
                    plt.show()

                connection_indices.append([k, metric_min[1]])

    print("\n")
    print(
        str(reconnect_counter)
        + " out of "
        + str(len(position[0]))
        + " trajectories reconnected"
    )

    if plot_flag:
        plt.show()

    return np.array(connection_indices)
