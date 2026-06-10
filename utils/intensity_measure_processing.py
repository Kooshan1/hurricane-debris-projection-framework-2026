import h5py
import numpy as np
import os

# Constants
RHO = 1  #1025 is not used to be compatible with Catalina's parameters  # Density of sea water in kg/m^3    # TODO: check if this is correct --> seems correct!


def load_data(folder_path, file_name, dataset_name):
    full_path = os.path.join(folder_path, file_name)
    with h5py.File(full_path, 'r') as file:
        data = np.array(file[dataset_name])
        data = np.where(data == -99999, 0, data)
        return data


def calculate_max_surge_depth(zeta_max, depth):
    # Only evaluate surge depth for inland locations
    zeta_processed = np.where((zeta_max < 0) | (depth >= 0), 0, zeta_max + depth)
    # Set negative values in zeta_processed to zero
    zeta_processed = np.where(zeta_processed < 0, 0, zeta_processed)
    return zeta_processed


def calculate_max_surge_level(zeta_max):
    # No extra calculation needed!
    return zeta_max


def get_wave_direction(swan_dir):
    # TODO: getting the maximum direction through time steps is wrong! However, we have to do this since the previous
    #  models by Catalina and Carl were trained with that!
    max_dir = np.max(swan_dir, axis=0)
    return max_dir


def get_wave_height_at_max_surge(swan_HS_max):
    # No extra calculation needed!
    return swan_HS_max


def get_max_wave_velocity_x(u_vel):

    # Find the maximum velocity magnitude across all time steps for each node
    max_velocity = np.max(u_vel, axis=0)

    return max_velocity


def get_max_wave_velocity_y(v_vel):

    # Find the maximum velocity magnitude across all time steps for each node
    max_velocity = np.max(v_vel, axis=0)

    return max_velocity


def get_max_wave_velocity(u_vel, v_vel):
    # Calculate the magnitude of velocity at each point
    velocity_magnitude = np.sqrt(u_vel ** 2 + v_vel ** 2)

    # Find the maximum velocity magnitude across all time steps for each node
    max_velocity = np.max(velocity_magnitude, axis=0)

    return max_velocity


def calculate_wind_direction(windx, windy):
    n_tp, n_nodes = windx.shape  # Correct dimensions: 188 time steps, 1141758 nodes
    wind_dir = np.zeros(n_nodes)

    for i in range(n_nodes):
        # Calculate wind speed at each time point for the current node
        ws_TH = np.sqrt(windx[:, i] ** 2 + windy[:, i] ** 2)
        V_max = np.max(ws_TH)
        V_thres = 0.7 * V_max
        ind = np.where(ws_TH > V_thres)[0]

        if len(ind) > 0:
            # Filter indices for the calculation based on the threshold
            angles = np.arctan2(windy[ind, i], windx[ind, i])
            # Convert angles to complex numbers for vector averaging
            complex_angles = np.exp(1j * angles)
            # Perform a weighted average of the complex representations
            weighted_complex_avg = np.sum(complex_angles * ws_TH[ind]) / np.sum(ws_TH[ind])
            # Also add 180 to the calculated angle to be consistent with debris volume model
            wind_dir[i] = np.angle(weighted_complex_avg, deg=True) + 180
        else:
            # Assign a default or invalid value if no time step meets the condition
            wind_dir[i] = np.nan  # This case handles nodes with wind speeds never exceeding the threshold

    # Ensure Wind_dir values wrap properly around 360 degrees
    wind_dir = np.mod(wind_dir, 360)

    return wind_dir


def calculate_wind_steadiness(windx, windy):
    U_avg = np.mean(windx, axis=0)
    V_avg = np.mean(windy, axis=0)
    Mv = np.sqrt(U_avg**2 + V_avg**2)
    Msp = np.mean(np.sqrt(windx**2 + windy**2), axis=0)
    return Mv / Msp  # TODO: we get nan values. Not sure we should keep them as nan!


def get_max_wind_velocity(windx, windy):
    # Calculate the magnitude of wind velocity at each point
    wind_velocity_magnitude = np.sqrt(windx ** 2 + windy ** 2)

    # Find the maximum wind velocity magnitude across all time steps for each node
    max_wind_velocity = np.max(wind_velocity_magnitude, axis=0)

    return max_wind_velocity


def calculate_momentum_flux(zeta, depth, u_vel, v_vel):
    # TODO: Check units for formulas --> seems correct!
    surge_depth = np.where((zeta < 0) | (depth >= 0), 0, zeta + depth)
    surge_depth = np.where(surge_depth < 0, 0, surge_depth)
    Fx = RHO * surge_depth * u_vel**2
    Fy = RHO * surge_depth * v_vel**2
    Ftotal = np.sqrt(Fx**2 + Fy**2)
    Ftotal_max = np.max(Ftotal, axis=0)  # TODO: I have used the maximum (shouldn't be mean?)
    return Ftotal_max


def calculate_inundation_duration(zeta, depth, time_step):
    # Calculate the surge depth
    surge_depth = np.where((zeta < 0) | (depth >= 0), 0, zeta + depth)
    surge_depth = np.where(surge_depth < 0, 0, surge_depth)

    # Count the number of positive surge depth values for each point
    positive_surge_count = np.sum(surge_depth > 0, axis=0)

    # Calculate the inundation duration in hours
    inun_dur = positive_surge_count * time_step

    return inun_dur
