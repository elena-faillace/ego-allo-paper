import pandas as pd
import numpy as np
import os
import pickle
import logging
import itertools
import random

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from scipy.linalg import qr, svd, inv
from sklearn.naive_bayes import GaussianNB

########## PATHS ##########

# TODO: change to path where the data is stored
root_dir = "/Users/elenafaillace/Library/CloudStorage/OneDrive-ImperialCollegeLondon/arena2.0/paper_code_review/"


########## PARAMETERS ##########

px_per_cm = 5.2
arena_cm = 20 * 7
arena_px = (
    arena_cm * px_per_cm
)  # The borders of the arena are from 0 to 700 on both axis
bin_size = 10 * px_per_cm  # Size of the bins for the decoder of position (in pixels)
well_radius_px = 15 / 2 * px_per_cm  # Radius of the wells in pixels (15 cm diameter)
strategies = {
    "H2226": {1: "ALLO", 2: "EGO"},
    "H2225": {1: "ALLO", 2: "EGO"},
    "H2230": {1: "ALLO", 2: "EGO"},
    "H2234": {1: "ALLO", 2: "EGO"},
    "H2241": {1: "ALLO", 2: "EGO"},
    "H2222": {1: "EGO", 2: "ALLO"},
    "H2224": {1: "EGO", 2: "ALLO"},
    "H2231": {1: "EGO", 2: "ALLO"},
    "H2235": {1: "EGO", 2: "ALLO"},  # ALLO here is not present
}
rats = list(strategies.keys())
# Define centers of wells
rw1_x, rw1_y = (20 + 20 / 2) * px_per_cm, (20 + 20 / 2) * px_per_cm
rw2_x, rw2_y = (20 * 5 + 20 / 2) * px_per_cm, rw1_y
rw3_x, rw3_y = (20 * 2 + 20 / 2) * px_per_cm, (20 * 3 + 20 / 2) * px_per_cm
rw4_x, rw4_y = (20 * 4 + 20 / 2) * px_per_cm, rw3_y
rw5_x, rw5_y = rw1_x, (20 * 5 + 20 / 2) * px_per_cm
rw6_x, rw6_y = rw2_x, rw5_y
# Define symmetrical combos and control combos
symmetrical_combos = {
    "N1": "S6",
    "S6": "N1",
    "W1": "E6",
    "E6": "W1",
    "S1": "N6",
    "N6": "S1",
    "E1": "W6",
    "W6": "E1",
    "N5": "S2",
    "S2": "N5",
    "W5": "E2",
    "E2": "W5",
    "S5": "N2",
    "N2": "S5",
    "E5": "W2",
    "W2": "E5",
    "N3": "S4",
    "S4": "N3",
    "W3": "E4",
    "E4": "W3",
    "S3": "N4",
    "N4": "S3",
    "E3": "W4",
    "W4": "E3",
}
control_combo = {
    "N1": "N2",
    "N2": "N1",
    "N3": "N4",
    "N4": "N3",
    "N5": "N6",
    "N6": "N5",
    "S1": "S2",
    "S2": "S1",
    "S3": "S4",
    "S4": "S3",
    "S5": "S6",
    "S6": "S5",
    "E1": "E5",
    "E5": "E1",
    "E2": "E6",
    "E6": "E2",
    "E3": "W4",
    "E4": "W3",
    "W1": "W5",
    "W5": "W1",
    "W2": "W6",
    "W6": "W2",
    "W3": "E4",
    "W4": "E3",
}


########## LOAD DATAFRAMES/DATA ##########


# Dataframe raw combined
def load_df_all_raw(rat, phase):
    """Load the dataframe of all the combined experiments, given a rat and a phase."""

    path_to_data = root_dir + "data/" + rat + "_phase" + str(phase) + "_all_raw.csv"
    try:
        df = pd.read_csv(path_to_data, low_memory=False)
    except FileNotFoundError:
        print("File of rat " + rat + " not found")
        print(path_to_data)
        df = None
    return df


# Dataframe raw + trials extracted
def load_df_with_trials(rat, phase):
    """Load the dataframe with all the information I added."""

    path_to_data = (
        root_dir + "data/" + rat + "_phase" + str(phase) + "_events_with_trials.csv"
    )
    try:
        df = pd.read_csv(path_to_data, low_memory=False)
    except FileNotFoundError:
        print("File of rat " + rat + " not found: " + path_to_data)
        df = None
    return df


# Informations on all trials
def load_all_trials_info(phase):
    """Load the dataframe with all the trials info."""

    path_to_data = root_dir + "/data/" + "phase" + str(phase) + "_all_trials_info.csv"
    try:
        df = pd.read_csv(path_to_data, low_memory=False)
    except FileNotFoundError:
        print("File with all the trials info not found")
        df = None
    return df


# Dataframe with raw + trials + firing rates
def load_df_all_data_firing_rates(rat, phase):
    """Load the dataframe with all the information I added."""

    path_to_data = (
        root_dir + "data/" + rat + "_phase" + str(phase) + "_data_with_firing_rates.csv"
    )
    try:
        df = pd.read_csv(path_to_data, low_memory=False)
    except FileNotFoundError:
        print("File of rat " + rat + " not found")
        df = None
    return df


# PCA transformations
def load_pca_transformations(rat, phase):
    """Load the dictionary with the PCA transformation of all the experiments of a rat."""

    path_to_data = (
        root_dir
        + "dictionaries/"
        + rat
        + "_phase"
        + str(phase)
        + "_all_experiments_pca_space.pkl"
    )
    try:
        with open(path_to_data, "rb") as f:
            pca_transformations = pickle.load(f)
    except FileNotFoundError:
        print("File with the PCA transformations not found")
        pca_transformations = None
    return pca_transformations


# Symmetrical trials data
def load_symmetrical_trials_data(phase):
    """
    Load the dictionary with the data of the symmetrical trajectories.
    'clened' means a file where I manually removed the trials that were not good but were selected automatically.
    """
    path_to_data = (
        root_dir + "dictionaries/phase" + str(phase) + "_symmetrical_trials_data.pkl"
    )
    try:
        with open(path_to_data, "rb") as f:
            symmetrical_trials_data = pickle.load(f)
    except FileNotFoundError:
        print("File with symmetrical trials data not found")
        symmetrical_trials_data = None
    return symmetrical_trials_data


# CCA correlations
def load_cca_correlations(phase, n_components):
    """
    Load the CCA correlations found on all rats in a phase using n_components for CCA allignment.
    They are in the form of rat -> symmetrical trials types -> combined trials -> correlations.
    The combined trials contain all the information of the 2 trials details.
    """

    path_to_file = (
        root_dir
        + "dictionaries/cca_phase"
        + str(phase)
        + "_comp"
        + str(n_components)
        + ".pkl"
    )
    try:
        with open(path_to_file, "rb") as f:
            cca_correlations = pickle.load(f)
    except FileNotFoundError:
        print("File with the CCA correlations not found")
        print(path_to_file)
        cca_correlations = None
    return cca_correlations


# Load decoder results
def load_decoder_results(phase):
    """Load the results from the decoder."""
    path_to_results = (
        root_dir + "dictionaries/phase" + str(phase) + "_decoding_results.pkl"
    )
    try:
        with open(path_to_results, "rb") as f:
            decoder_results = pickle.load(f)
    except FileNotFoundError:
        print("File with the decoder results not found: " + path_to_results)
        decoder_results = None
    return decoder_results


# Load summery for figures
def load_summary_figures_data(phase):
    """Load the summary data for the figures."""
    path_to_data = root_dir + "dictionaries/phase" + str(phase) + "_figures_data.csv"
    try:
        df = pd.read_csv(path_to_data, low_memory=False)
    except FileNotFoundError:
        print("File with the summary data for figures not found: " + path_to_data)
        df = None
    return df


########## CHECKING LOCATION STATUS ##########


def in_starting_box(posx, posy):
    """Return True if the rat is in the starting box"""

    # West box
    if posx < 0:
        in_box = True
    # East box
    elif posx > arena_px:
        in_box = True
    # North box
    elif posy < 0:
        in_box = True
    # South box
    elif posy > arena_px:
        in_box = True
    else:
        in_box = False

    return in_box


def in_rewarded_well(posx, posy):
    """Return True if the rat is in a rewarded well (any)."""

    # Check if the rat is in the rewarded wells
    if (posx - rw1_x) ** 2 + (posy - rw1_y) ** 2 < well_radius_px**2:
        in_rw = True
    elif (posx - rw2_x) ** 2 + (posy - rw2_y) ** 2 < well_radius_px**2:
        in_rw = True
    elif (posx - rw3_x) ** 2 + (posy - rw3_y) ** 2 < well_radius_px**2:
        in_rw = True
    elif (posx - rw4_x) ** 2 + (posy - rw4_y) ** 2 < well_radius_px**2:
        in_rw = True
    elif (posx - rw5_x) ** 2 + (posy - rw5_y) ** 2 < well_radius_px**2:
        in_rw = True
    elif (posx - rw6_x) ** 2 + (posy - rw6_y) ** 2 < well_radius_px**2:
        in_rw = True
    else:
        in_rw = False

    return in_rw


def type_box(posx, posy):
    """Returns the label of the box the rat is in ('False' if in no box)"""

    # West box
    if posx < 0:
        box_type = "W"
    # East box
    elif posx > arena_px:
        box_type = "E"
    # North box
    elif posy < 0:
        box_type = "N"
    # South box
    elif posy > arena_px:
        box_type = "S"
    else:
        box_type = False

    return box_type


def number_rw(posx, posy):
    """Returns the number of the rewarded well the rat is in (0 if in no well)."""

    # Check if the rat is in the rewarded wells
    if (posx - rw1_x) ** 2 + (posy - rw1_y) ** 2 < well_radius_px**2:
        rw = 1
    elif (posx - rw2_x) ** 2 + (posy - rw2_y) ** 2 < well_radius_px**2:
        rw = 2
    elif (posx - rw3_x) ** 2 + (posy - rw3_y) ** 2 < well_radius_px**2:
        rw = 3
    elif (posx - rw4_x) ** 2 + (posy - rw4_y) ** 2 < well_radius_px**2:
        rw = 4
    elif (posx - rw5_x) ** 2 + (posy - rw5_y) ** 2 < well_radius_px**2:
        rw = 5
    elif (posx - rw6_x) ** 2 + (posy - rw6_y) ** 2 < well_radius_px**2:
        rw = 6
    else:
        rw = 0

    return rw


########## PROCESS DATAFRAMES TO ADD TRIALS AND LOCATIONS ##########


def add_trials_to_df(df, experiments, df_idx):
    """
    Add column to the dataframe with the 'Trial' (0 or 1), and 'Correct_trial' (0 or 1).
    """
    rewarded_well = df["Rewarded_well"].unique()[0]
    n_trials = np.sum(df["Time(s)"] == 0.0)
    df.insert(5, "Trial", np.zeros(len(df)))
    df.insert(6, "Correct_trial", np.zeros(len(df)))
    try:  # TODO: delete this, just to see still get a dataset saved
        # indexs from which to start looking for trials
        time_beginning = df.index[df["Time(s)"] == 0.0].to_numpy()
        time_beginning = time_beginning - time_beginning[0]

        for t in range(n_trials):
            # Select the df only of this potential trial
            if t == n_trials - 1:
                mask = (df.index - df.index[0]) >= time_beginning[t]
            else:
                mask = ((df.index - df.index[0]) >= time_beginning[t]) & (
                    (df.index - df.index[0]) < time_beginning[t + 1]
                )
            # Get the combo specific to this trial
            combo = df[mask]["Combo"].to_numpy()[0]

            ### REMOVE THE STARTING BOX FROM THE POTENTIAL TRIAL ###
            mask = mask & (df["Location"] != combo[0])

            ### CHECK IF THE RAT GETS TO THE REWARDED WELL ###
            # Should always happen?
            if str(rewarded_well) in df[mask]["Location"].unique():
                # Get the index of the first time the rat gets to the rewarded well
                end_index = np.intersect1d(
                    np.where(df["Location"] == str(rewarded_well))[0], np.where(mask)
                )[0]
                start_index = end_index - list(mask[:end_index][::-1]).index(False)

                ### SELECT ONLY THE ARENA BEFORE THE CORRECT REWARDED WELL ###
                mask = (
                    mask
                    & ((df.index - df.index[0]) >= start_index)
                    & ((df.index - df.index[0]) <= end_index)
                )
                df.loc[mask, "Trial"] = np.ones(len(df[mask])) * (t + 1)

                ### CHECK IF THE TRIAL IS CORRECT ###
                # Check that the rat does not stop for more than 1s (20 frames) in another well during the trial, by going through the rows of the df
                # Check that the rat never goes outside the arena during the trial
                # Check that trials are no longer than 5s (100 frames)
                correct_trial = True
                for i, _ in enumerate(df[mask].iterrows()):
                    # Check previous 10 frames and see if the rat is in another well
                    if i >= 20:
                        loc10 = df[mask]["Location"].to_numpy()[i - 20 : i]
                        if (
                            np.all(loc10 == "1")
                            or np.all(loc10 == "2")
                            or np.all(loc10 == "3")
                            or np.all(loc10 == "4")
                            or np.all(loc10 == "5")
                            or np.all(loc10 == "6")
                        ):
                            correct_trial = False
                    # Check if the rat goes outside the arena
                    if df[mask]["Location"].to_numpy()[i] == "outside":
                        correct_trial = False
                    # Check if the trial is longer than 5s
                    if i >= 100:
                        correct_trial = False

                if correct_trial:
                    df.loc[mask, "Correct_trial"] = np.ones(len(df[mask]))

            else:
                print(
                    "Error: rat seems to never get to the rewarded well "
                    + str(rewarded_well)
                    + " in experiment "
                    + str(experiments[df_idx])
                    + " trial "
                    + str(t + 1)
                    + "."
                )
                print(df[mask]["Location"].unique())
    except:
        print(
            "Error in experiment "
            + str(experiments[df_idx])
            + ". Skipping its processing for now."
        )

    return df


def apply_tollerance_window(locations, var, rewarded_well):
    """
    Apply a tollerance window to the location of the rat when calculated, to avoid small oscillations in the location (10/20Hz)=0.5s.
    If the rat is the past 10 frames is in a location 'var', then a different one and ultimately the same location 'var', then the different location is changed to 'var'.
    Make an exeption for the reward well, don't delete it once it is reached.

    INPUTS:
    - locations: list of strings, the location of the rat in the past frames
    - var: string, the location we want to extend to previous frames if it was present in the last 10 frames with another location in between
    - rewarded_well: int, the number of the rewarded well, this is the exception to the rule

    OUTPUT:
    - locations: list of strings, updated location of the rat in the past frames
    """

    tollerance_window = 10
    first_var_present = False
    another_var_present = False
    if len(locations) > tollerance_window:
        # Check if the var was present before another var was
        for j in range(len(locations) - tollerance_window, len(locations)):
            if locations[j] == var:
                first_var_present = True
            if (
                (locations[j] != var)
                and (locations[j] != rewarded_well)
                and first_var_present
            ):
                another_var_present = True
        # In case update all the past 10 frames to be var
        if first_var_present and another_var_present:
            for j in range(len(locations) - tollerance_window, len(locations)):
                locations[j] = var
    return locations


def add_locations_to_df(df):
    """
    Add the 'Location' column to the dataframe. It can be: 'arena', 'outside', 'W', 'E', 'N', 'S', '1', '2', '3', '4', '5', '6'.
    """

    rewarded_well = df["Rewarded_well"].unique()[0]

    ##################################################
    ### Add the 'Location' column to the dataframe ###
    ##################################################

    locations = []
    # Go through the experiment dataframe and add the location of the rat
    for i, row in df.iterrows():
        x = (row["Rightear_x"] + row["Leftear_x"]) / 2
        y = (row["Rightear_y"] + row["Leftear_y"]) / 2
        rw = row["Rewarded_well"]

        ### CHECK FOR STARTING BOX ###
        if in_starting_box(x, y):
            sb = type_box(x, y)
            # if it was in a SB in the previous 10 frames, modify them to be in the SB
            locations = apply_tollerance_window(
                locations, sb, rewarded_well=rewarded_well
            )
            locations.append(sb)

        ### CHECK FOR REWARDED WELL ###
        # check if rat is in any rw and if the number of the rw is the current one
        elif in_rewarded_well(x, y) and number_rw(x, y) == rw:
            # if it was in a RW in the previous 10 frames, modify them to be in the RW
            locations = apply_tollerance_window(
                locations, str(rw), rewarded_well=rewarded_well
            )
            locations.append(str(rw))

        ### CHECK FOR ANY OTHER WELL ###
        elif in_rewarded_well(x, y) and number_rw(x, y) != rw:
            # if it was in any RW in the previous 10 frames, modify the locations to be in that RW
            locations = apply_tollerance_window(
                locations, str(number_rw(x, y)), rewarded_well=rewarded_well
            )
            locations.append(str(number_rw(x, y)))

        ### CHECK IF GOES OUTSIDE ARENA ###
        # if it is, write 'outside' and the trial is not correct
        # TODO: Could put a + some pixels for the borders
        elif x < 0 or x > arena_px or y < 0 or y > arena_px:
            locations = apply_tollerance_window(
                locations, "outside", rewarded_well=rewarded_well
            )
            locations.append("outside")

        ### OTHERWISE IT IS IN THE ARENA ###
        else:
            locations = apply_tollerance_window(
                locations, "arena", rewarded_well=rewarded_well
            )
            locations.append("arena")
    df.insert(4, "Location", locations)

    return df


def divide_df_sessions_stages(df):
    """Divide the dataframe in a list of dataframes where each df has a unique session and stage.
    OUTPUTS:
    - experiments = list of tuples (session, stage)
    - all_dfs = list of dataframes
    """

    experiments = []
    all_dfs = []

    sessions = df["Session"].unique()
    for session in sessions:
        stages = df[df["Session"] == session]["Stage"].unique()
        for stage in stages:
            experiments.append((session, stage))
            all_dfs.append(df[(df["Session"] == session) & (df["Stage"] == stage)])

    return experiments, all_dfs


def add_locations_and_trials_to_df(df):
    """
    Add columns to the dataframe with the location of the rat.
    Adding: Location, Trial, CorrectTrial.
    """

    # Divide the dataset into experiments: sessions and stages
    experiments, all_dfs = divide_df_sessions_stages(df)

    for df_idx in range(len(all_dfs)):
        # Select the dataframe of the experiment
        sel_df = all_dfs[df_idx]

        # Add the 'Location' column to the dataframe
        sel_df = add_locations_to_df(sel_df)

        # Add the 'Trial' and 'Correct_trial' columns to the dataframe
        sel_df = add_trials_to_df(sel_df, experiments, df_idx)

        print("Finished experiment ", experiments[df_idx])

    return all_dfs


########## TRANSFORM EVENTS TO FIRING RATES ##########


def save_df_all_data_firing_rates(rat, phase):
    """Re-save the dataframes such that the firing rates are present instead of the events."""

    # If the files are already there do not make them again
    if os.path.exists(
        root_dir + "data/" + rat + "_phase" + str(phase) + "_data_with_firing_rates.csv"
    ):
        print(
            "File already exists: "
            + root_dir
            + "data/"
            + rat
            + "_phase"
            + str(phase)
            + "_data_with_firing_rates.csv"
        )
        return None

    print("Starting saving firing rates for rat " + rat + "...")
    df_rat = load_df_with_trials(rat, phase=phase)
    sessions = df_rat["Session"].unique()
    for session in sessions:
        stages = df_rat[df_rat["Session"] == session]["Stage"].unique()
        for stage in stages:
            # Identify the indices of the rows you want to modify
            indices = df_rat[
                (df_rat["Session"] == session) & (df_rat["Stage"] == stage)
            ].index
            neurons_id = [col for col in df_rat if col.startswith(" C")]
            events = df_rat.loc[indices, neurons_id].values
            firing_rates = gaussian_filter1d(events.T, sigma=4).T
            df_rat.loc[indices, neurons_id] = firing_rates
    path_to_save = (
        root_dir + "data/" + rat + "_phase" + str(phase) + "_data_with_firing_rates.csv"
    )
    df_rat.to_csv(path_to_save, index=False)
    return None


########## SUMMARY TRIALS ##########


def get_summary_trials(phase):
    # If the files is already saved, skip
    if os.path.exists(root_dir + "data/phase" + str(phase) + "_all_trials_info.csv"):
        print(
            "Trials already found in phase: "
            + str(phase)
            + ": "
            + root_dir
            + "data/phase"
            + str(phase + 1)
            + "_all_trials_info.csv"
        )
        return
    columns = [
        "Rat",
        "Strategy",
        "Session",
        "Stage",
        "Combo",
        "Rewarded_well",
        "Trial",
        "Correct_trial",
    ]
    new_df = pd.DataFrame(columns=columns)

    for rat in rats:
        print("...starting to find trials in rat: ", rat)
        strategy = strategies[rat][phase]
        try:
            df = load_df_with_trials(rat, phase)
            # Get all the sessions
            sessions = df["Session"].unique()
            for session in sessions:
                # Get all the stages
                stages = df[df["Session"] == session]["Stage"].unique()
                for stage in stages:
                    # Get all the trials
                    trials = df[(df["Session"] == session) & (df["Stage"] == stage)][
                        "Trial"
                    ].unique()[1:]
                    for trial in trials:
                        # Get the rewarded well
                        rewarded_well = df[
                            (df["Session"] == session)
                            & (df["Stage"] == stage)
                            & (df["Trial"] == trial)
                        ]["Rewarded_well"].unique()[0]
                        # Get the combo (Start box - Rewarded well)
                        combo = df[
                            (df["Session"] == session)
                            & (df["Stage"] == stage)
                            & (df["Trial"] == trial)
                        ]["Combo"].unique()[0]
                        # Get the correctness of the trial
                        correct_trial = df[
                            (df["Session"] == session)
                            & (df["Stage"] == stage)
                            & (df["Trial"] == trial)
                        ]["Correct_trial"].unique()[0]

                        ### APPEND ROW TO THE DF ###
                        new_row = {
                            "Rat": rat,
                            "Strategy": strategy,
                            "Session": session,
                            "Stage": stage,
                            "Combo": combo,
                            "Rewarded_well": rewarded_well,
                            "Trial": int(trial),
                            "Correct_trial": int(correct_trial),
                        }
                        new_df = pd.concat(
                            [new_df, pd.DataFrame([new_row])], ignore_index=True
                        )
        except:
            print("Could not find trials in rat: ", rat)
    path_to_save = root_dir + "data/phase" + str(phase) + "_all_trials_info.csv"
    new_df.to_csv(path_to_save, index=False)
    print(new_df.head())
    print(
        "\tTrials found and saved in phase: "
        + str(phase)
        + ": "
        + root_dir
        + "data/phase"
        + str(phase)
        + "_all_trials_info.csv"
    )
    return


########## SYMMETRICAL TRIALS ##########


def get_trials_info_from_combo(df_info, rat, combo):
    """Get the trial's info associated to a specific combo using the trails info df.
    Select the correct rewards and only SAM+CHO."""
    df_combo = df_info[
        (df_info["Combo"].str.startswith(combo[0]))
        & (df_info["Rewarded_well"] == int(combo[1]))
        & (df_info["Correct_trial"] == 1)
        & (df_info["Rat"] == rat)
        & ((df_info["Stage"] == "SAM") | (df_info["Stage"] == "CHO"))
    ]
    return df_combo


def get_trials_data_from_combo(df_rat, df_info, rat, combo):
    """Get the trial's data (x,y,dataframe) associated to a specific combo,
    using the trails info df and rat df.
    INPUTS:
    - df_rat = dataframe with the data of the rat
    - df_info = dataframe with the trials info
    - rat = name of the rat
    - combo = name of the combo (e.g. 'W1')
    OUTPUTS:
    - data = dictionary with the data of the trials ()
    """

    # Get the trials to select informationt
    df_combo = get_trials_info_from_combo(df_info, rat, combo)

    # Extract the data
    data = {}
    trials = [
        (combo, row.Session, row.Stage, row.Trial) for row in df_combo.itertuples()
    ]
    for trial in trials:
        data[trial] = {}
        combo, session, stage, trial_num = trial
        df_trial = df_rat[
            (df_rat["Trial"] == trial_num)
            & (df_rat["Session"] == session)
            & (df_rat["Stage"] == stage)
            & (df_rat["Combo"].str.startswith(combo[0]))
            & (df_rat["Rewarded_well"] == int(combo[1]))
        ]
        # Get the trajectory of the trial
        data[trial]["x"] = (
            df_trial["Leftear_x"].to_numpy() + df_trial["Rightear_x"].to_numpy()
        ) / 2
        data[trial]["y"] = (
            df_trial["Leftear_y"].to_numpy() + df_trial["Rightear_y"].to_numpy()
        ) / 2
        # Save the all data of the trial
        data[trial]["all_data"] = df_trial
    return data


def save_symmetrical_trials_data(phase):
    """
    Go through each rat and find the symmetrical trials combos and save them in a .pkl
    The structure of the dictionary is: rats -> (combo, symm_combo) -> trials details -> (x,y,dataframe)
    """

    # If the file is already there do not make it again
    if os.path.exists(
        root_dir + "data/phase" + str(phase) + "_symmetrical_trials_data.pkl"
    ):
        print(
            "File already exists: "
            + root_dir
            + "data/phase"
            + str(phase)
            + "_symmetrical_trials_data.pkl"
        )
        return None

    # Load the dataframe with the info of each trial
    all_trials_info = load_all_trials_info(phase=phase)

    # Select only correct trials
    correct_trials_info = all_trials_info[all_trials_info["Correct_trial"] == 1]
    # Select only SAM and CHO
    correct_trials_info = correct_trials_info[
        (correct_trials_info["Stage"] == "SAM")
        | (correct_trials_info["Stage"] == "CHO")
    ]
    rats = correct_trials_info["Rat"].unique()

    # Save the data in here to pickle
    symmetrical_trials_data = {}
    for rat in rats:
        print("Starting rat " + rat + "...")
        symmetrical_trials_data[rat] = {}
        df_rat = load_df_with_trials(rat, phase)
        df_rat_info = correct_trials_info[correct_trials_info["Rat"] == rat]

        # Go through each possible combo and save the data
        possible_combos = list(symmetrical_combos.keys())
        for c in possible_combos:
            # Check if the combo was already found in a pair
            if (symmetrical_combos[c], c) not in list(
                symmetrical_trials_data[rat].keys()
            ):
                starting_box = c[0]
                rewarded_well = c[1]
                sel_df = df_rat_info[
                    (df_rat_info["Combo"].str.startswith(starting_box))
                    & (df_rat_info["Rewarded_well"] == int(rewarded_well))
                ]
                # If the combo is present in the dataframe look if its symmetrical combo is present too
                if len(sel_df) > 0:
                    symm_c = symmetrical_combos[c]
                    symm_starting_box = symm_c[0]
                    symm_rewarded_well = symm_c[1]
                    symm_sel_df = df_rat_info[
                        (df_rat_info["Combo"].str.startswith(symm_starting_box))
                        & (df_rat_info["Rewarded_well"] == int(symm_rewarded_well))
                    ]
                    # If both are present than we found a symmetrical pair of trials
                    if len(symm_sel_df) > 0:
                        print(
                            "...symmetrical pair of trials found: "
                            + c
                            + " and "
                            + symm_c
                        )
                        symmetrical_trials_data[rat][c, symm_c] = {}

                        # For each combo in the pair finds the trials' data
                        data_combo = get_trials_data_from_combo(
                            df_rat, all_trials_info, rat, c
                        )
                        data_symm_combo = get_trials_data_from_combo(
                            df_rat, all_trials_info, rat, symm_c
                        )

                        data = {**data_combo, **data_symm_combo}
                        symmetrical_trials_data[rat][c, symm_c] = data

    # Save the data
    path_to_data = root_dir + "data/phase" + str(phase) + "_symmetrical_trials_data.pkl"
    with open(path_to_data, "wb") as f:
        pickle.dump(symmetrical_trials_data, f)
    return None


########## PCA ###########


def transform_pca_space_all_sessions(phase):
    """
    Get all the experiments for each rat (session + stage) and apply PCA. Save the resulted matrix for future transformations of single trials (save in rat_all_experiments_pca_space.pkl).
    We do this to compensate for the small number of active neurons during each trial.
    """

    # Get information on all the experiments
    all_trials_info = load_all_trials_info(phase)
    rats = all_trials_info["Rat"].unique()
    for rat in rats:
        if os.path.exists(
            root_dir
            + "data/"
            + rat
            + "_phase"
            + str(phase)
            + "_all_experiments_pca_space.pkl"
        ):
            print(
                "PCA space already exists for phase "
                + str(phase)
                + " and rat:"
                + rat
                + ", at path: "
                + root_dir
                + "data/"
                + rat
                + "_phase"
                + str(phase)
                + "_all_experiments_pca_space.pkl"
            )
            continue
        all_experiments_pca = {}
        print("Starting PCA analysis of rat: " + rat)
        # Load the data from each rat
        df_rat = load_df_all_data_firing_rates(rat, phase)
        sessions = all_trials_info[all_trials_info["Rat"] == rat]["Session"].unique()
        for session in sessions:
            all_experiments_pca[session] = {}
            stages = all_trials_info[
                (all_trials_info["Rat"] == rat)
                & (all_trials_info["Session"] == session)
            ]["Stage"].unique()
            for stage in stages:
                all_experiments_pca[session][stage] = {}
                print("...starting session: " + session + " stage: " + stage)

                # Get the firing rates of this experiment
                sel_df = df_rat[
                    (df_rat["Session"] == session) & (df_rat["Stage"] == stage)
                ]
                sel_df = sel_df[
                    (sel_df["Trial"] > 0) & (sel_df["Correct_trial"] == 1.0)
                ]
                # Get the activity matrix
                all_neurons_id = [col for col in sel_df if col.startswith(" C")]
                sel_df = sel_df[all_neurons_id]
                # Drop neurons that have NaN values
                sel_df = sel_df.dropna(axis="columns")
                # Save the remaining neurons for this experiment
                sel_neurons_id = [col for col in sel_df if col.startswith(" C")]
                all_experiments_pca[session][stage]["neurons_id"] = sel_neurons_id
                print(sel_df.shape)
                # If there are not enough components don't do PCA
                if sel_df.shape[0] == 0:
                    print(
                        "Not enough neurons for PCA in rat: "
                        + rat
                        + " session: "
                        + session
                        + " stage: "
                        + stage
                    )
                    all_experiments_pca[session][stage]["components"] = 0
                    all_experiments_pca[session][stage]["firing_rates"] = 0
                    all_experiments_pca[session][stage]["explained_variance"] = 0
                    all_experiments_pca[session][stage]["neurons_id"] = 0
                else:
                    # Normalize the data between 0 and 1
                    sel_df = (sel_df - sel_df.min(axis=0)) / (
                        sel_df.max(axis=0) - sel_df.min(axis=0)
                    )
                    # Write 0 for the neurons that are not active
                    sel_df = sel_df.fillna(0)
                    # Apply PCA and save the matrix of components
                    pca = PCA()
                    pca.fit(sel_df.values)  # samples x features
                    components = (
                        pca.components_
                    )  # features x components (are the same if all components are kept)
                    all_experiments_pca[session][stage]["components"] = components
                    all_experiments_pca[session][stage]["firing_rates"] = (
                        sel_df.values
                    )  # samples x features
                    all_experiments_pca[session][stage]["explained_variance"] = (
                        pca.explained_variance_ratio_
                    )

        # Save the dictionary
        path_to_save = (
            root_dir
            + "dictionaries/"
            + rat
            + "_phase"
            + str(phase)
            + "_all_experiments_pca_space.pkl"
        )
        with open(path_to_save, "wb") as f:
            pickle.dump(all_experiments_pca, f)
        print("...saved the PCA space of all experiments for rat: " + rat)


def get_trial_pca_space(
    phase, trial, symmetrical_trials, rat, symmetrical_trials_type, n_components
):
    """For the experiment of the trial load the PCA of it (session + stage) and project the trial on it.
    INPUTS:
    - phase: what phase is being analysed (1 or 2)
    - trial: tuple with the information of the experiment (combo, session, stage, trial)
    - symmetrical_trials: dictionary with the dataframe of the symmetrical trials
    - rat: string with the rat name
    - symmetrical_trials_type: string with the two symmetrical trials
    - n_components: number of components to keep in the PCA
    OUTPUTS:
    - trial_pca: 2D array with the trial projected on the PCA space (samples x components)
    """

    # Load the trial data
    _, session, stage, _ = trial
    trial_df = symmetrical_trials[rat][symmetrical_trials_type][trial]["all_data"]
    neurons_id = [col for col in trial_df if col.startswith(" C")]
    trial_fr = trial_df[neurons_id]
    # Remove the neurons with NaNs values
    trial_fr = trial_fr.dropna(axis="columns")
    trial_neurons_id = [col for col in trial_fr if col.startswith(" C")]
    trial_fr = trial_fr[trial_neurons_id]
    # Normalize the data between 0 and 1
    trial_fr = (trial_fr - trial_fr.min(axis=0)) / (
        trial_fr.max(axis=0) - trial_fr.min(axis=0)
    )
    # Write 0 for the neurons that are not active, where the divided was 0
    trial_fr = trial_fr.fillna(0)

    # Load the PCA space of the experiment
    all_exp_pca = load_pca_transformations(rat, phase=phase)
    if isinstance(all_exp_pca[session][stage]["components"], int):
        print(
            "No PCA space available for trial: "
            + str(trial)
            + " of rat: "
            + rat
            + " in phase: "
            + str(phase)
        )
        return None
    pca_space_components = all_exp_pca[session][stage]["components"]
    pca_space_mean = all_exp_pca[session][stage]["firing_rates"].mean(axis=0)
    pca_neurons_id = all_exp_pca[session][stage]["neurons_id"]
    # Keep only the first n_components
    pca_space_components = pca_space_components[:n_components]

    # Check that the number of neurons is the same, otherwise trial has to remove the corresponsing mismatching columns
    if len(trial_neurons_id) > len(pca_neurons_id):
        trial_fr = trial_fr.values[:, np.isin(trial_neurons_id, pca_neurons_id)]
    else:
        # Check that the neurons in the trial are the same as the ones in the PCA space
        if not np.all(np.array(trial_neurons_id) == np.array(pca_neurons_id)):
            print(
                "The neurons in the trial are not the same as the ones in the experiment PCA space"
            )
            print("...trial neurons: " + str(trial_neurons_id))
            print("...pca neurons: " + str(pca_neurons_id))
            return None

    # Project the trial on the PCA space
    trial_pca = np.dot(trial_fr - pca_space_mean, pca_space_components.T)
    return trial_pca


########### CONTROL TRIALS ##########


def get_random_sample_neural_activity(n_samples, phase, strategy):
    """Given a lenght of a sample of neural activity, return n_samples random samples of neural activity of the same length from a random animal from a random time window, form SAM or CHO."""

    ### Prepare the data ###

    # Numbers come from observation on the trials used in CCA (see distribution plot)
    mean_lenght = 53
    std_lenght = 15
    n_components = 5  # For the PCA space
    # For each rat with the right strategy load the rats neural activity, speeds up the process
    sel_rats = [
        rat for rat in list(strategies.keys()) if strategies[rat][phase] == strategy
    ]
    all_data = {}
    rats_to_remove = []  # In case some dataframe is None
    for rat in sel_rats:
        data = load_df_all_data_firing_rates(rat, phase)
        if data is None:
            rats_to_remove.append(rat)
            continue
        all_data[rat] = {}
        all_data[rat]["data"] = data
        all_data[rat]["pca"] = load_pca_transformations(rat, phase=phase)
    for rat in rats_to_remove:
        sel_rats.remove(rat)

    ### Extract samples ###
    print(
        "Starting to extract random samples of neural activity from phase "
        + str(phase)
        + " and strategy "
        + strategy
        + "..."
    )

    # Save n_samples samples of neural activity
    neural_samples = []
    n = 0
    while n < n_samples:
        # Choose a random rat + stage + session
        rat = np.random.choice(sel_rats)
        sel_df = all_data[rat]["data"]
        stage = np.random.choice(["SAM", "CHO"])
        sel_df = sel_df[sel_df["Stage"] == stage]
        sessions = sel_df["Session"].unique()
        session = np.random.choice(sessions)
        sel_df = sel_df[sel_df["Session"] == session]
        # Get the neural activity of the rat
        all_neurons_id = [col for col in sel_df if col.startswith(" C")]
        firing_rates = sel_df[all_neurons_id]

        # Get the PCA space of the rat+stage
        pca_data = all_data[rat]["pca"]
        try:
            pca_space_components = pca_data[session][stage]["components"]
            pca_space_mean = pca_data[session][stage]["firing_rates"].mean(axis=0)
            pca_neurons_id = pca_data[session][stage]["neurons_id"]
            # Keep only the first n_components
            pca_space_components = pca_space_components[:n_components]
            # Keep only the neurons that are in the PCA space
            firing_rates = firing_rates[pca_neurons_id]
            # Transform to 0s the neurons that have NaNs
            firing_rates = firing_rates.fillna(0)

            # Get a random time window by estimating the window length from a gaussian distribution
            length_sample = int(np.random.normal(mean_lenght, std_lenght))
            start_time = np.random.randint(0, firing_rates.shape[0] - length_sample)
            firing_rates = firing_rates.iloc[start_time : start_time + length_sample, :]
            # Normalize the data between 0 and 1
            firing_rates = (firing_rates - firing_rates.min(axis=0)) / (
                firing_rates.max(axis=0) - firing_rates.min(axis=0)
            )
            # Insert 0 in non active neurons
            firing_rates = firing_rates.fillna(0).values

            # Project the trial on the PCA space
            trial_pca = np.dot(firing_rates - pca_space_mean, pca_space_components.T)
            # Check rank of the matrix
            if np.linalg.matrix_rank(trial_pca) >= n_components:
                neural_samples.append(trial_pca)
                n = n + 1
                if n % 10 == 0:
                    print("number of samples taken: " + str(n) + "/" + str(n_samples))
        except:
            pass
    print(
        "Finished extracting random samples of neural activity-"
        + str(len(neural_samples))
    )
    with open(
        root_dir + "dictionaries/phase" + str(phase) + "_" + strategy + "_control.pkl",
        "wb",
    ) as f:
        pickle.dump(neural_samples, f)
    return


########### CCA CORRELATIONS ##########


def interpolate_dataset(short_dataset, target_length):
    """Interpolate two datasets to have the same length."""
    x_old = np.linspace(0, 1, short_dataset.shape[0])
    x_new = np.linspace(0, 1, target_length)
    interpolator = interp1d(x_old, short_dataset, axis=0, kind="linear")
    new_dataset = interpolator(x_new)
    return new_dataset


def canoncorr(X: np.array, Y: np.array, fullReturn: bool = False) -> np.array:
    """
    Canonical Correlation Analysis (CCA)
    line-by-line port from Matlab implementation of `canoncorr`
    X,Y: (samples/observations) x (features) matrix, for both: X.shape[0] >> X.shape[1]
    fullReturn: whether all outputs should be returned or just `r` be returned (not in Matlab)

    returns: A,B,r,U,V
    A,B: Canonical coefficients for X and Y
    U,V: Canonical scores for the variables X and Y
    r:   Canonical correlations

    Signature:
    A,B,r,U,V = canoncorr(X, Y)
    """
    n, p1 = X.shape
    p2 = Y.shape[1]
    if p1 >= n or p2 >= n:
        logging.warning("Not enough samples, might cause problems")

    # Center the variables
    X = X - np.mean(X, 0)
    Y = Y - np.mean(Y, 0)

    # Factor the inputs, and find a full rank set of columns if necessary
    Q1, T11, perm1 = qr(X, mode="economic", pivoting=True, check_finite=True)

    rankX = sum(
        np.abs(np.diagonal(T11))
        > np.finfo(type((np.abs(T11[0, 0])))).eps * max([n, p1])
    )

    if rankX == 0:
        logging.error(f"stats:canoncorr:BadData = X")
    elif rankX < p1:
        logging.warning("stats:canoncorr:NotFullRank = X")
        Q1 = Q1[:, :rankX]
        T11 = T11[:rankX, :rankX]

    Q2, T22, perm2 = qr(Y, mode="economic", pivoting=True, check_finite=True)
    rankY = sum(
        np.abs(np.diagonal(T22))
        > np.finfo(type((np.abs(T22[0, 0])))).eps * max([n, p2])
    )

    if rankY == 0:
        logging.error(f"stats:canoncorr:BadData = Y")
    elif rankY < p2:
        logging.warning("stats:canoncorr:NotFullRank = Y")
        Q2 = Q2[:, :rankY]
        T22 = T22[:rankY, :rankY]

    # Compute canonical coefficients and canonical correlations.  For rankX >
    # rankY, the economy-size version ignores the extra columns in L and rows
    # in D. For rankX < rankY, need to ignore extra columns in M and D
    # explicitly. Normalize A and B to give U and V unit variance.
    d = min(rankX, rankY)
    L, D, M = svd(
        Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver="gesdd"
    )
    M = M.T

    if (
        np.isnan(T11).any()
        or np.isnan(T22).any()
        or np.isinf(T11).any()
        or np.isinf(T22).any()
    ):
        logging.error("stats:canoncorr:BadData = X or Y")
    A = inv(T11) @ L[:, :d] * np.sqrt(n - 1)
    B = inv(T22) @ M[:, :d] * np.sqrt(n - 1)
    r = D[:d]
    # remove roundoff errs
    r[r >= 1] = 1
    r[r <= 0] = 0

    if not fullReturn:
        return r

    ### FROME HERE ###
    # Put coefficients back to their full size and their correct order
    # Adaptation from MATLAB: Assign to A the correct size by taking the first elements of A and putting them in the order of perm1
    stackedA = np.vstack((A, np.zeros((p1 - rankX, d))))
    newA = np.zeros(stackedA.shape)
    for stackedA_idx, newA_idx in enumerate(perm1):
        newA[newA_idx, :] = stackedA[stackedA_idx, :]

    stackedB = np.vstack((B, np.zeros((p2 - rankY, d))))
    newB = np.zeros(stackedB.shape)
    for stackedB_idx, newB_idx in enumerate(perm2):
        newB[newB_idx, :] = stackedB[stackedB_idx, :]
    ### TO HERE ###

    # Compute the canonical variates
    U = X @ newA
    V = Y @ newB

    return newA, newB, r, U, V


def calculate_cca_correlations(phase, n_components_pca, n_components_cca):
    """
    For each rat finds the correlations between the CCA components and save the results.
    INPUTS:
    - n_components_pca: number of components to keep in the PCA transformation before the aligning with the CCA
    - n_components_cca: number of CCA components to keep
    OUTPUTS:
    - all_cca_correlations: saved as a .pkl file ?
    """
    with open(
        root_dir + "dictionaries/phase" + str(phase) + "_ALLO_control.pkl", "rb"
    ) as f:
        allo_controls = pickle.load(f)
    with open(
        root_dir + "dictionaries/phase" + str(phase) + "_EGO_control.pkl", "rb"
    ) as f:
        ego_controls = pickle.load(f)

    all_cca_correlations = {}

    # Load information about symmetrical trials
    symmetrical_trials = load_symmetrical_trials_data(phase=phase)
    rats = list(symmetrical_trials.keys())
    # Work on one rat at the time
    for rat in rats:
        print("Starting CCA analysis of rat: " + rat)
        all_cca_correlations[rat] = {}
        # Find the pair of trial types to compare
        symmetrical_trials_types = list(symmetrical_trials[rat].keys())
        for symmetrical_trials_type in symmetrical_trials_types:
            print("...looking into symmetrical trials: " + str(symmetrical_trials_type))
            all_cca_correlations[rat][symmetrical_trials_type] = {}
            # Details of the single experiments
            trials = list(symmetrical_trials[rat][symmetrical_trials_type].keys())

            # Do the analysis on the trials
            combinations_trials = list(itertools.combinations(trials, 2))
            for combination_trials in combinations_trials:
                # Get the PCA space of the 2 trials
                trial_1, trial_2 = combination_trials
                trial_1_pca = get_trial_pca_space(
                    phase=phase,
                    trial=trial_1,
                    symmetrical_trials=symmetrical_trials,
                    rat=rat,
                    symmetrical_trials_type=symmetrical_trials_type,
                    n_components=n_components_pca,
                )
                trial_2_pca = get_trial_pca_space(
                    phase=phase,
                    trial=trial_2,
                    symmetrical_trials=symmetrical_trials,
                    rat=rat,
                    symmetrical_trials_type=symmetrical_trials_type,
                    n_components=n_components_pca,
                )

                ### Get the CCA correlation between the 2 trials ###
                if np.any(trial_1_pca == None) | np.any(trial_2_pca == None):
                    print(
                        "Error in PCA... skipping this pair of trials because problems in neurons ID: "
                        + str(trial_1)
                        + " ---- "
                        + str(trial_2)
                    )
                    continue

                # Extend the shortest trial and shorten the longest one
                max_len = max(trial_1_pca.shape[0], trial_2_pca.shape[0])
                min_len = min(trial_1_pca.shape[0], trial_2_pca.shape[0])
                new_len = int((max_len + min_len) / 2)
                trial_1_pca = interpolate_dataset(trial_1_pca, new_len)
                trial_2_pca = interpolate_dataset(trial_2_pca, new_len)
                # Save the x and y of the trials
                trial_1_x = symmetrical_trials[rat][symmetrical_trials_type][trial_1][
                    "x"
                ]
                trial_1_y = symmetrical_trials[rat][symmetrical_trials_type][trial_1][
                    "y"
                ]
                trial_2_x = symmetrical_trials[rat][symmetrical_trials_type][trial_2][
                    "x"
                ]
                trial_2_y = symmetrical_trials[rat][symmetrical_trials_type][trial_2][
                    "y"
                ]
                # Save the active neurons in trial 1 and trial 2
                trial_1_neurons = [
                    c
                    for c in symmetrical_trials[rat][symmetrical_trials_type][trial_1][
                        "all_data"
                    ]
                    .dropna(axis=1)
                    .columns
                    if c.startswith(" C")
                ]
                trial_1_active = np.array(trial_1_neurons)[
                    (
                        symmetrical_trials[rat][symmetrical_trials_type][trial_1][
                            "all_data"
                        ]
                        .dropna(axis=1)[trial_1_neurons]
                        .sum(axis=0)
                        > 0
                    ).values
                ]
                trial_2_neurons = [
                    c
                    for c in symmetrical_trials[rat][symmetrical_trials_type][trial_2][
                        "all_data"
                    ]
                    .dropna(axis=1)
                    .columns
                    if c.startswith(" C")
                ]
                trial_2_active = np.array(trial_2_neurons)[
                    (
                        symmetrical_trials[rat][symmetrical_trials_type][trial_2][
                            "all_data"
                        ]
                        .dropna(axis=1)[trial_2_neurons]
                        .sum(axis=0)
                        > 0
                    ).values
                ]

                try:
                    print("Comparing experiments {} ---- {}".format(trial_1, trial_2))
                    A, B, r, U, V = canoncorr(trial_1_pca, trial_2_pca, fullReturn=True)
                    print("...CCA number of canonical components: " + str(len(r)))
                except:
                    print("Error in CCA")
                    continue

                # Save the CCA correlation only if n_components were kept
                if len(r) >= n_components_cca:
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ] = {}
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["r"] = r[:n_components_cca]
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["A"] = A[:, :n_components_cca]
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["B"] = B[:, :n_components_cca]
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["U"] = U[:, :n_components_cca]
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["V"] = V[:, :n_components_cca]
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["trial1"] = {}
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["trial2"] = {}
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["trial1"]["x"] = trial_1_x
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["trial1"]["y"] = trial_1_y
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["trial2"]["x"] = trial_2_x
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["trial2"]["y"] = trial_2_y
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["trial1"]["all_neurons"] = trial_1_neurons
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["trial2"]["all_neurons"] = trial_2_neurons
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["trial1"]["active_neurons"] = trial_1_active
                    all_cca_correlations[rat][symmetrical_trials_type][
                        combination_trials
                    ]["trial2"]["active_neurons"] = trial_2_active

            if strategies[rat][phase] == "ALLO":
                control_samples = allo_controls
            else:
                control_samples = ego_controls
            # Do the analysis on the controls
            for i, trial in enumerate(trials):
                print("Control trial: " + str(trial))
                trial_pca = get_trial_pca_space(
                    phase=phase,
                    trial=trial,
                    symmetrical_trials=symmetrical_trials,
                    rat=rat,
                    symmetrical_trials_type=symmetrical_trials_type,
                    n_components=n_components_pca,
                )
                if trial_pca is None:
                    print(
                        "Error in PCA... skipping this trial because problems getting the PCA space: "
                        + str(trial)
                    )
                    continue
                control_done = False
                counter = 0
                while (not control_done) and (counter < 10):
                    counter = counter + 1
                    print(counter)
                    control_trial = random.choice(control_samples)
                    # Extend the shortest trial and shorten the longest one
                    max_len = max(trial_pca.shape[0], control_trial.shape[0])
                    min_len = min(trial_pca.shape[0], control_trial.shape[0])
                    new_len = int((max_len + min_len) / 2)
                    trial_pca = interpolate_dataset(trial_pca, new_len)
                    control_trial = interpolate_dataset(control_trial, new_len)
                    try:
                        print("Comparing experiments {} ---- control".format(trial))
                        A, B, r, U, V = canoncorr(
                            trial_pca, control_trial, fullReturn=True
                        )
                        print("...CCA number of canonical components: " + str(len(r)))
                    except:
                        print("Error in CCA")
                        continue
                    # Save the CCA correlation only if n_components were kept
                    if len(r) >= n_components_cca:
                        all_cca_correlations[rat][symmetrical_trials_type][
                            (trial, "control")
                        ] = {}
                        all_cca_correlations[rat][symmetrical_trials_type][
                            (trial, "control")
                        ]["r"] = r[:n_components_cca]
                        all_cca_correlations[rat][symmetrical_trials_type][
                            (trial, "control")
                        ]["A"] = A[:, :n_components_cca]
                        all_cca_correlations[rat][symmetrical_trials_type][
                            (trial, "control")
                        ]["B"] = B[:, :n_components_cca]
                        all_cca_correlations[rat][symmetrical_trials_type][
                            (trial, "control")
                        ]["U"] = U[:, :n_components_cca]
                        all_cca_correlations[rat][symmetrical_trials_type][
                            (trial, "control")
                        ]["V"] = V[:, :n_components_cca]
                        control_done = True

    # Save the cca_correlations in .pkl
    with open(
        root_dir
        + "dictionaries/cca_phase"
        + str(phase)
        + "_comp"
        + str(n_components_cca)
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump(all_cca_correlations, f)

    return None


########### DECODING POSITION ###########


def get_binned_position(x, y, xdim, ydim, bin_size):
    """Get the position of the rat in the binned arena.
    INPUTS:
    x, y = 1D array with position of rat in pixels
    bin_size = number of pixels for 1 bin
    xdim, ydim = edges of coordinates x and y in the arena
    OUTPUTS:
    x_bin, y_bin =
    x_binned, y_binned =
    """

    # Get size in pixels of the arena
    dist_x = xdim[0] - xdim[1]
    dist_y = ydim[0] - ydim[1]
    # Get number of bins in each dimension
    n_bins_x = int(dist_x / bin_size)
    n_bins_y = int(dist_y / bin_size)
    # Bin the x and y position of the rat
    x_b = (x / bin_size).astype(int)
    y_b = (y / bin_size).astype(int)
    # If some bins accidentally happen to be over the boundary?
    x_b[x_b >= n_bins_x] = n_bins_x - 1
    y_b[y_b >= n_bins_y] = n_bins_y - 1
    x_b[x_b < 0] = 0
    y_b[y_b < 0] = 0

    return x_b, y_b, n_bins_x, n_bins_y


def get_decoders_results(phase, n_components_cca):
    # Load files only once
    cca_results = load_cca_correlations(phase, n_components_cca)
    symmetrical_trials_info = load_symmetrical_trials_data(phase)

    # Create data structure to save the results
    results = {}

    # Go through all CCAs results and do the decoder analysis
    rats = list(cca_results.keys())
    for rat in rats:
        print("Rat: " + rat)
        results[rat] = {}
        symmetrical_combos = list(cca_results[rat].keys())
        for symmetrical_combo in symmetrical_combos:
            results[rat][symmetrical_combo] = {}
            trials_combinations = list(cca_results[rat][symmetrical_combo].keys())
            for trials_combination in trials_combinations:
                results[rat][symmetrical_combo][trials_combination] = {}
                trial1 = trials_combination[0]
                trial2 = trials_combination[1]
                if trial2 != "control":
                    combo1, _, _, _ = trial1
                    combo2, _, _, _ = trial2

                    ### Get CCA results and space ###

                    # Select the trajectories of the 2 trials in the CCA space
                    trial1_cca_traj = cca_results[
                        rat
                    ][
                        symmetrical_combo
                    ][
                        trials_combination
                    ][
                        "U"
                    ]  # canonical scores for X (n_timepoints x n_components_cca) transformed X to canonical space
                    trial2_cca_traj = cca_results[
                        rat
                    ][
                        symmetrical_combo
                    ][
                        trials_combination
                    ][
                        "V"
                    ]  # canonical scores for Y (n_timepoints x n_components_cca) transformed Y to canonical space
                    results[rat][symmetrical_combo][trials_combination][
                        "trial1_cca_U"
                    ] = trial1_cca_traj
                    results[rat][symmetrical_combo][trials_combination][
                        "trial2_cca_V"
                    ] = trial2_cca_traj

                    ### Get the neural trajectories of the trials selected ###

                    trial1_df = symmetrical_trials_info[rat][symmetrical_combo][trial1][
                        "all_data"
                    ]
                    trial2_df = symmetrical_trials_info[rat][symmetrical_combo][trial2][
                        "all_data"
                    ]
                    # Get the neurons IDs of the two trials
                    trial1_neurons = trial1_df.columns[
                        trial1_df.columns.str.startswith(" C")
                    ].values
                    trial2_neurons = trial2_df.columns[
                        trial2_df.columns.str.startswith(" C")
                    ].values
                    # get the intersection of the neurons
                    common_neurons = np.intersect1d(trial1_neurons, trial2_neurons)
                    # get the firing rates of the trials
                    trial1_firing_rates = trial1_df[common_neurons]
                    trial1_firing_rates = trial1_firing_rates.fillna(0).values
                    trial2_firing_rates = trial2_df[common_neurons]
                    trial2_firing_rates = trial2_firing_rates.fillna(0).values

                    # Equalise the number of timepoints
                    max_len = max(
                        trial1_firing_rates.shape[0], trial2_firing_rates.shape[0]
                    )
                    min_len = min(
                        trial1_firing_rates.shape[0], trial2_firing_rates.shape[0]
                    )
                    new_len = int((max_len + min_len) / 2)

                    trial1_firing_rates_ = interpolate_dataset(
                        trial1_firing_rates, new_len
                    )
                    trial2_firing_rates_ = interpolate_dataset(
                        trial2_firing_rates, new_len
                    )
                    results[rat][symmetrical_combo][trials_combination][
                        "trial1_firing_rates"
                    ] = trial1_firing_rates_
                    results[rat][symmetrical_combo][trials_combination][
                        "trial2_firing_rates"
                    ] = trial2_firing_rates_

                    ### Save also the x,y positions of the trials ###

                    trial1_x = trial1_df["Cap_x"].values
                    trial1_y = trial1_df["Cap_y"].values
                    trial2_x = trial2_df["Cap_x"].values
                    trial2_y = trial2_df["Cap_y"].values
                    # If the trial is symmetrical, the trial2 needs to be inverted
                    if combo1 != combo2:
                        trial2_x = arena_px - trial2_x
                        trial2_y = arena_px - trial2_y

                    # euqalize the number of timepoints
                    trial1_x = interpolate_dataset(trial1_x, new_len)
                    trial1_y = interpolate_dataset(trial1_y, new_len)
                    trial2_x = interpolate_dataset(trial2_x, new_len)
                    trial2_y = interpolate_dataset(trial2_y, new_len)
                    # get the binned positions
                    trial1_xb, trial1_yb, _, _ = get_binned_position(
                        trial1_x, trial1_y, [arena_px, 0], [arena_px, 0], bin_size
                    )
                    trial2_xb, trial2_yb, _, _ = get_binned_position(
                        trial2_x, trial2_y, [arena_px, 0], [arena_px, 0], bin_size
                    )
                    results[rat][symmetrical_combo][trials_combination]["trial1_xb"] = (
                        trial1_xb
                    )
                    results[rat][symmetrical_combo][trials_combination]["trial1_yb"] = (
                        trial1_yb
                    )
                    results[rat][symmetrical_combo][trials_combination]["trial2_xb"] = (
                        trial2_xb
                    )
                    results[rat][symmetrical_combo][trials_combination]["trial2_yb"] = (
                        trial2_yb
                    )

                    ### Make a decoder on the first trial and test it on the second trial, both for CCA space and neural space ###

                    # remove the points that have 0 activity in all neurons
                    activity_trial1_mask = ~np.all(trial1_firing_rates_ == 0, axis=1)
                    activity_trial2_mask = ~np.all(trial2_firing_rates_ == 0, axis=1)
                    # remove the neurons that have 0 activity in all points
                    activity_neurons_mask = (
                        ~np.all(trial1_firing_rates_ == 0, axis=0)
                    ) | (~np.all(trial2_firing_rates_ == 0, axis=0))
                    trial1_firing_rates_ = trial1_firing_rates_[
                        :, activity_neurons_mask
                    ]
                    trial2_firing_rates_ = trial2_firing_rates_[
                        :, activity_neurons_mask
                    ]

                    ### Start with decoding neural space ###

                    (
                        mse1,
                        mse2,
                        chance1,
                        chance2,
                        trial2_pred_x,
                        trial2_pred_y,
                        trial1_pred_x,
                        trial1_pred_y,
                    ) = apply_decoders_gaussianNB(
                        trial1_firing_rates_[activity_trial1_mask],
                        trial1_xb[activity_trial1_mask],
                        trial1_yb[activity_trial1_mask],
                        trial2_firing_rates_[activity_trial2_mask],
                        trial2_xb[activity_trial2_mask],
                        trial2_yb[activity_trial2_mask],
                    )
                    results[rat][symmetrical_combo][trials_combination][
                        "accuracy_decoder1_neural_space"
                    ] = mse1
                    results[rat][symmetrical_combo][trials_combination][
                        "accuracy_decoder2_neural_space"
                    ] = mse2
                    results[rat][symmetrical_combo][trials_combination][
                        "chance_decoder1_neural_space"
                    ] = chance1
                    results[rat][symmetrical_combo][trials_combination][
                        "chance_decoder2_neural_space"
                    ] = chance2

                    ### Now decode from CCA space ###

                    (
                        mse1,
                        mse2,
                        chance1,
                        chance2,
                        trial2_pred_x,
                        trial2_pred_y,
                        trial1_pred_x,
                        trial1_pred_y,
                    ) = apply_decoders_gaussianNB(
                        trial1_cca_traj,
                        trial1_xb,
                        trial1_yb,
                        trial2_cca_traj,
                        trial2_xb,
                        trial2_yb,
                    )
                    results[rat][symmetrical_combo][trials_combination][
                        "accuracy_decoder1_cca_space"
                    ] = mse1
                    results[rat][symmetrical_combo][trials_combination][
                        "accuracy_decoder2_cca_space"
                    ] = mse2
                    results[rat][symmetrical_combo][trials_combination][
                        "chance_decoder1_cca_space"
                    ] = chance1
                    results[rat][symmetrical_combo][trials_combination][
                        "chance_decoder2_cca_space"
                    ] = chance2
    # save the results
    path_to_save = (
        root_dir + "dictionaries/phase" + str(phase) + "_decoding_results.pkl"
    )
    with open(path_to_save, "wb") as f:
        pickle.dump(results, f)
    return


def apply_decoders_gaussianNB(X1, Y1_x, Y1_y, X2, Y2_x, Y2_y):
    """Apply the decoders on the training and test sets."""

    model1_x = GaussianNB()
    model1_y = GaussianNB()
    model2_x = GaussianNB()
    model2_y = GaussianNB()

    model1_x.fit(X1, Y1_x)
    model1_y.fit(X1, Y1_y)
    # Get a score weightedby the number of classes in each model
    score1 = model1_x.score(X2, Y2_x) * len(model1_x.classes_) / (
        len(model1_x.classes_) + len(model1_y.classes_)
    ) + model1_y.score(X2, Y2_y) * len(model1_y.classes_) / (
        len(model1_x.classes_) + len(model1_y.classes_)
    )
    chance1 = (1 / len(model1_x.classes_) + 1 / len(model1_y.classes_)) / 2
    Y2_x_pred, Y2_y_pred = model1_x.predict(X2), model1_y.predict(X2)

    model2_x.fit(X2, Y2_x)
    model2_y.fit(X2, Y2_y)
    score2 = model2_x.score(X1, Y1_x) * len(model2_x.classes_) / (
        len(model2_x.classes_) + len(model2_y.classes_)
    ) + model2_y.score(X1, Y1_y) * len(model2_y.classes_) / (
        len(model2_x.classes_) + len(model2_y.classes_)
    )
    chance2 = (1 / len(model2_x.classes_) + 1 / len(model2_y.classes_)) / 2
    Y1_x_pred, Y1_y_pred = model2_x.predict(X1), model2_y.predict(X1)

    return score1, score2, chance1, chance2, Y2_x_pred, Y2_y_pred, Y1_x_pred, Y1_y_pred


########### SUMMARY RESULTS ###########


def make_summary_results(phase):
    """
    Puts the result in a convinient dictiornary for plotting later.
    """
    summary_results = {}
    cca_results = load_cca_correlations(
        phase=phase, n_components=5
    )  # TODO: what if I use only 2 ?
    decoder_results = load_decoder_results(phase=phase)
    rats = list(cca_results.keys())
    for rat in rats:
        # Set up the variables to save
        summary_results[rat] = {}
        cca_pairs = []
        same_or_symm_type = []
        symmetrical_combo = []
        strategy = []
        acc_neural_space = []
        acc_cca_space = []
        chance_neural_space = []
        chance_cca_space = []
        distance_2d_trajectories = []
        correlation_2d_trajectories = []
        perc_active_neurons = []  # percentage of neurons that are active between the two trials vs the total number of active neurons (to proove that neural similarity is not to matching neurons but topology)

        symmetrical_types = list(cca_results[rat].keys())
        for symmetrical_type in symmetrical_types:
            pairs_trials = list(cca_results[rat][symmetrical_type].keys())
            for pair_trials in pairs_trials:
                # Define trials information
                trial_1, trial_2 = pair_trials
                # Get CCA correlations
                corr = cca_results[rat][symmetrical_type][pair_trials]["r"]
                cca_pairs.append(np.array(corr).mean())
                if trial_2 != "control":
                    trial_combo_1, _, _, _ = trial_1
                    trial_combo_2, _, _, _ = trial_2

                    # Get the average sme for of the two decoders
                    acc_neural_space_avg = np.round(
                        (
                            decoder_results[rat][symmetrical_type][pair_trials][
                                "accuracy_decoder1_neural_space"
                            ]
                            + decoder_results[rat][symmetrical_type][pair_trials][
                                "accuracy_decoder2_neural_space"
                            ]
                        )
                        / 2,
                        3,
                    )
                    acc_cca_space_avg = np.round(
                        (
                            decoder_results[rat][symmetrical_type][pair_trials][
                                "accuracy_decoder1_cca_space"
                            ]
                            + decoder_results[rat][symmetrical_type][pair_trials][
                                "accuracy_decoder2_cca_space"
                            ]
                        )
                        / 2,
                        3,
                    )
                    chance_neural_space_avg = np.round(
                        (
                            decoder_results[rat][symmetrical_type][pair_trials][
                                "chance_decoder1_neural_space"
                            ]
                            + decoder_results[rat][symmetrical_type][pair_trials][
                                "chance_decoder2_neural_space"
                            ]
                        )
                        / 2,
                        3,
                    )
                    chance_cca_space_avg = np.round(
                        (
                            decoder_results[rat][symmetrical_type][pair_trials][
                                "chance_decoder1_cca_space"
                            ]
                            + decoder_results[rat][symmetrical_type][pair_trials][
                                "chance_decoder2_cca_space"
                            ]
                        )
                        / 2,
                        3,
                    )
                    acc_neural_space.append(acc_neural_space_avg)
                    acc_cca_space.append(acc_cca_space_avg)
                    chance_neural_space.append(chance_neural_space_avg)
                    chance_cca_space.append(chance_cca_space_avg)

                    # Get the distance between the trajectories
                    t1_x = cca_results[rat][symmetrical_type][pair_trials]["trial1"][
                        "x"
                    ]
                    t1_y = cca_results[rat][symmetrical_type][pair_trials]["trial1"][
                        "y"
                    ]
                    t2_x = cca_results[rat][symmetrical_type][pair_trials]["trial2"][
                        "x"
                    ]
                    t2_y = cca_results[rat][symmetrical_type][pair_trials]["trial2"][
                        "y"
                    ]
                    # Make the trials the same lenght
                    max_len = max(len(t1_x), len(t2_x))
                    min_len = min(len(t1_x), len(t2_x))
                    new_len = int((max_len + min_len) / 2)
                    t1_x = interpolate_dataset(t1_x, new_len)
                    t1_y = interpolate_dataset(t1_y, new_len)
                    t2_x = interpolate_dataset(t2_x, new_len)
                    t2_y = interpolate_dataset(t2_y, new_len)
                    # Get the distance
                    distance_2d_trajectories.append(
                        np.mean(np.sqrt((t1_x - t2_x) ** 2 + (t1_y - t2_y) ** 2))
                    )
                    # Get the correlation between the trajectories
                    correlation_2d_trajectories.append(
                        (np.corrcoef(t1_x, t2_x)[0, 1] + np.corrcoef(t1_y, t2_y)[0, 1])
                        / 2
                    )

                    # Get percentage of active neurons
                    shared_neurons = np.intersect1d(
                        cca_results[rat][symmetrical_type][pair_trials]["trial1"][
                            "active_neurons"
                        ],
                        cca_results[rat][symmetrical_type][pair_trials]["trial2"][
                            "active_neurons"
                        ],
                    )
                    total_neurons = np.union1d(
                        cca_results[rat][symmetrical_type][pair_trials]["trial1"][
                            "active_neurons"
                        ],
                        cca_results[rat][symmetrical_type][pair_trials]["trial2"][
                            "active_neurons"
                        ],
                    )
                    perc_active_neurons.append(
                        float(len(shared_neurons)) / float(len(total_neurons))
                    )

                    # Get the type of pair
                    if trial_combo_1 == trial_combo_2:
                        same_or_symm_type.append("Same-type")
                    else:
                        same_or_symm_type.append("Symmetrical-type")
                else:
                    same_or_symm_type.append("Control")
                    acc_neural_space.append(np.nan)
                    acc_cca_space.append(np.nan)
                    chance_neural_space.append(np.nan)
                    chance_cca_space.append(np.nan)
                    distance_2d_trajectories.append(np.nan)
                    correlation_2d_trajectories.append(np.nan)
                    perc_active_neurons.append(np.nan)

                # Save the symmetrical combo
                symmetrical_combo.append(symmetrical_type)
                # Save the strategy of the animal
                strategy.append(strategies[rat][phase])

        # Save the results
        summary_results[rat]["cca"] = cca_pairs
        summary_results[rat]["pair-type"] = same_or_symm_type
        summary_results[rat]["symm_combo"] = symmetrical_combo
        summary_results[rat]["strategy"] = strategy
        summary_results[rat]["acc_neural_space"] = acc_neural_space
        summary_results[rat]["chance_neural_space"] = chance_neural_space
        summary_results[rat]["acc_cca_space"] = acc_cca_space
        summary_results[rat]["chance_cca_space"] = chance_cca_space
        summary_results[rat]["distance_2d_trajectories"] = distance_2d_trajectories
        summary_results[rat]["correlation_2d_trajectories"] = (
            correlation_2d_trajectories
        )
        summary_results[rat]["perc_active_neurons"] = perc_active_neurons

    return summary_results


def save_figures_data(summary_results, phase):
    """Save the data for all the figures in a csv file."""

    animal_csv = []
    strategy_csv = []
    cca_csv = []
    same_or_symm_csv = []
    pair_of_combos_csv = []
    sme_neural_csv = []
    chance_neural_csv = []
    sme_cca_csv = []
    chance_cca_csv = []
    distance_2d_trajectories_csv = []
    correlation_2d_trajectories_csv = []
    perc_active_neurons_csv = []

    rats = list(summary_results.keys())
    for rat in rats:
        # Go through all the saved results for each rat and save them in a dataframe
        for idx in range(len(summary_results[rat]["cca"])):
            # Save the values to add to this line
            animal_csv.append(rat)
            strategy_csv.append(strategies[rat][phase])
            cca_csv.append(summary_results[rat]["cca"][idx])
            same_or_symm_csv.append(summary_results[rat]["pair-type"][idx])
            pair_of_combos_csv.append(summary_results[rat]["symm_combo"][idx])
            sme_neural_csv.append(summary_results[rat]["acc_neural_space"][idx])
            chance_neural_csv.append(summary_results[rat]["chance_neural_space"][idx])
            sme_cca_csv.append(summary_results[rat]["acc_cca_space"][idx])
            chance_cca_csv.append(summary_results[rat]["chance_cca_space"][idx])
            distance_2d_trajectories_csv.append(
                summary_results[rat]["distance_2d_trajectories"][idx]
            )
            correlation_2d_trajectories_csv.append(
                summary_results[rat]["correlation_2d_trajectories"][idx]
            )
            perc_active_neurons_csv.append(
                summary_results[rat]["perc_active_neurons"][idx]
            )

    # Create the dataframe
    final_df = pd.DataFrame(
        {
            "Animal": animal_csv,
            "Strategy": strategy_csv,
            "CCA_correlation": cca_csv,
            "Same_or_symm": same_or_symm_csv,
            "Accuracy_neural_space": sme_neural_csv,
            "Chance_neural_space": chance_neural_csv,
            "Accuracy_cca_space": sme_cca_csv,
            "Chance_cca_space": chance_cca_csv,
            "Pair_of_combos": pair_of_combos_csv,
            "Distance_2d_trajectories": distance_2d_trajectories_csv,
            "Correlation_2d_trajectories": correlation_2d_trajectories_csv,
            "Perc_active_neurons": perc_active_neurons_csv,
        }
    )

    # Save df
    final_df.to_csv(root_dir + "dictionaries/phase" + str(phase) + "_figures_data.csv")
    return None


############ PLOTTING FUNCTIONS ###########


def plot_violin_with_points(
    ax,
    data_groups,
    title="",
    ylabel="",
    xlim=None,
    ylim=None,
    show_violin=True,
    show_points=True,
    show_mean=True,
    show_errorbar=True,
    violin_alpha=0.3,
    point_alpha=0.4,
    point_size=15,
    mean_size=80,
    jitter_strength=0.05,
    errorbar_capsize=5,
    violin_bandwidth=None,
    violin_resolution=50,
):
    """
    Create violin plot with individual points, animal averages, and error bars.
    Data selection and preparation should be done in the notebook.

    Parameters:
    -----------
    ax : matplotlib axes
        The axes to plot on
    data_groups : list of dict
        List of data groups, each dict MUST contain:
        - 'data': pandas.Series or array-like with individual data points
        - 'animal_ids': pandas.Series or array-like with animal identifiers (same length as data)
        - 'position': float, x-position for this group
        - 'color': str, color for this group
        - 'label': str, label for this group (used for x-axis ticks)
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    xlim : tuple, optional
        X-axis limits (min, max)
    ylim : tuple, optional
        Y-axis limits (min, max)
    show_violin : bool, optional
        Whether to show violin density plot
    show_points : bool, optional
        Whether to show individual data points
    show_mean : bool, optional
        Whether to show animal means
    show_errorbar : bool, optional
        Whether to show error bars (SEM)
    violin_alpha : float, optional
        Transparency for violin plot
    point_alpha : float, optional
        Transparency for individual points
    point_size : int, optional
        Size of individual points
    mean_size : int, optional
        Size of animal mean points
    jitter_strength : float, optional
        Amount of horizontal jitter for points
    errorbar_capsize : int, optional
        Size of error bar caps
    violin_bandwidth : float, optional
        Bandwidth for violin smoothing (None for automatic). Lower values = less smooth
    violin_resolution : int, optional
        Number of points to use for violin curve (default: 50)

    Returns:
    --------
    dict : Dictionary containing plotted elements for further customization
    """

    plot_elements = {"points": [], "means": [], "errorbars": [], "violins": [], "medians": []}
    x_positions = []
    x_labels = []

    for group in data_groups:
        # Extract required group data (no defaults, all must be provided)
        data = group["data"]
        animal_ids = group["animal_ids"]
        position = group["position"]
        color = group["color"]
        label = group["label"]

        # Remove NaN values
        mask = ~pd.isna(data)
        data_clean = data[mask]
        animal_ids_clean = animal_ids[mask]

        if len(data_clean) == 0:
            continue

        x_positions.append(position)
        x_labels.append(label)

        # Calculate animal means and percentiles from individual data points
        animal_data = pd.DataFrame({"data": data_clean, "animal": animal_ids_clean})
        animal_means = animal_data.groupby("animal")["data"].mean()
        group_mean = animal_means.mean()
        
        # Calculate percentiles from ALL individual data points (not just animal means)
        data_q25 = np.percentile(data_clean, 25)  # 25th percentile of all individual points
        data_q75 = np.percentile(data_clean, 75)  # 75th percentile of all individual points

        # 1. Plot violin (density) if requested using matplotlib's violinplot (seaborn-style)
        if show_violin and len(data_clean) > 1:
            # Create matplotlib violin plot with custom smoothing controls
            if violin_bandwidth is not None:
                # Use custom KDE implementation with controlled bandwidth
                from scipy.stats import gaussian_kde
                
                # Create KDE with custom bandwidth
                kde = gaussian_kde(data_clean)
                kde.set_bandwidth(violin_bandwidth)
                
                # Generate violin shape
                y_range = np.linspace(data_clean.min(), data_clean.max(), violin_resolution)
                kde_values = kde(y_range)
                kde_values = kde_values / np.max(kde_values) * 0.15  # normalize width
                
                # Plot symmetric violin
                violin = ax.fill_betweenx(
                    y_range,
                    position - kde_values,
                    position + kde_values,
                    color=color,
                    alpha=violin_alpha,
                    edgecolor='white',
                    linewidth=0.8,
                    zorder=1,
                )
                plot_elements["violins"].append(violin)
            else:
                # Use matplotlib's default violinplot
                violin_parts = ax.violinplot(
                    [data_clean],
                    positions=[position],
                    widths=0.3,
                    showmeans=True,
                    showextrema=False,
                    showmedians=False,
                )

                # Style the violin to match seaborn aesthetics
                for pc in violin_parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(violin_alpha)
                    pc.set_edgecolor("white")
                    pc.set_linewidth(0.8)
                    # Make violin symmetric and smooth like seaborn
                    pc.set_linestyle("-")
                
                plot_elements["violins"].append(violin_parts)

        # 2. Plot individual points with jitter if requested
        if show_points:
            x_jitter = position + jitter_strength * np.random.randn(len(data_clean))
            points = ax.scatter(
                x_jitter,
                data_clean,
                alpha=point_alpha,
                s=point_size,
                color=color,
                zorder=3,
            )
            plot_elements["points"].append(points)

        # 3. Plot error bars (25th and 75th percentiles of individual data points)
        if show_errorbar and len(data_clean) > 1:  # Need at least 2 data points for percentiles
            # Use median as center point for percentile error bars (more robust)
            data_median = np.median(data_clean)
            
            # Calculate asymmetric error bars: distance from median to percentiles
            lower_err = data_median - data_q25
            upper_err = data_q75 - data_median
            
            # Ensure error values are non-negative (safety check)
            lower_err = max(0, lower_err)
            upper_err = max(0, upper_err)
            
            errorbar = ax.errorbar(
                position,
                data_median,  # Use median as center point
                yerr=[[lower_err], [upper_err]],  # Asymmetric error bars
                fmt="none",
                color="black",
                capsize=errorbar_capsize,
                capthick=1.5,
                linewidth=2,
                alpha=0.8,
                zorder=4,
            )
            plot_elements["errorbars"].append(errorbar)
            
            # Add a visible median line/marker
            median_line = ax.plot(
                [position - 0.15, position + 0.15],  # Short horizontal line
                [data_median, data_median],
                color='black',
                linewidth=3,
                alpha=1.0,
                zorder=6,  # Above error bars
            )
            plot_elements["medians"].append(median_line)

        # 4. Plot animal means as larger points
        if show_mean:
            # Add small jitter to animal means for visibility
            x_means = position + jitter_strength * 1 * np.random.randn(
                len(animal_means)
            )
            means = ax.scatter(
                x_means,
                animal_means.values,
                s=mean_size,
                color=color,
                edgecolors="black",
                linewidths=1.5,
                alpha=0.9,
                zorder=5,
            )
            plot_elements["means"].append(means)

    # Set up axes using labels from data groups with seaborn-like styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=10)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Apply seaborn-like styling
    ax.set_title(title, fontsize=12, pad=15, fontweight="normal")
    ax.set_ylabel(ylabel, fontsize=11)

    # Seaborn-style grid and spines
    ax.grid(True, alpha=0.3, zorder=0, color="white", linewidth=1.2)
    ax.set_facecolor('white')  # Light grey background like seaborn

    # Remove top and right spines, style remaining ones
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)

    # Style ticks
    #ax.tick_params(colors="grey", which="both")

    return plot_elements
