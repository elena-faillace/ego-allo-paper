import pandas as pd
import numpy as np
import os
import pickle

from scipy.ndimage import gaussian_filter1d


########## PATHS ##########

# TODO: change to path where the data is stored
root_dir = "/Users/elenafaillace/Library/CloudStorage/OneDrive-ImperialCollegeLondon/arena2.0/paper_code_review/"


########## PARAMETERS ##########

px_per_cm = 5.2
arena_cm = 20 * 7
arena_px = (
    arena_cm * px_per_cm
)  # The borders of the arena are from 0 to 700 on both axis
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
    "H2235": {1: "EGO", 2: "ALLO"},
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
