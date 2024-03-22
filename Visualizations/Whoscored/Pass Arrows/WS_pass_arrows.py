import pandas as pd
import numpy as np
from mplsoccer import Pitch, VerticalPitch
import matplotlib.pyplot as plt
from unidecode import unidecode

def get_pass_arrows_df(df) -> pd.DataFrame:
    """
    Extracts pass information DataFrame.

    Parameters:
    df (DataFrame): DataFrame containing event data.

    Returns:
    DataFrame: DataFrame containing pass arrows event data.
    """

    df["next_event"]=df["type_display_name"].shift(-1)

    df_passes = df[df.type_display_name == "Pass"][["team_id","player_name","period_display_name","minute", "second", "x", "y", "end_x", "end_y", "outcome_type_display_name", "next_event"]]
    df_passes["progressive"] = True
    df_passes.reset_index(drop=True)

    df_passes["beginning"] = np.sqrt(np.square(100-df_passes["x"]) + np.square(50 - df_passes["y"]))
    df_passes["end"] = np.sqrt(np.square(100-df_passes["end_x"]) + np.square(50 - df_passes["end_y"]))
    df_passes["progressive"] = df_passes.apply(lambda row: row["end"] / row["beginning"] < 0.75, axis=1)

    df_passes.loc[df_passes['outcome_type_display_name'] != "Unsuccessful", "pass_type"] = "Successful"
    df_passes.loc[df_passes["x"]>df_passes["end_x"],"progressive"] = False
    df_passes.loc[df_passes["progressive"]==True,"pass_type"] = "ProgressivePass"
    df_passes.loc[(df_passes["next_event"]=="MissedShots")|(df_passes["next_event"]=="SavedShot")|(df_passes["next_event"]=="ShotOnPost"),"pass_type"] = "keyPass"
    df_passes.loc[df_passes["next_event"]=="Goal","pass_type"] = "Assist"
    df_passes.loc[df_passes['outcome_type_display_name'] == "Unsuccessful", "pass_type"] = "Unsuccessful"

    df_passes.loc[df_passes['pass_type'] == "Unsuccessful", 'color'] = "#848585"
    df_passes.loc[df_passes['pass_type'] == "Successful", 'color'] = "#0793BC"
    df_passes.loc[df_passes['pass_type'] == "ProgressivePass", 'color'] = "#0CD127"
    df_passes.loc[df_passes['pass_type'] == "keyPass", 'color'] = "#DBE110"
    df_passes.loc[df_passes['pass_type'] == "Assist", 'color'] = "#F52825"

    return df_passes

# Home: home team passes
# Away: away team passes
# Player name: player passes

def plot_arrows(df, match_data, player="home"):
    """
    Plots pass arrows on the pitch.

    Parameters:
    df (DataFrame): DataFrame containing event data.
    match_data (dict): Dictionary containing match data including team information.
    player (str, optional): Specifies the player or team for which pass arrows are plotted.
        If "home" or "away", pass arrows for the corresponding team will be plotted.
        If a player name is provided, pass arrows for that player will be plotted.

    Returns:
    None
    """
    
    pitch = Pitch(pitch_type="opta")
    fig, axs = pitch.grid(nrows=1, ncols=1, figheight=8, grid_width=0.9, endnote_space=0.05, axis=False, title_space=0.01)
    plt.tight_layout()

    date = match_data['timeStamp'][0:10]
    main_color='#050732'

    axs['title'].text(0.5, 0.8, f"{match_data['home']['name']} vs {match_data['away']['name']}", ha='center', va='center', color=main_color, fontsize=22)
    axs['title'].text(0.5, 0.55, f"{date}", ha='center', va='center', color=main_color, fontsize=12)
    axs['title'].text(0.5, 0.35, "Passes attempted by", ha='center', va='center', color=main_color, fontsize=12)
    axs['endnote'].text(1, 1, '@Joel_AS', va='center', ha='right', color=main_color, fontsize=12)

    if (player == "home") | (player == "away"):
        team_id = match_data[player]['teamId']

        df_arrows = get_pass_arrows_df(df)
        df_arrows = df_arrows[df_arrows.team_id == team_id]

        pass_completed = df_arrows[df_arrows["outcome_type_display_name"] == "Successful"].shape[0]
        pass_att = df_arrows.shape[0]
        player_name = match_data[player]['name']
        

        axs['pitch'].set_title(f"{player_name}: {pass_completed}/{pass_att} ({pass_completed/pass_att:.2%}) passes completed ", color=main_color, fontsize=20)
    
    else:
        
        player = unidecode(player).lower().replace(" ","")

        df_arrows = get_pass_arrows_df(df)
        df_arrows["name_normalized"] = df_arrows["player_name"].apply(lambda x: unidecode(x).lower().replace(" ", ""))
        df_arrows = df_arrows[df_arrows["name_normalized"].str.contains(player)]

        pass_completed = df_arrows[df_arrows["outcome_type_display_name"] == "Successful"].shape[0]
        pass_att = df_arrows.shape[0]
        player_name = df_arrows.player_name.iloc[0]

        axs['pitch'].set_title(f"{player_name}: {pass_completed}/{pass_att} ({pass_completed/pass_att:.2%}) passes completed ", color=main_color, fontsize=20)

    legend_labels = ['Successful', 'Unsuccessful', 'Progressive pass', 'Key pass', 'Assist']
    legend_colors = ["#0793BC","#848585","#0CD127", "#DBE110", "#F52825"]
    legend_elements = [plt.Line2D([0], [0], marker='>', color='w', label=label, markerfacecolor=color, markersize=14)
                       for label, color in zip(legend_labels, legend_colors)]
    axs['pitch'].legend(handles=legend_elements, loc="best", fontsize = "small")
    pitch.arrows(df_arrows.x, df_arrows.y,df_arrows.end_x,df_arrows.end_y,ax=axs['pitch'], color=df_arrows.color, alpha=0.8, width=1.5)
    plt.show()