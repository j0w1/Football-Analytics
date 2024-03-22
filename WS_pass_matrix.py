from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import numpy as np
import pandas as pd
from typing import Tuple

def get_df_info(df) -> pd.DataFrame:
    """
    Extracts substitutions information DataFrame.

    Parameters:
    df (DataFrame): DataFrame containing event data.

    Returns:
    DataFrame: DataFrame containing substitutions event data.
    """

    subsIn = df[df.type_display_name == "SubstitutionOn"]["player_id"]
    subsOut = df[df.type_display_name == "SubstitutionOff"]["player_id"]

    df_info = df[["player_id", "player_name", "team_id", "shirt_no", "position", "is_first_eleven"]].copy().drop_duplicates()
    df_info["subbed_in"] = df_info["player_id"].isin(subsIn)
    df_info["subbed_out"] = df_info["player_id"].isin(subsOut)

    return df_info

def get_passes_df(df,team,match_data) -> pd.DataFrame:
    """
    Extracts passes information DataFrame.

    Parameters:
    df (DataFrame): DataFrame containing event data.
    team (str): Specifies the team for which passes are extracted.
        Options:
            - 'local': Extract passes for the local team.
            - 'away': Extract passes for the away team.
    match_data (dict): Dictionary containing match data including team information.

    Returns:
    DataFrame: DataFrame containing passes event data for the specified team.
    """

    df_info = get_df_info(df)

    team_id = match_data[team]['teamId']
    df2 = df[df['team_id']==team_id].copy()
    df2["receiver"] = df2["player_id"].shift(-1)

    passes_ids = df2.index[(df2.type_display_name == 'Pass')]
    df_passes = df2.loc[passes_ids, ["id", "x", "y", "end_x","end_y", "team_id", "player_id","shirt_no","position","is_first_eleven","player_name","receiver", "type_display_name", "outcome_type_display_name"]]
    df_passes = df_passes.merge(df_info[["player_id", "team_id","subbed_in", "subbed_out"]], on=["player_id","team_id"])

    return df_passes
    
def get_passes_between_df(df_passes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates passes counts, completion rates, average locations and pass amount between players.

    Parameters:
    df_passes (DataFrame): DataFrame containing passes event data.

    Returns:
    Tuple[DataFrame, DataFrame]: Tuple containing two DataFrames:
        1. DataFrame containing passes between players with their completion rates.
        2. DataFrame containing average locations and pass counts per player.
    """
    df_passes_total = df_passes.groupby('player_id').agg({'x': ['mean'], 'y': ['mean', 'count']})
    df_passes_total.columns = ['x', 'y', 'passes_attempted']

    df_passes_completed = df_passes[df_passes.outcome_type_display_name == 'Successful'].groupby('player_id').id.count().reset_index().rename({'id':'passes_completed'}, axis='columns')

    df_passes_total = df_passes_total.merge(df_passes_completed, on='player_id')
    df_passes_total["percentage_completed"] = round(df_passes_total.passes_completed/df_passes_total.passes_attempted*100,2)

    # Get the average loc. and the number of passes per player
    average_locs_and_count_df = df_passes_total.merge(df_passes[['player_id', 'player_name', 'shirt_no', 'position', 'is_first_eleven', "subbed_in", "subbed_out"]],on='player_id', how='left').set_index('player_id')
    average_locs_and_count_df = average_locs_and_count_df[average_locs_and_count_df.passes_completed > round(average_locs_and_count_df.passes_completed.max()*0.1,0)]

    # calculate the number of passes between each player in both directions (using player_id/receiver so we get passes in both ways)
    passes_player_ids_df = df_passes.loc[:, ['id', 'player_id', 'receiver', 'team_id']]
    passes_player_ids_df = passes_player_ids_df.groupby(['player_id','receiver']).id.count().reset_index().rename({'id': 'pass_count'}, axis='columns')

    # add on the location of each player so we have the start and end positions of the lines
    passes_between_df = passes_player_ids_df.merge(average_locs_and_count_df[['x','y','player_name','shirt_no','position']], left_on='player_id', right_index=True)
    passes_between_df = passes_between_df.merge(average_locs_and_count_df[['x','y','player_name','shirt_no','position']], left_on='receiver', right_index=True,suffixes=['', '_end']).drop_duplicates()
    #We only take that passes combinations that are higher than 10% of max. combination
    passes_between_df=passes_between_df[passes_between_df.pass_count > round(passes_between_df.pass_count.max()*0.1,0)]

    return passes_between_df, average_locs_and_count_df

def pass_network_visualization(ax, passes_between_df, average_locs_and_count_df, marker_label, flipped=False) -> Pitch:
    """
    Visualizes the passing network on a soccer pitch.

    Parameters:
    ax (mpl.axes.Axes): Matplotlib axes object to draw the pitch.
    passes_between_df (DataFrame): DataFrame containing passes between players.
    average_locs_and_count_df (DataFrame): DataFrame containing average locations and pass counts per player.
    marker_label (str): Specifies the type of marker label to be displayed. 
        Options are "Initials" or "Numbers".
    flipped (bool, optional): Specifies whether to flip the coordinates for the away team. Defaults to False.

    Returns:
    Pitch: mplsoccer Pitch object representing the soccer pitch.
    """
    MAX_LINE_WIDTH = 20
    MAX_MARKER_SIZE = 3000
    #Line width depends on the number of passes between players
    passes_between_df['width'] = (passes_between_df.pass_count*1.2 / passes_between_df.pass_count.max() *
                                  MAX_LINE_WIDTH)
    average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['passes_completed']
                                                / average_locs_and_count_df['passes_completed'].max() * MAX_MARKER_SIZE)
    
    # Only player that have made more than 5 passes during the match
    passes_between_df = passes_between_df[passes_between_df.pass_count>5]

    MIN_TRANSPARENCY = 0.3
    color = np.array(to_rgba('#507293'))
    color = np.tile(color, (len(passes_between_df), 1))
    c_transparency = passes_between_df.pass_count / passes_between_df.pass_count.max()
    c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency


    pitch = Pitch(pitch_type='opta', pitch_color='#0D182E', line_color='#5B6378')
    pitch.draw(ax=ax)

    # Printing the away team with flipped coordinates
    if flipped:
        passes_between_df.loc[:,'x'] = pitch.dim.right - passes_between_df.loc[:,'x']
        passes_between_df.loc[:,'y'] = pitch.dim.right - passes_between_df.loc[:,'y']
        passes_between_df.loc[:,'x_end'] = pitch.dim.right - passes_between_df.loc[:,'x_end']
        passes_between_df.loc[:,'y_end'] = pitch.dim.right - passes_between_df.loc[:,'y_end']
        average_locs_and_count_df['x'] = pitch.dim.right - average_locs_and_count_df['x']
        average_locs_and_count_df['y'] = pitch.dim.right - average_locs_and_count_df['y']

    # Setting the marker color conditions
    conditions_node = [
        (average_locs_and_count_df["percentage_completed"] <= 60),
        ((60 < average_locs_and_count_df["percentage_completed"]) & (average_locs_and_count_df["percentage_completed"] <= 70)),
        ((70 < average_locs_and_count_df["percentage_completed"]) & (average_locs_and_count_df["percentage_completed"] <= 80)),
        ((80 < average_locs_and_count_df["percentage_completed"]) & (average_locs_and_count_df["percentage_completed"] <= 85))
    ]
    conditions_markeredge = [
        (average_locs_and_count_df["subbed_in"] == True),
        ((average_locs_and_count_df["subbed_out"] == True))
    ]

    choices_node = ['#A61608', '#E57E03', '#E5DE05', '#63A51D']
    choices_markeredge = ['#018B22', '#A02C04']

    average_locs_and_count_df['color_node'] = np.select(conditions_node, choices_node, default='#58802E')
    average_locs_and_count_df['color_markeredge'] = np.select(conditions_markeredge, choices_markeredge, default='#FEFEFC')


    # Printing the lines and nodes
    pass_lines = pitch.lines(passes_between_df.x, passes_between_df.y,
                         passes_between_df.x_end, passes_between_df.y_end, lw=passes_between_df.width, 
                         color=color, zorder=1, ax=ax)
    pass_nodes = pitch.scatter(average_locs_and_count_df.x, average_locs_and_count_df.y,
                               s=average_locs_and_count_df.marker_size, marker='h',
                               c=average_locs_and_count_df.color_node, edgecolors=average_locs_and_count_df.color_markeredge, linewidth=3, alpha=0.1, ax=ax)
    

    # Setting up the marker label
    if marker_label == "Initials":
        for index, row in average_locs_and_count_df.iterrows():
            player_name = row["player_name"].split()
            player_initials = "".join(word[0] for word in player_name).upper()
            pitch.annotate(player_initials, xy=(row.x, row.y), c='#FEFEFC', va='center',
                           ha='center', size=14, ax=ax)
    
    elif marker_label == "Numbers":
        for index, row in average_locs_and_count_df.iterrows():
            player_name = row["shirt_no"]
            pitch.annotate(player_name, xy=(row.x, row.y), c='#FEFEFC', va='center',
                           ha='center', size=14, ax=ax)

    return pitch

def plot_pitch(df,match_data,marker_label="Numbers",team ="both"):
    """
    Plots the passing networks and top combinations by volume of passes for one or both teams.

    Parameters:
    df (DataFrame): DataFrame containing event data.
    match_data (dict): Dictionary containing match data including team information.
    marker_label (str, optional): Specifies the type of marker label to be displayed. Options are "Initials" or "Numbers". Defaults to "Numbers".
    team (str, optional): Specifies the team(s) for which passes are to be plotted. Options are "home", "away", or "both". Defaults to "both".
    """

    def plot_single_team(team_name, passes_df, marker_label):
        """
        Plots passing network for a single team.

        Parameters:
        team_name (str): Name of the team.
        passes_df (DataFrame): DataFrame containing passes event data for the team.
        marker_label (str): Specifies the type of marker label to be displayed.
        """

        pitch = Pitch(pitch_type='opta', pitch_color='#0D182E', line_color='#5B6378')
        fig, axs = pitch.grid(nrows=1, ncols=1, figheight=8, grid_width=0.9, endnote_space=0.05, axis=False, title_space=0.01)
        plt.tight_layout()
        fig.set_facecolor("#0D182E")

        main_color = '#FBFAF5'
        date = match_data['timeStamp'][0:10]

        axs['title'].text(0.5, 0.8, f"{match_data['home']['name']} vs {match_data['away']['name']}", ha='center', va='center', color=main_color, fontsize=22)
        axs['title'].text(0.5, 0.55, f"{date}", ha='center', va='center', color=main_color, fontsize=12)
        axs['title'].text(0.5, 0.35, "Passing networks and top combinations by volume of passes", ha='center', va='center', color=main_color, fontsize=12)
        axs['endnote'].text(1, 1, '@Joel_AS', va='center', ha='right', color=main_color, fontsize=12)
        axs['endnote'].text(0, 0.5, 'Min. 10% of total team passes\nMarker size: Amount of passes\nMarker edge color: Red - subbed out, Green - subbed in\nLines width: Amount of passes between players. The bigger the more passes.', color=main_color, fontsize=12)

        legend_labels = ['<= 60%', '60% - 70%', '70% - 80%', '80% - 85%', '+ 85%']
        legend_colors = ['#A61608', '#E57E03', '#E5DE05', '#63A51D', '#58802E']
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=14)
                           for label, color in zip(legend_labels, legend_colors)]
        axs['pitch'].legend(handles=legend_elements, loc='upper right', title="Pass completion %")

        passes_between_df, average_locs_and_count_df = get_passes_between_df(passes_df)
        comp = passes_df[passes_df.outcome_type_display_name == "Successful"]['id'].count()
        att = passes_df['id'].count()
        per = "{:.2f}%".format(comp / att * 100)

        pass_network_visualization(axs['pitch'], passes_between_df, average_locs_and_count_df, marker_label)
        axs['pitch'].set_title(f"{team_name} ({comp}/{att} ({per}))", color=main_color, fontsize=20)
        plt.show()

    if team == "home":
        plot_single_team(match_data['home']['name'], get_passes_df(df, team, match_data), marker_label=marker_label)

    elif team == "away":
        plot_single_team(match_data['away']['name'], get_passes_df(df, team, match_data), marker_label=marker_label)

    elif team == "both":

        pitch = Pitch(pitch_type='opta', pitch_color='#0D182E', line_color='#5B6378')
        # create plot
        fig, axs = pitch.grid(nrows=1, ncols=2, figheight=8, grid_width=0.9, endnote_space=0.05, axis=False, title_space=0.01)
        plt.tight_layout()
        fig.set_facecolor("#0D182E")
        # plot variables
        main_color = '#FBFAF5'

        date = match_data['timeStamp'][0:10]

        axs['title'].text(0.5,0.8,f"{match_data['home']['name']} vs {match_data['away']['name']}", ha='center', va='center', color=main_color, fontsize=22)
        axs['title'].text(0.5,0.55,f"{date}", ha='center', va='center', color=main_color, fontsize=12)
        axs['title'].text(0.5,0.35,f"Passing networks and top combinations by volume of passes", ha='center', va='center', color=main_color, fontsize=12)
        axs['endnote'].text(1, 1, '@Joel_AS', va='center', ha='right', color=main_color ,fontsize=12)
        axs['endnote'].text(0, 0.5, 'Min. 10% of total team passes\nMarker size: Amount of passes\nMarker edge color: Red - subbed out, Green - subbed in\nLines width: Amount of passes between players. The bigger the more passes.', color=main_color,fontsize=12)

        # Create legend
        legend_labels = ['<= 60%', '60% - 70%', '70% - 80%', '80% - 85%', '+ 85%']
        legend_colors = ['#A61608', '#E57E03', '#E5DE05', '#63A51D', '#58802E']
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=14)
                           for label, color in zip(legend_labels, legend_colors)]
        axs['pitch'][0].legend(handles=legend_elements, loc='upper right', title="Pass completion %")
        

        #home
        df_passes_home = get_passes_df(df, 'home', match_data)
        home_passes_between_df, home_average_locs_and_count_df = get_passes_between_df(df_passes_home)    
        home_comp = df_passes_home[df_passes_home.outcome_type_display_name=="Successful"]['id'].count()
        home_att = df_passes_home['id'].count()
        per_home = "{:.2f}%".format(home_comp/home_att*100)
        # home team viz
        pass_network_visualization(axs['pitch'][0], home_passes_between_df, home_average_locs_and_count_df, marker_label)
        axs['pitch'][0].set_title(f"{match_data['home']['name']} ({home_comp}/{home_att} ({per_home}))", color=main_color, fontsize=20)

        #away
        df_passes_away = get_passes_df(df, 'away', match_data)
        away_passes_between_df, away_average_locs_and_count_df = get_passes_between_df(df_passes_away)
        away_comp = df_passes_away[df_passes_away.outcome_type_display_name=="Successful"]['id'].count()
        away_att = df_passes_away['id'].count()
        per_away = "{:.2f}%".format(away_comp/away_att*100)
        # home team viz
        pass_network_visualization(axs['pitch'][1], away_passes_between_df, away_average_locs_and_count_df, marker_label,flipped=True)
        axs['pitch'][1].set_title(f"{match_data['away']['name']} ({away_comp}/{away_att} ({per_away}))", color=main_color, fontsize=20)
        plt.show()