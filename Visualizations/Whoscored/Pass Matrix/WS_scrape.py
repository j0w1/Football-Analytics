#Imports
import json
import time

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from selenium import webdriver

from supabase import create_client, Client

import re

#Functions

def convert_to_snake_case(column_name: str) -> str:
    """
    Convert a column name to snake_case.

    Parameters:
    - column_name (str): The column name to be converted.

    Returns:
    str: The converted column name in snake_case.
    """
    result = [column_name[0].lower()]
    for char in column_name[1:]:
        result.extend(['_', char.lower()] if char.isupper() else [char])
    return ''.join(result)

def get_matchdata_keys(url: str) -> Dict[str, Any]:
    """
    Retrieve match data from a Whoscored URL.

    Parameters:
    - url (str): The Whoscored URL for the desired match.

    Returns:
    Tuple[Dict[str, Any], KeysView[str]]: A tuple containing the match data dictionary
    and a view of its keys.
    """

    # Use a context manager to ensure the webdriver is closed properly
    with webdriver.Chrome() as driver:
        driver.get(url)
        # Use BeautifulSoup to parse the page content
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # Locate the script containing matchCentreData
        element = soup.select_one('script:-soup-contains("matchCentreData")')
        # Extract and parse the relevant JSON data
        matchdict = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])
        #Get matchdict keys
        matchdict_keys = matchdict.keys()

    return matchdict,matchdict_keys

def get_data(url: str, key: str) -> pd.DataFrame:
    """
    Extract and preprocess match data based on the specified key.

    Parameters:
    - url (str): The Whoscored URL for the desired match.
    - key (str): The key specifying the type of data to extract.

    Returns:
    pd.DataFrame: A DataFrame containing the extracted and processed match data.
    """
    match_data, match_keys = get_matchdata_keys(url)
    
    df = pd.DataFrame(match_data[key])
    df = df.dropna(subset='playerId')
    df = df.where(pd.notnull(df), None)
    
    # Convert column names to snake_case
    df.columns = [convert_to_snake_case(col) for col in df.columns]
    
    # Extract additional information from nested dictionaries
    df['period_display_name'] = df['period'].apply(lambda x: x['displayName'])
    df['type_display_name'] = df['type'].apply(lambda x: x['displayName'])
    df['outcome_type_display_name'] = df['outcome_type'].apply(lambda x: x['displayName'])
    df.drop(columns=["period", "type", "outcome_type"], inplace=True)
    
    # Reorder columns
    column_order = ['id', 'event_id', 'minute', 'second', 'team_id', 'player_id', 'x', 'y', 'end_x', 'end_y',
                    'qualifiers', 'is_touch', 'blocked_x', 'blocked_y', 'goal_mouth_z', 'goal_mouth_y', 'is_shot',
                    'card_type', 'is_goal', 'type_display_name', 'outcome_type_display_name', 'period_display_name']
    df = df[column_order]
    
    # Convert data types
    int_columns = ['id', 'event_id', 'minute', 'team_id', 'player_id']
    float_columns = ['second', 'x', 'y', 'end_x', 'end_y']
    bool_columns = ['is_shot', 'is_goal', 'card_type']
    
    df[int_columns] = df[int_columns].astype(np.int64)
    df[float_columns] = df[float_columns].astype(float)
    df[bool_columns] = df[bool_columns].fillna(False).astype(bool)
    
    # Replace NaN values in float columns with None
    for column in df.columns:
        if df[column].dtype == np.float64 or df[column].dtype == np.float32:
            df[column] = np.where(
                np.isnan(df[column]),
                None,
                df[column]
            )
    #Create match_id column
    timestamp = re.split('[-: ]', match_data['timeStamp'])
    teams = [str(i) for i in df.team_id.unique()]
    match_id = int(''.join(timestamp + teams))
    df['match_id'] = match_id
    #Create player_name column
    df_names = pd.DataFrame(list(match_data['playerIdNameDictionary'].items()), columns=['player_id', 'player_name'])
    df_names['player_id']=df_names['player_id'].astype(np.int64)
    df = df.merge(df_names, on='player_id', how ='left')

    #Create is_first_eleven & shirt_no columns
    def process_team(team_data, is_starter):
        player_data_list = []
        for index in range(len(team_data)):
            data_dict = team_data[index]
            player_info = {
                'player_id': data_dict['playerId'],
                'shirt_no': data_dict['shirtNo'],
                'name': data_dict['name'],
                'position': data_dict['position'],
                'is_first_eleven': data_dict['isFirstEleven'] if is_starter else False
            }
            player_data_list.append(player_info)
        return player_data_list
    # Home Team
    home_starters = process_team(list(match_data['home'].items())[7][1][:11], True)
    home_subs = process_team(list(match_data['home'].items())[7][1][11:], False)
    
    # Away Team
    away_starters = process_team(list(match_data['away'].items())[7][1][:11], True)
    away_subs = process_team(list(match_data['away'].items())[7][1][11:], False)
    
    # Combine all lists and create the df
    player_data_list = home_starters + home_subs + away_starters + away_subs
    df_all_players = pd.DataFrame(player_data_list)
    
    #Merge with original df
    df = df.merge(df_all_players[['player_id','shirt_no','is_first_eleven', 'position']], on='player_id', how ='left')
    
    return match_data, match_keys, df

def insert_match_events(df, supabase, table_name):
    """
    Insert match events data into a Supabase table.

    Parameters:
    - df (pd.DataFrame): DataFrame containing match events data.
    - supabase: Supabase client instance.
    - table_name (str): Name of the Supabase table to insert data into.
    """

    class MatchEvent(BaseModel):
        """
        Pydantic model representing a match event.
        Adjust attributes based on the structure of your data.
        """
        id: int
        event_id: int
        minute: int
        second: Optional[float] = None
        team_id: int
        player_id: int
        x: float
        y: float
        end_x: Optional[float] = None
        end_y: Optional[float] = None
        qualifiers: List[dict]
        is_touch: bool
        blocked_x: Optional[float] = None
        blocked_y: Optional[float] = None
        goal_mouth_z: Optional[float] = None
        goal_mouth_y: Optional[float] = None
        is_shot: bool
        card_type: bool
        is_goal: bool
        type_display_name: str
        outcome_type_display_name: str
        period_display_name: str
        match_id: int
        player_name: str
        shirt_no: int
        is_first_eleven: bool
        position: str

    # Convert DataFrame rows to a list of dictionaries using the MatchEvent model
    events = [
        MatchEvent(**x).model_dump()
        for x in df.to_dict(orient='records')
    ]

    # Perform an upsert operation to insert or update records in the Supabase table
    execution = supabase.table(table_name).upsert(events).execute()
