"""Scraper for https://fotmob.com/"""

from ast import Pass, Try
from urllib import response
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
from tqdm import tqdm
import re
import json

match_url = "https://www.fotmob.com/api/matchDetails?matchId={}"
league_url = "https://www.fotmob.com/api/leagues?id={}"
alt_league_url = "https://www.fotmob.com/_next/data/{}{}.json"
page_url = "https://www.fotmob.com/api"
leagues_url = "https://www.fotmob.com/api/allLeagues"


def get_league_url():
    response = requests.get(leagues_url).json()
    df_leagues = pd.concat([
    pd.json_normalize(response['countries'], record_path=['leagues']),
    pd.json_normalize(response['international'], record_path=['leagues'])
    ], 
    ignore_index=True
    )
    return pd.Series(df_leagues.name.values, index=df_leagues.id).to_dict()


dict_league_name = get_league_url()
api_delay = 1.0


def get_build_id():
    return re.search(r"\"buildId\"\:\"(\d+)\"", requests.get(page_url).text).group(1)


def enforce_delay():
    time.sleep(api_delay)


def get_single_match_data(match_id: int):
    enforce_delay()
    response = requests.get(match_url.format(match_id)).json()

    return response


def get_single_match_stats(match_id: int):
    response = get_single_match_data(match_id)

    df = pd.json_normalize(response["content"], ["stats", "stats", "stats"])
    df = df.T.reset_index(drop=True)
    df.columns = df.iloc[1]
    df = df[2:3]
    df = df.apply(pd.Series.explode)
    df = df.iloc[:2].reset_index(drop=True)
    df["match_id"] = match_id

    df.loc[::2, "team"] = response["content"]["liveticker"]["teams"][0]
    df.loc[1::2, "team"] = response["content"]["liveticker"]["teams"][1]
    df = df.loc[:, df.notna().any(axis=0)]
    df = df.loc[:, ~df.columns.duplicated()]

    df_in_rd = pd.json_normalize(response["content"]["matchFacts"], ["matchesInRound"])
    df_teams = pd.DataFrame()
    df_teams = pd.concat([df_teams,
        df_in_rd[["home.name", "home.shortName"]].rename(
            columns={"home.name": "team_name", "home.shortName": "team_short_name"}
        )]
    )
    df_teams = pd.concat([df_teams,
        df_in_rd[["away.name", "away.shortName"]].rename(
            columns={"away.name": "team_name", "away.shortName": "team_short_name"}
        )]
    )

    teams_dict = pd.Series(
        df_teams["team_short_name"].values, index=df_teams["team_name"]
    ).to_dict()
    df["team_short"] = df["team"].apply(lambda x: teams_dict.get(x))

    df["goals"] = pd.json_normalize(response["header"]["teams"])["score"]
    df["goals_against"] = df.loc[::-1, "goals"].reset_index(drop=True)

    df["match_round"] = response["general"]["matchRound"]
    df["match_round"] = df["match_round"].astype(int)

    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(" ", "_")

    df["pass_accuracy"] = (
        df["accurate_passes"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df["accurate_passes"] = df["accurate_passes"].str.split(" ").str.get(0)

    df["longball_accuracy"] = (
        df["long_balls_accurate"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df["long_balls_accurate"] = df["long_balls_accurate"].str.split(" ").str.get(0)

    df["cross_accuracy"] = (
        df["accurate_crosses"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df["accurate_crosses"] = df["accurate_crosses"].str.split(" ").str.get(0)

    df["tackles_succeeded_percentage"] = (
        df["tackles_succeeded"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df["tackles_succeeded"] = df["tackles_succeeded"].str.split(" ").str.get(0)

    df["ground_duels_won_percentage"] = (
        df["ground_duels_won"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df["ground_duels_won"] = df["ground_duels_won"].str.split(" ").str.get(0)

    df["aerial_duels_won_percentage"] = (
        df["aerials_won"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df["aerials_won"] = df["aerials_won"].str.split(" ").str.get(0)

    df["succeeded_dribbles_percentage"] = (
        df["dribbles_succeeded"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df["dribbles_succeeded"] = df["dribbles_succeeded"].str.split(" ").str.get(0)

    df.fillna(0, inplace=True)
    df = df.convert_dtypes()
    for column in df.columns[~df.columns.isin(['match_id', 'team', 'team_short'])]:
        if df[column].dtype == 'string':
            df[column] = df[column].astype(np.float32)

    return df


def get_league_fixtures(league_id: int):
    enforce_delay()
    df_fixtures = get_league_schedule(league_id)
    return list(df_fixtures.loc[(df_fixtures["status.finished"] == True) & (df_fixtures["status.started"] == True), "id"].astype(int))

def get_league_schedule(league_id: int):
    enforce_delay()
    league_name = dict_league_name[league_id]
    print("getting fixtures for {}".format(league_name))
    response = requests.get(league_url.format(league_id)).json()
    try:
        df_fixtures = pd.json_normalize(response['matches']['data']['matches'])
    except KeyError:
        df_fixtures = pd.json_normalize(response['matches']['allMatches'])

    return df_fixtures


def get_league_match_stats(league_id: int):
    enforce_delay()
    fixtures = get_league_fixtures(league_id)
    df_league_stats = pd.DataFrame()
    for l in tqdm(fixtures):
        df_league_stats = pd.concat([df_league_stats,get_single_match_stats(l)], axis=0).reset_index(
            drop=True
        )
    return df_league_stats


def get_league_team_stats(league_id: int):
    enforce_delay()
    df = get_league_match_stats(league_id)
    df = df.drop("match_id", axis=1)
    df = df.drop("match_round", axis=1)

    df["goals_against_over_expected"] = df["goals_against"].subtract(
        df["expected_goals_against_(xga)"]
    )
    df["goals_over_expected"] = df["goals"].subtract(df["expected_goals_(xg)"])

    df1 = df.groupby(["team", "team_short"]).mean().reset_index()
    df1.columns = [col + "_mean" for col in df1.columns]
    df1 = df1.rename(columns={"team_mean": "team", "team_short_mean": "team_short"})

    df2 = df.groupby(["team", "team_short"]).sum().reset_index()
    df2.columns = [col + "_sum" for col in df2.columns]
    df2 = df2.rename(columns={"team_sum": "team", "team_short_sum": "team_short"})
    df = df1.merge(df2, on=["team", "team_short"], how="left")

    return df


def get_single_match_shots(match_id: int):
    enforce_delay()
    response = get_single_match_data(match_id)

    df_shots = pd.json_normalize(response["content"]["shotmap"]["shots"])

    df_shots["match_id"] = match_id

    df_shots = df_shots.rename(
        columns={
            "id": "shot_id",
            "eventType": "event_type",
            "teamId": "team_id",
            "playerId": "player_id",
            "playerName": "player_name",
            "x": "x_coord",
            "y": "y_coord",
            "min": "minutes",
            "minAdded": "minutes_added",
            "isBlocked": "is_blocked",
            "isOnTarget": "is_on_target",
            "blockedX": "blocked_x_coord",
            "blockedY": "blocked_y_coord",
            "goalCrossedY": "goal_crossed_y_coord",
            "goalCrossedZ": "goal_crossed_z_coord",
            "expectedGoals": "expected_goals",
            "expectedGoalsOnTarget": "expected_goals_on_target",
            "shotType": "shot_type",
            "isOwnGoal": "is_own_goal",
            "firstName": "first_name",
            "lastName": "last_name",
            "onGoalShot.x": "on_goal_shot_x_coord",
            "onGoalShot.y": "on_goal_shot_y_coord",
            "onGoalShot.zoomRatio": "on_goal_shot_zoom_ratio",
        }
    )

    df_in_rd = pd.json_normalize(response["content"]["matchFacts"], ["matchesInRound"])

    df_teams = pd.DataFrame()
    df_teams = pd.concat([df_teams,
        df_in_rd[["home.name", "home.shortName", "home.id"]].rename(
            columns={
                "home.name": "team_name",
                "home.shortName": "team_short_name",
                "home.id": "team_id",
            }
        )], axis=0
    )
    df_teams = pd.concat([df_teams,
        df_in_rd[["away.name", "away.shortName", "away.id"]].rename(
            columns={
                "away.name": "team_name",
                "away.shortName": "team_short_name",
                "away.id": "team_id",
            }
        )], axis=0
    )

    team_id_dict = pd.Series(
        df_teams["team_name"].values, index=df_teams["team_id"].astype(int)
    ).to_dict()

    df_shots["team"] = df_shots["team_id"].apply(lambda x: team_id_dict.get(x))

    teams_dict = pd.Series(
        df_teams["team_short_name"].values, index=df_teams["team_name"]
    ).to_dict()
    df_shots["team_short"] = df_shots["team"].apply(lambda x: teams_dict.get(x))

    df_shots["minutes_added"].fillna(0, inplace=True)

    team_opponent_id_dict = pd.concat(
        [
            pd.Series(df_in_rd["home.id"].values.astype(int), index=df_in_rd["away.id"].astype(int)),
            pd.Series(df_in_rd["away.id"].values.astype(int), index=df_in_rd["home.id"].astype(int))
        ]
    ).to_dict()

    df_shots["opponent_id"] = df_shots["team_id"].apply(lambda x: team_opponent_id_dict.get(x))
    df_shots["opponent_team"] = df_shots["opponent_id"].apply(lambda x: team_id_dict.get(x))
    
    # get the keeper facing the shot
    df_lineup = get_single_match_lineup(match_id)
    df_lineup_gk = df_lineup[df_lineup.role == 'Keeper']
    df_shots['keeper_shot_faced'] = df_shots[['opponent_team','minutes']].apply(lambda x: df_lineup_gk.loc[(df_lineup_gk['team_name'] == x['opponent_team']) & (df_lineup_gk['time_subbed_on'] < x['minutes']) & ((df_lineup_gk['time_subbed_off'] >= x['minutes']) | (df_lineup_gk['time_subbed_off'].isnull())), 'player_last_name'].values[0], axis=1)
    
    return df_shots


def get_league_shots(league_id: int):
    enforce_delay()
    fixtures = get_league_fixtures(league_id)
    df_league_shots = pd.DataFrame()
    for l in tqdm(fixtures):
        df_league_shots = pd.concat([df_league_shots,get_single_match_shots(l)]).reset_index(drop=True)

    return df_league_shots

def get_missing_league_shots(league_id: int, df_league_shots: pd.DataFrame):
    enforce_delay()
    fixtures = get_league_fixtures(league_id)
    missing_games = [x for x in fixtures if x not in df_league_shots.match_id.unique()]
    for l in tqdm(missing_games):
        df_league_shots = pd.concat([df_league_shots, get_single_match_shots(l)], axis=0).reset_index(drop=True)

    return df_league_shots

def get_missing_league_match_stats(league_id: int, df_league_stats: pd.DataFrame):
    enforce_delay()
    fixtures = get_league_fixtures(league_id)
    missing_games = [x for x in fixtures if x not in df_league_stats.match_id.unique()]
    for l in tqdm(missing_games):
        df_league_stats = pd.concat([df_league_stats,get_single_match_stats(l)], axis=0).reset_index(
            drop=True
        )
    return df_league_stats

def get_missing_league_team_stats(league_id: int, df_match_stats: pd.DataFrame):
    enforce_delay()
    df = get_missing_league_match_stats(league_id, df_match_stats)
    df = df.drop("match_id", axis=1)
    df = df.drop("match_round", axis=1)

    df["goals_against_over_expected"] = df["goals_against"].subtract(
        df["expected_goals_against_(xga)"]
    )
    df["goals_over_expected"] = df["goals"].subtract(df["expected_goals_(xg)"])

    df1 = df.groupby(["team", "team_short"]).mean().reset_index()
    df1.columns = [col + "_mean" for col in df1.columns]
    df1 = df1.rename(columns={"team_mean": "team", "team_short_mean": "team_short"})

    df2 = df.groupby(["team", "team_short"]).sum().reset_index()
    df2.columns = [col + "_sum" for col in df2.columns]
    df2 = df2.rename(columns={"team_sum": "team", "team_short_sum": "team_short"})
    df = df1.merge(df2, on=["team", "team_short"], how="left")

    return df


def get_single_match_lineup(match_id: int):
    enforce_delay()
    response = get_single_match_data(match_id)
    df_lineup = pd.DataFrame()
    team_infos = pd.json_normalize(response["content"]["lineup"]["lineup"])[['teamId','teamName']].to_dict()
    lineup_team_0 = pd.DataFrame()
    for i in range(len(response["content"]["lineup"]["lineup"][0]['players'])):
        lineup_team_0 = pd.concat([lineup_team_0, pd.json_normalize(response["content"]["lineup"]["lineup"][0]['players'][i])])
    lineup_team_0['timeSubbedOn'] = lineup_team_0['timeSubbedOn'].fillna(0)
    bench_team_0 = pd.json_normalize(response["content"]["lineup"]["lineup"][0]['bench'])
    lineup_team_0 = pd.concat([lineup_team_0,bench_team_0])
    lineup_team_0['team_name'] = team_infos['teamName'][0]
    lineup_team_0['team_id'] = team_infos['teamId'][0]
    df_lineup = pd.concat([df_lineup,lineup_team_0])
    lineup_team_1 = pd.DataFrame()
    for i in range(len(response["content"]["lineup"]["lineup"][1]['players'])):
        lineup_team_1 = pd.concat([lineup_team_1,pd.json_normalize(response["content"]["lineup"]["lineup"][1]['players'][i])])
    lineup_team_1['timeSubbedOn'] = lineup_team_1['timeSubbedOn'].fillna(0)
    bench_team_1 = pd.json_normalize(response["content"]["lineup"]["lineup"][1]['bench'])
    lineup_team_1 = pd.concat([lineup_team_1,bench_team_1])
    lineup_team_1['team_name'] = team_infos['teamName'][1]
    lineup_team_1['team_id'] = team_infos['teamId'][1]
    df_lineup = pd.concat([df_lineup,lineup_team_1])
    df_lineup = df_lineup[['id', 'imageUrl', 'pageUrl', 'shirt', 'isHomeTeam', 'timeSubbedOn',
    'timeSubbedOff', 'role', 'minutesPlayed', 'positionStringShort',
    'name.firstName', 'name.lastName', 'team_name', 'team_id']]
    df_lineup.columns = df_lineup.columns.str.lower()
    df_lineup.columns = df_lineup.columns.str.replace(" ", "_")
    df_lineup.rename(columns={
    'id':'player_id',
    'imageurl':'player_image',
    'pageurl':'player_page',
    'shirt':'player_shirt_number',
    'ishometeam': 'is_home_team',
    'timesubbedon' : 'time_subbed_on',
    'timesubbedoff': 'time_subbed_off',
    'minutesplayed': 'minutes_played', 
    'positionstringshort': 'position_short',
    'name.firstname': 'player_first_name', 
        'name.lastname': 'player_last_name'
        }, inplace=True)

    return df_lineup


def get_single_match_player_stats(match_id: int):
    enforce_delay()
    response = get_single_match_data(match_id)
    df_lineup = pd.DataFrame()
    team_infos = pd.json_normalize(response["content"]["lineup"]["lineup"])[['teamId','teamName']].to_dict()

    lineup_team_0 = pd.DataFrame()
    for i in range(len(response["content"]["lineup"]["lineup"][0]['players'])):
        lineup_team_0 = pd.concat([lineup_team_0, pd.json_normalize(response["content"]["lineup"]["lineup"][0]['players'][i])])

    lineup_team_0['timeSubbedOn'] = lineup_team_0['timeSubbedOn'].fillna(0)
    bench_team_0 = pd.json_normalize(response["content"]["lineup"]["lineup"][0]['bench'])
    lineup_team_0 = pd.concat([lineup_team_0,bench_team_0])
    lineup_team_0['team_name'] = team_infos['teamName'][0]
    lineup_team_0['team_id'] = team_infos['teamId'][0]
    df_lineup = pd.concat([df_lineup,lineup_team_0])

    lineup_team_1 = pd.DataFrame()
    for i in range(len(response["content"]["lineup"]["lineup"][1]['players'])):
        lineup_team_1 = pd.concat([lineup_team_1,pd.json_normalize(response["content"]["lineup"]["lineup"][1]['players'][i])])
    lineup_team_1['timeSubbedOn'] = lineup_team_1['timeSubbedOn'].fillna(0)
    bench_team_1 = pd.json_normalize(response["content"]["lineup"]["lineup"][1]['bench'])
    lineup_team_1 = pd.concat([lineup_team_1,bench_team_1])
    lineup_team_1['team_name'] = team_infos['teamName'][1]
    lineup_team_1['team_id'] = team_infos['teamId'][1]
    df_lineup = pd.concat([df_lineup,lineup_team_1])
    df_lineup.columns = df_lineup.columns.str.lower()
    df_lineup.columns = df_lineup.columns.str.replace(" ", "_")
    df_lineup.rename(columns={
        'id':'player_id',
        'imageurl':'player_image',
        'pageurl':'player_page',
        'shirt':'player_shirt_number',
        'ishometeam': 'is_home_team',
        'timesubbedon' : 'time_subbed_on',
        'timesubbedoff': 'time_subbed_off',
        'minutesplayed': 'minutes_played', 
        'positionstringshort': 'position_short',
        'name.firstname': 'player_first_name', 
        'name.lastname': 'player_last_name'
        }, inplace=True)

    df_player_stats = pd.DataFrame()
    for i in range(len(df_lineup)):
        df_temp = pd.json_normalize(df_lineup.stats.iloc[i]).drop(columns=['title'])
        df_temp['player_id'] = df_lineup.player_id.iloc[i]
        df_temp = df_temp.groupby('player_id').bfill().head(1)
        df_temp['player_id'] = df_lineup.player_id.iloc[i]
        df_player_stats = pd.concat([df_player_stats, df_temp], ignore_index=True)

    df_lineup = df_lineup.merge(df_player_stats, on='player_id', how='left')
    df_lineup = df_lineup[[
        'player_id', 
        'player_shirt_number', 
        'is_home_team', 
        'time_subbed_on',
        'time_subbed_off', 
        'usualposition', 
        'role',
        'minutes_played', 
        'position_short',
        'player_first_name', 
        'player_last_name', 
        'team_name', 
        'team_id', 
        'stats.Saves', 
        'stats.Goals conceded',
        'stats.xGOT faced', 
        'stats.Accurate passes',
        'stats.Accurate long balls', 
        'stats.Diving save',
        'stats.Saves inside box', 
        'stats.Punches',
        'stats.Throws', 
        'stats.High claim', 
        'stats.Recoveries', 
        'stats.Touches',
        'stats.Goals', 
        'stats.Assists', 
        'stats.Total shots',
        'stats.Chances created', 
        'stats.Expected assists (xA)',
        'stats.Successful dribbles', 
        'stats.Accurate crosses', 
        'stats.Dispossessed', 
        'stats.Tackles won',
        'stats.Blocks', 
        'stats.Clearances', 
        'stats.Headed clearance',
        'stats.Interceptions', 
        'stats.Dribbled past', 
        'stats.Ground duels won',
        'stats.Aerial duels won', 
        'stats.Was fouled', 
        'stats.Fouls committed',
        'stats.Expected goals (xG)', 
        'stats.Shot accuracy',
        'stats.Expected goals on target (xGOT)', 
        'stats.Blocked shots', 
        'stats.Corners']]
    
    df_lineup = df_lineup.rename(columns={
        'stats.Saves': 'saves', 
        'stats.Goals conceded': 'goals_conceded',
        'stats.xGOT faced': 'xgot_faced', 
        'stats.Accurate passes': 'accurate_passes',
        'stats.Accurate long balls': 'accurate_long_balls',
        'stats.Diving save': 'diving_save',
        'stats.Saves inside box': 'saves_inside_box',
        'stats.Punches': 'punches',
        'stats.Throws': 'throws', 
        'stats.High claim': 'high_claim', 
        'stats.Recoveries': 'recoveries', 
        'stats.Touches': 'touches',
        'stats.Goals': 'goals', 
        'stats.Assists': 'assists', 
        'stats.Total shots': 'total_shots',
        'stats.Chances created': 'chances_created', 
        'stats.Expected assists (xA)': 'expected_assists_(xa)',
        'stats.Successful dribbles': 'successful_dribbles', 
        'stats.Accurate crosses': 'accurate_crosses', 
        'stats.Dispossessed': 'dispossessed', 
        'stats.Tackles won': 'tackles_won',
        'stats.Blocks': 'blocks', 
        'stats.Clearances': 'clearances', 
        'stats.Headed clearance': 'headed_clearance',
        'stats.Interceptions': 'interceptions', 
        'stats.Dribbled past': 'dribbled_past', 
        'stats.Ground duels won': 'ground_duels_won',
        'stats.Aerial duels won': 'aerial_duels_won', 
        'stats.Was fouled': 'was_fouled', 
        'stats.Fouls committed': 'fouls_committed',
        'stats.Expected goals (xG)': 'expected_goals_(xg)', 
        'stats.Shot accuracy': 'shot_accuracy',
        'stats.Expected goals on target (xGOT)': 'expected_goals_on_target_(xgot)', 
        'stats.Blocked shots': 'blocked_shots', 
        'stats.Corners': 'corners',
    })

    df_lineup["accurate_crosses_percentage"] = (
        df_lineup["accurate_crosses"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df_lineup["accurate_crosses"], df_lineup["attempted_crosses"] = df_lineup["accurate_crosses"].str.split(" ").str.get(0).str.split("/").str.get(0), df_lineup["accurate_crosses"].str.split(" ").str.get(0).str.split("/").str.get(1)

    df_lineup["accurate_passes_percentage"] = (
        df_lineup["accurate_passes"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df_lineup["accurate_passes"], df_lineup["attempted_passes"] = df_lineup["accurate_passes"].str.split(" ").str.get(0).str.split("/").str.get(0), df_lineup["accurate_passes"].str.split(" ").str.get(0).str.split("/").str.get(1)

    df_lineup["accurate_long_balls_percentage"] = (
        df_lineup["accurate_long_balls"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df_lineup["accurate_long_balls"], df_lineup["attempted_long_balls"] = df_lineup["accurate_long_balls"].str.split(" ").str.get(0).str.split("/").str.get(0), df_lineup["accurate_long_balls"].str.split(" ").str.get(0).str.split("/").str.get(1)

    df_lineup["successful_dribbles_percentage"] = (
        df_lineup["successful_dribbles"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df_lineup["successful_dribbles"], df_lineup["attempted_dribbles"] = df_lineup["successful_dribbles"].str.split(" ").str.get(0).str.split("/").str.get(0), df_lineup["successful_dribbles"].str.split(" ").str.get(0).str.split("/").str.get(1)

    df_lineup["tackles_won_percentage"] = (
        df_lineup["tackles_won"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df_lineup["tackles_won"], df_lineup["attempted_tackles"] = df_lineup["tackles_won"].str.split(" ").str.get(0).str.split("/").str.get(0), df_lineup["tackles_won"].str.split(" ").str.get(0).str.split("/").str.get(1)
    
    df_lineup["ground_duels_won_percentage"] = (
        df_lineup["ground_duels_won"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df_lineup["ground_duels_won"], df_lineup["attempted_ground_duels"] = df_lineup["ground_duels_won"].str.split(" ").str.get(0).str.split("/").str.get(0), df_lineup["ground_duels_won"].str.split(" ").str.get(0).str.split("/").str.get(1)
    
    df_lineup["aerial_duels_won_percentage"] = (
        df_lineup["aerial_duels_won"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df_lineup["aerial_duels_won"], df_lineup["attempted_aerial_duels"] = df_lineup["aerial_duels_won"].str.split(" ").str.get(0).str.split("/").str.get(0), df_lineup["aerial_duels_won"].str.split(" ").str.get(0).str.split("/").str.get(1)
    
    df_lineup["accurate_shots_percentage"] = (
        df_lineup["shot_accuracy"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df_lineup["accurate_shots"], df_lineup["attempted_shots"] = df_lineup["shot_accuracy"].str.split(" ").str.get(0).str.split("/").str.get(0), df_lineup["shot_accuracy"].str.split(" ").str.get(0).str.split("/").str.get(1)
    df_lineup.drop(columns=["shot_accuracy"], inplace=True)

    df_lineup = df_lineup.fillna(0)

    return df_lineup

def get_league_player_stats(league_id: int):
    enforce_delay()
    df_schedule = get_league_schedule(league_id)
    df_schedule = df_schedule[(df_schedule["status.finished"] == True) & (df_schedule["status.started"] == True)]
    df_league_player_stats = pd.DataFrame()
    for rd, id in tqdm(df_schedule[['round', 'id']].values):
        df_single_match_player_stats = get_single_match_player_stats(id)
        df_single_match_player_stats["match_id"] = id
        df_single_match_player_stats["round"] = rd
        df_league_player_stats = pd.concat([df_league_player_stats,df_single_match_player_stats]).reset_index(drop=True)

    return df_league_player_stats

def get_missing_league_player_stats(league_id: int, df_league_player_stats: pd.DataFrame):
    enforce_delay()
    df_schedule = get_league_schedule(league_id)
    missing_games = [x for x in df_schedule.id.values if x not in df_league_player_stats.match_id.unique()]
    df_schedule = df_schedule[(df_schedule["status.finished"] == True) & (df_schedule["status.started"] == True)]
    df_schedule = df_schedule[df_schedule.id.isin(missing_games)]
    df_league_player_stats = pd.DataFrame()
    for rd, id in tqdm(df_schedule[['round', 'id']].values):
        df_single_match_player_stats = get_single_match_player_stats(id)
        df_single_match_player_stats["match_id"] = id
        df_single_match_player_stats["round"] = rd
        df_league_player_stats = pd.concat([df_league_player_stats,df_single_match_player_stats]).reset_index(drop=True)

    return df_league_player_stats