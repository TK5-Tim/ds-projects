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
    df.columns = df.iloc[0]
    df = df[1:]
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
        df["accurate_long_balls"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df["accurate_long_balls"] = df["accurate_long_balls"].str.split(" ").str.get(0)

    df["cross_accuracy"] = (
        df["accurate_crosses"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df["accurate_crosses"] = df["accurate_crosses"].str.split(" ").str.get(0)

    df["tackles_won_percentage"] = (
        df["tackles_won"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df["tackles_won"] = df["tackles_won"].str.split(" ").str.get(0)

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
        df["aerial_duels_won"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df["aerial_duels_won"] = df["aerial_duels_won"].str.split(" ").str.get(0)

    df["successful_dribbles_percentage"] = (
        df["successful_dribbles"]
        .str.split(" ")
        .str.get(1)
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
        .str.replace("%", "", regex=True)
    )
    df["successful_dribbles"] = df["successful_dribbles"].str.split(" ").str.get(0)

    df["expected_goals_against_(xga)"] = df.loc[
        ::-1, "expected_goals_(xg)"
    ].reset_index(drop=True)
    df["xga_first_half"] = df.loc[::-1, "xg_first_half"].reset_index(drop=True)
    df["xga_second_half"] = df.loc[::-1, "xg_second_half"].reset_index(drop=True)
    df["xga_open_play"] = df.loc[::-1, "xg_open_play"].reset_index(drop=True)
    try:
        df["xga_set_play"] = df.loc[::-1, "xg_set_play"].reset_index(drop=True)
    except KeyError:
        df["xga_set_play"] = np.nan
        df["xg_set_play"] = np.nan
    try:
        df["xga_on_target_(xgaot)"] = df.loc[::-1, "xg_on_target_(xgot)"].reset_index(
            drop=True
        )
    except KeyError:
        df["xga_on_target_(xgaot)"] = np.nan
        df["xg_on_target_(xgot)"] = np.nan
    try:
        df["xga_penalty"] = df.loc[::-1, "xg_penalty"].reset_index(drop=True)
    except KeyError:
        df["xg_penalty"] = np.nan
        df["xga_penalty"] = np.nan

    df.fillna(0, inplace=True)
    df = df.astype(
        {
            "ball_possession": "int32",
            "expected_goals_(xg)": "float64",
            "total_shots": "int32",
            "big_chances": "int32",
            "big_chances_missed": "int32",
            "accurate_passes": "int32",
            "fouls_committed": "int32",
            "offsides": "int32",
            "corners": "int32",
            "shots_off_target": "int32",
            "shots_on_target": "int32",
            "blocked_shots": "int32",
            "hit_woodwork": "int32",
            "shots_inside_box": "int32",
            "shots_outside_box": "int32",
            "xg_first_half": "float64",
            "xg_second_half": "float64",
            "xg_open_play": "float64",
            "xg_set_play": "float64",
            "xg_on_target_(xgot)": "float64",
            "passes": "int32",
            "own_half": "int32",
            "opposition_half": "int32",
            "accurate_long_balls": "int32",
            "accurate_crosses": "int32",
            "throws": "int32",
            "tackles_won": "int32",
            "interceptions": "int32",
            "blocks": "int32",
            "clearances": "int32",
            "keeper_saves": "int32",
            "duels_won": "int32",
            "ground_duels_won": "int32",
            "aerial_duels_won": "int32",
            "successful_dribbles": "int32",
            "yellow_cards": "int32",
            "red_cards": "int32",
            "match_id": "int32",
            "team": "string",
            "pass_accuracy": "int32",
            "longball_accuracy": "int32",
            "cross_accuracy": "int32",
            "tackles_won_percentage": "int32",
            "ground_duels_won_percentage": "int32",
            "aerial_duels_won_percentage": "int32",
            "successful_dribbles_percentage": "int32",
            "expected_goals_against_(xga)": "float64",
            "xga_first_half": "float64",
            "xga_second_half": "float64",
            "xga_open_play": "float64",
            "xga_set_play": "float64",
            "xga_on_target_(xgaot)": "float64",
            "xg_penalty": "float64",
            "xga_penalty": "float64",
        }
    )

    return df


def get_league_fixtures(league_id: int):
    enforce_delay()
    df_fixtures = get_league_schedule(league_id)
    return list(df_fixtures.loc[(df_fixtures["status.finished"] == True) & (df_fixtures["status.started"] == True), "id"].astype(int))

def get_league_schedule(league_id: int):
    enforce_delay()
    league_name = dict_league_name[league_id]
    print("getting fixtures for {}".format(league_name))
    try:
        response = requests.get(league_url.format(league_id)).json()
        df_fixtures = pd.json_normalize(response['matches']['data']['allMatches'])
    except:
        build_id = get_build_id()
        response = requests.get(
            alt_league_url.format(build_id, dict_league_name[league_id])
        ).json()
        df_fixtures = pd.json_normalize(
            response["pageProps"]["initialState"]["league"][str(league_id)]["data"],
            record_path=["fixtures"],
        )

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
    df_shots['keeer_shot_faced'] = df_shots[['opponent_team','minutes']].apply(lambda x: df_lineup_gk.loc[(df_lineup_gk['team_name'] == x['opponent_team']) & (df_lineup_gk['time_subbed_on'] <= x['minutes']) & ((df_lineup_gk['time_subbed_off'] >= x['minutes']) | (df_lineup_gk['time_subbed_off'].isnull())), 'player_last_name'], axis=1)
    
    return df_shots


def get_league_shots(league_id: int):
    enforce_delay()
    fixtures = get_league_fixtures(league_id)
    df_league_shots = pd.DataFrame()
    for l in tqdm(fixtures):
        df_league_shots = pd.conact([df_league_shots,get_single_match_shots(l)]).reset_index(drop=True)

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
    df_lineup = df_lineup.drop(columns=['usingOptaId', 'usualPosition', 'positionRow', 'shotmap', 'stats', 'rating.num', 'rating.bgcolor',
        'rating.isTop.isTopRating', 'rating.isTop.isMatchFinished',
        'fantasyScore.bgcolor', 'events.sub.subbedOut',
        'events.yc', 'events.g', 'events.as', 'events.sub.subbedIn', 'teamData.home.color', 'teamData.away.color', 'teamData.home.id', 'teamData.away.id'])
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