# coding: utf-8

from itertools import combinations

import pandas as pd
import numpy as np

import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as table

from dash.dependencies import Input, Output

# dict to store league codes as per football-data website
league_codes = {
    'England': {
        'Premier League': 'E0',
        'Championship': 'E1',
        'League 1': 'E2',
        'League 2': 'E3',
        'Conference': 'EC'
        },
    'Scotland': {
        'Premier League': 'SC0',
        'Division 1': 'SC1',
        'Division 2': 'SC2',
        'Division 3': 'SC3'
        },
    'Germany': {
        'Bundesliga 1': 'D1',
        'Bundesliga 2': 'D2'
        },
    'Italy': {
        'Serie A': 'I1',
        'Serie B': 'I2'
        },
    'Spain': {
        'La Liga Primera Division': 'SP1',
        'La Liga Segunda Division': 'SP2'
        },
    'France': {
        'Le Championnat': 'F1',
        'Division 2': 'F2'
        },
    'Netherlands': {
        'Eredivisie': 'N1'
        },
    'Belgium': {
        'Jupiler League': 'B1'
        },
    'Portugal': {
        'Liga I': 'P1'
        },
    'Turkey': {
        'Futbol Ligi 1': 'T1'
        },
    'Greece': {
        'Ethniki Katigoria': 'G1'
        }
}


season = '1819'
country = 'England'
league = 'Premier League'

def calculate_points_per_game_difference(teams, modified_results):
    """Returns points per game difference for teams passed as input."""
    common_games = (modified_results[modified_results.team == teams[0]]
                    .replace({'opponent': f'{teams[1]}'}, 'H2H')
                    .assign(venue = lambda x: np.where(x.opponent == 'H2H', 'H2H', x.venue))
                    .merge((modified_results[modified_results.team == teams[1]]
                            .replace({'opponent': f'{teams[0]}'}, 'H2H')
                            .assign(venue = lambda x: np.where(x.opponent == 'H2H', 'H2H', x.venue))),
                           how='inner',
                           on=['opponent','venue']))
    
    if not common_games.empty:
        points_per_game_difference = ((common_games.points_x.sum()
                                       - common_games.points_y.sum())
                                      / common_games.shape[0])
    else:
        points_per_game_difference = 0
    
    return points_per_game_difference

def calculate_adjusted_points(country, league):

    # import results data for the current season as of Nov 23 2018
    results = pd.read_csv(f'http://www.football-data.co.uk/mmz4281/{season}/{league_codes[country][league]}.csv')

    # create modified results dataframe in required format
    home_results = (results
                   .assign(points = results.FTR.map({'H': 3, 'A': 0, 'D': 1}),
                           venue = 'home')
                   .rename(columns = {
                       'HomeTeam': 'team',
                       'AwayTeam': 'opponent'
                   })
                   .loc[:, ['team','opponent','venue','points']])

    away_results = (results
                   .assign(points = results.FTR.map({'A': 3, 'H': 0, 'D': 1}),
                           venue = 'away')
                   .rename(columns = {
                       'AwayTeam': 'team',
                       'HomeTeam': 'opponent'
                   })
                   .loc[:, ['team','opponent','venue','points']])

    modified_results = pd.concat([home_results, away_results], ignore_index=True, sort=False)

    teams = pd.unique(results.HomeTeam)
    teams.sort()

    # create the W matrix
    mapping_table = pd.DataFrame(index=list(combinations(teams, 2)), columns=teams).fillna(0)

    for idx, row in mapping_table.iterrows():
        mapping_table.loc[idx, idx[0]] = 1
        mapping_table.loc[idx, idx[1]] = -1

    # convert mapping table to numpy array to create 'W'
    W = mapping_table.values

    # create 'r' using the function defined above
    r = (mapping_table
         .index
         .map(lambda x: calculate_points_per_game_difference(x, modified_results))
         .values
         .reshape(mapping_table.shape[0], 1))

    # minimizing sum of squared errors to get x
    x = np.linalg.lstsq(W, r, rcond=None)[0]

    # multiplier used for scaling up equals number of matches played by team
    m = (modified_results
         .groupby('team')
         .points
         .count()
         .values
         .reshape(len(mapping_table.columns), 1))

    # add average points per team per game to x and multiply by number of matches
    adjusted_points = m * (x + modified_results.points.mean())
    actual_points = modified_results.groupby('team').points.sum().values

    combined_table = (pd.DataFrame({'Team': teams, 'Matches': m.flatten(), 
                                    'Actual Points': modified_results.groupby('team').points.sum().values,
                                    'Adjusted Points': adjusted_points.flatten()})
                     .assign(Actual_Position = lambda x: x['Actual Points'].rank(ascending=False, method='min'),
                             Adjusted_Position = lambda x: x['Adjusted Points'].rank(ascending=False, method='min'))
                     .rename(columns = lambda x: x.replace('_',' '))
                     .sort_values('Adjusted Points', ascending=False))

    return combined_table

def create_position_plot(combined_table):
    traces = []
    for idx, row in combined_table.iterrows():
        trace = go.Scatter(
                            x = ['Actual Points', 'Adjusted Points'],
                            y = [row['Actual Points'], row['Adjusted Points']],
                            showlegend = False,
                            mode = 'lines+markers+text',
                            text = row['Team'],
                            textposition = ['middle left', 'middle right'],
                            hoverinfo = 'text+x+y'
                            )
        traces.append(trace)

    layout = go.Layout(
                        hovermode = 'closest',
                        xaxis = {
                            'range': [-0.5, 1.5],
                            'side': 'top',
                            'fixedrange': True
                        },
                        yaxis = {
                            'fixedrange': True
                        },
                        margin = {
                            't': 50,
                            'r': 0,
                            'l': 50,
                            'b': 0 
                        }
                        )

    return {'data': traces, 'layout': layout}


from server import server

app = dash.Dash(name='app2', sharing=True, server=server, url_base_pathname='/salt/')

app.title = 'Schedule-adjusted league tables'

app.css.append_css({"external_url": "https://codepen.io/hkhare42/pen/eQzWNy.css"})

app.layout = html.Div(id='bodydiv2', style = {'width': '96vw'}, children = [
    html.Ul([
        html.Li(children = ['Idea and methodology for this has been developed by Constantinos Chappas.',
            html.A(href='https://twitter.com/cchappas', target="_blank", children='@cchappas', style={'color': 'Blue'})]),
        html.Li(children = ['Read more about it in his StatsBomb article here: ',
            html.A(href='https://statsbomb.com/2018/11/introducing-the-schedule-adjusted-league-table/', target="_blank", 
            children='Link to article', style={'color': 'Blue'})]),
        html.Li(children = ['Data used for this is being fetched from: ',
            html.A(href='http://www.football-data.co.uk/', target="_blank", 
            children='www.football-data.co.uk', style={'color': 'Blue'})])]),
    dcc.Dropdown(id='dropdown_country', style={'width': '25vw'}, options=
                            [{'label':country, 'value':country} 
                                for country in list(league_codes.keys())], value='England',
                                clearable=False),
    dcc.Dropdown(id='dropdown_league', style={'width': '25vw'},
                                clearable=False),
    table.DataTable(
        id='table_position',
        style_table={'width': '55vw', 'position':'absolute', 'left':'0vw'},
        style_cell={'fontFamily': 'Arial'},
        columns=[
            {"name": i, "id": i} for i in ['Team','Matches',
                                           'Actual Points','Adjusted Points',
                                           'Actual Position','Adjusted Position']
        ],
        sorting=True,
    ),
    dcc.Graph(id='plot_position', style={'width': '30vw', 'position':'absolute', 'left':'55vw', 'height': '80vh'},
        config={'displayModeBar': False})
])

@app.callback(
            Output('dropdown_league', 'options'),
            [Input('dropdown_country', 'value')])
def populate_leagues(country):
    return  [{'label':league, 'value':league} for league in list(league_codes[country].keys())]

@app.callback(
            Output('dropdown_league', 'value'),
            [Input('dropdown_league', 'options')])
def populate_league_value(available_options):
    return  available_options[0]['value']

@app.callback(
            Output('table_position', 'data'),
            [Input('dropdown_country', 'value'),
             Input('dropdown_league', 'value')])
def populate_table(country, league):
    combined_table = calculate_adjusted_points(country, league)
    combined_table.loc[:, 'Adjusted Points'] = combined_table['Adjusted Points'].round(2)
    return combined_table.to_dict('rows')

@app.callback(
            Output('plot_position', 'figure'),
            [Input('table_position', 'data')])
def create_position_figure(data_table):
    data_table = pd.DataFrame(data_table)
    return create_position_plot(data_table)

if __name__ == '__main__':
    app.run_server(
            debug=True, 
            port = 8080
        )