import math
from dateutil import parser

from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
import pandas as pd

from main import (create_df_from_params_vect, 
                  download_gcurve_params,
                  get_df_with_params, parse_trades)


app = Dash(__name__)

pd.options.display.float_format = '${:,.4f}'.format

df = create_df_from_params_vect()
params = get_df_with_params()
day_t = df.iloc[1]['tradedate'] 
trades = parse_trades( f"{day_t.day}.{day_t.month}.{day_t.year}",'gsecs', '#gsec_clean')


fig = px.line(df.iloc[0], x=df.drop('tradedate', axis=1).columns, 
                            y = df.drop('tradedate', axis=1).iloc[0].values,
                            title=f'График G-curve по состоянию на {day_t.day}.{day_t.month}.{day_t.year}')
fig.update_xaxes(title='duration')
fig.update_yaxes(title='yield')


app.layout = html.Div([
    dcc.Dropdown(df['tradedate'].unique(), df.iloc[0]['tradedate'], id='gcurve-tradedate'),
    html.Br(),
    html.Button("Обновить Gcurve", id='submit-gc-update', n_clicks=0),
    dcc.Graph(id='gcurve-fig', figure=fig),
    html.Div(["Дюрация: ", dcc.Input(id='gcurve-input', value='1.0', type='text')]),
    html.Br(),
    html.Div(id='gcurve-output'),
    html.Br(),
    html.Div(['Параметры ГЦБ:']),
    html.Br(),
    dash_table.DataTable(params.iloc[:1].to_dict('records'), [{'name': i, 'id':i} for i in params.columns], id='tbl'),
    html.Br(),
    html.Div(['Перечень ГЦБ:']),
    html.Br(),
    dash_table.DataTable(trades.to_dict('records'), [{'name': i, 'id':i} for i in trades.columns], id='tbl-trades')
    ], style={'width':'50%', 'margin-left':'10%'})


@app.callback(
    Output('gcurve-fig', 'figure'),
    Input('gcurve-tradedate', 'value'),
)
def update_fig(tradedate_v):
    day_t = parser.parse(tradedate_v)
    tmp_df = df.loc[df['tradedate']==tradedate_v]
    tmp_df = tmp_df.drop('tradedate', axis=1)
    fig = px.line(x = list(tmp_df.columns), 
                    y = tmp_df.values[0], 
                    title=f'График G-curve по состоянию на {day_t.day:02d}.{day_t.month:02d}.{day_t.year}')

    fig.update_xaxes(title='duration')
    fig.update_yaxes(title='yield')
    return fig

@app.callback(
    Output('tbl', 'data'),
    Input('gcurve-tradedate', 'value'),
)
def update_table(tradedate_v):
    tmp_df = params.loc[params['tradedate']==tradedate_v].copy()
    cols = list(set(tmp_df.columns)-set(['tradedate']))
    day_t = parser.parse(tradedate_v)
    tmp_df['tradedate'] = [f"{day_t.day:02d}.{day_t.month:02d}.{day_t.year}"] * tmp_df.shape[0]
    # tmp_df.loc[:,'tradedate'] = f'{day_t.day:2d}.{day_t.month:2d}.{day_t.year}'
    tmp_df[cols] = tmp_df[cols].applymap(lambda x: '{0:.4f}'.format(x))
    data = tmp_df.to_dict('records')
    return data 

@app.callback(
    Output('gcurve-output', 'children'),
    Input('gcurve-input', 'value'),
    Input('gcurve-tradedate', 'value'),
)
def update_div(input_value,tradedate_v):
    try:
        m = float(input_value)
        row = params.loc[params['tradedate']==tradedate_v]
        res = row["B0"] \
            +((row["B1"]+row["B2"])*(row["TAU"]/m)*(1-math.exp(-m/row["TAU"]))) \
            -row["B2"]*math.exp(-m/row["TAU"])
        return f'Значение кривой: {res.values[0]*100:.4f}%'
    except:
        return 'Значине введено некорректно'

@app.callback(
    Output('tbl-trades', 'data'),
    Input('gcurve-tradedate', 'value'),
)
def update_trades_table(tradedate_v):
    try:
        day_t = parser.parse(tradedate_v)
        trades = parse_trades( f"{day_t.day:02d}.{day_t.month:02d}.{day_t.year}",'gsecs', '#gsec_clean')
        trades = trades.sort_values('Дни до погашения')
        return trades.to_dict('records') 
    except:
        return 'Значине введено некорректно'

@app.callback(
    Output('gcurve-tradedate', 'options'),
    Input('submit-gc-update', 'n_clicks')
)
def update_gcurve(n_clicks):
    print('update gcurve started...')
    download_gcurve_params()
    global df, params
    df = create_df_from_params_vect()
    params = get_df_with_params()
    return [{'label': str(i), 'value': str(i)} for i in df['tradedate'].unique()]


if __name__=='__main__':
    app.run_server(debug=True)