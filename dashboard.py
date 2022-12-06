import math
from dateutil import parser
from datetime import datetime

from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
import pandas as pd

from main import (create_df_from_params_vect, 
                  get_df_with_params, parse_trades,
                  get_trades_from_file)

from bonds import Bond

from nelson_siegel_svensson.calibrate import calibrate_ns_ols


app = Dash(__name__)

pd.options.display.float_format = '{:,.4f}'.format


INL_TAU = 2.0

df = create_df_from_params_vect()
params = get_df_with_params()
day_t = df.iloc[1]['tradedate'] 
# trades = parse_trades( f"{day_t.day}.{day_t.month}.{day_t.year}",'gsecs', '#gsec_clean')

sht_cols = ['tradedate', 
        'Код', 'Режим','Минимальная цена','Максимальная цена', 
        'Цена первой сделки', 'Цена последней сделки', 
        'Количество сделок', 'Объем, млн KZT', 
        'Средневзвеш. цена', 'Yield, %', 'Дни до погашения']

sht_cols2 = ['tradedate', 'ticker', 'short_name', 'currency', 'high_price',
       'low_price', 'open_price', 'close_price', 'volume', 'Yield, %',
       'Duration, years']

fig = px.line(df.iloc[0], x=df.drop('tradedate', axis=1).columns, 
                            y = df.drop('tradedate', axis=1).iloc[0].values,
                            title=f'График G-curve по состоянию на {day_t.day}.{day_t.month}.{day_t.year}')
fig.update_xaxes(
    ticklen=10
)
fig.update_xaxes(title='duration')
fig.update_yaxes(title='yield')


app.layout = html.Div([
        dcc.Dropdown( [{'label': str(i), 'value': str(i)} for i in df['tradedate'].unique()], id='gcurve-tradedate'),
        html.Br(),
        html.Button("Обновить Gcurve", id='submit-gc-update', n_clicks=0),
        dcc.Graph(id='gcurve-fig', figure=fig),
        html.Div(["Дюрация: ", dcc.Input(id='gcurve-input', value='1.0', type='text')]),
        html.Br(),
        html.Div(id='gcurve-output'),
        html.Br(),
        html.Div(['Параметры ГЦБ:']),
        html.Br(),
        dash_table.DataTable(columns=[{'name': i, 'id':i} for i in params.columns], id='tbl'),
        html.Br(),
        html.Div(['Перечень ГЦБ из KASE:']),
        html.Br(),
        dash_table.DataTable(columns=[{'name': i, 'id': i} for i in sht_cols], id='tbl-trades', 
                             editable=True, row_deletable=True),
        html.Br(),
        dash_table.DataTable(columns=[{'name': i, 'id':i} for i in params.columns], id='gcurve-params-new1'),
        html.Br(),
        # html.Div(["Дюрация: ", dcc.Input(id='gcurve-input2', value='1.0', type='text')]),
        html.Br(),
        html.Div(id='gcurve-output2'),
        html.Br(),
        html.Div(['Перечень ГЦБ из TN:']),
        html.Br(),
        dash_table.DataTable(columns=[{'name': i, 'id': i} for i in sht_cols2], id='tbl-trades2',
                             editable=True, row_deletable=True),
        html.Br(),
        dash_table.DataTable(columns=[{'name': i, 'id':i} for i in params.columns], id='gcurve-params-new2'),
        html.Br(),
        # html.Div(["Дюрация: ", dcc.Input(id='gcurve-input3', value='1.0', type='text')]),
        html.Br(),
        html.Div(id='gcurve-output3'),
    ], style={'width': '50%', 'margin-left': '10%'})


@app.callback(
    Output('gcurve-fig', 'figure'),
    Input('gcurve-tradedate', 'value'),
)
def update_fig(tradedate_v):
    if not tradedate_v:
        return None
    day_t = parser.parse(tradedate_v)
    tmp_df = df.loc[df['tradedate']==tradedate_v]
    tmp_df = tmp_df.drop('tradedate', axis=1)
    fig = px.line(x = list(tmp_df.columns), 
                    y = tmp_df.values[0], 
                    title=f'График G-curve по состоянию на {day_t.day:02d}.{day_t.month:02d}.{day_t.year}')
    fig.update_xaxes(title='Дней до погашения')
    fig.update_yaxes(title='yield')
    return fig

@app.callback(
    Output('tbl', 'data'),
    Input('gcurve-tradedate', 'value'),
)
def update_table(tradedate_v):
    if not tradedate_v:
        return None
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
        return f'Значение кривой для дюрации {input_value} лет: {res.values[0]*100:.4f}%'
    except:
        return 'Значение введено некорректно'


@app.callback(
    Output('gcurve-output2', 'children'),
    Input('gcurve-input', 'value'),
    Input('gcurve-params-new1', 'data'),
)
def update_div2(input_value, data):
    try:
        m = float(input_value)
        row = pd.DataFrame(data)
        res = row["B0"] \
            +((row["B1"]+row["B2"])*(row["TAU"]/m)*(1-math.exp(-m/row["TAU"]))) \
            -row["B2"]*math.exp(-m/row["TAU"])
        return f'Значение кривой для дюрации {input_value} лет: {res.values[0]*100:.4f}%'
    except:
        return 'Значение введено некорректно'


@app.callback(
    Output('gcurve-output3', 'children'),
    Input('gcurve-input', 'value'),
    Input('gcurve-params-new2', 'data'),
)
def update_div3(input_value, data):
    try:
        m = float(input_value)
        row = pd.DataFrame(data)
        res = row["B0"] \
            +((row["B1"]+row["B2"])*(row["TAU"]/m)*(1-math.exp(-m/row["TAU"]))) \
            -row["B2"]*math.exp(-m/row["TAU"])
        return f'Значение кривой для дюрации {input_value} лет: {res.values[0]*100:.4f}%'
    except:
        return 'Значение введено некорректно'


@app.callback(
    Output('tbl-trades', 'data'),
    Input('gcurve-tradedate', 'value'),
)
def update_trades_table(tradedate_v):
    if not tradedate_v:
        return None
    day_t = parser.parse(tradedate_v)
    trades = parse_trades( f"{day_t.day:02d}.{day_t.month:02d}.{day_t.year}",'gsecs', '#gsec_clean')
    bonds = trades[['Код', 'Средневзвеш. цена']].values
    
    for kod, price in bonds:
        b = Bond.find_bond(code=kod, bond_price=price, rep_date=day_t)
        if b is None:
            continue
        r = b.get_ytm()
        trades.loc[trades["Код"]==kod, ['Yield, %']] = round(r * 100,2)
        # trades.loc[trades["Код"]==kod, ['Дни до погашения']] = int(b.get_duration(r, day_t))

    return trades.to_dict('records') 


@app.callback(
    Output('tbl-trades2', 'data'),
    Input('gcurve-tradedate', 'value'),
)
def update_trades_table2(tradedate_v):
    if not tradedate_v:
        return None
    day_t = parser.parse(tradedate_v)
    trades = get_trades_from_file(datetime(day_t.year, day_t.month, day_t.day, 21,0,0))
    trades['tradedate'] = trades['tradedate'].dt.strftime('%d.%m.%Y')
    bonds = trades[['ticker', 'close_price']].values
    
    print(trades.shape)
    
    if trades.shape[0] < 1:
        print('im here')
        return trades.to_dict('records')

    for kod, price in bonds:
        b = Bond.find_bond(code=kod.split('.')[0], bond_price=price, rep_date=day_t)
        if not b:
            continue
        r = b.get_ytm()
        trades.loc[trades["ticker"]==kod, ['Yield, %']] = round(r* 100,2)
        trades.loc[trades["ticker"]==kod, ['Duration, years']] = int(b.get_fix_days_before_mat(day_t))
    trades = trades.sort_values('Duration, years')
    return trades.to_dict('records')

# @app.callback(
#     Output('gcurve-tradedate', 'options'),
#     Input('submit-gc-update', 'n_clicks')
# )
# def update_gcurve(n_clicks):
#     print('update gcurve started...')
#     download_gcurve_params()
#     global df, params
#     df = create_df_from_params_vect()
#     params = get_df_with_params()
#     return [{'label': str(i), 'value': str(i)} for i in df['tradedate'].unique()]

@app.callback(
    Output('gcurve-params-new1', 'data'),
    Input('tbl-trades', 'data'),
)
def update_trades_table1(data):
    df = pd.DataFrame(data)
    if df.shape[0] < 1:
        return None
    df = df.loc[df['Yield, %']>0]
    df = df.loc[df['Код']!='KZ_06_4410']
    df = df.loc[df['Дни до погашения']<9999]
    # print(df)
    y = df['Yield, %'].values
    t = df['Дни до погашения'].values/365
    curve, status = calibrate_ns_ols(t, y, tau0=INL_TAU)
    # print('КАСЕ',curve)
    print(f"NelsonSiegelCurve({curve.beta0=:.4f}, {curve.beta1=:.4f}. {curve.beta2=:.4f}, {curve.tau=:.4f})")
    return pd.DataFrame([{'tradedate':df['tradedate'].iloc[0], 
                          'B0':curve.beta0/100, 'B1':curve.beta1/100, 'B2':curve.beta2/100, 'TAU':curve.tau}]).to_dict('records')


@app.callback(
    Output('gcurve-params-new2', 'data'),
    Input('tbl-trades2', 'data'),
)
def update_trades_table2(data):
    df = pd.DataFrame(data)
    if df.shape[0] < 1:
        return None
    df = df.loc[df['Yield, %']>0]
    df = df.loc[df['currency']=='KZT']
    df = df.loc[df['Duration, years']<9999]
    y = df['Yield, %'].values
    t = df['Duration, years'].values/365
    curve, status = calibrate_ns_ols(t, y, tau0=INL_TAU)
    return pd.DataFrame([{'tradedate':df['tradedate'].iloc[0],
                          'B0':curve.beta0/100, 
                          'B1':curve.beta1/100, 
                          'B2':curve.beta2/100, 
                          'TAU':curve.tau}]).to_dict('records')


if __name__=='__main__':
    app.run_server(debug=True)