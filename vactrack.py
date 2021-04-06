import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import scipy
import dash_bootstrap_components as dbc


def summarize(n):
    if n > 1_000_000:
        return '{:.2f}M'.format(n / 1_000_000)
    elif n > 1_000:
        return '{:.2f}K'.format(n / 1_000)
    else:
        if n - int(n) < 0.000001:
            return str(int(n))
        else:
            return '{:.2f}'.format(n) if n > 1 else '{:.3f}'.format(n)

########### CONSTANTS
ENGLAND_POPULATION = 55_270_000
SCOTLAND_POPULATION = 5_405_000
WALES_POPULATION = 3_082_000
NI_POPULATION = 1_862_000
UK_POPULATION = ENGLAND_POPULATION + SCOTLAND_POPULATION + WALES_POPULATION + NI_POPULATION

ONE_DOSE_IMMUNITY = 0.7
TWO_DOSE_IMMUNITY = 0.9

_first = 'cumPeopleVaccinatedFirstDoseByPublishDate'
_second = 'cumPeopleVaccinatedSecondDoseByPublishDate'

########### DATA

data = pd.read_csv("https://coronavirus.data.gov.uk/api/v1/data?filters=areaType=nation&structure=%7B%22areaType%22:%22areaType%22,%22areaName%22:%22areaName%22,%22areaCode%22:%22areaCode%22,%22date%22:%22date%22,%22cumPeopleVaccinatedFirstDoseByPublishDate%22:%22cumPeopleVaccinatedFirstDoseByPublishDate%22,%22cumPeopleVaccinatedSecondDoseByPublishDate%22:%22cumPeopleVaccinatedSecondDoseByPublishDate%22%7D&format=csv", 
                    parse_dates=['date'])
data_old = pd.read_csv('data_2020.csv', parse_dates=['date'])
data = pd.concat([data_old, data])
print(data.head())

date_range = data['date'].unique()
latest_date = date_range.max()
latest = data[data.date == latest_date]
print(latest_date)
data_up_to = data['date'].max() + pd.Timedelta(days=1)

total_first_doses = latest['cumPeopleVaccinatedFirstDoseByPublishDate'].sum()
total_second_doses = latest['cumPeopleVaccinatedSecondDoseByPublishDate'].sum()
total_doses = total_first_doses + total_second_doses

doses_england = latest.loc[latest.areaName == 'England', [_first, _second]].sum().sum()
doses_scotland = latest.loc[latest.areaName == 'Scotland', [_first, _second]].sum().sum()
doses_wales = latest.loc[latest.areaName == 'Wales', [_first, _second]].sum().sum()
doses_ni = latest.loc[latest.areaName == 'Northern Ireland', [_first, _second]].sum().sum()

doses_per_capita_england = doses_england / ENGLAND_POPULATION
doses_per_capita_scotland = doses_scotland / SCOTLAND_POPULATION
doses_per_capita_wales = doses_wales / WALES_POPULATION
doses_per_capita_ni = doses_ni / NI_POPULATION

estimated_r_reduction = (total_first_doses * ONE_DOSE_IMMUNITY + total_second_doses * TWO_DOSE_IMMUNITY) / UK_POPULATION

# print(doses_england)

cumdoses_by_date = data.groupby('date')[[_first, _second]].sum().sum(axis=1)
people_by_date = data.groupby('date')[_first].sum()
second_doses_by_date = data.groupby('date')[_second].sum()
first_doses_by_date = data.groupby('date')[_first].sum()
# print(cumdoses_by_date)

# print(cumdoses_by_date.index.to_series())
# print(cumdoses_by_date.index)

cumdoses = cumdoses_by_date.resample('D').interpolate('slinear')
daily_rates = cumdoses.diff(1)
weekly_rates = cumdoses.diff(7)

cumfirstdoses = first_doses_by_date.resample('D').interpolate('slinear')
weekly_first_rates = cumfirstdoses.diff(7)
weekly_first_rates.iloc[:8] = np.linspace(0, weekly_first_rates.iloc[7], 8)

daily_rates = daily_rates[daily_rates.index >= pd.Timestamp(year=2021, month=1, day=10)] # Only valid after daily data starts being published
# Need to correct initial NaNs
weekly_rates.iloc[:8] = np.linspace(0, weekly_rates.iloc[7], 8)

date_15mil = data_up_to + pd.Timedelta(days=(15_000_000 - total_doses) / (weekly_rates.values[-1] / 7))
print(date_15mil)
days_until_15feb = (pd.Timestamp(year=2021, month=2, day=16) - data_up_to).days
weekly_rate_needed_15mil = (15_000_000 - total_doses) / days_until_15feb * 7

date_36mil = data_up_to + pd.Timedelta(days=(36_000_000 - total_doses) / (weekly_rates.values[-1] / 7))
print(date_36mil)
days_until_15apr = (pd.Timestamp(year=2021, month=4, day=15) - data_up_to).days
weekly_rate_needed_36mil = (36_000_000 - total_doses) / days_until_15apr * 7

########## Modelling

print(data.groupby('date')[_first])
cum_firstdoses_by_date = data.groupby('date')[[_first]].sum().sum(axis=1).resample('D').interpolate('slinear')
cum_seconddoses_by_date = data.groupby('date')[[_second]].sum().sum(axis=1).resample('D').interpolate('slinear')
dose_offset = 11*7+3 # 11.5 weeks between doses
target = 52_100_000
# print(cum_firstdoses_by_date)
daily_rate = weekly_rates[-1] / 7
print(daily_rate)

def model_cumdoses(daily_rate, daily_rate_factor, daily_rate_func):
    model_daterange = pd.date_range(start=cum_firstdoses_by_date.index.min(), end='2021-12-01')

    model_first = []
    model_second = []
    for date in model_daterange:
        if daily_rate_func:
            daily_rate = daily_rate_func(date)

        # If we have data, we just reuse that data
        if date in cum_firstdoses_by_date.index:
            model_first.append(cum_firstdoses_by_date[date])
            model_second.append(cum_seconddoses_by_date[date])
        
        else:
            # Otherwise we model
            model_cum_second = model_second[-1]
            model_min_second = model_first[-dose_offset] if len(model_first) > dose_offset else 0
            model_daily_second = max(model_min_second - model_cum_second, 0) if model_first[-1] < target else daily_rate
            if model_cum_second > target:
                model_daily_second = 0

            # Assume 2nd dose recipients are ONLY those who need it
            model_second.append(model_cum_second + model_daily_second)
            model_daily_first = daily_rate - model_daily_second if model_first[-1] < target else 0
            if model_daily_first < 0:
                # AHHH
                print('[ERROR] Not enough 2nd doses on {}!'.format(date))

            model_first.append(model_first[-1] + model_daily_first)
            daily_rate *= daily_rate_factor

    # Make sure that first doses are monotonically increasing (remove where second doses later "borrow from first doses")
    for i in range(len(model_first)):
        model_first[i] = np.min(model_first[i:])

    df = pd.DataFrame()
    df['date'] = model_daterange
    df['first'] = model_first
    df['second'] = model_second
    return df

def cabinet_office(date):
    if date < pd.Timestamp(year=2021, month=7, day=31):
        return 2_700_000 // 7
    return 2_000_000 // 7

model_constant = model_cumdoses(daily_rate, 1, False)
model_increase = model_cumdoses(None, 1, cabinet_office)

########## DASH

external_stylesheets = ["https://use.typekit.net/dvr4nik.css", dbc.themes.BOOTSTRAP]#['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_scripts =[{'async': True, 'data-domain': "vaccine.mxbi.net", "defer": "defer", "src": "https://stats.mxbi.net/js/pla.js"}]# ["https://use.typekit.net/dvr4nik.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)
server = app.server

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
# fig_doses = px.line(cumdoses_by_date, name='Total')
fig_doses = go.Figure()
fig_doses.add_trace(go.Scatter(x=cumdoses_by_date.index, y=cumdoses_by_date, name='Total doses'))
fig_doses.add_trace(go.Scatter(x=people_by_date.index, y=people_by_date.values, name='1st doses'))
fig_doses.add_trace(go.Scatter(x=second_doses_by_date.index, y=second_doses_by_date.values, name='2nd doses'))
# fig_doses.data[0].update(mode='markers+lines')
fig_doses.update_layout(title="Total doses given in UK", xaxis_title="Date", yaxis_title="Cumulative doses", font=dict(size=15, family="nimbus-sans"), margin=dict(l=0, r=0, t=50, b=0))
fig_doses.update_yaxes(range=[0, cumdoses_by_date.max() * 1.05])

# fig_rate = px.scatter(daily_rates*7, mode='lines', name="Daily")
fig_rate = go.Figure()
fig_rate.add_trace(go.Scatter(x=daily_rates.index, y=daily_rates.values*7, name="1 day rate", line=dict(dash="dash")))
fig_rate.add_trace(go.Scatter(x=weekly_rates.index, y=weekly_rates.values, name="1 week rate"))
fig_rate.add_trace(go.Scatter(x=weekly_first_rates.index, y=weekly_first_rates.values, name="1 week first doses", line=dict(color="rgba(32, 201, 151, 0.2)")))
fig_rate.update_layout(title="Vaccination rate", xaxis_title="Date", yaxis_title="Doses/week", font=dict(size=15, family="nimbus-sans"), legend_title_text="Calculated over", margin=dict(l=0, r=0, t=50, b=0))
fig_rate.update_yaxes(range=[0, daily_rates.max() * 7 * 1.05])

# fig_rate.add_scatter(x=weekly_rates.index, y=weekly_rates.values, mode='lines', name="Weekly")

fig_model = go.Figure()
fig_model.add_trace(go.Scatter(x=model_constant.date, y=cum_firstdoses_by_date, name="First doses", line=dict(color="#636EFA")))
fig_model.add_trace(go.Scatter(x=model_constant.date, y=cum_seconddoses_by_date, name="Second doses", line=dict(color="#00CC96")))
fig_model.add_trace(go.Scatter(x=model_constant.date, y=model_constant['first'], name="First (model)", line=dict(dash="dash", color="#636EFA")))
fig_model.add_trace(go.Scatter(x=model_constant.date, y=model_constant.second, name="Second (model)", line=dict(dash="dash", color="#00CC96")))
fig_model.update_layout(title="Assuming supply is constant", xaxis_title="Date", yaxis_title="Total doses", font=dict(size=15, family="nimbus-sans"), margin=dict(l=0, r=0, t=50, b=0))
fig_model.add_shape(type='line', x0=model_constant.date.min(), x1=model_constant.date.max(), y0=15_000_000, y1=15_000_000, line=dict(color='rgba(171, 99, 250, 0.2)'), name="Group 4 (>70s+)")
fig_model.add_shape(type='line', x0=model_constant.date.min(), x1=model_constant.date.max(), y0=32_000_000, y1=32_000_000, line=dict(color='rgba(171, 99, 250, 0.2)'), name="Phase 1 (>50s+)")
fig_model.update_yaxes(range=[0, 52000000])
fig_model.update_xaxes(dtick="M1")

fig_model2 = go.Figure()
fig_model2.add_trace(go.Scatter(x=model_increase.date, y=cum_firstdoses_by_date, name="First doses", line=dict(color="#636EFA")))
fig_model2.add_trace(go.Scatter(x=model_increase.date, y=cum_seconddoses_by_date, name="Second doses", line=dict(color="#00CC96")))
fig_model2.add_trace(go.Scatter(x=model_increase.date, y=model_increase['first'], name="First (model)", line=dict(dash="dash", color="#636EFA")))
fig_model2.add_trace(go.Scatter(x=model_increase.date, y=model_increase.second, name="Second (model)", line=dict(dash="dash", color="#00CC96")))
fig_model2.update_layout(title="Cabinet Office Modelling Scenario (as of 31st March)", xaxis_title="Date", yaxis_title="Total doses", font=dict(size=15, family="nimbus-sans"), margin=dict(l=0, r=0, t=50, b=0))
fig_model2.add_shape(type='line', x0=model_increase.date.min(), x1=model_increase.date.max(), y0=15_000_000, y1=15_000_000, line=dict(color='rgba(171, 99, 250, 0.2)'), name="Group 4 (>70s+)")
fig_model2.add_shape(type='line', x0=model_increase.date.min(), x1=model_increase.date.max(), y0=32_000_000, y1=32_000_000, line=dict(color='rgba(171, 99, 250, 0.2)'), name="Phase 1 (>50s+)")
fig_model2.update_yaxes(range=[0, 52000000])
fig_model2.update_xaxes(dtick="M1")

app.layout = html.Div(children=[
    # dbc.NavbarSimple(brand="CovidTrack | Vaccine Rollout", dark=True, color="dark", expand=True),
    # html.H1("CovidTrack | Vaccine Rollout"),
    html.Div([
    # html.Script(**{'async': True, 'data-domain': "vaccine.mxbi.net", "defer": "defer", "src": "https://stats.mxbi.net/js/plausible.js"}),
    html.Div(children=[
    dcc.Markdown(f"""
# CovidTrack | Vaccine Rollout
**{summarize(total_doses)}** doses given in total, to **{summarize(total_first_doses)}** people.

**[See also: R rate tracker](http://mb2345.user.srcf.net/covidtrack/)**

#### Current rate of vaccination
Last day:  **{summarize(daily_rates.values[-1])}** doses/day ({summarize(daily_rates.values[-1] * 7)} doses/week)  
Last week: **{summarize(weekly_rates.values[-1])}** doses/week

Estimated population immunity due to vaccination: **{round(estimated_r_reduction*100, 1)}%** (crude)

The target of 15M doses was met on **February 12th, 2021**. 🎉  
The target of 36M doses was met on **April 1st, 2021**. 🎉  

#### Doses per capita
England: **{summarize(doses_per_capita_england)}** | Scotland: **{summarize(doses_per_capita_scotland)}**  | Wales: **{summarize(doses_per_capita_wales)}** | NI: **{summarize(doses_per_capita_ni)}**  
    """),
    ]),

    html.Div([
        html.Div([
            html.Div([dcc.Graph(id='example-graph',figure=fig_doses)], className="col-xl-6"),
            html.Div([dcc.Graph(id='graph2',figure=fig_rate)], className="col-xl-6")
        ], className="row"),
    ], className='container-fluid'),

    html.Hr(),

    dcc.Markdown(f"""
    ### Modelling

    Assuming supply **stays constant**, all adults will have their first dose by **{model_constant[model_constant['first'] > target]['date'].iloc[0].strftime('%B %d, %Y')}**, and be fully vaccinated by **{model_constant[model_constant['second'] > target]['date'].iloc[0].strftime('%B %d, %Y')}**  
    Assuming the **Cabinet Office internal planning scenario**, all adults will have their first dose by **{model_increase[model_increase['first'] > target]['date'].iloc[0].strftime('%B %d, %Y')}**, and be fully vaccinated by **{model_increase[model_increase['second'] > target]['date'].iloc[0].strftime('%B %d, %Y')}**

    """),

    html.Div([
        html.Div([
                html.Div([dcc.Graph(id='modelled-graph', figure=fig_model)], className="col-xl-6"),
                html.Div([dcc.Graph(id='modelled-graph2', figure=fig_model2)], className="col-xl-6")
        ], className="row"),
    ], className="container-fluid"),

    html.I(["Data up to {}. Data generally updates every day after 4pm.".format(data_up_to)]),
    html.Br(),
    dcc.Markdown("**Made by [Mikel Bober-Irizar](https://twitter.com/mikb0b)**. Data from [UK Coronavirus Dashboard](https://coronavirus.data.gov.uk/details/vaccinations)"),
    ], style={"margin-left": "2em", "margin-right": "2em", "margin-top": "1em", "font-family": '"nimbus-sans", sans-serif !important'})
])

if __name__ == '__main__':
    app.run_server(debug=True)
