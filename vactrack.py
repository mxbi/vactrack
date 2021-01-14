import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import scipy

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

ONE_DOSE_IMMUNITY = 0.8
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

print(doses_england)

cumdoses_by_date = data.groupby('date')[[_first, _second]].sum().sum(axis=1)
people_by_date = data.groupby('date')[_first].sum()
print(cumdoses_by_date)

# print(cumdoses_by_date.index.to_series())
print(cumdoses_by_date.index)

cumdoses = cumdoses_by_date.resample('D').interpolate('slinear')
daily_rates = cumdoses.diff(1)
weekly_rates = cumdoses.diff(7)

daily_rates = daily_rates[daily_rates.index >= pd.Timestamp(year=2021, month=1, day=10)] # Only valid after daily data starts being published
# Need to correct initial NaNs
weekly_rates.iloc[:8] = np.linspace(0, weekly_rates.iloc[7], 8)
print(weekly_rates)

########## DASH

external_stylesheets = ["https://use.typekit.net/dvr4nik.css"]#['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_scripts =[]# ["https://use.typekit.net/dvr4nik.css"]
# external_stylesheets = ["https://raw.githubusercontent.com/plotly/dash-app-stylesheets/master/dash-uber-ride-demo.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)
server = app.server

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
# fig_doses = px.line(cumdoses_by_date, name='Total')
fig_doses = go.Figure()
fig_doses.add_trace(go.Scatter(x=cumdoses_by_date.index, y=cumdoses_by_date, name='Doses'))
fig_doses.add_trace(go.Scatter(x=people_by_date.index, y=people_by_date.values, name='People'))
fig_doses.data[0].update(mode='markers+lines')

fig_doses.update_layout(title="Total doses given in UK", xaxis_title="Date", yaxis_title="Cumulative doses", font=dict(size=15, family="nimbus-sans"))

# fig_rate = px.scatter(daily_rates*7, mode='lines', name="Daily")
fig_rate = go.Figure()
fig_rate.add_trace(go.Scatter(x=daily_rates.index, y=daily_rates.values*7, name="1 day rate", line=dict(dash="dash")))
fig_rate.add_trace(go.Scatter(x=weekly_rates.index, y=weekly_rates.values, name="1 week rate"))
fig_rate.update_layout(title="Vaccination rate", xaxis_title="Date", yaxis_title="Doses/week", font=dict(size=15, family="nimbus-sans"), legend_title_text="Calculated over")
fig_rate.update_yaxes(range=[0, 2000000])
# fig_rate.add_scatter(x=weekly_rates.index, y=weekly_rates.values, mode='lines', name="Weekly")

app.layout = html.Div(children=[
    # html.H1(children='CovidTrack | Vaccine Rollout'),

    # html.Div(children=[
        # "Total doses: {}.\nTest".format(summarize(total_doses)),
        # 
        # ),
    html.Div(children=[
    dcc.Markdown(f"""
# CovidTrack | Vaccine Rollout

**{summarize(total_doses)}** doses given in total, to **{summarize(total_first_doses)}** people.

#### Current rate of vaccination
Last day:  **{summarize(daily_rates.values[-1])}** doses/day ({summarize(daily_rates.values[-1] * 7)} doses/week)  
Last week: **{summarize(weekly_rates.values[-1])}** doses/week

Estimated population immunity due to vaccination: **{round(estimated_r_reduction*100, 2)}%**

### Doses per capita
England: **{summarize(doses_per_capita_england)}**  
Scotland: **{summarize(doses_per_capita_scotland)}**  
Wales: **{summarize(doses_per_capita_wales)}**  
Northern Ireland: **{summarize(doses_per_capita_ni)}**  
    """),
    ]),

    html.Div([
        html.Div([
            dcc.Graph(
                id='example-graph',
                figure=fig_doses
            )
        ], className="six columns"),

        html.Div([dcc.Graph(id='graph2',figure=fig_rate)], className="six columns")
    ], className="row"),

    html.I(["Data up to {}. Data generally updates every day after 4pm.".format(daily_rates.index[-1] + pd.Timedelta(days=1))])
], style={"margin-left": "2em", "margin-right": "2em"})

if __name__ == '__main__':
    app.run_server(debug=True)
