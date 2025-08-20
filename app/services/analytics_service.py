# app/services/analytics_service.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache


class AnalyticsService:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

    @lru_cache(maxsize=32)  # Cache results to speed up dashboard filtering
    def get_filtered_data(self, platform='All'):
        if platform == 'All' or platform is None:
            return self.df
        return self.df[self.df['platform'] == platform].copy()

    def get_kpis(self, platform='All'):
        df = self.get_filtered_data(platform)
        total_customers = len(df)
        churned_customers = df['churn'].sum()
        churn_rate = (churned_customers / total_customers) * 100 if total_customers > 0 else 0

        active_users = df[df['churn'] == 0]
        avg_tenure = active_users['tenure_months'].mean()
        avg_watch_hours = active_users['monthly_watch_hours'].mean()

        return {
            'total_customers': f"{total_customers:,}",
            'churn_rate': f"{churn_rate:.2f}%",
            'avg_tenure': f"{avg_tenure:.1f} mo",
            'avg_watch_hours': f"{avg_watch_hours:.1f} hrs"
        }

    def get_churn_by_state_map(self, platform='All'):
        df = self.get_filtered_data(platform)
        churn_by_state = df.groupby('state')['churn'].mean().reset_index()
        churn_by_state['churn_rate'] = churn_by_state['churn'] * 100

        # Load India GeoJSON data
        # In a real app, this file would be in /static. For simplicity, we fetch it.
        # This requires an internet connection.
        fig = px.choropleth(
            churn_by_state,
            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea0826fc51e512/raw/e388c4cae20aa53cb5090210a42ebb9b765c3a36/india_states.geojson",
            featureidkey='properties.ST_NM',
            locations='state',
            color='churn_rate',
            color_continuous_scale="Reds",
            scope="asia",
            title="Churn Rate (%) by State"
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, title_x=0.5)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def get_content_strategy_plot(self, platform='All'):
        df = self.get_filtered_data(platform)
        genre_analysis = df.groupby('main_genre_focus').agg(
            popularity=('user_id', 'count'),
            retention_impact=('churn', lambda x: 1 - x.mean())  # 1 - churn_rate
        ).reset_index()
        genre_analysis['retention_impact'] *= 100

        fig = px.scatter(
            genre_analysis,
            x='popularity',
            y='retention_impact',
            size='popularity',
            color='main_genre_focus',
            hover_name='main_genre_focus',
            text='main_genre_focus',
            title="Content Strategy: Popularity vs. Retention Impact"
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(showlegend=False, yaxis_title="Retention Rate (%)", xaxis_title="Number of Viewers")
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def get_plan_revenue_plot(self, platform='All'):
        df = self.get_filtered_data(platform)
        df['estimated_revenue'] = df['monthly_price'] * df['tenure_months']
        fig = px.treemap(
            df,
            path=[px.Constant("All Plans"), 'plan_type'],
            values='estimated_revenue',
            color='churn',
            color_continuous_scale='RdYlGn_r',
            title='Revenue Contribution & Churn by Plan Type'
        )
        return fig.to_html(full_html=False, include_plotlyjs='cdn')