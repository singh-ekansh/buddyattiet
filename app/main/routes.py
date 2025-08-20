# app/main/routes.py
from flask import render_template, request, current_app
from . import bp
from app.services.analytics_service import AnalyticsService

# --- Initialization ---
# The service is now initialized on first request to ensure app context is available
analytics_service = None


def get_analytics_service():
    global analytics_service
    if analytics_service is None:
        # Get the DATA_PATH from the application's config
        data_path = current_app.config['DATA_PATH']
        analytics_service = AnalyticsService(data_path)
    return analytics_service


@bp.route('/')
def dashboard():
    service = get_analytics_service()
    selected_platform = request.args.get('platform', 'All')

    # ... rest of your dashboard route logic using 'service' ...
    # e.g., kpis = service.get_kpis(selected_platform)
    kpis = service.get_kpis(selected_platform)
    map_plot = service.get_churn_by_state_map(selected_platform)
    content_plot = service.get_content_strategy_plot(selected_platform)
    revenue_plot = service.get_plan_revenue_plot(selected_platform)

    sample_user_df = service.get_filtered_data(selected_platform).sample(1)
    sample_user = sample_user_df.drop(columns=['churn']).to_dict('records')[0]

    return render_template(
        'dashboard.html',
        platforms=['All', 'Disney+ Hotstar', 'Amazon Prime Video', 'Netflix', 'Zee5', 'SonyLIV'],
        selected_platform=selected_platform,
        kpis=kpis,
        map_plot=map_plot,
        content_plot=content_plot,
        revenue_plot=revenue_plot,
        sample_user=sample_user,
        form_options={
            'state': service.df['state'].unique().tolist(),
            'gender': ['Male', 'Female'],
            'plan_type': service.df['plan_type'].unique().tolist(),
            'main_genre_focus': service.df['main_genre_focus'].unique().tolist()
        }
    )