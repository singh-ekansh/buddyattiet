# app/data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_ott_data(num_customers=150000):
    print(f"Generating {num_customers} realistic records for 5 major Indian OTT platforms...")

    platforms = {
        'Disney+ Hotstar': {'share': 0.38, 'base_price': 299, 'focus': 'Sports', 'churn_factor': 1.5},
        'Amazon Prime Video': {'share': 0.22, 'base_price': 299, 'focus': 'International', 'churn_factor': 0.8},
        'Netflix': {'share': 0.20, 'base_price': 199, 'focus': 'International', 'churn_factor': 0.9},
        'Zee5': {'share': 0.10, 'base_price': 499, 'focus': 'Regional', 'churn_factor': 1.1},
        'SonyLIV': {'share': 0.10, 'base_price': 599, 'focus': 'Regional', 'churn_factor': 1.2}
    }

    states = ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh', 'West Bengal', 'Rajasthan', 'Gujarat',
              'Andhra Pradesh', 'Kerala']

    data = {'user_id': [], 'platform': [], 'state': [], 'age': [], 'gender': [], 'plan_type': [], 'monthly_price': [],
            'tenure_months': [], 'monthly_watch_hours': [], 'days_since_last_login': [], 'support_tickets': [],
            'main_genre_focus': [], 'watched_genres_count': [], 'churn': []}

    for platform, props in platforms.items():
        count = int(num_customers * props['share'])

        data['user_id'].extend([f'user_{platform.split()[0]}_{i}' for i in range(count)])
        data['platform'].extend([platform] * count)
        data['state'].extend(np.random.choice(states, size=count))
        data['age'].extend(np.random.randint(18, 50, size=count))
        data['gender'].extend(np.random.choice(['Male', 'Female'], p=[0.7, 0.3], size=count))

        # Platform-specific plan simulation
        if platform == 'Netflix':
            plans = ['Mobile', 'Basic', 'Standard', 'Premium']
            prices = [149, 199, 499, 649]
            plan_choices = np.random.choice(plans, p=[0.4, 0.3, 0.2, 0.1], size=count)
            price_map = dict(zip(plans, prices))
            data['plan_type'].extend(plan_choices)
            data['monthly_price'].extend([price_map[p] for p in plan_choices])
        else:  # Simplified for others
            plans = ['Annual', 'Monthly']
            annual_price = props['base_price']
            monthly_price = int(annual_price / 12 * 1.5)  # Monthly is more expensive
            plan_choices = np.random.choice(plans, p=[0.6, 0.4], size=count)
            data['plan_type'].extend(plan_choices)
            data['monthly_price'].extend([annual_price if p == 'Annual' else monthly_price for p in plan_choices])

        data['tenure_months'].extend(np.random.randint(1, 36, size=count))
        data['monthly_watch_hours'].extend(np.random.gamma(2, 15, size=count))
        data['days_since_last_login'].extend(np.random.randint(0, 120, size=count))
        data['support_tickets'].extend(np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05], size=count))

        # Content focus simulation
        focus_genre = props['focus']
        other_genres = ['Bollywood', 'Hollywood', 'Regional', 'Sports']
        genre_choices = np.random.choice([focus_genre] + other_genres, p=[0.6] + [0.1] * 4, size=count)
        data['main_genre_focus'].extend(genre_choices)
        data['watched_genres_count'].extend(np.random.randint(1, 5, size=count))

        # --- Platform-Specific Churn Logic ---
        # Convert recent data additions to NumPy arrays for calculations
        days_since_login_np = np.array(data['days_since_last_login'][-count:])
        tenure_months_np = np.array(data['tenure_months'][-count:])
        support_tickets_np = np.array(data['support_tickets'][-count:])
        main_genre_focus_np = np.array(data['main_genre_focus'][-count:])
        watched_genres_count_np = np.array(data['watched_genres_count'][-count:])

        base_churn_score = np.zeros(count)
        base_churn_score += (days_since_login_np / 120) * 0.5
        base_churn_score -= (tenure_months_np / 36) * 0.2
        base_churn_score += (support_tickets_np / 3) * 0.2

        if platform == 'Disney+ Hotstar':
            # High churn for sports-only watchers
            # Use NumPy's logical AND (&) for element-wise comparison
            base_churn_score += ((main_genre_focus_np == 'Sports') & (watched_genres_count_np == 1)) * 0.4

        churn_prob = 1 / (1 + np.exp(-10 * (base_churn_score - 0.4) * props['churn_factor']))
        data['churn'].extend((np.random.rand(count) < churn_prob).astype(int))

    return pd.DataFrame(data)