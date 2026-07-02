import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import joblib

# 1. Load Data
df = pd.read_csv('ball_by_ball_ipl.csv')
print("Columns found in CSV:", df.columns.tolist())

# 2. Dynamic Column Detection
# This mapping logic finds the correct names automatically
def find_col(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

col_inning = find_col(['Innings', 'inning', 'inn', 'match_inning'])
col_match = find_col(['Match ID', 'match_id', 'id'])
col_bat = find_col(['Bat First', 'batting_team', 'batting_side'])
col_bowl = find_col(['Bat Second', 'bowling_team', 'bowling_side'])
col_over = find_col(['Over', 'over'])
col_bat_runs = find_col(['Batter Runs', 'batsman_runs', 'runs_off_bat'])
col_extra = find_col(['Extra Runs', 'extra_runs', 'extras'])
col_wicket = find_col(['Wicket', 'wicket', 'dismissal_kind', 'player_dismissed'])

# 3. Clean Data
df = df[df[col_inning] == 1].copy()
df['runs_off_bat'] = pd.to_numeric(df[col_bat_runs], errors='coerce').fillna(0)
df['extras'] = pd.to_numeric(df[col_extra], errors='coerce').fillna(0)
df['total_runs'] = df['runs_off_bat'] + df['extras']
df['is_wicket'] = df[col_wicket].notnull().astype(int)

# 4. Engineering Features
over_df = df.groupby([col_match, col_bat, col_bowl, col_over]).agg(
    runs_in_over=('total_runs', 'sum'),
    wickets_in_over=('is_wicket', 'sum')
).reset_index()

# Standardize names for the model
over_df = over_df.rename(columns={
    col_match: 'match_id',
    col_bat: 'batting_team',
    col_bowl: 'bowling_team',
    col_over: 'over'
})

over_df = over_df.sort_values(['match_id', 'over'])
over_df['cumulative_runs'] = over_df.groupby('match_id')['runs_in_over'].cumsum()
over_df['cumulative_wickets'] = over_df.groupby('match_id')['wickets_in_over'].cumsum()
over_df['current_run_rate'] = over_df['cumulative_runs'] / (over_df['over'] + 1)
over_df['overs_remaining'] = 20 - (over_df['over'] + 1)
over_df['wickets_remaining'] = 10 - over_df['cumulative_wickets']
over_df['rolling_run_rate'] = over_df.groupby('match_id')['runs_in_over'].transform(lambda x: x.rolling(3, min_periods=1).mean())
over_df['pressure_index'] = (over_df['cumulative_wickets'] * (over_df['over'] + 1) / 20)

# Merge Target
final_scores = df.groupby(col_match)['total_runs'].sum().reset_index()
final_scores.columns = ['match_id', 'final_score']
over_df = over_df.merge(final_scores, on='match_id')
over_df = over_df[over_df['overs_remaining'] > 0]

# 5. Build and Save Model
features = ['over', 'cumulative_runs', 'cumulative_wickets', 'current_run_rate', 
            'overs_remaining', 'wickets_remaining', 'rolling_run_rate', 
            'pressure_index', 'batting_team', 'bowling_team']

X = over_df[features]
y = over_df['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['batting_team', 'bowling_team']),
    ('num', 'passthrough', [f for f in features if f not in ['batting_team', 'bowling_team']])
])

rf_pipeline = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=200, random_state=42))])
xgb_pipeline = Pipeline([('preprocessor', preprocessor), ('model', xgb.XGBRegressor(n_estimators=200, learning_rate=0.05))])

rf_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train)

best_model = rf_pipeline if r2_score(y_test, rf_pipeline.predict(X_test)) > r2_score(y_test, xgb_pipeline.predict(X_test)) else xgb_pipeline

joblib.dump(best_model, 'cricket_model.pkl')
joblib.dump(sorted(over_df['batting_team'].unique().tolist()), 'teams.pkl')
print("Success! Model saved.")