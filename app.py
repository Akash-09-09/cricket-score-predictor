import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="T20 Innings Score Predictor", 
    page_icon="🏏", 
    layout="wide"
)

# Custom premium styling (modern sports analytics branding)
st.markdown("""
<style>
    /* Gradient Title */
    .title-gradient {
        background: linear-gradient(90deg, #f43f5e, #fb7185);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    /* Increased subtitle size and contrast for light theme */
    .subtitle {
        color: #374151; /* Dark gray for high readability on white */
        font-size: 1.25rem;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    
    /* Styled Metric Card */
    .metric-card {
        background-color: #1f2937;
        border-radius: 12px;
        padding: 1.5rem 1rem;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #f43f5e;
        margin-top: 0.5rem;
    }
    
    /* Make metric title larger, bold, and high-contrast on dark card */
    .metric-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f3f4f6; /* Bright white-gray */
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Aggressive Sidebar overrides */
    section[data-testid="stSidebar"] label p, 
    section[data-testid="stSidebar"] label span,
    section[data-testid="stSidebar"] [class*="stWidgetLabel"] p {
        font-size: 1.25rem !important; /* Large, clear text */
        font-weight: 700 !important;   /* Bold headers */
        color: #0f172a !important;     /* Deep dark slate */
    }
    
    /* Text inside selectboxes and inputs */
    section[data-testid="stSidebar"] div[data-baseweb="select"] div,
    section[data-testid="stSidebar"] input {
        font-size: 1.15rem !important;
        font-weight: 500 !important;
        color: #0f172a !important;
    }
    
    /* Radio buttons text options */
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #1e293b !important;
    }

    /* Slider min/max and current value numbers */
    section[data-testid="stSidebar"] div[data-testid="stSlider"] div,
    section[data-testid="stSidebar"] div[class*="stSlider"] span {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    /* Footer caption overrides */
    .footer-caption {
        font-size: 1rem !important;
        color: #4b5563 !important;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD ARTIFACTS & CONFIGS
# ==========================================
# Load model and teams
model = joblib.load('cricket_model.pkl')
teams = joblib.load('teams.pkl')

# IPL Team Colors mapping for dynamic widget styling
TEAM_COLORS = {
    "Chennai Super Kings": {"bg": "#F5CD05", "border": "#ca8a04", "text": "#0f172a", "title": "#854d0e"}, # Official CSK Yellow
    "Mumbai Indians": {"bg": "#1e3a8a", "border": "#3b82f6", "text": "#ffffff", "title": "#93c5fd"}, # Blue
    "Royal Challengers Bangalore": {"bg": "#7f1d1d", "border": "#ef4444", "text": "#ffffff", "title": "#fca5a5"}, # Red
    "Royal Challengers Bengaluru": {"bg": "#7f1d1d", "border": "#ef4444", "text": "#ffffff", "title": "#fca5a5"},
    "Kolkata Knight Riders": {"bg": "#4c1d95", "border": "#8b5cf6", "text": "#ffffff", "title": "#ddd6fe"}, # Purple
    "Delhi Capitals": {"bg": "#1d4ed8", "border": "#3b82f6", "text": "#ffffff", "title": "#93c5fd"}, # Blue
    "Delhi Daredevils": {"bg": "#1d4ed8", "border": "#3b82f6", "text": "#ffffff", "title": "#93c5fd"},
    "Rajasthan Royals": {"bg": "#db2777", "border": "#f472b6", "text": "#ffffff", "title": "#fbcfe8"}, # Pink
    "Kings XI Punjab": {"bg": "#b91c1c", "border": "#f87171", "text": "#ffffff", "title": "#fca5a5"}, # Red
    "Punjab Kings": {"bg": "#b91c1c", "border": "#f87171", "text": "#ffffff", "title": "#fca5a5"},
    "Sunrisers Hyderabad": {"bg": "#c2410c", "border": "#f97316", "text": "#ffffff", "title": "#ffedd5"}, # Orange
    "Deccan Chargers": {"bg": "#0f172a", "border": "#475569", "text": "#ffffff", "title": "#94a3b8"}, # Navy/Grey
    "Gujarat Titans": {"bg": "#0b132b", "border": "#d97706", "text": "#ffffff", "title": "#fde68a"}, # Navy/Gold
    "Lucknow Super Giants": {"bg": "#155e75", "border": "#06b6d4", "text": "#ffffff", "title": "#cffafe"} # Cyan
}

def get_team_theme(team_name, default_type="batting"):
    # Clean matching in case of minor naming differences
    for name, theme in TEAM_COLORS.items():
        if name.lower() in team_name.lower():
            return theme
            
    # Default fallbacks
    if default_type == "batting":
        return {"bg": "#1e3a8a", "border": "#3b82f6", "text": "#ffffff", "title": "#93c5fd"}
    else:
        return {"bg": "#581c87", "border": "#a855f7", "text": "#ffffff", "title": "#e9d5ff"}

# ==========================================
# APP LAYOUT
# ==========================================
st.markdown('<div class="title-gradient">🏏 IPL Match Score Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time final score projection from current match state — powered by XGBoost & Random Forest</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar inputs ──
st.sidebar.header("⚙️ Match State Input")
match_type = st.sidebar.radio("Innings Type", ["1st Innings", "2nd Innings (Chase)"])
team1 = st.sidebar.selectbox("Team 1", teams, index=0) # Chennai Super Kings by default
team2 = st.sidebar.selectbox("Team 2", [t for t in teams if t != team1], index=0)
batting_first = st.sidebar.selectbox("Who Bats First?", [team1, team2])
bowling_first = team2 if batting_first == team1 else team1

# Flip teams automatically in the 2nd Innings
if match_type == "1st Innings":
    batting_team = batting_first
    bowling_team = bowling_first
else:
    batting_team = bowling_first
    bowling_team = batting_first

st.sidebar.markdown(f"🏏 **Batting Now:** {batting_team}")
st.sidebar.markdown(f"🥎 **Bowling Now:** {bowling_team}")
st.sidebar.markdown("---")

over = st.sidebar.slider("Overs Completed", 1, 19, 10)
cumulative_runs = st.sidebar.number_input("Runs Scored So Far", 0, 350, 80)
cumulative_wickets = st.sidebar.slider("Wickets Lost", 0, 9, 2)

target = None
if match_type == "2nd Innings (Chase)":
    target = st.sidebar.number_input("Target to Chase", 100, 350, 170)

# ── Derived features ──
current_run_rate = round(cumulative_runs / over, 2) if over > 0 else 0.0
overs_remaining = 20 - over
wickets_remaining = 10 - cumulative_wickets
rolling_run_rate = current_run_rate # approximation for live input
pressure_index = round((cumulative_wickets * over) / 20, 4)
runs_needed = (target - cumulative_runs) if target else None
required_run_rate = (round(runs_needed / overs_remaining, 2) if (target and overs_remaining > 0) else None)

# ── Predict ──
input_df = pd.DataFrame([{
    'over': over,
    'cumulative_runs': cumulative_runs,
    'cumulative_wickets': cumulative_wickets,
    'current_run_rate': current_run_rate,
    'overs_remaining': overs_remaining,
    'wickets_remaining': wickets_remaining,
    'rolling_run_rate': rolling_run_rate,
    'pressure_index': pressure_index,
    'batting_team': batting_team,
    'bowling_team': bowling_team
}])

predicted_score = int(model.predict(input_df)[0])
margin = 8
low, high = predicted_score - margin, predicted_score + margin

# ── Top metrics row ──
col1, col2, col3, col4 = st.columns(4)

if match_type == "1st Innings":
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">🎯 Predicted Final Score</div>'
            f'<div class="metric-value">{predicted_score}</div>'
            f'</div>', 
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">📊 Predicted Range</div>'
            f'<div class="metric-value">{low} – {high}</div>'
            f'</div>', 
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">⚡ Current Run Rate</div>'
            f'<div class="metric-value" style="color: #3b82f6;">{current_run_rate:.2f}</div>'
            f'</div>', 
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">🪣 Wickets Remaining</div>'
            f'<div class="metric-value" style="color: #f59e0b;">{wickets_remaining}</div>'
            f'</div>', 
            unsafe_allow_html=True
        )
else: # 2nd Innings (Chase)
    balls_remaining = 120 - (over * 6)
    
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">🎯 Runs Needed</div>'
            f'<div class="metric-value">{runs_needed}</div>'
            f'</div>', 
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">🥎 Balls Remaining</div>'
            f'<div class="metric-value" style="color: #f59e0b;">{balls_remaining}</div>'
            f'</div>', 
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">⚡ Current Run Rate</div>'
            f'<div class="metric-value" style="color: #3b82f6;">{current_run_rate:.2f}</div>'
            f'</div>', 
            unsafe_allow_html=True
        )

    with col4:
        if required_run_rate is not None:
            delta = round(required_run_rate - current_run_rate, 2)
            color = "#10b981" if delta <= 0 else "#ef4444" # Green if below CRR, Red if above
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-title">🏹 Required Run Rate</div>'
                f'<div class="metric-value" style="color: {color};">{required_run_rate:.2f}</div>'
                f'</div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-title">🏹 Required Run Rate</div>'
                f'<div class="metric-value" style="color: #ef4444;">N/A</div>'
                f'</div>', 
                unsafe_allow_html=True
            )

# ── Calculate Win Probability ──
if match_type == "1st Innings":
    # 1st Innings Win Probability: based on predicted score relative to average T20 target (165)
    bat_prob = 1 / (1 + np.exp(-(predicted_score - 165) / 15))
    bat_prob = round(float(bat_prob) * 100, 1)
    bowl_prob = round(100 - bat_prob, 1)
else:
    # 2nd Innings (Chase) Win Probability: based on predicted score relative to the target
    if cumulative_wickets >= 10:
        bat_prob = 0.0
        bowl_prob = 100.0
    elif cumulative_runs >= target:
        bat_prob = 100.0
        bowl_prob = 0.0
    else:
        bat_prob = 1 / (1 + np.exp(-(predicted_score - target) / 12))
        bat_prob = round(float(bat_prob) * 100, 1)
        bat_prob = max(1.0, min(99.0, bat_prob)) # boundary buffer
        bowl_prob = round(100 - bat_prob, 1)

# Look up team colors dynamically
bat_theme = get_team_theme(batting_team, "batting")
bowl_theme = get_team_theme(bowling_team, "bowling")

# Display Win Probability
st.markdown("---")
st.write("### 📊 Live Win Probability")
col_p1, col_p2 = st.columns(2)

with col_p1:
    st.markdown(
        f'<div class="metric-card" style="background-color: {bat_theme["bg"]}; border-color: {bat_theme["border"]}; padding: 1.25rem 1rem;">'
        f'<div class="metric-title" style="color: {bat_theme["title"]};">🏏 {batting_team} (Batting)</div>'
        f'<div class="metric-value" style="color: {bat_theme["text"]};">{bat_prob}%</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.progress(bat_prob / 100)

with col_p2:
    st.markdown(
        f'<div class="metric-card" style="background-color: {bowl_theme["bg"]}; border-color: {bowl_theme["border"]}; padding: 1.25rem 1rem;">'
        f'<div class="metric-title" style="color: {bowl_theme["title"]};">🥎 {bowling_team} (Bowling)</div>'
        f'<div class="metric-value" style="color: {bowl_theme["text"]};">{bowl_prob}%</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.progress(bowl_prob / 100)


# ── Chase result banner ──
if match_type == "2nd Innings (Chase)" and target:
    st.markdown("---")
    if predicted_score >= target:
        st.success(f"✅ **{batting_team}** is predicted to successfully chase! "
                   f"Projected to finish at **{predicted_score}** "
                   f"(Target: {target} | Win margin: ~{predicted_score - target} runs)")
    else:
        st.error(f"❌ **{batting_team}** is predicted to fall short by "
                 f"**{target - predicted_score} runs** "
                 f"(Projected: {predicted_score} | Target: {target})")

# ── Footer ──
st.markdown("---")
st.markdown('<div class="footer-caption">Built by Akash | Python · Scikit-Learn · XGBoost · Streamlit | Data: IPL Ball-by-Ball Dataset (Kaggle)</div>', unsafe_allow_html=True)