import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Set page config
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("style.css")
except FileNotFoundError:
    st.warning("style.css not found. Using default styles.")

# --- Data Loading & Preprocessing ---

@st.cache_data
def load_and_prep_data():
    files = [
        "head_to_head_French_Ligue_1.csv",
        "head_to_head_Italian_Serie_A.csv",
        "head_to_head_English_Premier_League.csv"
    ]
    
    dfs = []
    for f in files:
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))
        else:
            st.error(f"File not found: {f}")
            
    if not dfs:
        return None, None, None, None

    df = pd.concat(dfs, ignore_index=True)

    # Basic cleaning based on notebook logic
    # Fill missing rankings with median
    if 'homeClassment' in df.columns:
         df['homeClassment'] = df.groupby('strHomeTeam')['homeClassment'].transform(lambda x: x.fillna(x.median()))
         # Global fallback if team specific median is NaN
         df['homeClassment'] = df['homeClassment'].fillna(df['homeClassment'].median())
         
    if 'awayClassment' in df.columns:
         df['awayClassment'] = df.groupby('strAwayTeam')['awayClassment'].transform(lambda x: x.fillna(x.median()))
         # Global fallback if team specific median is NaN
         df['awayClassment'] = df['awayClassment'].fillna(df['awayClassment'].median())
    
    # Drop rows with missing essential values if any remain
    df = df.dropna(subset=['intHomeScore', 'intAwayScore', 'strHomeTeam', 'strAwayTeam'])

    # Feature Engineering / Selection
    # We need to ensure we have the columns used in the notebook
    # The notebook used: 'strLeague' ,'strHomeTeam', 'strAwayTeam', 'homeClassement', 'awayClassement'
    # Note: CSV headers might be 'homeClassment' or 'homeClassement' (spelling check)
    
    # Normalize column names if needed
    if 'homeClassment' in df.columns and 'homeClassement' not in df.columns:
        df.rename(columns={'homeClassment': 'homeClassement'}, inplace=True)
    if 'awayClassment' in df.columns and 'awayClassement' not in df.columns:
        df.rename(columns={'awayClassment': 'awayClassement'}, inplace=True)

    # Select features
    features = ['strLeague', 'strHomeTeam', 'strAwayTeam', 'homeClassement', 'awayClassement']
    targets = ['intHomeScore', 'intAwayScore']
    
    # Filter for existing columns
    features = [c for c in features if c in df.columns]
    
    if len(features) < 5:
        st.error("Missing required columns in dataset.")
        return None, None, None, None

    # Final check for NaNs in features
    X = df[features].fillna(0) # Last resort fill
    y = df[targets]
    
    return X, y, df, features

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# --- Model Training ---

@st.cache_resource
def train_models(X, y, features):
    from sklearn.impute import SimpleImputer
    
    # Identify categorical and numerical columns
    categorical_features = [col for col in features if col in ['strLeague', 'strHomeTeam', 'strAwayTeam']]
    numerical_features = [col for col in features if col in ['homeClassement', 'awayClassement']]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features)
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models
    models = {}
    
    # Linear Regression
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', MultiOutputRegressor(LinearRegression()))
    ])
    lr_pipeline.fit(X_train, y_train)
    models['Linear Regression'] = lr_pipeline

    # SVM (SVR)
    svr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)))
    ])
    svr_pipeline.fit(X_train, y_train)
    models['SVM'] = svr_pipeline

    # Random Forest
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))
    ])
    rf_pipeline.fit(X_train, y_train)
    models['Random Forest'] = rf_pipeline

    # Gradient Boosting
    gb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)))
    ])
    gb_pipeline.fit(X_train, y_train)
    models['Gradient Boosting'] = gb_pipeline

    return models, X_test, y_test

# --- Visualization Functions ---

def plot_goal_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['intHomeScore'], color='blue', label='Home Goals', kde=True, ax=ax, alpha=0.6)
    sns.histplot(df['intAwayScore'], color='red', label='Away Goals', kde=True, ax=ax, alpha=0.6)
    ax.set_title("Distribution of Home vs Away Goals")
    ax.legend()
    return fig

def plot_actual_vs_predicted(y_test, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Home Goals
    axes[0].scatter(y_test['intHomeScore'], y_pred[:, 0], alpha=0.5, color='blue')
    axes[0].plot([y_test['intHomeScore'].min(), y_test['intHomeScore'].max()], 
                 [y_test['intHomeScore'].min(), y_test['intHomeScore'].max()], 'k--', lw=2)
    axes[0].set_xlabel("Actual Home Goals")
    axes[0].set_ylabel("Predicted Home Goals")
    axes[0].set_title("Home Goals: Actual vs Predicted")
    
    # Away Goals
    axes[1].scatter(y_test['intAwayScore'], y_pred[:, 1], alpha=0.5, color='red')
    axes[1].plot([y_test['intAwayScore'].min(), y_test['intAwayScore'].max()], 
                 [y_test['intAwayScore'].min(), y_test['intAwayScore'].max()], 'k--', lw=2)
    axes[1].set_xlabel("Actual Away Goals")
    axes[1].set_ylabel("Predicted Away Goals")
    axes[1].set_title("Away Goals: Actual vs Predicted")
    
    return fig

def plot_correlation(df):
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
    ax.set_title("Feature Correlation Matrix")
    return fig

# --- Main App Layout ---

def main():
    st.title(" Football Match Predictor")
    st.markdown("Predict match scores using **Machine Learning** (Linear Regression & SVM).")

    X, y, full_df, feature_cols = load_and_prep_data()
    
    if X is not None:
        with st.spinner("Training models... This might take a moment."):
            models, X_test, y_test = train_models(X, y, feature_cols)
        
        # Sidebar
        st.sidebar.header("Configuration")
        model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))
        
        st.sidebar.divider()
        st.sidebar.header("Match Details")
        
        # Get unique values for dropdowns
        leagues = sorted(full_df['strLeague'].unique().astype(str))
        teams = sorted(set(full_df['strHomeTeam'].unique().astype(str)) | set(full_df['strAwayTeam'].unique().astype(str)))
        
        selected_league = st.sidebar.selectbox("League", leagues)
        
        # Filter teams by league
        league_teams = set(full_df[full_df['strLeague'] == selected_league]['strHomeTeam'].unique()) | \
                       set(full_df[full_df['strLeague'] == selected_league]['strAwayTeam'].unique())
        league_teams = sorted(list(league_teams))
        
        home_team = st.sidebar.selectbox("Home Team", league_teams)
        away_team = st.sidebar.selectbox("Away Team", [t for t in league_teams if t != home_team])
        
        # Rankings (Input manually or fetch median)
        st.sidebar.subheader("Team Stats (Current Season)")
        
        # Default to median ranking if available
        default_home_rank = int(full_df[full_df['strHomeTeam'] == home_team]['homeClassement'].median()) if not full_df[full_df['strHomeTeam'] == home_team].empty else 10
        default_away_rank = int(full_df[full_df['strAwayTeam'] == away_team]['awayClassement'].median()) if not full_df[full_df['strAwayTeam'] == away_team].empty else 10
        
        home_rank = st.sidebar.slider("Home Team Rank", 1, 20, default_home_rank)
        away_rank = st.sidebar.slider("Away Team Rank", 1, 20, default_away_rank)

        # Tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Evaluation"])

        with tab1:
            # Prediction Logic
            if st.button("Predict Score", type="primary"):
                # Create input dataframe
                input_data = pd.DataFrame({
                    'strLeague': [selected_league],
                    'strHomeTeam': [home_team],
                    'strAwayTeam': [away_team],
                    'homeClassement': [home_rank],
                    'awayClassement': [away_rank]
                })
                
                model = models[model_choice]
                prediction = model.predict(input_data)
                
                pred_home_score = max(0, round(prediction[0][0]))
                pred_away_score = max(0, round(prediction[0][1]))
                
                # Display Result
                st.markdown("### Prediction Result")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.markdown(f"<div style='text-align: center;'><h3>{home_team}</h3></div>", unsafe_allow_html=True)
                    st.metric("Home Score", int(pred_home_score))
                    
                with col2:
                    st.markdown("<div style='text-align: center; padding-top: 20px;'><h1>VS</h1></div>", unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"<div style='text-align: center;'><h3>{away_team}</h3></div>", unsafe_allow_html=True)
                    st.metric("Away Score", int(pred_away_score))
                
                st.success(f"Predicted Outcome: **{home_team} {int(pred_home_score)} - {int(pred_away_score)} {away_team}**")

        with tab2:
            st.header("Data Analysis")
            st.markdown("### Goal Distribution")
            st.pyplot(plot_goal_distribution(full_df))
            
            st.markdown("### Feature Correlations")
            st.pyplot(plot_correlation(full_df))
            
            st.markdown("### Dataset Statistics")
            st.write(full_df.describe())

        with tab3:
            st.header(f"Model Evaluation: {model_choice}")
            
            model = models[model_choice]
            y_pred_test = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
            col2.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
            col3.metric("R² Score", f"{r2:.2f}")
            
            st.markdown("### Actual vs Predicted Scores")
            st.pyplot(plot_actual_vs_predicted(y_test, y_pred_test))
            
            st.info("Note: The scatter plots show how well the model predicts goals. Points closer to the diagonal line indicate better predictions.")

    else:
        st.error("Could not load data. Please check if CSV files exist.")

if __name__ == "__main__":
    main()
