import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------
# Page Configuration & Aesthetics
# ----------------------------------------------------
st.set_page_config(
    page_title="Spotify UK Chart Dynamics",
    page_icon="https://www.scdn.co/i/_global/twitter_card.jpg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Spotify-inspired Dark Mode CSS (Modernized Premium Gradient and Glassmorphism by Antigravity)
st.markdown("""
    <style>
    /* Main Background and Text with deep premium dark-mode gradient */
    .stApp {
        background-color: #0d0e12;
        background-image: 
            radial-gradient(at 0% 0%, rgba(29, 185, 84, 0.08) 0px, transparent 50%),
            radial-gradient(at 50% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.05) 0px, transparent 50%);
        background-attachment: fixed;
        color: #f3f4f6;
    }
    
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sidebar Styling - Sleek semi-transparent dark pane */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 11, 15, 0.9) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Custom KPI Container */
    .kpi-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-bottom: 30px;
    }
    
    /* Glassmorphic KPI Cards with custom glowing top borders */
    .kpi-card {
        background: rgba(22, 24, 33, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        flex: 1;
        min-width: 220px;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #1db954, #6366f1);
        opacity: 0.8;
    }
    
    /* Unique colored accent lines for each KPI card */
    .kpi-card-0::before { background: linear-gradient(90deg, #8b5cf6, #ec4899); }
    .kpi-card-1::before { background: linear-gradient(90deg, #06b6d4, #3b82f6); }
    .kpi-card-2::before { background: linear-gradient(90deg, #10b981, #14b8a6); }
    .kpi-card-3::before { background: linear-gradient(90deg, #f59e0b, #ef4444); }
    .kpi-card-4::before { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    }
    
    /* Custom colored hover glows for cards */
    .kpi-card-0:hover { box-shadow: 0 10px 25px rgba(139, 92, 246, 0.15); }
    .kpi-card-1:hover { box-shadow: 0 10px 25px rgba(6, 182, 212, 0.15); }
    .kpi-card-2:hover { box-shadow: 0 10px 25px rgba(16, 185, 129, 0.15); }
    .kpi-card-3:hover { box-shadow: 0 10px 25px rgba(245, 158, 11, 0.15); }
    .kpi-card-4:hover { box-shadow: 0 10px 25px rgba(99, 102, 241, 0.15); }
    
    .kpi-title {
        color: #9ca3af;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #ffffff;
    }
    
    /* High-contrast metrics gradients for readability and premium look */
    .kpi-card-0 .kpi-value { background: linear-gradient(120deg, #c084fc, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .kpi-card-1 .kpi-value { background: linear-gradient(120deg, #22d3ee, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .kpi-card-2 .kpi-value { background: linear-gradient(120deg, #34d399, #2dd4bf); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .kpi-card-3 .kpi-value { background: linear-gradient(120deg, #fbbf24, #f87171); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .kpi-card-4 .kpi-value { background: linear-gradient(120deg, #818cf8, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    
    .kpi-desc {
        color: #9ca3af;
        font-size: 12px;
        line-height: 1.4;
    }
    
    /* Track Row Styling */
    .track-row {
        background-color: rgba(22, 24, 33, 0.6);
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        display: flex;
        align-items: center;
        transition: all 0.2s ease;
    }
    .track-row:hover {
        background-color: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Tabs Customization - Glass and High-Contrast Accent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(22, 24, 33, 0.8);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #9ca3af;
        font-weight: 600;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1db954, #1ed760) !important;
        color: #000000 !important;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #1db954, #1ed760);
        color: #000000;
        border-radius: 24px;
        border: none;
        padding: 10px 28px;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.2);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .stButton>button:hover {
        color: #000000;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(29, 185, 84, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Data Loading & Processing (Cached)
# ----------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    csv_path = "Atlantic_United_Kingdom.csv"
        
    df = pd.read_csv(csv_path, encoding="latin1")
    
    # Standard clean-up
    df['artist'] = df['artist'].astype(str).str.strip()
    df['song'] = df['song'].astype(str).str.strip()
    
    # Resolve Daily Top 50 Integrity (keep first 50 rows per day to prune anomalies)
    df_clean = df.groupby('date').head(50).reset_index(drop=True)
    df_clean['parsed_date'] = pd.to_datetime(df_clean['date'], format='%d-%m-%Y', errors='coerce')
    df_clean = df_clean.sort_values('parsed_date').reset_index(drop=True)
    
    # Artist Splitting logic, supporting known compound names
    def split_artists(artist_str):
        known_groups = [
            'Chase & Status', 
            'Richy Mitch & The Coal Miners', 
            'Earth, Wind & Fire', 
            'Bobby "Boris" Pickett & The Crypt-Kickers'
        ]
        temp = artist_str
        for group in known_groups:
            temp = temp.replace(group, group.replace(' & ', ' _AND_ '))
        
        parts = [p.strip() for p in temp.split('&')]
        
        restored_parts = []
        for p in parts:
            restored_p = p
            for group in known_groups:
                restored_p = restored_p.replace(group.replace(' & ', ' _AND_ '), group)
            restored_parts.append(restored_p)
        return restored_parts

    df_clean['artist_list'] = df_clean['artist'].apply(split_artists)
    df_clean['is_collaboration'] = df_clean['artist_list'].apply(lambda x: len(x) > 1)
    df_clean['num_artists'] = df_clean['artist_list'].apply(len)
    
    # Explicitness boolean mapping
    df_clean['is_explicit_bool'] = df_clean['is_explicit'].astype(str).str.upper().str.strip() == 'TRUE'
    
    # Track Duration conversion
    df_clean['duration_min'] = df_clean['duration_ms'] / 60000
    df_clean['duration_type'] = pd.cut(
        df_clean['duration_min'], 
        bins=[0, 2.5, 3.5, 100], 
        labels=['Short-form (<2.5m)', 'Standard (2.5-3.5m)', 'Long-form (>3.5m)']
    )
    
    # Categorization buckets
    df_clean['popularity_bucket'] = pd.cut(
        df_clean['popularity'], 
        bins=[0, 70, 80, 90, 101], 
        labels=['<70', '70-79', '80-89', '90+'],
        right=False
    )
    
    df_clean['rank_bucket'] = pd.cut(
        df_clean['position'], 
        bins=[0, 10, 20, 30, 40, 50], 
        labels=['Top 10', '11-20', '21-30', '31-40', '41-50']
    )
    
    # Define rank_group for collaboration analysis
    df_clean['rank_group'] = np.where(df_clean['position'] <= 10, 'Top 10', 'Ranks 11-50')
    
    return df_clean

df_all = load_and_preprocess_data()

# ----------------------------------------------------
# Machine Learning Model Pipeline (Cached)
# ----------------------------------------------------
@st.cache_resource
def train_machine_learning_pipeline(df_clean):
    # Compute artist appearances for artist power
    df_exploded = df_clean.explode('artist_list')
    artist_appearances = df_exploded['artist_list'].value_counts().to_dict()
    
    # Group by song and artist to get song-level records
    song_groups = df_clean.groupby(['song', 'artist'])
    song_data = []
    
    for (song, artist), group in song_groups:
        artists_list = group['artist_list'].iloc[0]
        artist_power = max([artist_appearances.get(a, 0) for a in artists_list])
        
        song_data.append({
            'song': song,
            'artist': artist,
            'max_popularity': group['popularity'].max(),
            'best_position': group['position'].min(),
            'avg_position': group['position'].mean(),
            'days_charted': len(group),
            'duration_min': group['duration_min'].mean(),
            'is_explicit_bool': int(group['is_explicit_bool'].max()),
            'is_collaboration': int(group['is_collaboration'].max()),
            'total_tracks': group['total_tracks'].mean(),
            'album_type': group['album_type'].iloc[0],
            'artist_power': artist_power,
            'reached_top_10': int(group['position'].min() <= 10)
        })
    
    df_songs = pd.DataFrame(song_data)
    
    # Album type mapping
    album_type_map = {'single': 1, 'album': 2, 'compilation': 3}
    df_songs['album_type_encoded'] = df_songs['album_type'].map(album_type_map).fillna(0)
    
    # Features and targets
    features = ['duration_min', 'total_tracks', 'is_explicit_bool', 'is_collaboration', 'artist_power', 'album_type_encoded']
    X = df_songs[features].fillna(0)
    y_class = df_songs['reached_top_10']
    y_reg = df_songs['max_popularity']
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X, y_class)
    
    # Train regressor
    reg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    reg.fit(X, y_reg)
    
    # Clustering
    scaler = StandardScaler()
    cluster_features = ['avg_position', 'max_popularity', 'days_charted']
    X_cluster = scaler.fit_transform(df_songs[cluster_features])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_songs['cluster'] = kmeans.fit_predict(X_cluster)
    
    # Map clusters stably based on chart longevity
    cluster_means = df_songs.groupby('cluster')['days_charted'].mean().sort_values(ascending=False)
    cluster_mapping = {}
    labels = ["Mega Blockbuster Hits", "Steady Catalog Performers", "Short-Run Album Tracks", "Transient / Anomalous Entries"]
    for i, (cluster_idx, _) in enumerate(cluster_means.items()):
        cluster_mapping[cluster_idx] = labels[i]
    df_songs['performance_profile'] = df_songs['cluster'].map(cluster_mapping)
    
    # Train a cluster predictor (predicts profile from metadata)
    cluster_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    cluster_clf.fit(X, df_songs['performance_profile'])
    
    return clf, reg, cluster_clf, df_songs, features, artist_appearances

# Train ML components
clf_model, reg_model, cluster_clf, df_songs, ml_features, artist_appearance_dict = train_machine_learning_pipeline(df_all)

# Compute the global unique artist list for the dropdown filter
df_exploded_all = df_all.explode('artist_list')
all_artists = ["All Artists"] + sorted(list(df_exploded_all['artist_list'].dropna().unique()))

# ----------------------------------------------------
# Sidebar Controls & Filtering
# ----------------------------------------------------
st.sidebar.markdown(
    "<h1 style='text-align: center; color: #1db954; font-size: 26px; margin-bottom: 20px;'>🎵 Spotify UK Insights</h1>", 
    unsafe_allow_html=True
)

st.sidebar.markdown("### 📅 Date Range Selector")
min_date = df_all['parsed_date'].min().date()
max_date = df_all['parsed_date'].max().date()

start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

# Check for valid date range input
if start_date > end_date:
    st.sidebar.error("Error: Start Date must be before or equal to End Date.")
    st.stop()

# Filtering dataset based on selected dates
df_filtered = df_all[
    (df_all['parsed_date'].dt.date >= start_date) & 
    (df_all['parsed_date'].dt.date <= end_date)
].copy()

if df_filtered.empty:
    st.warning("No data found for the selected date range.")
    st.stop()

# Search Box for filtering
st.sidebar.markdown("### 🔍 Filter by Artist and Song")
search_artist = st.sidebar.selectbox("Select an Artist to Explore:", all_artists)

# Dynamically filter song list based on selected artist
if search_artist != "All Artists":
    df_artist_subset = df_all[df_all['artist_list'].apply(lambda x: search_artist in x)]
    available_songs = ["All Songs"] + sorted(list(df_artist_subset['song'].dropna().unique()))
    df_filtered = df_filtered[df_filtered['artist_list'].apply(lambda x: search_artist in x)]
else:
    available_songs = ["All Songs"] + sorted(list(df_all['song'].dropna().unique()))

search_song = st.sidebar.selectbox("Select a Song to Explore:", available_songs)

if search_song != "All Songs":
    df_filtered = df_filtered[df_filtered['song'] == search_song]




# ----------------------------------------------------
# Key Statistics & KPI Calculations
# ----------------------------------------------------
df_exploded = df_filtered.explode('artist_list')
artist_appearances = df_exploded['artist_list'].value_counts()
num_unique_artists = len(artist_appearances)
total_appearances = artist_appearances.sum()

# Artist Concentration Metrics
top_5_appearances = artist_appearances.head(5).sum()
top_10_appearances = artist_appearances.head(10).sum()
top_5_share = top_5_appearances / total_appearances if total_appearances > 0 else 0
top_10_share = top_10_appearances / total_appearances if total_appearances > 0 else 0

# HHI (Herfindahl-Hirschman Index)
artist_shares = artist_appearances / total_appearances if total_appearances > 0 else pd.Series(dtype=float)
hhi = (artist_shares ** 2).sum() if not artist_shares.empty else 0

# Shannon Entropy & Pielou's Evenness Index (Content Variety Index)
shannon_entropy = - (artist_shares * np.log(artist_shares)).sum() if not artist_shares.empty else 0
pielou_evenness = shannon_entropy / np.log(num_unique_artists) if num_unique_artists > 1 else 0

# Collaborations
overall_collab_ratio = df_filtered['is_collaboration'].mean() if len(df_filtered) > 0 else 0

# Release Formats
album_type_shares = df_filtered['album_type'].value_counts(normalize=True)
single_pct = album_type_shares.get('single', 0)
album_pct = album_type_shares.get('album', 0)

# Content Explicitness
explicit_ratio = df_filtered['is_explicit_bool'].mean() if len(df_filtered) > 0 else 0

# ----------------------------------------------------
# Header Section
# ----------------------------------------------------
st.markdown(
    f"<h1 style='margin-bottom: 5px; font-weight: 700;'>UK Spotify Chart Dynamics</h1>"
    f"<p style='color: #a7a7a7; font-size: 14px;'>Analyzing the UK Top 50 Spotify charts from "
    f"<strong>{start_date.strftime('%B %d, %Y')}</strong> to <strong>{end_date.strftime('%B %d, %Y')}</strong></p>", 
    unsafe_allow_html=True
)

# ----------------------------------------------------
# Main View - Tab System
# ----------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Dashboard Overview", 
    "🎤 Artist Dominance", 
    "🤝 Collaboration Network", 
    "📦 Album & Format Analysis", 
    "🔞 Content Explicitness",
    "📅 Daily Chart Browser",
    "🔮 Machine Learning Insights"
])

# ----------------------------------------------------
# Tab 1: Dashboard Overview & KPIs
# ----------------------------------------------------
with tab1:
    st.markdown("### 📊 Market Structure Key Performance Indicators")
    
    # HTML formatted KPI cards
    kpi_cols = st.columns(5)
    
    with kpi_cols[0]:
        st.markdown(f"""
            <div class="kpi-card kpi-card-0">
                <div class="kpi-title">Artist Concentration</div>
                <div class="kpi-value">{top_10_share * 100:.2f}%</div>
                <div class="kpi-desc">Share of chart appearances dominated by the Top 10 artists (Concentration Index)</div>
            </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[1]:
        st.markdown(f"""
            <div class="kpi-card kpi-card-1">
                <div class="kpi-title">Unique Artists</div>
                <div class="kpi-value">{num_unique_artists}</div>
                <div class="kpi-desc">Total unique artists making appearances on the UK Top 50 charts in this timeframe</div>
            </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[2]:
        st.markdown(f"""
            <div class="kpi-card kpi-card-2">
                <div class="kpi-title">Collaboration Ratio</div>
                <div class="kpi-value">{overall_collab_ratio * 100:.2f}%</div>
                <div class="kpi-desc">Percentage of collaborative tracks (multiple credited artists) appearing on the chart</div>
            </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[3]:
        st.markdown(f"""
            <div class="kpi-card kpi-card-3">
                <div class="kpi-title">Singles / Albums</div>
                <div class="kpi-value">{single_pct*100:.1f}% / {album_pct*100:.1f}%</div>
                <div class="kpi-desc">Proportion of chart appearances represented by singles versus album tracks</div>
            </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[4]:
        st.markdown(f"""
            <div class="kpi-card kpi-card-4">
                <div class="kpi-title">Variety Index (Evenness)</div>
                <div class="kpi-value">{pielou_evenness:.4f}</div>
                <div class="kpi-desc">Pielou's Evenness of artist shares (0 = monopoly, 1 = perfectly equal distribution)</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
    
    col_chart_left, col_chart_right = st.columns(2)
    
    with col_chart_left:
        st.markdown("#### 🏆 Top 10 Artists by Chart Occurrences")
        top_artists_df = artist_appearances.head(10).reset_index()
        top_artists_df.columns = ['Artist', 'Appearances (Days)']
        
        # Sleek Seaborn Plot
        fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0d0e12')
        ax.set_facecolor('#0d0e12')
        sns.barplot(
            data=top_artists_df, 
            x='Appearances (Days)', 
            y='Artist', 
            palette="viridis",
            ax=ax
        )
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.spines['bottom'].set_color('#282828')
        ax.spines['left'].set_color('#282828')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.title("Most Represented Artists in UK Charts", color='white', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_chart_right:
        st.markdown("#### 🎵 Top Charted Songs in Selected Period")
        # Find songs with most occurrences
        top_songs = df_filtered.groupby(['song', 'artist'])['position'].count().reset_index()
        top_songs.columns = ['Song', 'Artist', 'Days on Chart']
        top_songs = top_songs.sort_values(by='Days on Chart', ascending=False).head(10)
        
        # Display as a styled dataframe
        st.dataframe(
            top_songs,
            column_config={
                "Song": st.column_config.TextColumn("Track Title"),
                "Artist": st.column_config.TextColumn("Artist"),
                "Days on Chart": st.column_config.NumberColumn("Days on Chart", format="%d 🔥")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Add basic trend indicators
        avg_pop_over_time = df_filtered.groupby('parsed_date')['popularity'].mean().reset_index()
        fig_trend, ax_trend = plt.subplots(figsize=(10, 3.2), facecolor='#0d0e12')
        ax_trend.set_facecolor('#0d0e12')
        sns.lineplot(
            data=avg_pop_over_time, 
            x='parsed_date', 
            y='popularity', 
            color='#1db954',
            linewidth=2,
            ax=ax_trend
        )
        ax_trend.tick_params(colors='white')
        ax_trend.xaxis.label.set_color('white')
        ax_trend.yaxis.label.set_color('white')
        ax_trend.spines['bottom'].set_color('#282828')
        ax_trend.spines['left'].set_color('#282828')
        ax_trend.spines['top'].set_visible(False)
        ax_trend.spines['right'].set_visible(False)
        plt.title("Average Streaming Popularity Trend Over Time", color='white', fontsize=12, fontweight='bold', pad=10)
        plt.xlabel("Date", color='white')
        plt.ylabel("Popularity", color='white')
        plt.tight_layout()
        st.pyplot(fig_trend)

# ----------------------------------------------------
# Tab 2: Artist Dominance & Concentration Detail
# ----------------------------------------------------
with tab2:
    st.markdown("### 🎤 Artist Dominance & Market Share Analysis")
    
    col_dom_left, col_dom_right = st.columns([3, 2])
    
    with col_dom_left:
        st.markdown("#### Market Concentration Context")
        st.write(
            f"The Herfindahl-Hirschman Index (HHI) for the current selection is **{hhi:.6f}** "
            f"(or **{hhi*10000:.2f}** on the standard 0-10000 scale). "
            f"According to antitrust standards, an HHI below 0.015 (150 scale) represents a "
            f"highly competitive and diversified market. Here, the artist market shares show "
            f"competitive diversity outside of a few hyper-frequent performers."
        )
        
        st.write(
            f"The **Content Variety Index (Pielou's Evenness)** is **{pielou_evenness:.4f}**, "
            f"which signifies that the distribution of artist appearances is relatively balanced. "
            f"While chart-toppers receive substantial streaming real-estate, a wide cohort of artists "
            f"populate the remaining positions on a daily basis."
        )
        
        # Explanatory metric table
        st.markdown("#### Concentration Thresholds")
        st.table(pd.DataFrame({
            "Metric": ["Top 5 Share", "Top 10 Share (Concentration Index)", "HHI (0-10000 scale)", "Pielou's Evenness"],
            "Value": [f"{top_5_share * 100:.2f}%", f"{top_10_share * 100:.2f}%", f"{hhi*10000:.2f}", f"{pielou_evenness:.4f}"],
            "Interpretation": [
                "Dominance of the absolute elite five artists",
                "Dominance of the top 10 artists (moderate at 15-25%)",
                "Market concentration (<150 indicates highly competitive, unconcentrated)",
                "Closeness of distribution to perfect equality (1.0 = equal shares)"
            ]
        }))
        
    with col_dom_right:
        st.markdown("#### 🔍 Deep-Dive into Individual Artist Trajectory")
        all_artists = sorted(list(artist_appearances.index))
        selected_artist = st.selectbox("Select an Artist to Explore:", all_artists, index=all_artists.index("Taylor Swift") if "Taylor Swift" in all_artists else 0)
        
        # Retrieve stats for selected artist
        artist_data = df_filtered[df_filtered['artist_list'].apply(lambda x: selected_artist in x)]
        
        if not artist_data.empty:
            total_days = artist_data['date'].nunique()
            best_rank = artist_data['position'].min()
            avg_rank = artist_data['position'].mean()
            unique_songs = artist_data['song'].nunique()
            
            st.markdown(f"**Selected Artist Summary:** `{selected_artist}`")
            col_art_kpi1, col_art_kpi2 = st.columns(2)
            col_art_kpi1.metric("Chart Appearances (Days)", total_days)
            col_art_kpi1.metric("Unique Charted Songs", unique_songs)
            col_art_kpi2.metric("Best Position Achieved", f"#{best_rank}")
            col_art_kpi2.metric("Average Chart Position", f"{avg_rank:.1f}")
            
            # List top songs for selected artist
            st.markdown("**Charted Tracks & Occurrences:**")
            artist_songs = artist_data.groupby('song')['position'].agg(['count', 'min', 'mean']).reset_index()
            artist_songs.columns = ['Song Title', 'Chart Days', 'Best Rank', 'Avg Rank']
            st.dataframe(artist_songs.sort_values(by='Chart Days', ascending=False), use_container_width=True, hide_index=True)
        else:
            st.write("Selected artist has no entries matching the current filter filters.")

# ----------------------------------------------------
# Tab 3: Collaboration Structure
# ----------------------------------------------------
with tab3:
    st.markdown("### 🤝 Collaboration Prevalence & Artist Networks")
    
    col_collab_left, col_collab_right = st.columns([1, 1])
    
    with col_collab_left:
        st.markdown("#### Partnership & Team-up Statistics")
        st.write(
            f"Collaborations represent **{overall_collab_ratio * 100:.2f}%** of all chart placements. "
            f"When artist team-ups occur, the tracks feature an average of **{df_filtered[df_filtered['is_collaboration']]['num_artists'].mean():.2f}** "
            f"artists. The overall average of collaborators across all entries stands at **{df_filtered['num_artists'].mean():.2f}**."
        )
        
        # Collab ratio by Rank group
        st.markdown("#### Collaboration Rates in Chart Segments")
        collab_by_rank = df_filtered.groupby('rank_group', observed=False)['is_collaboration'].mean().reset_index()
        collab_by_rank.columns = ['Chart Segment', 'Collaboration Ratio']
        collab_by_rank['Collaboration Ratio'] = collab_by_rank['Collaboration Ratio'].apply(lambda x: f"{x * 100:.2f}%")
        st.table(collab_by_rank)
        
        st.markdown("""
            > **Market Dynamics Insight**:
            > Collaboration rates remain remarkably consistent across the charts (spanning from the elite Top 10 down to ranks 11-50). 
            > Cross-artist features are not merely used as temporary popularity hacks to secure the absolute top ranks, 
            > but represent an industry-wide release standard across all chart segments.
        """)
        
    with col_collab_right:
        st.markdown("#### 🕸️ Artist Collaboration Network Graph")
        
        # Build undirected graph of collaborations
        G = nx.Graph()
        for artists in df_filtered['artist_list']:
            if len(artists) > 1:
                for i in range(len(artists)):
                    for j in range(i + 1, len(artists)):
                        a1, a2 = artists[i], artists[j]
                        if G.has_edge(a1, a2):
                            G[a1][a2]['weight'] += 1
                        else:
                            G.add_edge(a1, a2, weight=1)
                            
        # Filter edges to keep only significant partnerships for clarity
        edge_threshold = st.slider("Select Minimum Collaboration Days (Threshold):", min_value=1, max_value=50, value=20)
        
        significant_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= edge_threshold]
        subG = G.edge_subgraph(significant_edges).copy()
        subG.remove_nodes_from(list(nx.isolates(subG)))
        
        if subG.number_of_nodes() > 0:
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0d0e12')
            ax.set_facecolor('#0d0e12')
            pos = nx.spring_layout(subG, k=0.5, seed=42)
            
            # Node degrees for sizing
            degrees = dict(subG.degree())
            node_sizes = [v * 120 for v in degrees.values()]
            
            # Edge weights for width
            edge_widths = [subG[u][v]['weight'] / 10 for u, v in subG.edges()]
            
            # Draw network elements styled for dark mode
            nx.draw_networkx_nodes(
                subG, pos, ax=ax,
                node_size=node_sizes, 
                node_color='#1db954', 
                edgecolors='white', 
                linewidths=1.0
            )
            nx.draw_networkx_edges(
                subG, pos, ax=ax,
                width=edge_widths, 
                edge_color='#555555', 
                alpha=0.6
            )
            nx.draw_networkx_labels(
                subG, pos, ax=ax,
                font_size=8, 
                font_family='sans-serif', 
                font_weight='bold',
                font_color='white'
            )
            
            plt.title(f"UK Spotify Top 50 Collaboration Network (Min. {edge_threshold} Days)", color='white', fontsize=12, pad=15)
            plt.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info(f"No collaborations found with a strength of at least {edge_threshold} days in this timeframe. Try lowering the threshold slider.")

# ----------------------------------------------------
# Tab 4: Release Strategy & Format Analysis
# ----------------------------------------------------
with tab4:
    st.markdown("### 📦 Album Structure & Track Formats")
    
    col_fmt_left, col_fmt_right = st.columns(2)
    
    with col_fmt_left:
        st.markdown("#### Single vs Album Format Presence")
        # Plot release type share
        fig_pie, ax_pie = plt.subplots(figsize=(8, 6), facecolor='#0d0e12')
        ax_pie.set_facecolor('#0d0e12')
        
        # Colors: Green, Blue, Crimson
        pie_colors = ['#1db954', '#4c72b0', '#c44e52']
        
        ax_pie.pie(
            album_type_shares.values, 
            labels=album_type_shares.index, 
            autopct='%1.1f%%', 
            colors=pie_colors[:len(album_type_shares)],
            startangle=140, 
            textprops={'color': 'white', 'fontsize': 12},
            explode=[0.02] * len(album_type_shares)
        )
        plt.title("Distribution of Release Formats (Entries)", color='white', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig_pie)
        
        correlation = df_filtered[['total_tracks', 'position', 'popularity']].corr()
        st.markdown("#### Correlation Matrix")
        st.write(
            "Below is the correlation between the size of the album (`total_tracks`), "
            "the chart position, and overall track popularity."
        )
        st.dataframe(correlation.style.background_gradient(cmap='Greens', axis=None), use_container_width=True)
        
    with col_fmt_right:
        st.markdown("#### Track Duration Distribution")
        
        fig_dur, ax_dur = plt.subplots(figsize=(10, 5.5), facecolor='#0d0e12')
        ax_dur.set_facecolor('#0d0e12')
        
        sns.histplot(
            data=df_filtered, 
            x='duration_min', 
            kde=True, 
            color='teal',
            ax=ax_dur
        )
        mean_dur = df_filtered['duration_min'].mean()
        ax_dur.axvline(mean_dur, color='red', linestyle='--', label=f"Mean: {mean_dur:.2f}m")
        ax_dur.tick_params(colors='white')
        ax_dur.xaxis.label.set_color('white')
        ax_dur.yaxis.label.set_color('white')
        ax_dur.spines['bottom'].set_color('#282828')
        ax_dur.spines['left'].set_color('#282828')
        ax_dur.spines['top'].set_visible(False)
        ax_dur.spines['right'].set_visible(False)
        ax_dur.legend()
        plt.title("Track Duration Distribution (UK Top 50)", color='white', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Duration (Minutes)", color='white')
        plt.ylabel("Number of Tracks", color='white')
        plt.tight_layout()
        st.pyplot(fig_dur)
        
        # Duration Shares Table
        st.markdown("#### Duration Category Shares")
        duration_shares = df_filtered['duration_type'].value_counts(normalize=True).reset_index()
        duration_shares.columns = ['Duration Type', 'Share of Chart Placements']
        duration_shares['Share of Chart Placements'] = duration_shares['Share of Chart Placements'].apply(lambda x: f"{x * 100:.2f}%")
        st.table(duration_shares)

# ----------------------------------------------------
# Tab 5: Content Explicitness
# ----------------------------------------------------
with tab5:
    st.markdown("### 🔞 Content Explicitness Analysis")
    
    col_exp_left, col_exp_right = st.columns([2, 3])
    
    with col_exp_left:
        st.markdown("#### Overall Explicitness Share")
        st.write(
            f"Across the charts in the selected timeframe, explicit content accounts for **{explicit_ratio * 100:.2f}%** "
            f"of all chart entries. This indicates that roughly 1 out of 3 chart spots is occupied by tracks "
            f"containing explicit lyrical content."
        )
        
        st.markdown("#### Explicitness Rate by Chart Segment")
        explicit_by_bucket = df_filtered.groupby('rank_bucket', observed=False)['is_explicit_bool'].mean().reset_index()
        explicit_by_bucket.columns = ['Rank Bucket', 'Explicit Content Share']
        explicit_by_bucket['Explicit Content Share'] = explicit_by_bucket['Explicit Content Share'].apply(lambda x: f"{x * 100:.2f}%")
        st.table(explicit_by_bucket)
        
        st.markdown("""
            > **Cultural Insight**:
            > Lyrical explicitness has a positive correlation with higher performance ranks. 
            > As we move closer to the Top 10, the proportion of explicit tracks increases. 
            > This suggests a high tolerance and demand for explicit tracks in the core listening demographics of the UK market.
        """)
        
    with col_exp_right:
        st.markdown("#### Explicitness Rate vs Chart Rank")
        
        # Aggregate by rank position (1 to 50)
        explicit_by_rank = df_filtered.groupby('position')['is_explicit_bool'].mean().reset_index()
        explicit_by_rank.columns = ['Chart Position', 'Explicit Share (%)']
        explicit_by_rank['Explicit Share (%)'] = explicit_by_rank['Explicit Share (%)'] * 100
        
        fig_exp, ax_exp = plt.subplots(figsize=(10, 6.5), facecolor='#0d0e12')
        ax_exp.set_facecolor('#0d0e12')
        
        sns.lineplot(
            data=explicit_by_rank, 
            x='Chart Position', 
            y='Explicit Share (%)', 
            marker='o', 
            color='crimson',
            linewidth=2,
            ax=ax_exp
        )
        ax_exp.tick_params(colors='white')
        ax_exp.xaxis.label.set_color('white')
        ax_exp.yaxis.label.set_color('white')
        ax_exp.spines['bottom'].set_color('#282828')
        ax_exp.spines['left'].set_color('#282828')
        ax_exp.spines['top'].set_visible(False)
        ax_exp.spines['right'].set_visible(False)
        ax_exp.set_ylim(0, 100)
        plt.title("Explicit Content % by Chart Position", color='white', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Chart Position (1 to 50)", color='white')
        plt.ylabel("Explicit Share (%)", color='white')
        plt.tight_layout()
        st.pyplot(fig_exp)

# ----------------------------------------------------
# Tab 6: Daily Chart Browser
# ----------------------------------------------------
with tab6:
    st.markdown("### 📅 Browse Daily Top 50 Charts")
    st.write("Select a specific date to view the official Top 50 Spotify chart for that day, complete with album covers, durations, and details.")
    
    # Extract list of available dates
    available_dates = sorted(df_filtered['parsed_date'].dropna().unique())
    
    if len(available_dates) == 0:
        st.info("No dates available in this range.")
    else:
        # Date selection box
        selected_parsed_date = st.selectbox(
            "Select Date:", 
            options=available_dates, 
            format_func=lambda x: pd.to_datetime(x).strftime('%B %d, %Y')
        )
        
        # Filter chart for selected date
        daily_chart = df_filtered[df_filtered['parsed_date'] == selected_parsed_date].sort_values(by='position').reset_index(drop=True)
        
        st.markdown(f"#### 📅 Spotify UK Top 50 Chart on **{pd.to_datetime(selected_parsed_date).strftime('%A, %B %d, %Y')}**")
        
        # Display the 50 tracks inside a nice visual dashboard
        for index, row in daily_chart.iterrows():
            # Build HTML row layout
            col_rank, col_img, col_info, col_pop, col_dur, col_exp = st.columns([1, 1.2, 5, 2, 2, 1.2])
            
            with col_rank:
                st.markdown(f"<h3 style='margin-top:12px; color:#1db954; font-weight:700;'>#{row['position']}</h3>", unsafe_allow_html=True)
            
            with col_img:
                if pd.notna(row['album_cover_url']) and str(row['album_cover_url']).startswith('http'):
                    st.image(row['album_cover_url'], width=64)
                else:
                    st.markdown("<div style='background-color:#282828; width:64px; height:64px; border-radius:4px; display:flex; align-items:center; justify-content:center;'>🎵</div>", unsafe_allow_html=True)
                    
            with col_info:
                collab_badge = "🤝 " if row['is_collaboration'] else ""
                st.markdown(
                    f"<div style='margin-top: 5px;'>"
                    f"<strong style='font-size:16px; color:#ffffff;'>{row['song']}</strong><br/>"
                    f"<span style='color:#a7a7a7; font-size:13px;'>{collab_badge}{row['artist']}</span>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
                
            with col_pop:
                st.markdown(f"<p style='margin-top:16px; color:#b3b3b3; font-size:14px;'>📈 Pop: <strong>{row['popularity']}</strong></p>", unsafe_allow_html=True)
                
            with col_dur:
                st.markdown(f"<p style='margin-top:16px; color:#b3b3b3; font-size:14px;'>⏱️ {row['duration_min']:.2f} min</p>", unsafe_allow_html=True)
                
            with col_exp:
                if row['is_explicit_bool']:
                    st.markdown(
                        "<div style='margin-top: 14px; background-color: #3e3e3e; color: #ffffff; "
                        "padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; "
                        "display: inline-block; border: 1px solid #555555; text-align: center; width: 45px;'>EXPLICIT</div>", 
                        unsafe_allow_html=True
                    )
            
            # Separator line
            st.markdown("<hr style='margin: 8px 0; border: 0; border-top: 1px solid #282828;' />", unsafe_allow_html=True)

# ----------------------------------------------------
# Tab 7: Machine Learning Insights
# ----------------------------------------------------
with tab7:
    st.markdown("### 🔮 Machine Learning Models & Predictive Insights")
    st.write(
        "Using **Random Forest** algorithms trained on song-level chart histories, "
        "we can segment tracks into performance cohorts and predict future success based on track metadata."
    )
    
    col_ml_left, col_ml_right = st.columns([1.1, 1.3])
    
    with col_ml_left:
        st.markdown("#### 🔮 Predict Track Success Probability")
        st.write("Input a track's metadata to calculate its likelihood of reaching the Top 10 and its estimated popularity.")
        
        # Build interactive inputs
        # Create an autocomplete selectbox for artist selection to estimate artist power
        sorted_artists = sorted(list(artist_appearance_dict.keys()))
        selected_artist = st.selectbox("Lead Artist:", options=["-- Custom / New Artist --"] + sorted_artists)
        
        if selected_artist == "-- Custom / New Artist --":
            custom_appearances = st.number_input("Artist's Prior Total Chart Appearances (Days):", min_value=0, max_value=2000, value=0)
            artist_power_val = custom_appearances
        else:
            artist_power_val = artist_appearance_dict[selected_artist]
            st.info(f"Selected artist has {artist_power_val} prior chart-day appearances in this dataset.")
            
        dur_min = st.slider("Track Duration (minutes):", min_value=1.0, max_value=8.0, value=3.3, step=0.1)
        is_explicit = st.checkbox("Explicit Lyrical Content", value=False)
        is_collab = st.checkbox("Collaborative Track (Multiple Artists)", value=False)
        rel_format = st.selectbox("Release Format:", ["single", "album", "compilation"])
        total_tracks_num = st.number_input("Total Tracks on Release Album:", min_value=1, max_value=100, value=12)
        
        # Encode inputs
        album_type_map = {'single': 1, 'album': 2, 'compilation': 3}
        album_type_encoded_val = album_type_map[rel_format]
        
        # Run model inference on button click
        if st.button("Run ML Analysis"):
            input_vector = [[
                dur_min,
                total_tracks_num,
                int(is_explicit),
                int(is_collab),
                artist_power_val,
                album_type_encoded_val
            ]]
            
            # Predict Top 10 Probability
            top_10_prob = clf_model.predict_proba(input_vector)[0][1] * 100
            
            # Predict Popularity
            pred_popularity_score = reg_model.predict(input_vector)[0]
            
            # Predict Performance Profile
            pred_cohort_label = cluster_clf.predict(input_vector)[0]
            
            # Style the predictions using HTML cards matching our theme
            st.markdown(f"""
                <div style="background: rgba(22, 24, 33, 0.9); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; margin-top: 15px;">
                    <h4 style="color:#ffffff; margin-top:0; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:10px;">🔮 Prediction Output</h4>
                    <div style="display:flex; justify-content:space-between; margin-bottom:15px; margin-top:10px;">
                        <div>
                            <span style="font-size:12px; color:#9ca3af; text-transform:uppercase;">Top 10 Entry Prob.</span><br/>
                            <strong style="font-size:24px; color:#1db954;">{top_10_prob:.1f}%</strong>
                        </div>
                        <div>
                            <span style="font-size:12px; color:#9ca3af; text-transform:uppercase;">Est. Popularity Score</span><br/>
                            <strong style="font-size:24px; color:#6366f1;">{pred_popularity_score:.1f} / 100</strong>
                        </div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border-radius:8px; padding:12px; border-left:4px solid #8b5cf6;">
                        <span style="font-size:11px; color:#9ca3af; text-transform:uppercase; font-weight:bold;">Predicted Performance Profile</span><br/>
                        <strong style="font-size:16px; color:#ffffff;">{pred_cohort_label}</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
    with col_ml_right:
        st.markdown("#### 📊 Customer/Track Cohort Clustering")
        st.write("Applying K-Means clustering to partition tracks into stable performance groups based on chart longevity and popularity:")
        
        # Color mapping for stable performance profiles
        cohort_colors = {
            "Mega Blockbuster Hits": "#1db954",        # Spotify green
            "Steady Catalog Performers": "#6366f1",    # Indigo
            "Short-Run Album Tracks": "#fbbf24",       # Amber
            "Transient / Anomalous Entries": "#ef4444"  # Red
        }
        
        # Create Scatter Plot: Days Charted vs Peak Position
        fig_ml, ax_ml = plt.subplots(figsize=(10, 5.5), facecolor='#0d0e12')
        ax_ml.set_facecolor('#0d0e12')
        
        # Plot each profile separately to map colors
        for profile, color in cohort_colors.items():
            subset = df_songs[df_songs['performance_profile'] == profile]
            ax_ml.scatter(
                subset['days_charted'], 
                subset['best_position'], 
                label=profile, 
                color=color, 
                alpha=0.75, 
                edgecolors='none', 
                s=45
            )
            
        ax_ml.invert_yaxis()  # Invert so position 1 (the best) is at the top
        ax_ml.set_xlabel("Total Days Charted", color='white', fontsize=11)
        ax_ml.set_ylabel("Peak Chart Position (Rank)", color='white', fontsize=11)
        ax_ml.set_title("Track Performance Profiles (K-Means Clustering)", color='white', fontsize=13, pad=15)
        
        # Style legend and axes
        legend = ax_ml.legend(facecolor='#0d0e12', edgecolor='#333333', loc='lower right')
        plt.setp(legend.get_texts(), color='white')
        
        ax_ml.spines['top'].set_visible(False)
        ax_ml.spines['right'].set_visible(False)
        ax_ml.spines['left'].set_color('#333333')
        ax_ml.spines['bottom'].set_color('#333333')
        ax_ml.tick_params(colors='white')
        ax_ml.grid(True, color='#222222', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        st.pyplot(fig_ml)
        
        # Feature Importance Plot
        st.markdown("#### ⚡ Random Forest Feature Importance (Top 10 Classifier)")
        
        # Calculate feature importance
        importances = clf_model.feature_importances_
        feature_names = ["Duration (min)", "Album Size (Tracks)", "Is Explicit", "Is Collab", "Artist Power", "Format Encoded"]
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
        
        fig_imp, ax_imp = plt.subplots(figsize=(10, 3.5), facecolor='#0d0e12')
        ax_imp.set_facecolor('#0d0e12')
        
        # Custom gradient color for barplot
        sns.barplot(
            data=imp_df, 
            y='Feature', 
            x='Importance', 
            palette='viridis', 
            ax=ax_imp
        )
        
        ax_imp.set_xlabel("Relative Importance Weight", color='white')
        ax_imp.set_ylabel("")
        ax_imp.set_title("Which attributes drive UK Top 10 Entry?", color='white', fontsize=12, pad=10)
        
        # Style axes
        ax_imp.spines['top'].set_visible(False)
        ax_imp.spines['right'].set_visible(False)
        ax_imp.spines['bottom'].set_color('#333333')
        ax_imp.spines['left'].set_color('#333333')
        ax_imp.tick_params(colors='white')
        ax_imp.grid(True, color='#222222', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        st.pyplot(fig_imp)

# ----------------------------------------------------
# Bottom Info
# ----------------------------------------------------
st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #555555; font-size: 12px;'>"
    "Spotify UK Chart Insights Explorer | Powered by Streamlit & Python"
    "</p>", 
    unsafe_allow_html=True
)
