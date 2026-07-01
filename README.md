# United Kingdom Top 50 Playlist Market Structure, Artist Diversity & ContentLocalization Analysis

A interactive data visualization and predictive analytics platform that explores the UK Spotify Top 50 charts. Built with **Streamlit**, **Pandas**, **NetworkX**, and **Scikit-Learn**, this application provides deep insights into music market structure, artist concentration, collaboration networks, and predicting hit songs using Machine Learning.

---

## 🚀 Features

### 1. Interactive Dashboard Overview
* **Market KPIs**: Real-time stats on Artist Concentration (Top 10 share), Unique Artists, Collaboration Ratio, Format shares, and Content Variety Index (Pielou's Evenness).
* **Top Performers**: Interactive visual graphs highlighting top-charted artists and songs.
* **Popularity Trends**: Line chart showing streaming popularity fluctuations over time.

### 2. Artist Dominance & Market Share
* **Market Concentration Analysis**: Utilizes the Herfindahl-Hirschman Index (HHI) and Pielou's Evenness to assess market diversity.
* **Artist Deep-Dive**: Choose any artist from the dataset to track their daily chart presence, best rank, average rank, and individual song performance.

### 3. Collaboration Network
* **Interactive Graph**: Visualizes artist partnership networks using `NetworkX`.
* **Adjustable Thresholds**: A slider allows filtering out casual pairings to display only the most significant or repeated artist collaborations.

### 4. Release Strategy & Track Formats
* **Single vs. Album Shares**: Pie chart visualizing the proportion of chart entries that are singles, album tracks, or compilations.
* **Correlation Analysis**: Computes correlations between album size (`total_tracks`), peak positions, and streaming popularity.
* **Duration Distribution**: Explores track length patterns (Short-form, Standard, Long-form) in the charts.

### 5. Content Explicitness Analysis
* **Explicitness Share**: Computes the percentage of explicit content on the charts.
* **Position Analysis**: Identifies how explicit content relates to final chart positions.

### 6. Daily Chart Browser
* Select a specific date to view the official Spotify Top 50 chart.
* Features responsive, stylized row representations containing album art (if URLs are active), artist/song details, popularity scores, duration, and custom explicit badges.

### 7. Machine Learning & Predictive Insights
* **Interactive Track Success Predictor**: Input custom track metadata (duration, album size, explicit label, collab status, format, and lead artist chart power) to predict:
  * **Top 10 Entry Probability (%)** (Random Forest Classifier)
  * **Estimated Popularity Score** (Random Forest Regressor)
  * **Predicted Performance Cohort** (Random Forest Cluster Predictor)
* **K-Means Clustering Scatter Plot**: Visualizes track cohorts based on popularity and longevity:
  * *Mega Blockbuster Hits*
  * *Steady Catalog Performers*
  * *Short-Run Album Tracks*
  * *Transient / Anomalous Entries*
* **Feature Importances**: Highlights the main attributes (e.g. artist power, track duration) that drive Top 10 chart entries.

---

## 📁 Repository Structure

```
├── .venv/                         # Python Virtual Environment
├── app.py                         # Main Streamlit web application
├── Atlantic_United_Kingdom.csv    # Dataset (UK Top 50 Spotify chart data)
└── ML/
    ├── Demo.ipynb                 # Jupyter Notebook containing exploratory analysis
    ├── requirements.txt           # Python dependency requirements
    └── plots/                     # Output directory for exported visualizations
```

### Dataset Columns
The underlying dataset (`Atlantic_United_Kingdom.csv`) contains:
* `date`: The chart date (standardized to `DD-MM-YYYY`).
* `position`: Chart rank (1 to 50).
* `song`: Name of the track.
* `artist`: Primary or collaborative artists.
* `popularity`: Stream-based popularity score (0 to 100).
* `duration_ms`: Duration in milliseconds.
* `album_type`: Format (single, album, compilation).
* `total_tracks`: Number of tracks on the release.
* `is_explicit`: Boolean indicator for explicit content.
* `album_cover_url`: Link to the album artwork.

---

## 🛠️ Installation & Setup

1. **Clone or Navigate to the Directory**:
   Ensure you are in the project's root directory:
   ```bash
   cd "UM Project- 2"
   ```

2. **Activate the Virtual Environment**:
   * **Windows** (Command Prompt):
     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **Windows** (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

3. **Install Dependencies**:
   Install the required libraries listed in `ML/requirements.txt`:
   ```bash
   pip install -r ML/requirements.txt
   ```

4. **Launch the Dashboard**:
   Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
   *This will open the application in your default web browser (usually at `http://localhost:8501`).*

5. **Run the Notebook (Optional)**:
   For step-by-step exploratory data analysis and prototyping:
   ```bash
   jupyter notebook ML/Demo.ipynb
   ```

---

## 🧠 Machine Learning Details

The predictive models in `app.py` extract features from raw chart histories by grouping records at the song/artist level.

* **Extracted Features**:
  * `duration_min`: Average track duration in minutes.
  * `total_tracks`: Size of the album.
  * `is_explicit_bool`: Whether the song contains explicit lyrics.
  * `is_collaboration`: Whether the track credits multiple artists.
  * `artist_power`: Measured by the total number of historical chart appearances of the lead artist.
  * `album_type_encoded`: Numerical mapping for release format (single = 1, album = 2, compilation = 3).
* **Models**:
  * **Classification**: `RandomForestClassifier(n_estimators=100, max_depth=6)` predicts the probability of a song reaching the Top 10.
  * **Regression**: `RandomForestRegressor(n_estimators=100, max_depth=6)` predicts the peak popularity score.
  * **Clustering**: `KMeans(n_clusters=4)` segments songs into longevity/popularity cohorts. A separate Random Forest Classifier is trained to predict these cluster labels from basic metadata features.
