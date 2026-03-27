# Air Quality Visual Analytics Dashboard

This project provides an interactive dashboard for analyzing and forecasting air quality data. The application is built using the Dash framework and incorporates machine learning to provide insights into pollutant levels and their correlations with health outcomes.

## Project Structure

- **app.py**: The main application script containing the fully functional Dash dashboard layout and interactive analytics logic.
- **process_air_quality.py**: The source code script used for pre-processing raw sensor datasets (merging multi-year data, cleaning timestamps, and aggregating by pollutant).
- **requirements.txt**: A list of Python dependencies required to run the application.
- **data/**: A directory containing the processed datasets in CSV format.
  - **cleaned_air_quality_merged.csv**: The primary dataset used for forecasting models.
  - **city_pollutant_health_merged_v2.csv**: A dataset linking air quality indicators with regional health statistics.
  - **station_id.csv**: A reference file for monitoring station identifiers.
  - (Other supplemental datasets are included for extended analysis).

## Components Implemented by Students

This functional prototype implements the entire designed visual analytics workflow. The following components were developed and implemented by the students:

- **Data Preprocessing**: Custom scripts (`process_air_quality.py`) for automated cleaning, resampling, and alignment of multi-year sensor data.
- **ML Integration**: Interactive training of Random Forest Regressors within the Dash environment, featuring dynamic feature selection and hyperparameter tuning.
- **Interactive Visualizations**:
  - **Forecasting Dashboard**: Time-series plots with error spike detection and interactive selection.
  - **Model Inspection**: High-contrast Divergence Bar Charts showing real-time impact of model modifications (tree-level disabling).
  - **Random Forest Explorer**: Dynamic visualization of individual decision tree structures and per-tree feature importance analysis.
  - **Correlation Explorer**: Interactive Heatmaps and Scatter Plots with OLS trendlines for health-pollutant analysis.
- **State Management**: Robust handling of model state, selected data points, and ensemble modifications without page reloads.

## Installation & Setup

Ensure you have Python 3.8 or higher installed.

1. Navigate to the project directory:
   ```bash
   cd final
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To start the dashboard, execute:
```bash
python app.py
```

Open your browser to `http://127.0.0.1:8052` (or the address shown in the terminal).

## Dependencies

The project relies on the following core libraries:
- `dash` & `dash-bootstrap-components`: Front-end framework and UI components.
- `pandas` & `numpy`: Data manipulation.
- `plotly`: Interactive data visualizations.
- `scikit-learn`: Machine learning model implementation and metrics.
- `matplotlib`: Background tree visualization generation.
- `statsmodels`: Statistical trendlines for correlation analysis.

## Features

### Forecasting and Tuning
Users can select a target pollutant and various input predictors to train a Random Forest model. The dashboard visualizes forecast accuracy and allows for detailed inspection of "spikes" (high-error regions).

### Active Model Modification
Unique to this prototype, users can interactively disable specific trees from the ensemble to observe the impact on prediction error and model stability via a real-time divergence graph.

### Correlation Explorer
Investigate relationships between pollutants and health outcomes across cities and time periods using interactive heatmaps and scatter analysis.

