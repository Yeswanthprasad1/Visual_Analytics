# Air Quality Visual Analytics Dashboard

This project provides an interactive dashboard for analyzing and forecasting air quality data. The application is built using the Dash framework and incorporates machine learning to provide insights into pollutant levels and their correlations with health outcomes.

## Project Structure

- **app.py**: The main application script containing the dashboard layout and interactive logic.
- **requirements.txt**: A list of Python dependencies required to run the application.
- **data/**: A directory containing the necessary datasets in CSV format.
  - **cleaned_air_quality_merged.csv**: The primary dataset used for forecasting models.
  - **city_pollutant_health_merged_v2.csv**: A dataset linking air quality indicators with regional health statistics.
  - **station_id.csv**: A reference file for monitoring station identifiers.
  - (Other supplemental datasets are included for extended analysis).

## Installation

Ensure you have Python 3.8 or higher installed on your system. It is recommended to use a virtual environment to manage dependencies.

1. Navigate to the project directory:
   ```bash
   cd final
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To start the dashboard, execute the following command:
```bash
python app.py
```

Once the server is running, open your web browser and navigate to the address provided in the terminal (standardly http://127.0.0.1:8050).

## Features

### Forecasting and Tuning
Users can select a target pollutant and various input predictors to train a Random Forest model. The dashboard visualizes the forecast accuracy and allows for detailed inspection of specific data points where prediction errors are high. It also provides a tree explorer to understand the decision-making process of the model components.

### Correlation Explorer
This section allows for the investigation of relationships between different air pollutants and health conditions. Interactive heatmaps and scatter plots with trendlines help identify significant associations across different cities and time periods.

## Data Sources
The datasets provided in the data directory are pre-processed and ready for analysis within the scope of this dashboard. Any additional data should be formatted according to the existing CSV structures to ensure compatibility.
