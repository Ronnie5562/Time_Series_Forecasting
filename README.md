# Air Quality Forecasting Project

A time series forecasting project to predict PM2.5 air quality levels using meteorological and environmental data.

## Dataset

The dataset contains hourly air quality measurements with the following features:

- **Meteorological**: Temperature (TEMP), Dew Point (DEWP), Pressure (PRES)
- **Wind**: Speed (Iws), Direction (cbwd_NW, cbwd_SE, cbwd_cv)
- **Precipitation**: Snow (Is), Rain (Ir)
- **Target**: PM2.5 concentration levels

## Project Structure

```
├── data_exploration.ipynb    # Comprehensive EDA and preprocessing
├── model_training.ipynb      # Time series model implementation
├── data/                     # Dataset files
└── README.md
```

## Key Features

### Data Analysis

- **Missing Value Handling**: Strategic imputation using forward fill, interpolation, and seasonal means
- **Outlier Detection**: IQR and Z-score methods with temporal pattern analysis
- **Seasonal Decomposition**: Trend, seasonal, and residual component analysis
- **Feature Engineering**: Cyclical encoding, lag features, and rolling statistics

### Preprocessing Pipeline

- **Temporal Features**: Hour, day, month encoding with cyclical transformations
- **Scaling**: RobustScaler for features, StandardScaler for target
- **Time Windows**: Configurable lag periods and rolling window statistics
- **Data Quality**: Comprehensive missing data and outlier handling

### Visualizations

- Time series plots with trend and seasonal patterns
- Correlation heatmaps and scatter plots
- Distribution analysis before/after preprocessing
- Seasonal and hourly pattern analysis

## Key Insights

- **Seasonal Patterns**: Strong monthly and daily variations in PM2.5 levels
- **Weather Impact**: Temperature and wind conditions significantly affect air quality
- **Data Quality**: ~X% missing values handled through multi-strategy approach
- **Feature Importance**: Lag features and meteorological data are key predictors

## Next Steps

- Implement LSTM/GRU models for time series forecasting
- Add external data sources (traffic, industrial activity)
- Deploy model for real-time predictions
- Implement model monitoring and retraining pipeline
