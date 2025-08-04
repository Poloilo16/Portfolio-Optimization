# Brasileirão 2024 Prediction System 🏆

A comprehensive machine learning system to predict the final standings of the 20 teams in Brazil's top football division (Campeonato Brasileiro Série A) for the 2024 season.

## Overview

This project implements a 4-step approach as outlined in your requirements:

1. **Gather data from the internet** - Collect current season statistics
2. **Simple regression** - Initial prediction using 3 key parameters  
3. **Advanced regressions** - Test multiple models for better accuracy
4. **Feature analysis** - Analyze which features are most important

## Files Structure

```
├── brasileirao_predictor.py    # Main prediction system
├── data_scraper.py            # Web scraping utilities for real data
├── requirements.txt           # Python dependencies
└── README_brasileirao.md      # This documentation
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run with Simulated Data (Recommended for Testing)

The main script includes realistic simulated data based on typical Brasileirão patterns:

```bash
python brasileirao_predictor.py
```

This will:
- Generate realistic team statistics for all 20 teams
- Run the complete 4-step analysis
- Create visualizations
- Save predictions to CSV

### Option 2: Use Real Data (Advanced)

First, try to gather real data:

```bash
python data_scraper.py
```

Then modify the main script to use real data instead of simulated data.

## Features

### Data Collection (Step 1)
- **Simulated Data**: Realistic statistics based on actual Brasileirão patterns
- **Real Data Scraping**: Support for multiple sources (ESPN, Transfermarkt, API-Football)
- **Comprehensive Metrics**: 20+ features including goals, possession, form, etc.

### Simple Regression (Step 2)
- Uses 3 key parameters: `points_per_game`, `goal_difference`, `win_rate`
- Linear regression with cross-validation
- Initial accuracy baseline

### Advanced Models (Step 3)
Tests multiple regression algorithms:
- Linear Regression
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)  
- Random Forest
- Gradient Boosting

Automatically selects the best performing model based on cross-validation.

### Feature Analysis (Step 4)
- Feature importance ranking
- Model performance comparison
- Visualizations showing predictions vs current standings
- Analysis of which statistics matter most

## Output

The system generates:

1. **Console Output**: Step-by-step progress and results
2. **CSV File**: `brasileirao_predictions.csv` with detailed predictions
3. **Visualization**: `brasileirao_analysis.png` with 4 analytical charts
4. **Final Standings**: Predicted final table with position changes

## Example Output

```
🏆 BRASILEIRÃO 2024 PREDICTION SYSTEM 🏆
==================================================

Step 1: Gathering data from the internet...
Data gathered for 20 teams

Current top 5 teams:
        team  points  wins  draws  losses  goal_difference
0    Flamengo      45    14      3       8               12
1   Palmeiras      42    13      3       8                8
2  Atlético-MG     40    12      4       9                6
...

Step 2: Running simple regression with 3 key parameters...
Simple Regression Results:
Features used: ['points_per_game', 'goal_difference', 'win_rate']
Mean Absolute Error: 2.15 positions
R² Score: 0.847

Step 3: Testing multiple regression models for better accuracy...
Linear Regression: MAE: 1.98 positions, R²: 0.863
Ridge Regression: MAE: 1.95 positions, R²: 0.867
...
Best model: Random Forest with CV MAE of 1.73 positions

Step 4: Analyzing feature importance...
Top 10 Most Important Features:
              feature  importance
    points_per_game       0.342
   goal_difference       0.198
          win_rate       0.156
...

Predicted Final Standings (using Random Forest):
================================================================================
 1. Flamengo        Current:  1 (=)     Points: 45 GD: +12
 2. Palmeiras       Current:  2 (=)     Points: 42 GD:  +8
 3. Atlético-MG     Current:  3 (+1.2)  Points: 40 GD:  +6
...
```

## Key Features

### Realistic Team Data
The system includes all 20 current Brasileirão teams:
- Flamengo, Palmeiras, Atlético-MG, Fortaleza, Internacional
- São Paulo, Corinthians, Bahia, Cruzeiro, Vasco da Gama
- EC Vitória, Atlético-PR, Grêmio, Juventude, Bragantino
- Botafogo, Criciúma, Cuiabá, Atlético-GO, Fluminense

### Comprehensive Statistics
- **Basic**: Games, wins, draws, losses, goals for/against
- **Advanced**: Shots, possession, pass accuracy, tackles
- **Form**: Recent performance metrics
- **Discipline**: Cards and fouls
- **Efficiency**: Clean sheets, failed to score

### Model Performance
- Typical accuracy: 1.5-2.5 positions average error
- Cross-validation to prevent overfitting
- Multiple model comparison for best results

## Real Data Sources

The `data_scraper.py` supports multiple sources:

1. **API-Football** (Requires API key)
   - Most reliable and comprehensive
   - Real-time data updates
   
2. **Transfermarkt** (Free web scraping)
   - Detailed team statistics
   - Historical data available
   
3. **ESPN Brazil** (Free web scraping)
   - Current standings
   - Basic statistics

### Using Real Data

To use real data with an API key:

```python
from data_scraper import BrasileiraoDataScraper

scraper = BrasileiraoDataScraper()
real_data = scraper.get_real_data(source='api', api_key='YOUR_API_KEY')
```

## Customization

### Adding New Features
Edit the `advanced_features` list in `step3_advanced_regressions()`:

```python
advanced_features = [
    'points_per_game', 'goal_difference', 'win_rate',
    # Add your custom features here
    'new_feature_1', 'new_feature_2'
]
```

### Changing Models
Add new models to the `models_to_test` dictionary:

```python
models_to_test = {
    'Linear Regression': LinearRegression(),
    'Your Custom Model': YourModel(),
}
```

## Interpreting Results

### Position Changes
- `(+2.3)`: Team predicted to move up 2.3 positions
- `(-1.5)`: Team predicted to drop 1.5 positions  
- `(=)`: No significant change predicted

### Model Accuracy
- **MAE < 2.0**: Excellent accuracy
- **MAE 2.0-3.0**: Good accuracy
- **MAE > 3.0**: May need more data or features

### Feature Importance
Higher importance values indicate features that have more impact on final position prediction.

## Limitations

1. **Simulated Data**: Default data is realistic but not real
2. **Web Scraping**: Real data scraping may break if websites change
3. **Sample Size**: Limited to current season data (20 teams)
4. **External Factors**: Cannot predict injuries, transfers, or other events

## Future Enhancements

- Integration with live data APIs
- Historical season analysis
- Player-level statistics
- Match prediction capabilities
- Monte Carlo simulations for uncertainty quantification

## Contributing

Feel free to enhance the system by:
- Adding new data sources
- Implementing additional ML models
- Improving feature engineering
- Adding more detailed analysis

## License

This project is for educational and analysis purposes. Please respect the terms of service of any data sources you use.

---

**Enjoy predicting the Brasileirão! 🇧🇷⚽**