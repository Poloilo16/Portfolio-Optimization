#!/bin/bash

echo "🏆 Starting Brasileirão 2024 Prediction Analysis..."
echo "=================================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found"
    echo "Please install Python 3 first"
    exit 1
fi

# Run the main analysis
echo "Running Brasileirão prediction analysis..."
python3 brasileirao_predictor.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Analysis completed successfully!"
    echo ""
    echo "Generated files:"
    echo "📊 brasileirao_predictions.csv - Final predictions"
    echo "📈 brasileirao_analysis.png - Visualizations"
    echo "📖 README_brasileirao.md - Documentation"
    echo ""
else
    echo "❌ Analysis failed. Please check the error messages above."
    exit 1
fi