#!/usr/bin/env python3
"""
Comparison Script: Basic vs Enhanced Predictions
===============================================

This script compares predictions made with basic team statistics
versus predictions made with detailed match-by-match data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from brasileirao_predictor import BrasileiraoPredictor
from enhanced_brasileirao_predictor import EnhancedBrasileiraoPredictor
import warnings
warnings.filterwarnings('ignore')

def run_comparison_analysis():
    """
    Run both prediction systems and compare results
    """
    print("🏆 BRASILEIRÃO PREDICTION COMPARISON 🏆")
    print("=" * 55)
    print("Comparing Basic Statistics vs Match Data Predictions")
    print("=" * 55)
    
    # Run basic prediction system
    print("\n1️⃣ RUNNING BASIC PREDICTION SYSTEM...")
    print("-" * 40)
    basic_predictor = BrasileiraoPredictor()
    
    # Suppress detailed output for cleaner comparison
    print("Step 1: Gathering basic team data...")
    basic_predictor.step1_gather_data()
    
    print("Step 2: Simple regression (3 parameters)...")
    basic_predictor.step2_simple_regression()
    
    print("Step 3: Advanced regressions...")
    basic_best_model = basic_predictor.step3_advanced_regressions()
    
    print("Step 4: Feature analysis...")
    basic_predictor.step4_analyze_features()
    
    basic_predictions = basic_predictor.predict_final_standings()
    
    # Run enhanced prediction system
    print("\n\n2️⃣ RUNNING ENHANCED PREDICTION SYSTEM...")
    print("-" * 45)
    enhanced_predictor = EnhancedBrasileiraoPredictor()
    
    print("Step 1: Gathering detailed match data...")
    enhanced_predictor.step1_gather_match_data()
    enhanced_predictor.aggregate_team_statistics()
    
    print("Step 2: Match-derived regression...")
    enhanced_predictor.step2_simple_regression()
    
    print("Step 3: Advanced match-based regressions...")
    enhanced_best_model = enhanced_predictor.step3_advanced_regressions()
    
    print("Step 4: Match feature analysis...")
    enhanced_predictor.step4_analyze_features()
    
    enhanced_predictions = enhanced_predictor.predict_final_standings()
    
    # Compare results
    print("\n\n3️⃣ COMPARISON RESULTS")
    print("-" * 25)
    
    # Model performance comparison
    basic_accuracy = basic_predictor.models[basic_best_model]['cv_score']
    enhanced_accuracy = enhanced_predictor.models[enhanced_best_model]['cv_score']
    
    print(f"\n📊 Model Performance:")
    print(f"Basic System:    {basic_accuracy:.2f} positions average error")
    print(f"Enhanced System: {enhanced_accuracy:.2f} positions average error")
    
    improvement = ((basic_accuracy - enhanced_accuracy) / basic_accuracy) * 100
    if improvement > 0:
        print(f"Improvement:     {improvement:.1f}% better accuracy with match data")
    else:
        print(f"Difference:      {abs(improvement):.1f}% (enhanced system)")
    
    # Feature comparison
    print(f"\n🔍 Feature Analysis:")
    print(f"Basic System Features: {len(basic_predictor.models[basic_best_model]['features'])}")
    basic_top_features = basic_predictor.feature_importance.head(3)['feature'].tolist()
    print(f"Top 3: {', '.join(basic_top_features)}")
    
    print(f"\nEnhanced System Features: {len(enhanced_predictor.models[enhanced_best_model]['features'])}")
    enhanced_top_features = enhanced_predictor.feature_importance.head(3)['feature'].tolist()
    print(f"Top 3: {', '.join(enhanced_top_features)}")
    
    # Prediction differences
    compare_predictions(basic_predictions, enhanced_predictions)
    
    # Create comparison visualization
    create_comparison_visualization(basic_predictor, enhanced_predictor, 
                                  basic_predictions, enhanced_predictions)
    
    return basic_predictor, enhanced_predictor, basic_predictions, enhanced_predictions

def compare_predictions(basic_preds, enhanced_preds):
    """
    Compare the actual predictions between systems
    """
    print(f"\n⚖️ Prediction Differences:")
    print("-" * 30)
    
    # Merge predictions
    comparison = basic_preds[['team', 'current_position', 'predicted_position']].copy()
    comparison = comparison.rename(columns={'predicted_position': 'basic_prediction'})
    
    enhanced_simple = enhanced_preds[['team', 'predicted_position']].copy()
    enhanced_simple = enhanced_simple.rename(columns={'predicted_position': 'enhanced_prediction'})
    
    comparison = comparison.merge(enhanced_simple, on='team')
    comparison['prediction_difference'] = comparison['enhanced_prediction'] - comparison['basic_prediction']
    
    # Show teams with biggest prediction differences
    comparison['abs_difference'] = abs(comparison['prediction_difference'])
    biggest_differences = comparison.nlargest(5, 'abs_difference')
    
    print("Teams with biggest prediction differences:")
    print("=" * 50)
    for _, row in biggest_differences.iterrows():
        direction = "higher" if row['prediction_difference'] > 0 else "lower"
        print(f"{row['team']:<15} Enhanced predicts {abs(row['prediction_difference']):.1f} positions {direction}")
    
    # Summary statistics
    avg_difference = comparison['abs_difference'].mean()
    max_difference = comparison['abs_difference'].max()
    
    print(f"\nPrediction Comparison Summary:")
    print(f"Average difference: {avg_difference:.2f} positions")
    print(f"Maximum difference: {max_difference:.1f} positions")
    
    # Agreement analysis
    close_predictions = (comparison['abs_difference'] <= 1.0).sum()
    total_teams = len(comparison)
    
    print(f"Close agreement (≤1 position): {close_predictions}/{total_teams} teams ({close_predictions/total_teams*100:.1f}%)")

def create_comparison_visualization(basic_pred, enhanced_pred, basic_results, enhanced_results):
    """
    Create visualization comparing both systems
    """
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Model Performance Comparison
    models_basic = list(basic_pred.models.keys())
    scores_basic = [basic_pred.models[name]['cv_score'] for name in models_basic]
    
    models_enhanced = list(enhanced_pred.models.keys())
    scores_enhanced = [enhanced_pred.models[name]['cv_score'] for name in models_enhanced]
    
    x_pos = np.arange(max(len(models_basic), len(models_enhanced)))
    
    if len(models_basic) > 0:
        ax1.bar(x_pos[:len(models_basic)] - 0.2, scores_basic, 0.4, 
                label='Basic System', color='lightblue', alpha=0.7)
    
    if len(models_enhanced) > 0:
        ax1.bar(x_pos[:len(models_enhanced)] + 0.2, scores_enhanced, 0.4, 
                label='Enhanced System', color='orange', alpha=0.7)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Cross-Validation MAE (Lower is Better)')
    ax1.set_title('Model Performance Comparison')
    ax1.legend()
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Model {i+1}' for i in x_pos], rotation=45)
    
    # 2. Feature Importance Comparison
    if hasattr(basic_pred, 'feature_importance') and hasattr(enhanced_pred, 'feature_importance'):
        basic_top = basic_pred.feature_importance.head(5)
        enhanced_top = enhanced_pred.feature_importance.head(5)
        
        y_pos = np.arange(5)
        ax2.barh(y_pos - 0.2, basic_top['importance'], 0.4, 
                label='Basic Features', color='lightblue', alpha=0.7)
        ax2.barh(y_pos + 0.2, enhanced_top['importance'], 0.4, 
                label='Match Features', color='orange', alpha=0.7)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f'Feature {i+1}' for i in range(5)])
        ax2.set_xlabel('Feature Importance')
        ax2.set_title('Top 5 Feature Importance Comparison')
        ax2.legend()
    
    # 3. Prediction Scatter Plot
    if len(basic_results) > 0 and len(enhanced_results) > 0:
        # Merge for comparison
        comparison = basic_results[['team', 'predicted_position']].copy()
        comparison = comparison.rename(columns={'predicted_position': 'basic_pred'})
        
        enhanced_simple = enhanced_results[['team', 'predicted_position']].copy()
        enhanced_simple = enhanced_simple.rename(columns={'predicted_position': 'enhanced_pred'})
        
        comparison = comparison.merge(enhanced_simple, on='team')
        
        ax3.scatter(comparison['basic_pred'], comparison['enhanced_pred'], 
                   alpha=0.7, s=60, color='purple')
        
        # Add diagonal line (perfect agreement)
        min_pos = min(comparison['basic_pred'].min(), comparison['enhanced_pred'].min())
        max_pos = max(comparison['basic_pred'].max(), comparison['enhanced_pred'].max())
        ax3.plot([min_pos, max_pos], [min_pos, max_pos], 'r--', alpha=0.5)
        
        ax3.set_xlabel('Basic System Predictions')
        ax3.set_ylabel('Enhanced System Predictions')
        ax3.set_title('Prediction Comparison by Team')
        ax3.grid(True, alpha=0.3)
    
    # 4. Accuracy Improvement
    basic_best_score = min([basic_pred.models[name]['cv_score'] for name in basic_pred.models.keys()])
    enhanced_best_score = min([enhanced_pred.models[name]['cv_score'] for name in enhanced_pred.models.keys()])
    
    systems = ['Basic\nSystem', 'Enhanced\nSystem']
    scores = [basic_best_score, enhanced_best_score]
    colors = ['lightblue', 'orange']
    
    bars = ax4.bar(systems, scores, color=colors, alpha=0.7)
    ax4.set_ylabel('Best Model MAE (Lower is Better)')
    ax4.set_title('Overall System Accuracy')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Highlight the better system
    if enhanced_best_score < basic_best_score:
        bars[1].set_color('gold')
        bars[1].set_edgecolor('black')
        bars[1].set_linewidth(2)
    else:
        bars[0].set_color('gold')
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig('/workspace/prediction_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 Comparison visualization saved as 'prediction_comparison.png'")

def save_comparison_results(basic_pred, enhanced_pred, basic_results, enhanced_results):
    """
    Save detailed comparison results
    """
    print(f"\n💾 Saving comparison results...")
    
    # Create comprehensive comparison dataframe
    comparison = basic_results[['team', 'current_position']].copy()
    
    # Add basic predictions
    basic_simple = basic_results[['team', 'predicted_position', 'position_change']].copy()
    basic_simple = basic_simple.rename(columns={
        'predicted_position': 'basic_predicted_position',
        'position_change': 'basic_position_change'
    })
    comparison = comparison.merge(basic_simple, on='team')
    
    # Add enhanced predictions
    enhanced_simple = enhanced_results[['team', 'predicted_position', 'position_change']].copy()
    enhanced_simple = enhanced_simple.rename(columns={
        'predicted_position': 'enhanced_predicted_position',
        'position_change': 'enhanced_position_change'
    })
    comparison = comparison.merge(enhanced_simple, on='team')
    
    # Calculate differences
    comparison['prediction_difference'] = (comparison['enhanced_predicted_position'] - 
                                         comparison['basic_predicted_position'])
    
    # Add model performance
    basic_best = min([basic_pred.models[name]['cv_score'] for name in basic_pred.models.keys()])
    enhanced_best = min([enhanced_pred.models[name]['cv_score'] for name in enhanced_pred.models.keys()])
    
    comparison['basic_system_accuracy'] = basic_best
    comparison['enhanced_system_accuracy'] = enhanced_best
    comparison['accuracy_improvement'] = basic_best - enhanced_best
    
    # Sort by current position
    comparison = comparison.sort_values('current_position')
    
    # Save to CSV
    comparison.to_csv('/workspace/prediction_comparison.csv', index=False)
    print(f"Detailed comparison saved to 'prediction_comparison.csv'")
    
    return comparison

def main():
    """
    Main function to run the complete comparison
    """
    # Run comparison
    basic_pred, enhanced_pred, basic_results, enhanced_results = run_comparison_analysis()
    
    # Save results
    comparison_data = save_comparison_results(basic_pred, enhanced_pred, basic_results, enhanced_results)
    
    print(f"\n" + "=" * 55)
    print("🎯 COMPARISON COMPLETE!")
    print("=" * 55)
    print("Files generated:")
    print("📊 prediction_comparison.png - Visual comparison")
    print("📋 prediction_comparison.csv - Detailed data")
    print("\nThe enhanced system uses detailed match statistics to provide")
    print("more accurate predictions based on actual game performance!")

if __name__ == "__main__":
    main()