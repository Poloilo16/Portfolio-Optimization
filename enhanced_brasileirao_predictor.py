#!/usr/bin/env python3
"""
Enhanced Brasileirão Prediction System
======================================

This enhanced version uses detailed match-by-match data to create
more accurate predictions based on actual in-game statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from match_data_scraper import BrasileiraoMatchScraper
import warnings
warnings.filterwarnings('ignore')

class EnhancedBrasileiraoPredictor:
    def __init__(self):
        self.match_data = None
        self.team_stats = None
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def step1_gather_match_data(self, use_real_data=False, api_key=None):
        """
        Step 1: Gather detailed match-by-match data
        """
        print("Step 1: Gathering detailed match-by-match data...")
        
        scraper = BrasileiraoMatchScraper()
        
        if use_real_data:
            matches = scraper.get_comprehensive_match_data(source='api' if api_key else 'auto', api_key=api_key)
        else:
            matches = scraper.generate_realistic_match_data(380)  # Full season
        
        self.match_data = pd.DataFrame(matches)
        
        print(f"Collected {len(self.match_data)} matches")
        print(f"Date range: {self.match_data['date'].min()} to {self.match_data['date'].max()}")
        print(f"Available columns: {len(self.match_data.columns)}")
        
        # Save the match data
        self.match_data.to_csv('/workspace/enhanced_match_data.csv', index=False)
        
        return self.match_data
    
    def aggregate_team_statistics(self):
        """
        Aggregate match-level data into team-level statistics
        """
        print("Aggregating team statistics from match data...")
        
        teams = list(set(self.match_data['home_team'].unique().tolist() + 
                        self.match_data['away_team'].unique().tolist()))
        
        team_stats = []
        
        for team in teams:
            # Get all matches for this team
            home_matches = self.match_data[self.match_data['home_team'] == team]
            away_matches = self.match_data[self.match_data['away_team'] == team]
            
            # Calculate basic statistics
            total_matches = len(home_matches) + len(away_matches)
            
            if total_matches == 0:
                continue
            
            # Goals
            goals_for = (home_matches['home_score'].sum() + away_matches['away_score'].sum())
            goals_against = (home_matches['away_score'].sum() + away_matches['home_score'].sum())
            
            # Wins, draws, losses
            home_wins = len(home_matches[home_matches['home_score'] > home_matches['away_score']])
            away_wins = len(away_matches[away_matches['away_score'] > away_matches['home_score']])
            total_wins = home_wins + away_wins
            
            home_draws = len(home_matches[home_matches['home_score'] == home_matches['away_score']])
            away_draws = len(away_matches[away_matches['away_score'] == away_matches['home_score']])
            total_draws = home_draws + away_draws
            
            total_losses = total_matches - total_wins - total_draws
            
            # Points
            points = total_wins * 3 + total_draws
            
            # Advanced statistics (if available)
            stats = {
                'team': team,
                'matches_played': total_matches,
                'wins': total_wins,
                'draws': total_draws,
                'losses': total_losses,
                'points': points,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'goal_difference': goals_for - goals_against,
                'points_per_game': points / total_matches if total_matches > 0 else 0,
                'goals_per_game': goals_for / total_matches if total_matches > 0 else 0,
                'goals_conceded_per_game': goals_against / total_matches if total_matches > 0 else 0,
                'win_rate': total_wins / total_matches if total_matches > 0 else 0,
            }
            
            # Advanced match statistics
            if 'home_total_shots' in self.match_data.columns:
                home_shots = home_matches['home_total_shots'].sum()
                away_shots = away_matches['away_total_shots'].sum()
                stats['total_shots'] = home_shots + away_shots
                stats['shots_per_game'] = stats['total_shots'] / total_matches if total_matches > 0 else 0
                
            if 'home_shots_on_target' in self.match_data.columns:
                home_sot = home_matches['home_shots_on_target'].sum()
                away_sot = away_matches['away_shots_on_target'].sum()
                stats['shots_on_target'] = home_sot + away_sot
                stats['shots_on_target_per_game'] = stats['shots_on_target'] / total_matches if total_matches > 0 else 0
                
                # Shot accuracy
                if stats['total_shots'] > 0:
                    stats['shot_accuracy'] = stats['shots_on_target'] / stats['total_shots'] * 100
                else:
                    stats['shot_accuracy'] = 0
            
            if 'home_possession' in self.match_data.columns:
                home_poss = home_matches['home_possession'].mean()
                away_poss = away_matches['away_possession'].mean()
                stats['avg_possession'] = (home_poss + away_poss) / 2 if not pd.isna(home_poss) and not pd.isna(away_poss) else 50
                
            if 'home_pass_accuracy' in self.match_data.columns:
                home_pass_acc = home_matches['home_pass_accuracy'].mean()
                away_pass_acc = away_matches['away_pass_accuracy'].mean()
                stats['avg_pass_accuracy'] = (home_pass_acc + away_pass_acc) / 2 if not pd.isna(home_pass_acc) and not pd.isna(away_pass_acc) else 75
            
            if 'home_fouls' in self.match_data.columns:
                home_fouls = home_matches['home_fouls'].sum()
                away_fouls = away_matches['away_fouls'].sum()
                stats['total_fouls'] = home_fouls + away_fouls
                stats['fouls_per_game'] = stats['total_fouls'] / total_matches if total_matches > 0 else 0
            
            if 'home_yellow_cards' in self.match_data.columns:
                home_yellows = home_matches['home_yellow_cards'].sum()
                away_yellows = away_matches['away_yellow_cards'].sum()
                stats['yellow_cards'] = home_yellows + away_yellows
                
            if 'home_red_cards' in self.match_data.columns:
                home_reds = home_matches['home_red_cards'].sum()
                away_reds = away_matches['away_red_cards'].sum()
                stats['red_cards'] = home_reds + away_reds
            
            if 'home_corner_kicks' in self.match_data.columns:
                home_corners = home_matches['home_corner_kicks'].sum()
                away_corners = away_matches['away_corner_kicks'].sum()
                stats['corner_kicks'] = home_corners + away_corners
                stats['corners_per_game'] = stats['corner_kicks'] / total_matches if total_matches > 0 else 0
            
            # Recent form (last 5 matches)
            recent_matches = pd.concat([
                home_matches.tail(3), 
                away_matches.tail(2)
            ]).sort_values('date').tail(5)
            
            if len(recent_matches) > 0:
                recent_points = 0
                for _, match in recent_matches.iterrows():
                    if match['home_team'] == team:
                        if match['home_score'] > match['away_score']:
                            recent_points += 3
                        elif match['home_score'] == match['away_score']:
                            recent_points += 1
                    else:  # away team
                        if match['away_score'] > match['home_score']:
                            recent_points += 3
                        elif match['away_score'] == match['home_score']:
                            recent_points += 1
                
                stats['recent_form_points'] = recent_points
                stats['recent_form_ppg'] = recent_points / len(recent_matches)
            else:
                stats['recent_form_points'] = 0
                stats['recent_form_ppg'] = 0
            
            # Home vs Away performance
            stats['home_points'] = home_wins * 3 + home_draws
            stats['away_points'] = away_wins * 3 + away_draws
            stats['home_ppg'] = stats['home_points'] / len(home_matches) if len(home_matches) > 0 else 0
            stats['away_ppg'] = stats['away_points'] / len(away_matches) if len(away_matches) > 0 else 0
            
            team_stats.append(stats)
        
        self.team_stats = pd.DataFrame(team_stats)
        
        # Sort by points to create current standings
        self.team_stats = self.team_stats.sort_values('points', ascending=False).reset_index(drop=True)
        self.team_stats['current_position'] = range(1, len(self.team_stats) + 1)
        
        print(f"Aggregated statistics for {len(self.team_stats)} teams")
        print("\nCurrent top 5 teams:")
        print(self.team_stats[['team', 'points', 'wins', 'draws', 'losses', 'goal_difference']].head())
        
        return self.team_stats
    
    def step2_simple_regression(self):
        """
        Step 2: Simple regression with 3 key parameters from match data
        """
        print("\nStep 2: Running simple regression with 3 match-derived parameters...")
        
        if self.team_stats is None:
            self.aggregate_team_statistics()
        
        # Use match-derived features
        features = ['goals_per_game', 'shot_accuracy', 'recent_form_ppg']
        
        # Ensure features exist
        available_features = []
        for feature in features:
            if feature in self.team_stats.columns:
                available_features.append(feature)
            else:
                # Fallback features
                if feature == 'shot_accuracy' and 'shots_per_game' in self.team_stats.columns:
                    available_features.append('shots_per_game')
                elif feature == 'recent_form_ppg' and 'points_per_game' in self.team_stats.columns:
                    available_features.append('points_per_game')
        
        if len(available_features) < 2:
            # Add basic features as fallback
            available_features = ['points_per_game', 'goal_difference', 'win_rate']
        
        features = available_features[:3]  # Use top 3 available features
        
        X = self.team_stats[features].fillna(0)
        y = self.team_stats['current_position']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Simple Linear Regression
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
        
        self.models['simple_regression_match_data'] = {
            'model': model,
            'features': features,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'cv_score': -cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"Simple Regression Results (Match Data):")
        print(f"Features used: {features}")
        print(f"Mean Absolute Error: {mae:.2f} positions")
        print(f"R² Score: {r2:.3f}")
        print(f"Cross-validation MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
        
        return self.models['simple_regression_match_data']
    
    def step3_advanced_regressions(self):
        """
        Step 3: Advanced regressions using all available match-derived features
        """
        print("\nStep 3: Testing multiple regression models with match-derived features...")
        
        # Use all available numeric features
        numeric_cols = self.team_stats.select_dtypes(include=[np.number]).columns
        excluded_cols = ['current_position', 'matches_played', 'wins', 'draws', 'losses', 'points']
        advanced_features = [col for col in numeric_cols if col not in excluded_cols]
        
        print(f"Using {len(advanced_features)} match-derived features:")
        for feature in sorted(advanced_features):
            print(f"  - {feature}")
        
        X = self.team_stats[advanced_features].fillna(0)
        y = self.team_stats['current_position']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features for linear models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Test multiple models
        models_to_test = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = float('inf')
        
        for name, model in models_to_test.items():
            # Train model
            if 'Forest' in name or 'Boosting' in name:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cv_score = -cv_scores.mean()
            
            self.models[name] = {
                'model': model,
                'features': advanced_features,
                'mae': mae,
                'r2': r2,
                'cv_score': cv_score,
                'cv_std': cv_scores.std()
            }
            
            print(f"\n{name}:")
            print(f"  MAE: {mae:.2f} positions")
            print(f"  R²: {r2:.3f}")
            print(f"  CV MAE: {cv_score:.2f} ± {cv_scores.std():.2f}")
            
            # Track best model
            if cv_score < best_score:
                best_score = cv_score
                best_model = name
        
        print(f"\nBest model: {best_model} with CV MAE of {best_score:.2f} positions")
        
        return best_model
    
    def step4_analyze_features(self):
        """
        Step 4: Analyze match-derived feature importance
        """
        print("\nStep 4: Analyzing match-derived feature importance...")
        
        # Get the best performing model
        best_model_name = min(self.models.keys(), 
                             key=lambda x: self.models[x]['cv_score'])
        best_model_info = self.models[best_model_name]
        best_model = best_model_info['model']
        
        print(f"Analyzing features for best model: {best_model_name}")
        
        # Feature importance analysis
        features = best_model_info['features']
        
        if hasattr(best_model, 'feature_importances_'):
            # Tree-based models
            importance = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            # Linear models
            importance = abs(best_model.coef_)
        else:
            print("Cannot extract feature importance for this model type")
            return
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance_df
        
        print("\nTop 10 Most Important Match-Derived Features:")
        print(feature_importance_df.head(10).to_string(index=False))
        
        # Create enhanced visualizations
        self.create_enhanced_visualizations()
        
        return feature_importance_df
    
    def create_enhanced_visualizations(self):
        """
        Create enhanced visualizations showing match data insights
        """
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Feature Importance (Match-derived)
        top_features = self.feature_importance.head(10)
        bars = ax1.barh(top_features['feature'], top_features['importance'])
        ax1.set_title('Top 10 Match-Derived Feature Importance', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Importance Score')
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 2. Goals vs Shots Relationship
        if 'goals_per_game' in self.team_stats.columns and 'shots_per_game' in self.team_stats.columns:
            scatter = ax2.scatter(self.team_stats['shots_per_game'], 
                                self.team_stats['goals_per_game'],
                                c=self.team_stats['current_position'], 
                                cmap='RdYlGn_r', 
                                s=100, alpha=0.7)
            ax2.set_xlabel('Shots per Game')
            ax2.set_ylabel('Goals per Game')
            ax2.set_title('Shot Efficiency by League Position')
            plt.colorbar(scatter, ax=ax2, label='League Position')
            
            # Add trend line
            z = np.polyfit(self.team_stats['shots_per_game'], self.team_stats['goals_per_game'], 1)
            p = np.poly1d(z)
            ax2.plot(self.team_stats['shots_per_game'], p(self.team_stats['shots_per_game']), "r--", alpha=0.8)
        else:
            ax2.text(0.5, 0.5, 'Shot data not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Shot Efficiency Analysis')
        
        # 3. Form vs Position
        if 'recent_form_ppg' in self.team_stats.columns:
            bars = ax3.bar(range(len(self.team_stats)), 
                          self.team_stats['recent_form_ppg'],
                          color=plt.cm.RdYlGn(self.team_stats['recent_form_ppg']/3))
            ax3.set_xlabel('Current League Position')
            ax3.set_ylabel('Recent Form (Points per Game)')
            ax3.set_title('Recent Form by League Position')
            ax3.set_xticks(range(0, len(self.team_stats), 2))
            ax3.set_xticklabels(range(1, len(self.team_stats)+1, 2))
        else:
            ax3.text(0.5, 0.5, 'Form data not available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Recent Form Analysis')
        
        # 4. Model Performance Comparison
        model_names = list(self.models.keys())
        cv_scores = [self.models[name]['cv_score'] for name in model_names]
        
        bars = ax4.bar(range(len(model_names)), cv_scores, 
                      color=['red' if score == min(cv_scores) else 'skyblue' for score in cv_scores])
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Cross-Validation MAE (Lower is Better)')
        ax4.set_title('Enhanced Model Performance (Match Data)')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=0, fontsize=9)
        
        # Highlight best model
        best_idx = cv_scores.index(min(cv_scores))
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig('/workspace/enhanced_brasileirao_analysis.png', dpi=300, bbox_inches='tight')
        print("\nEnhanced visualization saved as 'enhanced_brasileirao_analysis.png'")
        
        return fig
    
    def predict_final_standings(self):
        """
        Generate final standings prediction using match data
        """
        print("\nGenerating Enhanced Final Standings Prediction...")
        
        # Use best model
        best_model_name = min(self.models.keys(), 
                             key=lambda x: self.models[x]['cv_score'])
        best_model_info = self.models[best_model_name]
        
        # Make predictions
        X_all = self.team_stats[best_model_info['features']].fillna(0)
        if 'Forest' in best_model_name or 'Boosting' in best_model_name:
            predicted_positions = best_model_info['model'].predict(X_all)
        else:
            X_all_scaled = self.scaler.transform(X_all)
            predicted_positions = best_model_info['model'].predict(X_all_scaled)
        
        # Create prediction dataframe
        predictions_df = self.team_stats.copy()
        predictions_df['predicted_position'] = predicted_positions
        predictions_df['position_change'] = predictions_df['current_position'] - predictions_df['predicted_position']
        
        # Sort by predicted position
        final_standings = predictions_df.sort_values('predicted_position')[
            ['team', 'current_position', 'predicted_position', 'position_change', 
             'points', 'goal_difference', 'recent_form_ppg']
        ].round(1)
        
        print(f"\nEnhanced Predicted Final Standings (using {best_model_name}):")
        print("=" * 90)
        for i, row in final_standings.iterrows():
            change_str = f"({row['position_change']:+.1f})" if abs(row['position_change']) > 0.1 else "(=)"
            form_str = f"Form: {row.get('recent_form_ppg', 0):.1f}" if 'recent_form_ppg' in row else ""
            print(f"{int(row['predicted_position']):2d}. {row['team']:<15} "
                  f"Current: {int(row['current_position']):2d} {change_str:<6} "
                  f"Points: {row['points']:2.0f} GD: {row['goal_difference']:+3.0f} {form_str}")
        
        return final_standings
    
    def run_enhanced_analysis(self, use_real_data=False, api_key=None):
        """
        Run the complete enhanced 4-step analysis using match data
        """
        print("🏆 ENHANCED BRASILEIRÃO PREDICTION SYSTEM 🏆")
        print("(Using Detailed Match Data)")
        print("=" * 55)
        
        # Step 1: Gather match data
        self.step1_gather_match_data(use_real_data, api_key)
        
        # Aggregate into team statistics
        self.aggregate_team_statistics()
        
        # Step 2: Simple regression with match-derived features
        self.step2_simple_regression()
        
        # Step 3: Advanced regressions
        best_model = self.step3_advanced_regressions()
        
        # Step 4: Feature analysis
        self.step4_analyze_features()
        
        # Final prediction
        final_standings = self.predict_final_standings()
        
        print("\n" + "=" * 55)
        print("Enhanced Analysis Complete! 🎉")
        print(f"Best model achieved {self.models[best_model]['cv_score']:.2f} positions average error")
        print("Using detailed match statistics for more accurate predictions!")
        
        return final_standings

def main():
    """
    Main function to run the enhanced Brasileirão prediction system
    """
    predictor = EnhancedBrasileiraoPredictor()
    
    # You can set use_real_data=True and provide api_key for real data
    final_standings = predictor.run_enhanced_analysis(use_real_data=False)
    
    # Save results
    final_standings.to_csv('/workspace/enhanced_brasileirao_predictions.csv', index=False)
    print(f"\nEnhanced results saved to 'enhanced_brasileirao_predictions.csv'")

if __name__ == "__main__":
    main()