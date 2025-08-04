#!/usr/bin/env python3
"""
Brasileirão Data Scraper
========================

This script provides functions to scrape real Brasileirão data from various sources.
This can be used as an alternative to the simulated data in the main prediction script.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from urllib.parse import urljoin

class BrasileiraoDataScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def scrape_espn_standings(self):
        """
        Scrape current standings from ESPN Brazil
        """
        print("Scraping current standings from ESPN...")
        
        try:
            url = "https://www.espn.com.br/futebol/liga/_/nome/bra.1"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # This is a simplified example - actual scraping would need to be
            # adapted based on the current website structure
            teams_data = []
            
            # Look for table rows with team data
            table_rows = soup.find_all('tr', class_='Table__TR')
            
            for row in table_rows:
                # Extract team name, points, goals, etc.
                # This would need to be customized based on ESPN's actual HTML structure
                team_data = self.extract_team_data_from_row(row)
                if team_data:
                    teams_data.append(team_data)
            
            return pd.DataFrame(teams_data)
            
        except Exception as e:
            print(f"Error scraping ESPN: {e}")
            return None
    
    def scrape_transfermarkt_data(self):
        """
        Scrape detailed team statistics from Transfermarkt
        """
        print("Scraping detailed statistics from Transfermarkt...")
        
        try:
            # Base URL for Brasileirão Serie A
            base_url = "https://www.transfermarkt.com/campeonato-brasileiro-serie-a/tabelle/wettbewerb/BRA1"
            
            response = requests.get(base_url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract table data
            teams_data = []
            
            # Find the main table
            table = soup.find('table', class_='items')
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    team_data = self.extract_transfermarkt_data(row)
                    if team_data:
                        teams_data.append(team_data)
            
            return pd.DataFrame(teams_data)
            
        except Exception as e:
            print(f"Error scraping Transfermarkt: {e}")
            return None
    
    def extract_team_data_from_row(self, row):
        """
        Extract team data from a table row (ESPN format)
        """
        try:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 8:
                return None
            
            team_data = {
                'team': cells[1].get_text(strip=True),
                'games_played': int(cells[2].get_text(strip=True)),
                'wins': int(cells[3].get_text(strip=True)),
                'draws': int(cells[4].get_text(strip=True)),
                'losses': int(cells[5].get_text(strip=True)),
                'goals_for': int(cells[6].get_text(strip=True)),
                'goals_against': int(cells[7].get_text(strip=True)),
                'points': int(cells[8].get_text(strip=True))
            }
            
            # Calculate additional metrics
            team_data['goal_difference'] = team_data['goals_for'] - team_data['goals_against']
            team_data['points_per_game'] = team_data['points'] / team_data['games_played']
            team_data['win_rate'] = team_data['wins'] / team_data['games_played']
            
            return team_data
            
        except (ValueError, IndexError, AttributeError):
            return None
    
    def extract_transfermarkt_data(self, row):
        """
        Extract team data from Transfermarkt table row
        """
        try:
            cells = row.find_all('td')
            if len(cells) < 10:
                return None
            
            # Extract team name (usually in an anchor tag)
            team_cell = cells[1]
            team_name = team_cell.find('a')
            if team_name:
                team_name = team_name.get_text(strip=True)
            else:
                team_name = team_cell.get_text(strip=True)
            
            team_data = {
                'team': team_name,
                'games_played': int(cells[2].get_text(strip=True)),
                'wins': int(cells[3].get_text(strip=True)),
                'draws': int(cells[4].get_text(strip=True)),
                'losses': int(cells[5].get_text(strip=True)),
                'goals_for': int(cells[6].get_text(strip=True)),
                'goals_against': int(cells[7].get_text(strip=True)),
                'goal_difference': int(cells[8].get_text(strip=True)),
                'points': int(cells[9].get_text(strip=True))
            }
            
            # Calculate additional metrics
            if team_data['games_played'] > 0:
                team_data['points_per_game'] = team_data['points'] / team_data['games_played']
                team_data['win_rate'] = team_data['wins'] / team_data['games_played']
                team_data['goals_per_game'] = team_data['goals_for'] / team_data['games_played']
                team_data['goals_conceded_per_game'] = team_data['goals_against'] / team_data['games_played']
            
            return team_data
            
        except (ValueError, IndexError, AttributeError):
            return None
    
    def scrape_api_football_data(self, api_key=None):
        """
        Scrape data from API-Football (requires API key)
        """
        if not api_key:
            print("API key required for API-Football")
            return None
        
        print("Fetching data from API-Football...")
        
        try:
            headers = {
                'X-RapidAPI-Key': api_key,
                'X-RapidAPI-Host': 'api-football-v1.p.rapidapi.com'
            }
            
            # Get league standings
            url = "https://api-football-v1.p.rapidapi.com/v3/standings"
            querystring = {"season": "2024", "league": "71"}  # 71 is Brasileirão Serie A
            
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            
            data = response.json()
            
            teams_data = []
            if 'response' in data and len(data['response']) > 0:
                standings = data['response'][0]['league']['standings'][0]
                
                for team_info in standings:
                    team_data = {
                        'team': team_info['team']['name'],
                        'current_position': team_info['rank'],
                        'points': team_info['points'],
                        'games_played': team_info['all']['played'],
                        'wins': team_info['all']['win'],
                        'draws': team_info['all']['draw'],
                        'losses': team_info['all']['lose'],
                        'goals_for': team_info['all']['goals']['for'],
                        'goals_against': team_info['all']['goals']['against'],
                        'goal_difference': team_info['goalsDiff']
                    }
                    
                    # Calculate additional metrics
                    if team_data['games_played'] > 0:
                        team_data['points_per_game'] = team_data['points'] / team_data['games_played']
                        team_data['win_rate'] = team_data['wins'] / team_data['games_played']
                        team_data['goals_per_game'] = team_data['goals_for'] / team_data['games_played']
                        team_data['goals_conceded_per_game'] = team_data['goals_against'] / team_data['games_played']
                    
                    teams_data.append(team_data)
            
            return pd.DataFrame(teams_data)
            
        except Exception as e:
            print(f"Error fetching from API-Football: {e}")
            return None
    
    def get_real_data(self, source='auto', api_key=None):
        """
        Get real Brasileirão data from the best available source
        """
        print("Attempting to gather real Brasileirão data...")
        
        if source == 'api' and api_key:
            data = self.scrape_api_football_data(api_key)
            if data is not None and not data.empty:
                print(f"Successfully gathered data from API-Football: {len(data)} teams")
                return data
        
        if source in ['auto', 'transfermarkt']:
            print("Trying Transfermarkt...")
            time.sleep(1)  # Be respectful with requests
            data = self.scrape_transfermarkt_data()
            if data is not None and not data.empty:
                print(f"Successfully gathered data from Transfermarkt: {len(data)} teams")
                return data
        
        if source in ['auto', 'espn']:
            print("Trying ESPN...")
            time.sleep(1)  # Be respectful with requests
            data = self.scrape_espn_standings()
            if data is not None and not data.empty:
                print(f"Successfully gathered data from ESPN: {len(data)} teams")
                return data
        
        print("Could not gather real data from any source. Consider using simulated data.")
        return None
    
    def save_data(self, data, filename):
        """
        Save scraped data to CSV file
        """
        if data is not None and not data.empty:
            data.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        else:
            print("No data to save")

def main():
    """
    Example usage of the data scraper
    """
    scraper = BrasileiraoDataScraper()
    
    # Try to get real data (will fall back through sources if needed)
    real_data = scraper.get_real_data(source='auto')
    
    if real_data is not None:
        print("\nReal data sample:")
        print(real_data.head())
        
        # Save the data
        scraper.save_data(real_data, '/workspace/real_brasileirao_data.csv')
        
        print(f"\nTotal teams found: {len(real_data)}")
        print(f"Columns available: {list(real_data.columns)}")
    else:
        print("Failed to gather real data. Please check your internet connection or use simulated data.")

if __name__ == "__main__":
    main()