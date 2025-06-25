import requests
import time
import pandas as pd
import os
from datetime import datetime
import random

class RiotAPI:
    def __init__(self, api_key, region="na1"):
        self.api_key = api_key
        self.region = region
        self.base_url = f"https://{region}.api.riotgames.com/lol"
        self.headers = {"X-Riot-Token": api_key}
        self.region_v5 = self._get_region_v5(region)
        self.base_url_v5 = f"https://{self.region_v5}.api.riotgames.com/lol"
        
        # Rate limit tracking
        self.short_limit_count = 0
        self.short_limit_start = time.time()
        self.long_limit_count = 0
        self.long_limit_start = time.time()
    
    def _get_region_v5(self, region):
        region_mapping = {
            "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
            "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe",
            "kr": "asia", "jp1": "asia",
            "oc1": "sea", "ph2": "sea", "sg2": "sea", "th2": "sea", "tw2": "sea", "vn2": "sea"
        }
        return region_mapping.get(region, "americas")
    
    def _respect_rate_limits(self):
        """Manage rate limits: 20 requests/1s and 100 requests/2min"""
        current_time = time.time()
        
        # Short limit: 20 requests every 1 second
        if current_time - self.short_limit_start >= 1.0:
            # Reset short window counter if more than 1 second has passed
            self.short_limit_count = 0
            self.short_limit_start = current_time
        elif self.short_limit_count >= 20:
            # Sleep until the 1 second window is over
            sleep_time = 1.0 - (current_time - self.short_limit_start)
            if sleep_time > 0:
                print(f"Short rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            self.short_limit_count = 0
            self.short_limit_start = time.time()
        
        # Long limit: 100 requests every 2 minutes
        if current_time - self.long_limit_start >= 120.0:
            # Reset long window counter if more than 2 minutes have passed
            self.long_limit_count = 0
            self.long_limit_start = current_time
        elif self.long_limit_count >= 100:
            # Sleep until the 2 minute window is over
            sleep_time = 120.0 - (current_time - self.long_limit_start)
            if sleep_time > 0:
                print(f"Long rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            self.long_limit_count = 0
            self.long_limit_start = time.time()
    
    def _make_request(self, url, params=None):
        """Make API request with rate limit handling"""
        self._respect_rate_limits()
        
        response = requests.get(url, headers=self.headers, params=params)
        
        # Update rate limit counters
        self.short_limit_count += 1
        self.long_limit_count += 1
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            # Rate limit exceeded according to server
            retry_after = int(response.headers.get('Retry-After', 1))
            print(f"Rate limit exceeded (server). Waiting {retry_after} seconds.")
            time.sleep(retry_after)
            # Try again (recursive call)
            return self._make_request(url, params)
        else:
            print(f"Error: {response.status_code}, URL: {url}")
            if response.status_code == 403:
                print("API key may be invalid or expired")
            return None
    
    def get_challenger_league(self, queue="RANKED_SOLO_5x5"):
        url = f"{self.base_url}/league/v4/challengerleagues/by-queue/{queue}"
        return self._make_request(url)
    
    def get_summoner_by_id(self, summoner_id):
        url = f"{self.base_url}/summoner/v4/summoners/{summoner_id}"
        return self._make_request(url)
    
    def get_match_ids(self, puuid, count=20, start=0, queue=None):
        url = f"{self.base_url_v5}/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"count": count, "start": start}
        if queue:
            params["queue"] = queue
        return self._make_request(url, params)
    
    def get_match_details(self, match_id):
        url = f"{self.base_url_v5}/match/v5/matches/{match_id}"
        return self._make_request(url)
