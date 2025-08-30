import requests
import logging
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from googlesearch import search
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinkedInScraper:
    def __init__(self, api_key: str, api_host: str = "linkedin-api8.p.rapidapi.com"):
        """
        Initialize the LinkedIn scraper with RapidAPI credentials
        
        Args:
            api_key: RapidAPI key for LinkedIn API
            api_host: RapidAPI host for LinkedIn API
        """
        self.api_url = f"https://{api_host}/"
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": api_host
        }
        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def find_profiles(self, job_role: str, location: str = None, num_results: int = 5) -> List[str]:
        """
        Use Google dorking to find LinkedIn profiles matching criteria
        
        Args:
            job_role: The job role to search for
            location: Optional location filter
            num_results: Number of results to return
            
        Returns:
            List of LinkedIn usernames
        """
        location_term = f'"{location}"' if location else ""
        query = f'site:linkedin.com/in/ "{job_role}" {location_term} -jobs -careers'
        
        return self.extract_linkedin_usernames(query, num_results)
        
    def extract_linkedin_usernames(self, query: str, num_results: int = 5) -> List[str]:
        """
        Extract LinkedIn usernames from Google search results
        
        Args:
            query: Google dork query
            num_results: Number of search results to process
            
        Returns:
            List of LinkedIn usernames
        """
        usernames = []
        logger.info("Searching Google with query: %s", query)
        
        try:
            for url in search(query, num_results=num_results, lang="en"):
                parsed_url = urlparse(url)
                # Check if the URL contains '/in/'
                if "linkedin.com/in/" in url:
                    # Split the path and filter out empty strings
                    parts = [p for p in parsed_url.path.strip("/").split("/") if p]
                    # We expect the first part to be "in" and the second part to be the username
                    if parts and parts[0].lower() == "in" and len(parts) >= 2:
                        username = parts[1]
                        if username and username not in usernames:
                            usernames.append(username)
                    else:
                        logger.debug("URL did not have the expected structure: %s", url)
                        
            logger.info("Extracted LinkedIn usernames: %s", usernames)
            return usernames
        except Exception as e:
            logger.error(f"Error extracting LinkedIn usernames: {e}")
            return []
    
    def get_profile_details(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve profile details for a LinkedIn username using RapidAPI
        
        Args:
            username: LinkedIn username/profile ID
            
        Returns:
            Dictionary containing profile data or None if retrieval failed
        """
        querystring = {"username": username}
        logger.info(f"Fetching profile for username: {username}")
        
        try:
            response = requests.get(self.api_url, headers=self.headers, params=querystring)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get profile for {username}, status code: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving profile for {username}: {e}")
            return None
            
    def process_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw profile data into a structured format
        
        Args:
            profile_data: Raw profile data from API
            
        Returns:
            Processed profile with relevant fields extracted and embedding
        """
        if not profile_data:
            return {}
            
        # Extract basic info and create profile text as in agent.py
        profile_text = f"{profile_data.get('name', '')}\n{profile_data.get('headline', '')}\n{profile_data.get('summary', '')}"
        
        # Create the structured profile
        processed = {
            "username": profile_data.get('username', ''),
            "profile_text": profile_text,
            "source_url": f"https://www.linkedin.com/in/{profile_data.get('username', '')}",
            "name": profile_data.get('name', ''),
            "headline": profile_data.get('headline', ''),
            "summary": profile_data.get('summary', '')
        }
        
        # Extract experience details if available
        if 'experience' in profile_data:
            processed['experience'] = profile_data.get('experience', [])
            
            # Add experience details to profile text
            for exp in processed['experience']:
                profile_text += f"\nRole: {exp.get('title', '')}\nCompany: {exp.get('company', '')}\n"
                if 'date_range' in exp:
                    profile_text += f"Duration: {exp.get('date_range', '')}\n"
                if 'description' in exp:
                    profile_text += f"Description: {exp.get('description', '')}\n"
        
        # Extract education details if available
        if 'education' in profile_data:
            processed['education'] = profile_data.get('education', [])
            
            # Add education details to profile text
            for edu in processed['education']:
                profile_text += f"\nEducation: {edu.get('school', '')}, {edu.get('degree', '')}, {edu.get('field', '')}\n"
        
        # Extract skills if available
        if 'skills' in profile_data:
            processed['skills'] = profile_data.get('skills', [])
            profile_text += f"\nSkills: {', '.join(processed['skills'])}"
        
        # Update the profile text with all the additional information
        processed["profile_text"] = profile_text
        
        # Generate embedding
        processed["embedding"] = self.model.encode(profile_text).tolist()
        
        return processed
        
    def get_profiles(self, job_role: str, location: Optional[str] = None, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Complete profile extraction pipeline - find usernames, get details, process data
        
        Args:
            job_role: The job role to search for
            location: Optional location filter
            num_results: Number of results to return
            
        Returns:
            List of processed profile data dictionaries
        """
        # Find profile usernames
        usernames = self.find_profiles(job_role, location, num_results)
        
        # Get and process profiles
        profiles = []
        for username in usernames:
            profile_json = self.get_profile_details(username)
            if profile_json:
                processed_profile = self.process_profile(profile_json)
                if processed_profile:
                    profiles.append(processed_profile)
        
        logger.info(f"Fetched and processed {len(profiles)} profiles for '{job_role}'")
        return profiles