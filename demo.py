import os
import logging
from dotenv import load_dotenv
from agents.profile_scraper_agent import ProfileScraperAgent

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Initialize the profile scraper agent
    scraper_agent = ProfileScraperAgent()
    
    # Define job role and description
    job_role = "Python Developer"
    job_description = """
    We are looking for an experienced Python Developer to join our team.
    
    Requirements:
    - 3+ years of experience with Python development
    - Strong understanding of web frameworks (Django, Flask)
    - Experience with databases (SQL, NoSQL)
    - Knowledge of RESTful APIs
    - Familiarity with cloud services (AWS, GCP)
    - Good problem-solving skills
    
    Preferred:
    - Experience with machine learning libraries
    - Knowledge of Docker and Kubernetes
    - Understanding of CI/CD pipelines
    """
    location = "India"  # Optional location filter
    
    # Step 1: Collect real profiles
    logger.info(f"Collecting profiles for {job_role} in {location}")
    profiles = scraper_agent.collect_profiles(job_role, location, num_results=5)
    logger.info(f"Collected {len(profiles)} profiles")
    
    # Step 2: Analyze candidates using RAG
    if profiles:
        logger.info("Analyzing candidates using RAG")
        analysis = scraper_agent.analyze_candidates(job_role, job_description)
        
        # Print results
        print("\n" + "="*50)
        print("CANDIDATE ANALYSIS RESULTS")
        print("="*50 + "\n")
        
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
        else:
            print(analysis["analysis"])
    else:
        logger.warning("No profiles collected, skipping analysis")

if __name__ == "__main__":
    main()