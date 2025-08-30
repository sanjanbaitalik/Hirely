from dotenv import load_dotenv
from tasks.hr_tasks import HRTasks
from crewai import Crew, Process
import os

load_dotenv()

def main():
    hr_query = input("HR, please enter your job-role query: ")

    hr_tasks = HRTasks()

    # Step 1: Interpret HR's query first
    query_crew = Crew(
        agents=[hr_tasks.hr_query_agent()],
        tasks=[hr_tasks.handle_hr_query(hr_query)],
        verbose=True
    )

    # Get the result from CrewOutput object
    crew_output = query_crew.kickoff()
    # Access the actual string result
    job_details = str(crew_output)
    job_role = job_details.strip().replace("Job Role:", "").strip()
    
    print(f"Interpreted job role: {job_role}")

    # Now start subsequent tasks with interpreted role
    hr_crew = Crew(
        agents=[
            hr_tasks.profile_scraper_agent(job_role),
            hr_tasks.cv_screening_agent(),
            hr_tasks.communication_agent(),
            hr_tasks.interview_scheduler_agent(),
            hr_tasks.reporting_agent(),
        ],
        tasks=[
            hr_tasks.scrape_profiles(job_role),
            hr_tasks.screen_cvs(job_role),
            hr_tasks.communicate(),
            hr_tasks.schedule_interviews(),
            hr_tasks.generate_report()
        ],
        verbose=True,
        process=Process.sequential
    )

    results = hr_crew.kickoff()
    print("Final Results:")
    print(results)

if __name__ == "__main__":
    main()