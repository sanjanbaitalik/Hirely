from crewai import Agent, Task
from agents.profile_scraper_agent import ProfileScraperAgent
from agents.cv_screening_agent import CVScreeningAgent
from agents.communication_agent import CommunicationAgent
from agents.interview_scheduler_agent import InterviewSchedulerAgent
from agents.reporting_agent import ReportingAgent
from agents.hr_query_agent import HRQueryAgent

class HRTasks:
    def hr_query_agent(self):
        return HRQueryAgent.agent()

    def profile_scraper_agent(self, job_role):
        return ProfileScraperAgent.agent(job_role=job_role)

    def cv_screening_agent(self):
        return CVScreeningAgent.agent()

    def communication_agent(self):
        return CommunicationAgent.agent()

    def interview_scheduler_agent(self):
        return InterviewSchedulerAgent.agent()

    def reporting_agent(self):
        return ReportingAgent.agent()

    def handle_hr_query(self, hr_query):
        return Task(
            description=(
                f"Interpret this HR query: '{hr_query}'. Clearly specify the exact job role and essential skills. "
                "Pass these details to subsequent tasks for scraping and screening."
            ),
            agent=self.hr_query_agent(),
            expected_output="Clearly identified job role and essential skills from HR's query."
        )

    def scrape_profiles(self, job_role):
        return Task(
            description=f"Scrape candidate profiles matching the role: '{job_role}'.",
            agent=self.profile_scraper_agent(job_role=job_role),
            expected_output="Excel file of candidate profiles for given role."
        )

    def screen_cvs(self, job_role):
        return Task(
            description=f"Screen and score CVs for candidates relevant to '{job_role}'.",
            agent=self.cv_screening_agent(),
            expected_output="CSV file with scored CVs for the role."
        )

    def communicate(self):
        return Task(
            description="Communicate interview information to shortlisted candidates.",
            agent=self.communication_agent(),
            expected_output="Record of communications sent to candidates."
        )

    def schedule_interviews(self):
        return Task(
            description="Schedule interviews with candidates based on availability.",
            agent=self.interview_scheduler_agent(),
            expected_output="Confirmed schedule of interviews."
        )

    def generate_report(self):
        return Task(
            description="Generate a comprehensive recruitment summary report.",
            agent=self.reporting_agent(),
            expected_output="Recruitment report document."
        )
