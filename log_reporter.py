import os
import sys
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# Check for required API key
if "SERPER_API_KEY" not in os.environ:
    print("Error: SERPER_API_KEY environment variable is not set.")
    sys.exit(1)

print("Initializing SerperDevTool...")
search_tool = SerperDevTool()
print("SerperDevTool initialized successfully.")

# Define Agents
log_researcher = Agent(
    role="MacOS Log Researcher",
    goal="Find the best locations for system logs related to crashes and restarts on macOS",
    backstory="You are an expert in macOS systems with deep knowledge of log file locations and structures.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)

log_analyzer = Agent(
    role="Log File Analyzer",
    goal="Analyze log files to identify app crashes, system crashes, and machine restarts",
    backstory="You are a skilled data analyst specializing in parsing and interpreting system log files.",
    verbose=True,
    allow_delegation=False
)

data_organizer = Agent(
    role="Data Organizer",
    goal="Organize and categorize crash and restart data from log analysis",
    backstory="You excel at structuring complex data into clear, meaningful categories.",
    verbose=True,
    allow_delegation=False
)

report_compiler = Agent(
    role="Report Compiler",
    goal="Compile a comprehensive report on system crashes and restarts",
    backstory="You are an expert at synthesizing technical information into clear, actionable reports.",
    verbose=True,
    allow_delegation=False
)

# Define Tasks
task1 = Task(
    description="Research and list the most relevant log file locations for crashes and restarts on macOS.",
    agent=log_researcher
)

task2 = Task(
    description="Analyze the identified log files for app crashes, system crashes, and machine restarts. Provide a summary of findings.",
    agent=log_analyzer
)

task3 = Task(
    description="Organize the crash and restart data into categories. Identify patterns or frequent issues.",
    agent=data_organizer
)

task4 = Task(
    description="Compile a comprehensive report on the system crashes and restarts, including locations of log files, summary of findings, and organized data.",
    agent=report_compiler
)

# Create Crew
crew = Crew(
    agents=[log_researcher, log_analyzer, data_organizer, report_compiler],
    tasks=[task1, task2, task3, task4],
    verbose=2,
    process=Process.sequential
)

try:
    print("Starting crew tasks execution...")
    # Execute the crew's tasks
    result = crew.kickoff()
    print("Crew tasks completed successfully.")

    print("\n======== MacOS Log Analysis Report ========\n")
    print(result)
except Exception as e:
    print(f"An error occurred during execution: {str(e)}")
    print("Detailed error information:")
    import traceback
    traceback.print_exc()