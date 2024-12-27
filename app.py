import json
import requests
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.title("GitHub User, Repo, and Commit Info")

# Input username
userName = st.text_input("Enter GitHub Username")

# Agent class for Generative AI interaction
class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append(SystemMessage(content=self.system))
    
    def __call__(self, message):
        self.messages.append(HumanMessage(content=message))
        result = self.execute()
        self.messages.append(AIMessage(content=result))
        return result
    
    def execute(self):
        chat = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)
        result = chat.invoke(self.messages)
        return result.content

if userName:
    st.subheader("User Info")

    # User Info from User class
    class User:
        def __init__(self, Username):
            self.Username = Username
            self.UserURL = f'https://api.github.com/users/{self.Username}'
        
        def get_user_stats(self):
            try:
                UserDataFromGithub = requests.get(self.UserURL).json()
                DataNeeded = [
                    'name',
                    'type',
                    'company',
                    'blog',
                    'location',
                    'email',
                    'public_repos',
                    'followers'
                ]
                self.UserData = {k: v for k, v in UserDataFromGithub.items() if k in DataNeeded}
                return self.UserData
            except Exception as e:
                st.error(f"Error fetching user info: {e}")
                return {}

    user = User(userName)
    user_data = user.get_user_stats()
    st.json(user_data)

    st.subheader("Repositories Info")

    # Repository Info from Repo class
    class Repo:
        def __init__(self, username):
            self.username = username
            self.repo_url = f'https://api.github.com/users/{self.username}/repos'
        
        def get_all_repos(self):
            try:
                repos = requests.get(self.repo_url).json()
                if isinstance(repos, list):
                    repo_stats = []
                    DataNeeded = [
                        'name',
                        'html_url',
                        'description',
                        'forks',
                        'open_issues',
                        'language',
                        'git_url',
                    ]
                    for repo in repos:
                        if isinstance(repo, dict):
                            repo_data = {k: repo.get(k, 'N/A') for k in DataNeeded}
                            repo_stats.append(repo_data)
                    return repo_stats
                else:
                    return []
            except Exception as e:
                st.error(f"Error fetching repositories: {e}")
                return []

    repo = Repo(userName)
    all_repos = repo.get_all_repos()

    if all_repos:
        for idx, repo_data in enumerate(all_repos, start=1):
            st.write(f"Repository {idx}")
            st.json(repo_data)

            # Fetch the README file and analyze it using the Gemini API
            readme_url = f'https://api.github.com/repos/{userName}/{repo_data["name"]}/readme'
            try:
                readme_response = requests.get(readme_url).json()
                if 'content' in readme_response:
                    readme_content = readme_response['content']
                    try:
                        readme_decoded = requests.utils.unquote(readme_content)
                        bot = Agent("Analyze the README content of a GitHub repository.")
                        analysis_result = bot(readme_decoded)
                        st.subheader(f"README Analysis for Repository: {repo_data['name']}")
                        st.write(analysis_result)
                    except Exception:
                        st.subheader(f"README Analysis for Repository: {repo_data['name']}")
                        st.write("NULL")
                else:
                    st.subheader(f"README Analysis for Repository: {repo_data['name']}")
                    st.write("NULL")
            except Exception as e:
                st.error(f"Error fetching README for {repo_data['name']}: {e}")
    else:
        st.write("No repositories found or there was an error fetching repositories.")

    st.subheader("Commit Info")

    # Commit Info from Commit class
    class Commit:
        def __init__(self, username, project_id, sha):
            self.username = username
            self.project_id = project_id
            self.sha = sha
            self.commit_url = f'https://api.github.com/repos/{self.username}/{self.project_id}/commits/{self.sha}'
        
        def get_commit_stats(self):
            try:
                commit_data = requests.get(self.commit_url).json()
                return {
                    'committer': commit_data.get('commit', {}).get('committer', {}),
                    'commit': commit_data.get('commit', {}),
                    'message': commit_data.get('commit', {}).get('message', 'N/A')
                }
            except Exception as e:
                st.error(f"Error fetching commit stats: {e}")
                return {}

    if all_repos:
        first_repo_name = all_repos[0]['name']
        st.write(f"Fetching commits for repository: {first_repo_name}")
        repo_url = f'https://api.github.com/repos/{userName}/{first_repo_name}/commits'
        try:
            commits = requests.get(repo_url).json()
            if isinstance(commits, list):
                for commit in commits[:5]:
                    if isinstance(commit, dict):
                        st.json({
                            'SHA': commit.get('sha', 'N/A'),
                            'Message': commit.get('commit', {}).get('message', 'N/A'),
                            'Author': commit.get('commit', {}).get('author', {}),
                        })
            else:
                st.write("No commits found or an error occurred.")
        except Exception as e:
            st.error(f"Error fetching commits: {e}")
else:
    st.write("Please enter a GitHub username.")


def evaluate_repository_with_gemini(repo_data, username):
    """
    Evaluate a GitHub repository using the Gemini API based on a 20-point rubric.

    Args:
        repo_data (dict): Data about the repository, including its name.
        username (str): GitHub username.

    Returns:
        str: Evaluation summary scored out of 100 marks.
    """
    # Define the rubric (example points, can be expanded)
    rubric = [
        "Number of pull requests accepted",
        "Frequency of commits",
        "Quality of README file",
        "Number of stars",
        "Number of forks",
        "Number of contributors",
        "Open issues to closed issues ratio",
        "Use of automated tests",
        "Presence of CI/CD pipelines",
        "Activity in the past month",
        "Diversity of programming languages used",
        "Presence of a well-documented CONTRIBUTING file",
        "Code coverage (if reported)",
        "Security vulnerabilities flagged",
        "Number of releases",
        "Use of GitHub Actions or equivalent",
        "Community engagement in discussions",
        "Quality of documentation other than README",
        "License presence and clarity",
        "Use of tags for versioning"
    ]

    # Initialize the Agent
    bot = Agent("Evaluate a GitHub repository based on a 20-point rubric, scoring each point out of 5 marks.")

    # Create a prompt including the rubric and repository details
    prompt = f"""
    Evaluate the GitHub repository '{repo_data['name']}' for the user '{username}'.
    Use the following rubric to score the repository out of 100 marks. Each point is worth 5 marks:
    """ + "\n".join([f"{i + 1}. {point}" for i, point in enumerate(rubric)]) + "\n\nProvide a detailed evaluation and the final score."

    # Execute the evaluation
    try:
        result = bot(prompt)
        return result
    except Exception as e:
        return f"Error during evaluation: {e}"


def evaluate_all_repositories(username):
    """
    Fetch all repositories for a GitHub user and calculate an average score using the Gemini API.

    Args:
        username (str): GitHub username.

    Returns:
        str: Overall evaluation summary and average score.
    """
    repo_url = f'https://api.github.com/users/{username}/repos'
    try:
        repos = requests.get(repo_url).json()
        if not isinstance(repos, list):
            return "Error fetching repositories or user has no repositories."

        total_score = 0
        repo_count = 0
        detailed_results = []

        for repo in repos:
            if isinstance(repo, dict):
                repo_data = {
                    'name': repo.get('name', 'N/A'),
                    'html_url': repo.get('html_url', 'N/A')
                }
                evaluation_result = evaluate_repository_with_gemini(repo_data, username)
                detailed_results.append({
                    'repository': repo_data['name'],
                    'evaluation': evaluation_result
                })

                # Debugging: Log the evaluation result
                print(f"Evaluation Result for {repo_data['name']}: {evaluation_result}")

                # Extract the score from the result (update parsing logic)
                try:
                    # Adjust the extraction logic based on actual format
                    score_line = evaluation_result.splitlines()[-1]  # Assuming score is on the last line
                    score = int([s for s in score_line.split() if s.isdigit()][0])  # Extract first number
                    total_score += score
                    repo_count += 1
                except Exception as e:
                    print(f"Error parsing score for {repo_data['name']}: {e}")

        if repo_count == 0:
            return "No valid scores could be calculated for the user's repositories."

        average_score = total_score / repo_count
        summary = f"Overall Average Score for {username}: {average_score}/100\n\n"
        for result in detailed_results:
            summary += f"Repository: {result['repository']}\nEvaluation: {result['evaluation']}\n\n"

        return summary

    except Exception as e:
        return f"Error fetching or evaluating repositories: {e}"


if userName and all_repos:
    st.subheader(f"Overall Evaluation for User: {userName}")
    overall_evaluation = evaluate_all_repositories(userName)
    st.write(overall_evaluation)
