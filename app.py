import json
import requests
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import re

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.title("GitHub User, Repo, and Commit Info")

# Input username
userName = st.text_input("Enter GitHub Username")

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

def fetch_repo_data(repo_name, username):
    """
    Fetch all required repository data for Gemini evaluation.
    """
    try:
        # Fetch pull request data
        pulls_url = f'https://api.github.com/repos/{username}/{repo_name}/pulls?state=all'
        pulls = requests.get(pulls_url)
        pulls.raise_for_status()
        pulls = pulls.json()
        num_pulls = len(pulls) if isinstance(pulls, list) else 0

        # Fetch commit data
        commits_url = f'https://api.github.com/repos/{username}/{repo_name}/commits'
        commits = requests.get(commits_url)
        commits.raise_for_status()
        commits = commits.json()
        num_commits = len(commits) if isinstance(commits, list) else 0

        # Fetch README content
        readme_url = f'https://api.github.com/repos/{username}/{repo_name}/readme'
        readme_response = requests.get(readme_url)
        readme_response.raise_for_status()
        readme_content = requests.utils.unquote(readme_response.json().get('content', '')) if 'content' in readme_response.json() else 'N/A'

        # Fetch languages
        languages_url = f'https://api.github.com/repos/{username}/{repo_name}/languages'
        languages = requests.get(languages_url)
        languages.raise_for_status()
        languages = languages.json()

        # Fetch contributors
        contributors_url = f'https://api.github.com/repos/{username}/{repo_name}/contributors'
        contributors = requests.get(contributors_url)
        contributors.raise_for_status()
        contributors = contributors.json()
        num_contributors = len(contributors) if isinstance(contributors, list) else 0

        # Extract forks directly from the repo data
        forks_count = commits[0].get('commit', {}).get('committer', {}).get('name', 'N/A') if commits else 0

        return {
            'num_pulls': num_pulls,
            'num_commits': num_commits,
            'readme_content': readme_content,
            'languages': languages,
            'num_contributors': num_contributors,
            'forks': forks_count,
            'license': commits[0].get('commit', {}).get('committer', {}).get('date', 'N/A')  # Assumes license exists in commits
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data for repository {repo_name}: {e}")
        return {}

def extract_score_from_result(result):
    """
    Extract score from the result.
    """
    try:
        match = re.search(r'Score:\s*(\d+)/100', result)
        if match:
            score = int(match.group(1))
            return score
        else:
            raise ValueError("Score not found in result.")
    except Exception as e:
        st.error(f"Error extracting score: {e}")
        return None

def evaluate_repository_with_gemini(repo_name, username):
    """
    Evaluate a GitHub repository using the Gemini API based on available metrics.
    """
    repo_data = fetch_repo_data(repo_name, username)
    if not repo_data:
        return "Unable to fetch repository data for evaluation."

    rubric = [
        "Number of pull requests accepted",
        "Frequency of commits",
        "Quality of README",
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

    prompt = f"""
    Evaluate the GitHub repository '{repo_name}' for the user '{username}'.
    Use the following rubric to score the repository out of 100 marks. Each point is worth 5 marks:
    """ + "\n".join([f"{i + 1}. {point}" for i, point in enumerate(rubric)]) + "\n\nProvide a detailed evaluation and the final score."

    bot = Agent("Evaluate a GitHub repository based on a 20-point rubric, scoring each point out of 5 marks.")
    try:
        result = bot(prompt)
        score = extract_score_from_result(result)
        return result if score is not None else "Score could not be extracted."
    except Exception as e:
        return f"Error during evaluation: {e}"

def evaluate_all_repositories(username):
    """
    Fetch all repositories for a GitHub user and calculate an average score using the Gemini API.
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
                repo_name = repo.get('name', 'N/A')
                evaluation_result = evaluate_repository_with_gemini(repo_name, username)
                detailed_results.append({
                    'repository': repo_name,
                    'evaluation': evaluation_result
                })

                # Debugging: Log the evaluation result
                print(f"Evaluation Result for {repo_name}: {evaluation_result}")

                # Extract the score from the result (update parsing logic)
                try:
                    score = extract_score_from_result(evaluation_result)
                    if score is not None:
                        total_score += score
                        repo_count += 1
                except Exception as e:
                    print(f"Error parsing score for {repo_name}: {e}")

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
    st.subheader(f"Overall Evaluation for {userName}")
    evaluation_summary = evaluate_all_repositories(userName)
    st.text_area(label="Evaluation Summary", value=evaluation_summary, height=600)
