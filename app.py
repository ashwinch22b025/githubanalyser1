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

    def fetch_repo_data(repo_name, username):
        """
        Fetch all required repository data for Gemini evaluation.
        """
        try:
            # Fetch pull request data
            pulls_url = f'https://api.github.com/repos/{username}/{repo_name}/pulls?state=all'
            pulls = requests.get(pulls_url).json()
            num_pulls = len(pulls) if isinstance(pulls, list) else 0

            # Fetch commit data
            commits_url = f'https://api.github.com/repos/{username}/{repo_name}/commits'
            commits = requests.get(commits_url).json()
            num_commits = len(commits) if isinstance(commits, list) else 0

            # Fetch README content
            readme_url = f'https://api.github.com/repos/{username}/{repo_name}/readme'
            readme_response = requests.get(readme_url).json()
            readme_content = requests.utils.unquote(readme_response.get('content', '')) if 'content' in readme_response else 'N/A'

            # Fetch languages
            languages_url = f'https://api.github.com/repos/{username}/{repo_name}/languages'
            languages = requests.get(languages_url).json()

            # Fetch contributors
            contributors_url = f'https://api.github.com/repos/{username}/{repo_name}/contributors'
            contributors = requests.get(contributors_url).json()
            num_contributors = len(contributors) if isinstance(contributors, list) else 0

            return {
                'num_pulls': num_pulls,
                'num_commits': num_commits,
                'readme_content': readme_content,
                'languages': languages,
                'num_contributors': num_contributors,
                'stargazers_count': requests.get(f'https://api.github.com/repos/{username}/{repo_name}').json().get('stargazers_count', 0),
                'forks': requests.get(f'https://api.github.com/repos/{username}/{repo_name}').json().get('forks', 0),
                'license': requests.get(f'https://api.github.com/repos/{username}/{repo_name}').json().get('license', {}).get('name', 'N/A')
            }
        except Exception as e:
            st.error(f"Error fetching data for repository {repo_name}: {e}")
            return {}

    def evaluate_repository_with_gemini(repo_name, username):
        """
        Evaluate a GitHub repository using the Gemini API based on available metrics.
        """
        repo_data = fetch_repo_data(repo_name, username)
        if not repo_data:
            return "Unable to fetch repository data for evaluation."

        rubric = [
            "Number of pull requests accepted (30%)",
            "Frequency of commits (20%)",
            "Quality of README file (15%)",
            "Number of contributors (10%)",
            "Diversity of programming languages used (10%)",
            "Number of stars (10%)",
            "Number of forks (5%)",
            "License (10%)"
        ]

        bot = Agent("Evaluate a GitHub repository based on the provided metrics.")
        prompt = f"""
        Evaluate the repository '{repo_name}' for user '{username}' using the following data:
        - Number of pull requests: {repo_data['num_pulls']}
        - Number of commits: {repo_data['num_commits']}
        - README content: {repo_data['readme_content'][:200]}...
        - Number of contributors: {repo_data['num_contributors']}
        - Languages: {', '.join(repo_data['languages'].keys())}
        - Number of stars: {repo_data['stargazers_count']}
        - Number of forks: {repo_data['forks']}
        - License: {repo_data['license']}

        Use the following rubric:
        """ + "\n".join([f"{i + 1}. {point}" for i, point in enumerate(rubric)]) + "\n\nProvide a detailed evaluation and a score out of 100."

        try:
            result = bot(prompt)
            return result
        except Exception as e:
            return f"Error during evaluation: {e}"

    def evaluate_all_repositories(username):
        """
        Fetch all repositories for a GitHub user and calculate an average score using the Gemini API.
        """
        try:
            if not all_repos:
                return "No repositories found for the user."

            total_score = 0
            repo_count = 0
            detailed_results = []

            for repo in all_repos:
                repo_name = repo.get('name', 'N/A')
                if repo_name != 'N/A':
                    evaluation_result = evaluate_repository_with_gemini(repo_name, username)
                    detailed_results.append({
                        'repository': repo_name,
                        'evaluation': evaluation_result
                    })

                    # Extract score (update parsing logic based on AI response)
                    try:
                        score_line = evaluation_result.splitlines()[-1]  # Assuming score is on the last line
                        score = int([s for s in score_line.split() if s.isdigit()][0])
                        total_score += score
                        repo_count += 1
                    except Exception as e:
                        print(f"Error parsing score for {repo_name}: {e}")

            if repo_count == 0:
                return "No valid scores could be calculated for the user's repositories."

            average_score = total_score / repo_count
            summary = f"Overall Average Score for {username}: {average_score:.2f}/100\n\n"
            for result in detailed_results:
                summary += f"Repository: {result['repository']}\nEvaluation: {result['evaluation']}\n\n"

            return summary

        except Exception as e:
            return f"Error fetching or evaluating repositories: {e}"

    if st.button("Evaluate Repositories"):
        st.subheader(f"Overall Evaluation for User: {userName}")
        overall_evaluation = evaluate_all_repositories(userName)
        st.write(overall_evaluation)
else:
    st.write("Please enter a GitHub username.")
