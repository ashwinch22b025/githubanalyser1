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
        def __init__(self, username):
            self.username = username
            self.user_url = f'https://api.github.com/users/{self.username}'
        
        def get_user_stats(self):
            try:
                user_data = requests.get(self.user_url).json()
                data_needed = [
                    'name', 'type', 'company', 'blog', 'location', 'email', 
                    'public_repos', 'followers'
                ]
                return {k: v for k, v in user_data.items() if k in data_needed}
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
                    data_needed = [
                        'name', 'html_url', 'description', 'forks', 
                        'open_issues', 'language', 'git_url',
                    ]
                    for repo in repos:
                        if isinstance(repo, dict):
                            repo_data = {k: repo.get(k, 'N/A') for k in data_needed}
                            repo_stats.append(repo_data)
                    return repo_stats
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
        Fetch all required repository data for evaluation.
        """
        try:
            repo_info = {}
            urls = {
                'pulls': f'https://api.github.com/repos/{username}/{repo_name}/pulls?state=all',
                'commits': f'https://api.github.com/repos/{username}/{repo_name}/commits',
                'readme': f'https://api.github.com/repos/{username}/{repo_name}/readme',
                'languages': f'https://api.github.com/repos/{username}/{repo_name}/languages',
                'contributors': f'https://api.github.com/repos/{username}/{repo_name}/contributors',
                'repo': f'https://api.github.com/repos/{username}/{repo_name}'
            }
            
            for key, url in urls.items():
                response = requests.get(url)
                if key == 'readme' and response.status_code == 200:
                    repo_info[key] = requests.utils.unquote(response.json().get('content', ''))
                elif key in ['pulls', 'commits', 'contributors']:
                    repo_info['num_' + key] = len(response.json()) if response.ok else 0
                elif key == 'languages':
                    repo_info[key] = response.json()
                else:
                    repo_info.update(response.json())

            return repo_info
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
            "Number of pull requests accepted",
            "Frequency of commits",
            "Quality of README file",
            "Number of contributors",
            "Diversity of programming languages used",
            "Languages",
            "Number of stars",
            "Number of forks",
            "License"
        ]

        bot = Agent("Evaluate a GitHub repository based on the provided metrics.")
        prompt = f"""
        Evaluate the repository '{repo_name}' for user '{username}' using the following data:
        - Number of pull requests: {repo_data.get('num_pulls', 0)}
        - Number of commits: {repo_data.get('num_commits', 0)}
        - README content: {repo_data.get('readme', 'N/A')[:200]}...
        - Number of contributors: {repo_data.get('num_contributors', 0)}
        - Languages: {', '.join(repo_data.get('languages', {}).keys()) or 'N/A'}
        - Number of stars: {repo_data.get('stargazers_count', 0)}
        - Number of forks: {repo_data.get('forks', 0)}
        - License: {repo_data.get('license', {}).get('name', 'N/A')}

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

            average_score = total_score / repo_count if repo_count else 0
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
