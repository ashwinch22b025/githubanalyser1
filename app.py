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

if userName:
    st.subheader("User Info")

    # User Info from User class
    class User:
        def __init__(self, Username):
            self.Username = Username
            self.UserURL = f'https://api.github.com/users/{self.Username}'
        
        def get_user_stats(self):
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
            # Ensure the response is valid and is a list of repositories
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
                        if isinstance(repo, dict):  # Ensure each repo is a dictionary
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

    # Check if repositories are available and display them
    if all_repos:
        for idx, repo_data in enumerate(all_repos, start=1):
            st.write(f"Repository {idx}")
            st.json(repo_data)

            # Fetch the README file and analyze it using the Gemini API
            readme_url = f'https://api.github.com/repos/{userName}/{repo_data["name"]}/readme'
            readme_response = requests.get(readme_url).json()

            # Check if README exists for the repository
            if 'content' in readme_response:
                readme_content = readme_response['content']
                readme_decoded = requests.utils.unquote(readme_content)  # Decode base64 content if needed

                # Analyze README content with Gemini API
                bot = Agent("Analyze the README content of a GitHub repository.")
                analysis_result = bot(readme_decoded)
                st.subheader(f"README Analysis for Repository: {repo_data['name']}")
                st.write(analysis_result)
            else:
                st.write("README not available for this repository.")
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
            commit_data = requests.get(self.commit_url).json()
            return {
                'committer': commit_data['commit']['committer'],
                'commit': commit_data['commit'],
                'message': commit_data['commit']['message']
            }

    # Fetch commits for the first repository as an example
    if all_repos:
        first_repo_name = all_repos[0]['name']
        st.write(f"Fetching commits for repository: {first_repo_name}")
        repo_url = f'https://api.github.com/repos/{userName}/{first_repo_name}/commits'
        commits = requests.get(repo_url).json()
        if commits:
            for commit in commits[:5]:  # Limit to 5 commits for display
                st.json({
                    'SHA': commit.get('sha'),
                    'Message': commit['commit']['message'],
                    'Author': commit['commit']['author'],
                })
else:
    st.write("Please enter a GitHub username.")

# Agent class for Generative AI interaction (same as in your provided code)
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
        # Use Gemini API without specifying the version
        chat = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)
        result = chat.invoke(self.messages)
        return result.content
