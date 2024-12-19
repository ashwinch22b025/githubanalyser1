import json
import requests
import streamlit as st

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
            repos = requests.get(self.repo_url).json()
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
                repo_data = {k: repo.get(k, 'N/A') for k in DataNeeded}
                repo_stats.append(repo_data)
            return repo_stats

    repo = Repo(userName)
    all_repos = repo.get_all_repos()
    for idx, repo_data in enumerate(all_repos, start=1):
        st.write(f"Repository {idx}")
        st.json(repo_data)

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
