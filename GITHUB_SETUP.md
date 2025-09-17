# GitHub Repository Setup

Follow these steps to create the GitHub repository and push your code:

## Step 1: Create Repository on GitHub

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" button in the top right and select "New repository"
3. Fill in the repository details:
   - **Repository name**: `agentic-bte`
   - **Description**: `Production-ready biomedical research platform combining LLMs with BioThings Explorer knowledge graphs. Features MCP Server and LangGraph multi-agent workflows for drug discovery and biomedical question answering.`
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have our files)

4. Click "Create repository"

## Step 2: Add GitHub Remote and Push

After creating the repository, GitHub will show you commands. Use these commands in your terminal:

```bash
# Navigate to your project directory
cd /Users/mastorga/Documents/agentic-bte

# Add the GitHub repository as remote origin
git remote add origin https://github.com/YOUR_USERNAME/agentic-bte.git

# Push your code to GitHub
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 3: Verify Upload

1. Go to your repository on GitHub: `https://github.com/YOUR_USERNAME/agentic-bte`
2. Verify that all files are uploaded correctly
3. Check that the README.md displays properly

## Alternative: Using SSH

If you prefer SSH authentication:

```bash
# Add remote using SSH
git remote add origin git@github.com:YOUR_USERNAME/agentic-bte.git

# Push to GitHub
git push -u origin main
```

## Repository Features to Enable

After pushing, consider enabling these GitHub features:

### Issues & Discussions
- Go to Settings → General → Features
- Enable Issues for bug reports and feature requests
- Enable Discussions for community Q&A

### Branch Protection
- Go to Settings → Branches
- Add a branch protection rule for `main`
- Require pull request reviews for changes

### Topics/Tags
Add these topics to your repository for discoverability:
- `biomedical-research`
- `biothings-explorer`
- `mcp-server`
- `langgraph`
- `drug-discovery`
- `knowledge-graph`
- `biomedical-nlp`
- `trapi`

## Next Steps

Once your repository is live on GitHub:

1. **Update README**: The README.md should display properly
2. **Create Releases**: Tag your first release as `v0.1.0`
3. **Set up CI/CD**: Consider GitHub Actions for testing
4. **Documentation**: The MCP_SETUP.md will help users get started
5. **Community**: Enable Discussions for user support

## Troubleshooting

If you get authentication errors:
1. Make sure you're signed in to GitHub
2. For HTTPS: Use a personal access token instead of password
3. For SSH: Ensure your SSH keys are set up correctly

Your repository is now ready for the biomedical research community!