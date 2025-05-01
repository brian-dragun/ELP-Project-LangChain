## 🌲 Trunk-Based Development Workflow

Trunk-based development focuses on keeping the main branch (trunk) always deployable, with small, frequent commits and integrations.

### Key Principles
- Work in small batches
- Commit frequently to the main branch
- Use feature flags to hide incomplete work
- Merge feature branches within 1-2 days

### Daily Workflow
```bash
# Start by updating local main
git checkout main
git pull

# Option 1: Direct trunk work (for small changes)
# Make changes directly on main
git add .
git commit -m "Add small feature or fix"
git push

# Option 2: Short-lived feature branches (for larger changes)
# Create a branch that will live less than a day
git checkout -b feature/quick-change

# Make changes
git add .
git commit -m "Add feature component"

# Update with latest trunk changes before merging
git fetch origin main
git rebase origin/main

# Push and merge back to trunk quickly (same day if possible)
git push -u origin feature/quick-change
git checkout main
git merge feature/quick-change
git push
```

### Using Feature Flags
For larger features that take multiple days, use feature flags to hide unfinished work:

```python
# Example feature flag in Python code
if Config.ENABLE_LANGSMITH_FEATURE:
    # New LangSmith code here
    langsmith_client = Client(api_key=Config.LANGCHAIN_API_KEY)
else:
    # Original code or placeholder
    langsmith_client = None
```

This allows you to commit incomplete features to main while keeping the application functional.