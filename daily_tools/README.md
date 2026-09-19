# Full-Stack-GenAI-Bootcamp-1.0

### 👨‍🏫 Meet Your Mentor

🔗 [Visit Mentor Profile](https://sunnysavita10.github.io/)

### Update the repository

Run this command from the repository root to pull the latest code from the
official bootcamp repository (the `upstream` remote):

```bash
./daily_tools/pull_latest.sh
```

The script uses a fast-forward-only pull and will stop instead of creating a
merge commit, overwriting local commits, or updating a working tree with
uncommitted changes.

To pull from your fork instead, pass `origin` explicitly:

```bash
./daily_tools/pull_latest.sh origin
```

The official repository must be configured as `upstream`:

```bash
git remote add upstream https://github.com/sunnysavita10/Full-Stack-GenAI-Bootcamp-1.0.git
```

See [daily_tools/README.md](daily_tools/README.md) for documentation of the
repository tools.
