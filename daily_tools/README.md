# Full-Stack-GenAI-Bootcamp-1.0

### 👨‍🏫 Meet Your Mentor

🔗 [Visit Mentor Profile](https://sunnysavita10.github.io/)

### Update the repository

Run this command from the repository root to merge the latest `main` branch
from the official bootcamp repository (`upstream`) into your current branch:

```bash
./daily_tools/pull_latest.sh
```

The script stashes tracked and untracked changes, fetches upstream, merges
`upstream/main` without discarding local commits, then restores the stash after
a successful merge. If the merge has conflicts, the stash is left intact until
you resolve or abort the merge.

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
