#!/usr/bin/env bash
set -euo pipefail

remote=${1:-upstream}
branch=$(git branch --show-current)

if [[ -z "$branch" ]]; then
    printf '%s\n' 'The repository is in detached HEAD state; switch to a branch first.' >&2
    exit 1
fi

if ! git remote get-url "$remote" >/dev/null 2>&1; then
    printf 'Remote "%s" is not configured. Available remotes:\n' "$remote" >&2
    git remote >&2
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    printf '%s\n' 'Your working tree has uncommitted changes. Commit or stash them before syncing.' >&2
    exit 1
fi

printf 'Updating %s from %s/%s...\n' "$branch" "$remote" "$branch"
git pull --ff-only "$remote" "$branch"
printf 'Repository is up to date.\n'
