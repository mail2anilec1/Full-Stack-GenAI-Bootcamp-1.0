#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s [remote]\n' "$0"
    printf 'Default remote: upstream\n'
}

remote=upstream
remote_set=false

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            printf 'Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ "$remote_set" == true ]]; then
                printf '%s\n' 'Specify at most one remote.' >&2
                usage >&2
                exit 2
            fi
            remote=$1
            remote_set=true
            ;;
    esac
    shift
done

branch=$(git branch --show-current)

if [[ -z "$branch" ]]; then
    printf '%s\n' 'The repository is in detached HEAD state; switch to a branch first.' >&2
    exit 1
fi

remote_branch=main

if ! git remote get-url "$remote" >/dev/null 2>&1; then
    printf 'Remote "%s" is not configured. Available remotes:\n' "$remote" >&2
    git remote >&2
    exit 1
fi

stash_ref=
if [[ -n "$(git status --porcelain)" ]]; then
    previous_stash=$(git rev-parse --verify -q refs/stash || true)
    printf 'Stashing tracked and untracked changes before syncing...\n'
    git stash push --include-untracked -m "auto-stash before syncing ${remote}/${remote_branch}"
    current_stash=$(git rev-parse --verify -q refs/stash || true)

    if [[ -z "$current_stash" || "$current_stash" == "$previous_stash" ]]; then
        printf '%s\n' 'Unable to create a stash; aborting before fetch.' >&2
        exit 1
    fi

    stash_ref=$(git stash list -1 --format='%gd')
    if [[ -n "$(git status --porcelain)" ]]; then
        printf 'Some changes remain unstashed. Your stash is %s; aborting before fetch.\n' \
            "$stash_ref" >&2
        exit 1
    fi
fi

restore_stash() {
    [[ -z "$stash_ref" ]] && return 0

    printf 'Restoring saved work from %s...\n' "$stash_ref"
    if git stash pop "$stash_ref"; then
        stash_ref=
    else
        printf 'Stash restoration had conflicts; your saved work remains in %s.\n' \
            "$stash_ref" >&2
        return 1
    fi
}

printf 'Fetching %s/%s...\n' "$remote" "$remote_branch"
if ! git fetch "$remote" "+refs/heads/${remote_branch}:refs/remotes/${remote}/${remote_branch}"; then
    printf 'Fetch failed; restoring your saved work.\n' >&2
    restore_stash || true
    exit 1
fi

printf 'Merging %s/%s into %s...\n' "$remote" "$remote_branch" "$branch"
if ! git merge --no-ff "${remote}/${remote_branch}"; then
    if [[ -n "$stash_ref" ]]; then
        printf 'Merge failed. Resolve or abort the merge, then restore your saved work with: git stash pop %s\n' \
            "$stash_ref" >&2
    else
        printf '%s\n' 'Merge failed. Resolve the merge conflicts before continuing.' >&2
    fi
    exit 1
fi

if ! restore_stash; then
    exit 1
fi

printf 'Sync complete.\n'
