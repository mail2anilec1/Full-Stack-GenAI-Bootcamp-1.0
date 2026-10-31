#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s <repository-url> [branch]\n' "$0"
    printf 'Default branch: main\n'
}

if (($# == 0)); then
    usage >&2
    exit 2
fi

repository_url=
branch=main
argument_count=0

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
            argument_count=$((argument_count + 1))
            if ((argument_count == 1)); then
                repository_url=$1
            elif ((argument_count == 2)); then
                branch=$1
            else
                printf '%s\n' 'Specify a repository URL and at most one branch.' >&2
                usage >&2
                exit 2
            fi
            ;;
    esac
    shift
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf '%s\n' 'Run this script from inside a Git working tree.' >&2
    exit 1
fi

if ! git check-ref-format --branch "$branch" >/dev/null 2>&1; then
    printf 'Invalid branch name: %s\n' "$branch" >&2
    exit 2
fi

current_branch=$(git branch --show-current)
if [[ -z "$current_branch" ]]; then
    printf '%s\n' 'The repository is in detached HEAD state; switch to a branch first.' >&2
    exit 1
fi

stash_ref=
if [[ -n "$(git status --porcelain)" ]]; then
    previous_stash=$(git rev-parse --verify -q refs/stash || true)
    printf 'Stashing tracked and untracked changes before syncing...\n'
    git stash push --include-untracked -m "auto-stash before syncing ${repository_url} ${branch}"
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

printf 'Fetching %s/%s...\n' "$repository_url" "$branch"
if ! git fetch -- "$repository_url" "refs/heads/${branch}"; then
    printf 'Fetch failed.\n' >&2
    if ! restore_stash; then
        printf 'Restore the saved work manually with: git stash pop %s\n' "$stash_ref" >&2
    fi
    exit 1
fi

printf 'Merging %s/%s into %s...\n' "$repository_url" "$branch" "$current_branch"
if ! git merge --no-ff FETCH_HEAD; then
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
