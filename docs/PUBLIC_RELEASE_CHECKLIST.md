# Public Release Checklist

Review these items before changing the repository visibility to Public.

## 1) Secrets / Privacy

- [ ] No API keys / tokens / passwords in code or notebooks
- [ ] No personal private data (faces, phone numbers, emails, student IDs)
- [ ] No restricted dataset raw files

Quick scan:

```bash
rg -n "token|api[_-]?key|secret|password|passwd|AKIA|BEGIN PRIVATE KEY" .
```

## 2) Large / Generated Files

- [ ] Local archives are excluded (for example `matlab.zip`)
- [ ] Temporary outputs are excluded (debug dumps / checkpoints)
- [ ] Datasets and training cache are excluded

Quick scan (>10MB):

```bash
find . -type f -not -path './.git/*' -size +10M -print
```

## 3) Repo Usability

- [ ] Top-level `README.md` includes project overview, layout, and run commands
- [ ] `pytorch/README.md` and `matlab/README.md` are runnable independently
- [ ] Key commands work when executed from repository root

## 4) Git Hygiene

- [ ] `git status` is clean before tagging or release
- [ ] Only showcase-relevant files are committed
- [ ] Commit message clearly describes showcase restructuring

Optional preflight:

```bash
git add -A
git status --short
git diff --cached --stat
```
