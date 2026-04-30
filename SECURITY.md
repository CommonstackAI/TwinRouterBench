# Security policy

## Credentials

- **Never commit** `.env` or any file containing live API keys, tokens, or passwords. This repository lists `.env` in `.gitignore`; use `.env.example` as the template only.
- If you accidentally pushed a secret, revoke the credential at your provider immediately, then remove it from Git history (e.g. `git filter-repo` or GitHub support) and force-push.

## Reporting vulnerabilities

Please report security issues privately to the maintainers (e.g. GitHub Security Advisories for this repository, or the contact listed in the organization profile) instead of opening a public issue.
