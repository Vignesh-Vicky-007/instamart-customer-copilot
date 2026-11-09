#!/usr/bin/env bash
# Everything goes on the single branch main.
# Usage: bash git_setup.sh https://github.com/<you>/<repo>.git
set -e
git init -b main
git add -A
git commit -m "Instamart Customer Intelligence and Support Co-Pilot: Parts A-D"
if [ -n "$1" ]; then git remote add origin "$1" && git push -u origin main; fi
