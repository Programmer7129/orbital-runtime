#!/bin/zsh
cd ~/Documents/orbital-runtime || exit 1
exec claude --bg -n hardening2 --model claude-opus-4-8 --permission-mode=bypassPermissions "$(cat docs/reviews/HARDENING2-BRIEF.md)"
