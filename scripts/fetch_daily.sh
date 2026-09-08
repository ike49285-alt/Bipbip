#!/usr/bin/env bash
# Local fallback for the GitHub Actions collector.
#
# Prefer the Actions workflow: a laptop cron only fires when the laptop is
# awake, which is exactly how bar archives end up full of holes. Use this when
# working offline, or as a belt-and-braces second collector.
#
# Install (weekdays at 18:30 local, after the close):
#   crontab -e
#   30 18 * * 1-5 /path/to/Bipbip/scripts/fetch_daily.sh >> /tmp/bipbip-fetch.log 2>&1

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "=== $(date -u +'%Y-%m-%dT%H:%M:%SZ') fetching ==="

# The full 30-day window, so a missed day leaves no permanent gap.
python -m bipbip.cli fetch --symbols SPY TQQQ --lookback 30
python -m bipbip.cli coverage

if [[ "${BIPBIP_AUTOCOMMIT:-0}" == "1" ]]; then
  if ! git diff --quiet -- data/bars; then
    git add data/bars
    git commit -m "Archive intraday bars for $(date -u +%Y-%m-%d)"
    echo "Committed. Push manually, or set up the Actions workflow instead."
  fi
fi
