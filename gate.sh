#!/bin/bash
#
# gate.sh — publish gate for this repo. Run before every push.
#
#   1. SECRET SCAN      — key-shaped strings (fatal)
#   2. IDENTITY SCAN    — employer / machine / chat identifiers (fatal)
#   3. LOCAL PATH SCAN  — absolute /Users/<name> paths (fatal: must be ~ or $HOME)
#
# Exits non-zero on any finding. Nothing is committed or pushed by this script.
#
set -uo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO" || exit 1

# Only scan what git would publish.
FILES="$(git ls-files 2>/dev/null || find . -type f -not -path './.git/*')"
if [ -z "$FILES" ]; then
  echo "no files to scan"; exit 0
fi
SCAN="$(echo "$FILES" | grep -vE '\.(png|jpg|jpeg|gif|pdf|ico)$')"
# this script's own pattern list is a literal self-match — never scan ourselves
SCAN="$(echo "$SCAN" | grep -vE '^(\./)?gate\.sh$')"

PAT_SECRET='sk-[A-Za-z0-9]{28,}'
PAT_SECRET="$PAT_SECRET"'|ghp_[A-Za-z0-9]{30,}'
PAT_SECRET="$PAT_SECRET"'|gho_[A-Za-z0-9]{30,}'
PAT_SECRET="$PAT_SECRET"'|github_pat_[A-Za-z0-9_]{50,}'
PAT_SECRET="$PAT_SECRET"'|tskey-[a-z]+-[A-Za-z0-9]{20,}'
PAT_SECRET="$PAT_SECRET"'|hf_[A-Za-z0-9]{30,}'
PAT_SECRET="$PAT_SECRET"'|r8_[A-Za-z0-9]{30,}'
PAT_SECRET="$PAT_SECRET"'|AIza[A-Za-z0-9_-]{33}'
PAT_SECRET="$PAT_SECRET"'|xox[baprs]-[A-Za-z0-9-]{20,}'
PAT_SECRET="$PAT_SECRET"'|[0-9]{8,10}:[A-Za-z0-9_-]{30,}'
PAT_SECRET="$PAT_SECRET"'|-----BEGIN [A-Z ]*PRIVATE KEY'
PAT_SECRET="$PAT_SECRET"'|Bearer [A-Za-z0-9._-]{40,}'

PAT_IDENT='OZA|Mini-Seis|Raven RV|RPM|192\.168\.|VM_DATABASE|Peak Group|vms-hub'
PAT_IDENT="$PAT_IDENT"'|1004373661209|1340301920|shing69|369hermes'

PAT_PATH='/Users/[A-Za-z]+/'

fail=0
echo "════ gate ════"

echo "── 1. secret-shaped strings ──"
hits=$(echo "$SCAN" | xargs grep -lIE "$PAT_SECRET" 2>/dev/null)
if [ -n "$hits" ]; then echo "❌❌ SECRETS:"; echo "$hits" | sed 's|^|     |'; fail=1
else echo "   ✅ none"; fi

echo "── 2. personal / employer identifiers ──"
hits=$(echo "$SCAN" | xargs grep -lIE "$PAT_IDENT" 2>/dev/null)
if [ -n "$hits" ]; then echo "❌❌ IDENTIFIERS:"; echo "$hits" | sed 's|^|     |'; fail=1
else echo "   ✅ none"; fi

echo "── 3. absolute local paths ──"
hits=$(echo "$SCAN" | xargs grep -lIE "$PAT_PATH" 2>/dev/null)
if [ -n "$hits" ]; then echo "❌❌ /Users/<name> PATHS:"; echo "$hits" | sed 's|^|     |'; fail=1
else echo "   ✅ none"; fi

echo
if [ "$fail" != 0 ]; then
  echo "❌ gate FAILED — do not push"; exit 1
fi
echo "✅ gate passed"
