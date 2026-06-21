#!/usr/bin/env bash
# Securely add/replace an API key in ~/CAISc/.env.
# The key is read from a HIDDEN prompt — it never appears in shell history, in logs,
# or in the Claude chat. Contains NO secrets itself.
#
# Usage (run in a bash shell ON THE SERVER):
#   bash ~/mech_interp/scripts/set_api_key.sh SARVAM_API_KEY
set -euo pipefail

VAR="${1:-SARVAM_API_KEY}"
ENV_FILE="$HOME/CAISc/.env"

read -rsp "Paste value for ${VAR} (hidden, then press Enter): " VAL
echo
if [ -z "${VAL}" ]; then
  echo "empty input — aborted, nothing written."
  exit 1
fi

mkdir -p "$(dirname "$ENV_FILE")"
touch "$ENV_FILE"
# drop any existing line for this VAR, then append the fresh value
grep -v "^${VAR}=" "$ENV_FILE" > "${ENV_FILE}.tmp" 2>/dev/null || true
mv "${ENV_FILE}.tmp" "$ENV_FILE"
printf '%s=%s\n' "$VAR" "$VAL" >> "$ENV_FILE"
chmod 600 "$ENV_FILE" 2>/dev/null || true

echo "✓ ${VAR} written to ${ENV_FILE}  (length ${#VAL}, value not shown)"
unset VAL
