#!/bin/bash
set -u
echo "HOST $(hostname)"
echo "--- BEFORE"
cat /etc/resolv.conf
cp -a /etc/resolv.conf "/etc/resolv.conf.bak-$(date +%s)"
python3 - <<'PY'
from pathlib import Path
path = Path("/etc/resolv.conf")
lines = path.read_text().splitlines()
if "nameserver 114.114.114.114" not in lines:
    lines.insert(1, "nameserver 114.114.114.114")
    path.write_text("\n".join(lines) + "\n")
PY
echo "--- AFTER"
cat /etc/resolv.conf
echo "--- DNS"
timeout 8 getent hosts api.wandb.ai || true
echo "--- CURL"
env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy \
  timeout 15 curl -sS -o /dev/null -w '%{http_code}\n' https://api.wandb.ai || true
