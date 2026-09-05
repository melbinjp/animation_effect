#!/bin/bash
# One-shot bootstrap for a fresh RunPod CPU pod (or any bare Ubuntu/Debian
# box) to run this native pipeline. Packages up everything that was done by
# hand, one SSH command at a time, across many rented pods this session:
#   - apt packages ffmpeg needs, plus libegl1/libgl1/libgles2 -- without
#     these, MediaPipe's CPU delegate fails outright with
#     "libEGL.so.1: cannot open shared object file", not just a warning.
#   - this repo itself, via `git clone`/`git pull` -- pulled from GitHub
#     directly on the pod's own (fast, datacenter) connection instead of
#     scp'd from your PC, so your home upload speed is only ever spent on
#     the actual input video, never on source code.
#   - both requirements files (CLI + webui).
#
# The repo is private, so cloning needs a key with read access -- this
# script expects a read-only deploy key already sitting at
# ~/.ssh/animation_effect_deploy_key on the pod (copy it up with scp
# before running this, e.g. as part of your own pod-bootstrap sequence).
# It does NOT fetch itself from a public URL for the same reason; copy this
# file up (scp/cat via SSH) alongside the deploy key, then:
#   bash setup_pod.sh
#
# Idempotent: safe to re-run on a pod that's already set up (git pulls
# instead of failing to clone into a non-empty dir, apt/pip installs are
# no-ops when already satisfied).

set -e

REPO_SSH_URL="git@github.com:melbinjp/animation_effect.git"
REPO_DIR="/workspace/animation_effect"
DEPLOY_KEY="$HOME/.ssh/animation_effect_deploy_key"

echo "== apt packages =="
apt-get update -qq
apt-get install -y -qq ffmpeg git libegl1 libgl1 libgles2 python3-pip > /dev/null

echo "== repo =="
if [ -f "$DEPLOY_KEY" ]; then
  chmod 600 "$DEPLOY_KEY"
  export GIT_SSH_COMMAND="ssh -i $DEPLOY_KEY -o StrictHostKeyChecking=accept-new"
fi
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone --depth 1 "$REPO_SSH_URL" "$REPO_DIR"
fi

echo "== python deps =="
cd "$REPO_DIR/native"
pip install --quiet -r requirements.txt -r requirements-webui.txt

echo "== sanity checks =="
python3 -c "
import os
try:
    cores = len(os.sched_getaffinity(0))
except AttributeError:
    cores = os.cpu_count()
print(f'Real allocated cores (sched_getaffinity): {cores}')
print(f'os.cpu_count() (often wrong inside a container -- do not use for --workers): {os.cpu_count()}')
"
echo "ffmpeg hardware encoders available:"
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'h264_(nvenc|vaapi|qsv)' || echo "  (none -- will use libx264 software encode)"

mkdir -p "$REPO_DIR/native/webui_uploads"

echo
echo "== done =="
echo "Upload your input video into: $REPO_DIR/native/webui_uploads/"
echo "  (or use the webui's own upload button once it's running -- either works)"
echo
echo "Start the UI with:"
echo "  cd $REPO_DIR/native && python3 webui.py --host 127.0.0.1 --port 8765"
echo
echo "Then, from your own PC, open an SSH tunnel to reach it:"
echo "  ssh -i <your_key> -p <pod_ssh_port> -L 8765:localhost:8765 root@<pod_ip>"
echo "and browse to http://127.0.0.1:8765/ -- renders you start there will"
echo "auto-download back to your machine's browser download folder when done."
