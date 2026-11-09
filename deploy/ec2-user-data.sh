#!/bin/bash
# EC2 user data for Ubuntu 24.04: installs Docker and clones DocPilot. Runs once, as root, on
# first boot. Progress: /var/log/cloud-init-output.log. Then follow deploy/README.md.
set -euxo pipefail

REPO=https://github.com/scatyf3/DocPilot.git
BRANCH=backend-multitenant

# 2 GB of swap, so building the Next.js image fits on a 2 GB instance.
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

curl -fsSL https://get.docker.com | sh
usermod -aG docker ubuntu

sudo -u ubuntu git clone --branch "$BRANCH" "$REPO" /home/ubuntu/DocPilot
