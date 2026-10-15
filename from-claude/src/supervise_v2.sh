#!/bin/sh
cd /home/claude/tnr/src
until grep -q XONG ../runs/run_002_main.log 2>/dev/null; do python3 run_v2.py >> ../runs/run_002_main.log 2>&1; done
