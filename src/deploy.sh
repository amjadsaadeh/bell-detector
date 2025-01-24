#! /bin/bash

# Simple script to deploy the application to the mic server

scp src/data_collector.py "${MIC_SSH_URI}:~/data_collector/data_collector.py"
scp params.yaml "${MIC_SSH_URI}:~/data_collector/params.yaml"
