#!/bin/bash

# Data fetching: Download the dataset
wget http://gwa.ewi.tudelft.nl/fileadmin/pds/trace-archives/grid-workloads-archive/datasets/gwa-t-12/rnd.zip

# Extract the zip file
unzip rnd.zip -d targetdir
