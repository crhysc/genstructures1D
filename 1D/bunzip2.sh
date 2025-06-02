#!/bin/bash
set -e
mkdir -p interim
cd interim
bunzip2 ../raw/*
mv ../raw/* .
