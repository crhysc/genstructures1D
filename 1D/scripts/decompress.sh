#!/bin/bash
mkdir -p interim
cd interim
mv ../raw/* .
cp * ../raw/
bunzip2 *
