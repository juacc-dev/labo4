#!/bin/bash

rm -rf ./pylabo
printf "y\n" | pip uninstall pylabo
mkdir pylabo && cp -r ../pylabo/* pylabo
pip install ./pylabo
rm -rf ./pylabo
