#!/bin/bash

mtt train options.yaml
mtt eval model.pt eval.yaml -e extensions/ -o outputs.mts
