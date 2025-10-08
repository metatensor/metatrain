#!/bin/bash

mtt train options.yaml -o model.pt
mtt train options-llpr.yaml -o model-llpr.pt
