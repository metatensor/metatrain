#!/bin/bash

mtt train options_no_zbl.yaml -o model_no_zbl.pt
mtt train options_zbl.yaml -o model_zbl.pt
