#!/usr/bin/env bash

python main_new.py --config train-config-office311.yaml --dist-url 'tcp://localhost:13110' --loss_weight 0.01 --loss_penalty 0.00 --prot_start 5
