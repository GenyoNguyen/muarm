#!/usr/bin/env bash
python learning/play.py --robot panda --task reach_obstacles --algo differentiable_mpc --episodes 100 --seed 0 --headless --horizon 300 --n-substeps 100 --tensorboard
