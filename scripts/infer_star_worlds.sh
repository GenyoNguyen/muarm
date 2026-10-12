#!/usr/bin/env bash
python learning/play.py --robot panda --task reach_obstacles --algo star_worlds --episodes 100 --seed 0 --headless --horizon 500 --n-substeps 6 --physics-timestep 0.004166666666666667 --distance-threshold 0.02 --tensorboard
