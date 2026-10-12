#!/usr/bin/env bash
python learning/play.py --robot panda --task reach_obstacles --algo ik_rrt_connect --episodes 100 --seed 0 --headless --horizon 300 --tensorboard
