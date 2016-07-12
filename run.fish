#!/usr/bin/fish
parallel ./main ::: (ls ~/kitti/dataset/sequences/)
