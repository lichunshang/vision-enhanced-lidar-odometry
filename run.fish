#!/usr/bin/fish
for f in ~/kitti/dataset/sequences/*
    ./main (basename $f) &
end
