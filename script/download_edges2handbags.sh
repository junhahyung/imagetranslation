#!/bin/bash
mkdir datasets/edges2handbags/train1 -p
mkdir datasets/edges2handbags/train0 -p
mkdir datasets/edges2handbags/test1 -p
mkdir datasets/edges2handbags/test0 -p
for f in datasets/edges2handbags/train/*; do convert -quality 100 -crop 50%x100% +repage $f datasets/edges2handbags/train%d/${f##*/}; done;
for f in datasets/edges2handbags/val/*; do convert -quality 100 -crop 50%x100% +repage $f datasets/edges2handbags/test%d/${f##*/}; done;
mv datasets/edges2handbags/train0 datasets/edges2handbags/trainA
mv datasets/edges2handbags/train1 datasets/edges2handbags/trainB
mv datasets/edges2handbags/test0 datasets/edges2handbags/testA
mv datasets/edges2handbags/test1 datasets/edges2handbags/testB
