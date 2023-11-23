echo "The shell is running!"


python train.py --dof 6 --gpu_ids 0 --hidden 1024 --name "cnf" --epoch 500 --voxel 3456 --lr 0.002 --batch_size 30 --dataroot "T:\Workspace\Master\\neural-nets-on-motion-planning\datasets"