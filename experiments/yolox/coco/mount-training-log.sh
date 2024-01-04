#unmount
#fusermount -u /path/to/mount_folder

echo "================= mount training log from 8x2080Ti_Opt.sh.intel.com ==================="
sshfs linjiaojiao@8x2080Ti_Opt.sh.intel.com:/mnt/disk1/data_for_linjiaojiao/projects/pytorch-detection/experiments/yolox/coco/output /mnt/disk2/projects/pytorch-detection/experiments/yolox/coco/output-2080ti

echo "================= mount training log from a100 ==================="
sshfs linjiaojiao@10.67.109.29:/mnt/disk1/data_for_linjiaojiao/projects/pytorch-detection/experiments/yolox/coco/output /mnt/disk2/projects/pytorch-detection/experiments/yolox/coco/output-a100

echo "Done."

