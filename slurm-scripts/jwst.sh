#!/bin/bash
################################################################################
# Set variables                                                                #
################################################################################

# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=5000M

# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:1

#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=168:00:00
#SBATCH -e experiments/cvae/linear/new/06/stderr.txt
#SBATCH -o experiments/cvae/linear/new/06/stdout.txt

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo ""
# python3 -u train_direct.py ../data experiments/cvae/linear/new/06 @experiments/cvae/linear/new/06/args.txt
python3 -u train_cvae.py ../data experiments/cvae/linear/new/06 @experiments/cvae/linear/new/06/args.txt
