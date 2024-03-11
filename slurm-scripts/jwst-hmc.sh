#!/bin/bash
################################################################################
# Set variables                                                                #
################################################################################

# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=5 --mem=15000M

# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:1

#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=62:00:00
#SBATCH -e experiments/hmc/marg/submission-08/stderr.txt
#SBATCH -o experiments/hmc/marg/submission-08/stdout.txt

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo ""
python3 -u posterior_samples.py ../data/full experiments/hmc/marg/submission-08/01 experiments/direct-likelihood/linear/08 no-cvae marginalize &
python3 -u posterior_samples.py ../data/full experiments/hmc/marg/submission-08/02 experiments/direct-likelihood/linear/08 no-cvae marginalize &
python3 -u posterior_samples.py ../data/full experiments/hmc/marg/submission-08/03 experiments/direct-likelihood/linear/08 no-cvae marginalize &
python3 -u posterior_samples.py ../data/full experiments/hmc/marg/submission-08/04 experiments/direct-likelihood/linear/08 no-cvae marginalize &
python3 -u posterior_samples.py ../data/full experiments/hmc/marg/submission-08/05 experiments/direct-likelihood/linear/08 no-cvae marginalize &
wait
