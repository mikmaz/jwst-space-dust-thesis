#!/bin/sh
cat <<EoF
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
#SBATCH -e $1/stderr.txt
#SBATCH -o $1/stdout.txt

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo "$CUDA_VISIBLE_DEVICES"
python3 -u posterior_samples.py ../data/full $1/01 $2 $3 $4 &
python3 -u posterior_samples.py ../data/full $1/02 $2 $3 $4 &
python3 -u posterior_samples.py ../data/full $1/03 $2 $3 $4 &
python3 -u posterior_samples.py ../data/full $1/04 $2 $3 $4 &
python3 -u posterior_samples.py ../data/full $1/05 $2 $3 $4 &
wait
EoF
