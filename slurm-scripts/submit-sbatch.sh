#!/bin/sh
# Clear script
:> ./slurm-scripts/jwst.sh
# Create script
./slurm-scripts/make-sbatch-script.sh "$1" >> ./slurm-scripts/jwst.sh
# Give permissions
chmod 777 ./slurm-scripts/jwst.sh
# Submit job
sbatch ./slurm-scripts/jwst.sh
