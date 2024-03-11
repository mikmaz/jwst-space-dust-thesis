#!/bin/sh
# Clear script
:> ./slurm-scripts/jwst-hmc.sh
# Create script
./slurm-scripts/make-hmc-script.sh "$1" "$2" "$3" "$4" >> ./slurm-scripts/jwst-hmc.sh
# Give permissions
chmod 777 ./slurm-scripts/jwst-hmc.sh
# Submit job
sbatch ./slurm-scripts/jwst-hmc.sh
