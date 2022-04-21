#!/bin/bash -l
## run in /bin/bash NOTE the -l flag!
#
#######################################################
# example Slurm batch script for a plain MPI program
# using 512 cores (16 nodes x 32 cores per node) with
# 32 MPI tasks per node executed under the intel runtime
#######################################################
#
## do not join stdout and stderr
#SBATCH -o Grid_100.%j.out
#SBATCH -e Grid_100.%j.err
## name of the job
#SBATCH -J Grid_100
## execute job from the current working directory
## this is default slurm behavior
#SBATCH -D ./
## do not send mail
#SBATCH --mail-type=NONE
## Total number of tasks must be a multiple of 28
## request xx nodes (x 32 cores)
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
## run for 4 hours
#SBATCH -t 24:00:00
module load gcc python3 openmpi2 lib/boost
## to save memory at high core counts, the "connectionless user datagram protocol" 
## can be enabled (might come at the expense of speed)
## see https://software.intel.com/en-us/articles/dapl-ud-support-in-intel-mpi-library
# export I_MPI_DAPL_UD=1
##gather MPI statistics to be analyzed with itac's mps tool
# export I_MPI_STATS=all
##gather MPI debug information (high verbosity)
# export I_MPI_DEBUG=5
srun output_Boltz_Grid100.out
 
