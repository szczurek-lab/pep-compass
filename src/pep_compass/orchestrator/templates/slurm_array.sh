#!/bin/bash
#SBATCH --array=0-${array_max}
${directives}
set -euo pipefail
cd ${working_directory}
${command}
