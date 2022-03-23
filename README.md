# Neuro#

## Running the code:
Run the Docker container
0. `docker run --rm -it modelcountanon/modelcount_v1:latest /bin/bash`

From within the container:
1. `tar zxf data.tgz`
2. `cd sat-rl/pysat-master`
3. `./rebuild.sh`
4. `./rebuild.sh`      # Yes, (x2). Sorry about the embarrassement :/
5. `cd /code`
6. Run experiments with: `./run_experiments.sh [1-15] [vanilla]` 
   (Where first parameter is experiment number, second parameter is empty for the pre-trained model, "vanilla" for vanilla sharpSAT heuristics.)

