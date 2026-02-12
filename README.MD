## Running with Docker

Build the environment:

docker build -t ml-core-algorithms .

Run any algorithm:

docker run ml-core-algorithms python supervised/DecisionTree.py
docker run ml-core-algorithms python unsupervised/GMM.py
