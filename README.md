## DetCGD: Compressed Gradient Descent with Matrix Stepsizes for Non-Convex Optimization

### [File-structure]:
```
├─  Experiment_code.py
├─  plot.py
├─  README.md
├─  dataset
├─  figures
├─  result_exp_1
├─  result_exp_2
├─  result_exp_3
└─  result_exp_4
```

Codes for running experiments are in file `Experiment_code.py`, there are functions for a total of four experiments in this file.


### [Corrspondence]:
The following experiments are done for the single node setting:
- Experiment $1$ in the code corresponds to the section **D.1.1** "Comparison to CGD with scalar stepsize, scalar smoothness constant".
- Experiment $2$ in the code corresponds to the section **D.1.2** "Comparison of the two algorithms under the same stepsize".
- Experiment $3$ is mainly a trial on different sampling strategies and has no correspondence in the paper.

The following experiment is done for the distributed setting:
- Experiment $4$ in the code corresponds to the section **D.2.1** "Comparison to standard DCGD in the distributed case"

### [Dependencies]:
Python packages dependencies:
- Numpy
- Scipy 
- Matplotlib
- Scikit-learn

### [Run]:
In order to run the experiment, we first need to donwload the corresponding dataset from **LIBSVM** into `./dataset/[DATASET].txt`, the result of each experiment will be placed into the corresponding result folder `./result_exp_[EXP_INDEX]/`, there are a number of parameters we can change for each experiment, 
```
-i    # Should be within [1, 2, 3, 4], specifying the experiments to run.
-d    # Name of the data set, for example 'a1a'.
-l    # Value of lambda, default is 0.2
-c    # Number of clients for experiment 4
-e    # The value of epsilon squared, for experiment 4
-K    # Number of iterations needed
```
We can create one enviroment using anaconda and switch to the root directory. 
For a sample run of experiment 1, use the command
```
python Experiment_code.py -i=1 -d=a1a -l=0.3 -K=3000
```
For a sample run of experiment 2, use the command
```
python Experiment_code.py -i=2 -d=a1a -l=0.3 -K=3000
```
For a sample run of experiment 4, use the command
```
python Experiment_code.py -i=4 -d=a1a -l=0.1 -K=2000 -c=100 -e=0.0001
```

### Visulization
For visualization of results of experiment $1$ with parameter $\lambda = 0.3$, using `a1a.txt` running for $3000$ iterations, we can use the command
```
python Plot.py -i=1 -d=a1a -K=3000 -l=0.3
```
For visualization of results of experiment $2$ with parameter $\lambda = 0.3$, using `a1a.txt` running for $3000$ iterations with rand-$30$ sketch, we can use the command
```
python Plot.py -i=2 -d=a1a -K=3000 -l=0.3 -t=30
```
For visualization of resuts of experiment $4$ with parameter $\lambda=0.1$, using `a1a.txt` running for $2000$ iterations with $100$ clients, setting $\epsilon^2 = 0.0001$, use the command 
```
python Plot.py -i=4 -d=a1a -K=2000 -l=0.1 -c=100 -e=0.0001
```
All the figures will appear in the folder `./figures`.
