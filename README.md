# Privacy Attacks on Graph Embeddings
Given a graph and an embedding trained on that graph. One node and it's embedding vector is deleted. Is it possible to 
reconstruct information about the neighborhood structure of the deleted node from the embedding vectors of the other 
nodes?

This project includes a algorithm to retrieving this information.

# Setup Environment
## Install conda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source .bashrc (or restart the terminal)
```
## Setup conda cnvironment
```
conda env create -f environment.yml
conda activate mt
conda env update
```
# Setup Embeddings
## Setup snap
Snap should be cloned in the same directory as `ma---cssh`
```
git clone https://github.com/snap-stanford/snap.git
cd snap/examples/node2vec
make
cd ../../..
```
On some machines ` __exception is not defined`, which causes the compilation to fail.
Quick and dirty fix: Add the following at line 4 to gitlab-core/bd.cpp

```
struct __exception {
    int    type;      /* Exception type */    
    char*  name;      /* Name of function causing exception */
    double arg1;      /* 1st argument to function */
    double arg2;      /* 2nd argument to function */
    double retval;    /* Function return value */
};
```

## Setup LINE

The algorithm is available for windows and Linux. They can be found in github:
```
git clone https://github.com/tangjianpku/LINE.git
```

To build the project on Linux with (GSL has to be installed):
```
cd LINE/linux
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result line.cpp -o line -lgsl -lm -lgslcblas -I/home/mellers/include/ -L/home/mellers/lib/ -static -static-libgcc -static-libstdc++
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result normalize.cpp -o normalize
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result concatenate.cpp -o concatenate
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result reconstruct.cpp -o reconstruct
```

### use statically compiled LINE
If the installation does not work I included a statically complied Version in the directory. This might be less efficient because the code is not compiled for the machine but avoids headaches installing gsl. 

To use those copy the LINE folder to the base directory:
```
cp -r priv_attack_on_graph_embeddings/LINE .
```

## Setup GEM Embeddings
Clone the embedding in the same directory as this project. (If you prefere another location adjust path in config.py)
```
git clone https://github.com/palash1992/GEM.git
```

# Example
A simple execution can be found in ``code/examples/example_execution.py``: