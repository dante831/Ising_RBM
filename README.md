# Ising_RBM

RBM learning for the Ising model. 

This code suite samples from the Ising model and produce spin configurations at different temperatures and external field 
strengths of the Ising model. Then, it fits the Restricted Boltzmann machine (RBM) to the spin configurations. The 
conclusion is that the parameters of the RBM correponds directly to the temperature parameter of the Ising model. 

The IsingModel.py file is forked from [christianb93](https://github.com/christianb93/MachineLearning/blob/master/IsingModel.py), 
and the impementation of Restricted Boltzmann machine using Tensorflow is forked from 
[Yelysei Bondarenko](https://github.com/yell/boltzmann-machines). 

## Usage

In a command-line window, type the following code to generate spin configurations of the Ising model
    
    python IsingModel.py 
    
Nest, use the following command to train the Restricted Boltzmann machine on the Ising model

    python RBM_ising.py
    
You might want to use nohup to run the program in the background, since it takes quite some time to finish the sampling. 
Hence, alternatively to the above code, you can use

    nohup python -u IsingModel.py > data.out &
    nohup python -u RBM_ising.py > fit.out &
    
You can check the sampling progress in data.out and the training progress in fit.out

## License

This code is distributed under the [MIT](http://opensource.org/licenses/mit-license.php) license
