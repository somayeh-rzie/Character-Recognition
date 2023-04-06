# Character-Recognition
A simple character recognition written in python <br />



# About This Project
Implementing a multilayer perceptron neural network with feedforward algorithm. This network uses simple OCR Train Set in Data to train perceptron and then test it by using OCR Test Set in Data and changing some parameters. This evaluation is based on *Error* and *Number of iterations that network will be converge*. Results have shown below : <br /><br />
1.parameter: **number of hidden layer neurons** = (Respectively) 8, 16, 32<br />
(constants : batch size = 1, learning rate = 0.1)<br />
![Alt text](https://s2.uupload.ir/files/1_sd7z.png)<br /><br />
Error = 28.6% , Number of Iterations = 64<br /><br /><br /><br />

![Alt text](https://s2.uupload.ir/files/2_prli.png)<br />
Error = 9.5% , Number of Iterations = 22<br /><br /><br /><br />
![Alt text](https://s2.uupload.ir/files/3_qpkb.png)<br />
Error = 23.8% , Number of Iterations = 13<br /><br /><br />

**Note** : The small number of neurons in the hidden layer causes underfit due to the over-simplicity of the model and the inability to find an optimal algorithm, it causes high error rate in calculating the output , and requires a high number of iterations in order to learn the model.<br/> 
On the other hand, a high number also causes complexity and calculation error because it quickly converges towards the solution with a low number of iterations and insufficient training of the model, it reaches a learning that is not optimal. The optimal answers are in the range between these two.<br/>

2.parameter: **different initial values for weights and biases** = try random values 3 times<br />
(constants : batch size = 1, learning rate = 0.1)<br />



# Built With
- [python](https://www.python.org/) <br /><br />

# Getting Started
### Prerequisites
- put data.xlsx in your project path
- xlrd <br />
    `pip install xlrd`
    
<br /><br />
# License
Distributed under the MIT License. See `LICENSE.txt` for more information
<br /><br />

# Contact
rezaie.somayeh79@gmail.com
