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
(constants : batch size = 1, learning rate = 0.1, number of hidden layer neurons = 16)<br />
I)Error = 14.3% , Number of Iterations = 35<br />
II)Error = 19% , Number of Iterations = 24<br />
III)Error = 19% , Number of Iterations = 25<br />
**Note** : Since these weights and biases generated randomly, you may get different results. But we know weight matrix indicates the effectiveness of a particular input. The greater the weight of the input, the more it will affect the network and accelerate the activation function. On the other hand, the bias is like an added interval in a linear equation and is an additional parameter in the neural network that is used to adjust the output along with the weighted sum of the neuron's inputs and helps to control the value at which the activation function is triggered. At first, we place both of them randomly because we don't need any initial view of the importance of each neuron or interruption, so we may give too much weight to a less important neuron or vice versa, but gradually we improve the network. The effect of initial weight and bias cannot be ignored and affects the number of repetitions as well as the final error, but anyway, due to our little information at the beginning of the work; It is unavoidable.

3.parameter: **learning rate** = (Respectively) 0.01, 0.1, 0.2, 0.5, 0.9<br />
For each learning rate our results are shown below :<br />
I)learning rate = 0.01  :  Error = 23.8% , Number of Iterations = 153 <br />
II)learning rate = 0.1  :  Error = 9.5% , Number of Iterations = 24 <br />
III)learning rate = 0.2  :  Error = 19% , Number of Iterations = 17 <br />
IV)learning rate = 0.5  :  Error = 14.3% , Number of Iterations = 9 <br />
V)learning rate = 0.9  :  Error = 28.6% , Number of Iterations = 13 <br />
![Alt text](https://s2.uupload.ir/files/4_ouj9.png)<br /><br />
**Note** : Graph is actually discrete, but its points have plotted continuous. <br />
Learning rate controls the speed at which the model adapts to the problem. Smaller learning rates require more number of iterations, due to smaller changes in the weights per update, while larger learning rates result in faster changes and require fewer iterations. High value of learning rate can cause quickly converge to a suboptimal solution, In the other hand low value of learning rate can cause the process to stuck.<br />


# Built With
- [python](https://www.python.org/) <br /><br />

# Getting Started
### Prerequisites
- put Data in your project path
- numpy <br />
    `pip install numpy`<br />
- matplotlib <br />
    `pip install matplotlib`<br />
    
<br /><br />
# License
Distributed under the MIT License. See `LICENSE.txt` for more information
<br /><br />

# Contact
rezaie.somayeh79@gmail.com
