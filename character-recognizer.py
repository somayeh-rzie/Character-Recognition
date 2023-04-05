import numpy as np
import matplotlib.pyplot as plt
import os


ROWS = 9
COLS = 7
NUMBER_OF_CHARACTERS = 63

batch_size = 1
number_of_hidden_cells = 16
number_of_correct_estimations = 0
maximum_cost = 4

errors = []

dictionary = {
  "A": 0,
  "B": 1,
  "C": 2,
  "D": 3,
  "E": 4,
  "J": 5,
  "K": 6
}


# Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Load Dataset
def read_from_file(address):
    path = os.path.join('data hw1',address)
    result_set = []

    counter = 0

    # iterate through all file
    for file in os.listdir(path):
        # Check whether file is in txt format or not
        if file.endswith(".txt"):
            counter = counter+1
            file_path = os.path.join(path, file)
            temp = open(file_path,'r').read().splitlines()
            txt = np.zeros((NUMBER_OF_CHARACTERS, 1))
            i=0
            for line in temp:
                for char in line:
                    if(char=='.'):
                        txt[i, 0] = 0.
                    elif(char=='#'):
                        txt[i, 0] = 1.
                    else:
                        txt[i, 0] = 0.5
                    i = i+1
            # first character of file name is the correct label e.g:A1.txt shows A
            label_value = file[0]
            label = np.zeros((7, 1))
            label[dictionary[label_value], 0] = 1
            result_set.append((txt, label))
    return result_set

train_set = read_from_file('TrainSetHW1')
test_set = read_from_file('TestSetHW1')


def plot_result(x, y, label_x, label_y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel=label_x, ylabel=label_y, title='Histogram')
    ax.grid()
    fig.savefig("test.png")
    plt.show()


def train(learning_rate):

    total_costs = []
    cost = 100
    number_of_epochs = 0
    # Initialize W with random normal distribution for each layer.
    W1 = np.random.normal(size=(number_of_hidden_cells, NUMBER_OF_CHARACTERS))
    W2 = np.random.normal(size=(7, number_of_hidden_cells))

    # print('First Weight Matrix is :\n', W1)
    # print('Second Weight Matrix is :\n', W2)

    # Initialize b = 0, for each layer.
    b1 = np.random.normal(size=(number_of_hidden_cells, 1))
    b2 = np.random.normal(size=(7, 1))

    # print('First Bias is :\n', b1)
    # print('Second Bias is :\n', b2)


    while(cost > maximum_cost):
        batches = [train_set[x:x+batch_size] for x in range(0, len(train_set), batch_size)]
        for batch in batches:
            # allocate grad_W matrix for each layer
            grad_W1 = np.zeros((number_of_hidden_cells, NUMBER_OF_CHARACTERS))
            grad_W2 = np.zeros((7, number_of_hidden_cells))
            # allocate grad_b for each layer
            grad_b1 = np.zeros((number_of_hidden_cells, 1))
            grad_b2 = np.zeros((7, 1))
            
            for txt, label in batch:
                # compute the output (image is equal to a0)
                a1 = sigmoid(W1 @ txt + b1)
                a2 = sigmoid(W2 @ a1 + b2)
                
                # ---- Last layer
                # weight
                for j in range(grad_W2.shape[0]):
                    for k in range(grad_W2.shape[1]):
                        grad_W2[j, k] += 2 * (a2[j, 0] - label[j, 0]) * a2[j, 0] * (1 - a2[j, 0]) * a1[k, 0]
                
                # bias
                for j in range(grad_b2.shape[0]):
                        grad_b2[j, 0] += 2 * (a2[j, 0] - label[j, 0]) * a2[j, 0] * (1 - a2[j, 0])      
                

                # ---- 2nd layer
                # activation
                delta = np.zeros((number_of_hidden_cells, 1))
                for k in range(number_of_hidden_cells):
                    for j in range(7):
                        delta[k, 0] += 2 * (a2[j, 0] - label[j, 0]) * a2[j, 0] * (1 - a2[j, 0]) * W2[j, k]
                
                # weight
                for m in range(grad_W1.shape[0]):
                    for v in range(grad_W1.shape[1]):
                        grad_W1[m, v] += delta[m, 0] * a1[m,0] * (1 - a1[m, 0]) * txt[v, 0]
                        
                # bias
                for m in range(grad_b1.shape[0]):
                    grad_b1[m, 0] += delta[m, 0] * a1[m, 0] * (1 - a1[m, 0])
            
            W2 = W2 - (learning_rate * (grad_W2 / batch_size))
            W1 = W1 - (learning_rate * (grad_W1 / batch_size))
            
            b2 = b2 - (learning_rate * (grad_b2 / batch_size))
            b1 = b1 - (learning_rate * (grad_b1 / batch_size))
        
        # calculate cost average per epoch
        cost = 0
        for train_data in train_set:
            a0 = train_data[0]
            a1 = sigmoid(W1 @ a0 + b1)
            a2 = sigmoid(W2 @ a1 + b2)

            for j in range(7):
                cost += np.power((a2[j, 0] - train_data[1][j,  0]), 2)
                
        cost /= 2
        total_costs.append(cost)
        number_of_epochs = number_of_epochs + 1    
    total_train = 0
    number_of_correct_estimations = 0
    for train_data in train_set:
        a0 = train_data[0]
        a1 = sigmoid(W1 @ a0 + b1)
        a2 = sigmoid(W2 @ a1 + b2)
        
        predicted_number = np.where(a2 == np.amax(a2))
        real_number = np.where(train_data[1] == np.amax(train_data[1]))

        total_train = total_train+1
        
        if predicted_number == real_number:
            number_of_correct_estimations += 1
            
    # print(f"Error of Train: {round(100 - (number_of_correct_estimations/total_train)*100 , 1)}")
    return W1, W2, b1, b2, number_of_epochs, total_costs


def test(weight1, weight2, bias1, bias2):
    total_test = 0
    number_of_correct_estimations = 0
    for test_data in test_set:
        a0 = test_data[0]
        a1 = sigmoid(weight1 @ a0 + bias1)
        a2 = sigmoid(weight2 @ a1 + bias2)
        
        predicted_number = np.where(a2 == np.amax(a2))
        real_number = np.where(test_data[1] == np.amax(test_data[1]))

        total_test = total_test+1
        
        if predicted_number == real_number:
            number_of_correct_estimations += 1
            
    print(f"Error of Test: {round(100 - (number_of_correct_estimations/total_test)*100 , 1)}")
    return round(100 - (number_of_correct_estimations/total_test)*100 , 1)


learning_rates = [0.01, 0.1, 0.2, 0.5, 0.9]

for learning_rate in learning_rates:
    weight1, weight2, bias1, bias2, epoch, total_costs = train(learning_rate)
    print('Number of Iterations:',epoch)
    error = test(weight1, weight2, bias1, bias2)
    errors.append(error)


plot_result(learning_rates, errors, 'Learning Rate', 'Error')

# weight1, weight2, bias1, bias2, epoch, total_costs = train(learning_rate=0.1)
# test(weight1, weight2, bias1, bias2)

# print('Number of Iterations:',epoch)

# epoch_size = [x for x in range(epoch)]
# plot_result(epoch_size, total_costs, 'epoch size', 'Error')