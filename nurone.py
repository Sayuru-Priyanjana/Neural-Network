inputs = [1,2,3]

weights = [[0.2, 0.8, -0.5],
           [0.5,-0.91,0.26],
           [-1.2, 1.3, 0.8],
           [0.1, 0.4, 0.9 ],
           [0.3, 0.2, 0.5 ]]

bias = [2,3,0.5,1,2]
neuron_outputs = []

for bias_index, weight_list in zip(bias, weights):
    
    neuron_output = 0
    for i in range(len(inputs)):
        print(bias_index , "weights ",  i , "inputs ", i)
        output = inputs[i] * weight_list[i] 
        neuron_output += output
    neuron_outputs.append(neuron_output+ bias_index)

print(neuron_outputs)
  