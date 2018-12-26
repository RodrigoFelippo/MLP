import os
from netCDF4 import Dataset
import numpy as np

convergence_path = 'C:/Users/suporte/Desktop/Outros/poli/APAC/convergencia'    #path of folder of images    
not_convergence_path = 'C:/Users/suporte/Desktop/Outros/poli/APAC/nao-convergencia'

convergence_list = os.listdir(convergence_path)

not_convergence_list = os.listdir(not_convergence_path)

input_cappi_matrix = []
output_cappi_matrix = []

def reduce_matrix(input_matrix, size_reduce_matrix):
    value = 0
    output_matrix = []
    initial_first_aux_size_reduce_matrix = 0
    first_aux_size_reduce_matrix = size_reduce_matrix
    initial_second_aux_size_reduce_matrix = 0
    second_aux_size_reduce_matrix = size_reduce_matrix
    
    while(first_aux_size_reduce_matrix <= len(input_matrix)): 
        while(second_aux_size_reduce_matrix <= len(input_matrix)):
            for i in range(initial_first_aux_size_reduce_matrix, first_aux_size_reduce_matrix):
                for j in range(initial_second_aux_size_reduce_matrix, second_aux_size_reduce_matrix):
                    value += input_matrix[i][j]
            initial_second_aux_size_reduce_matrix += size_reduce_matrix
            second_aux_size_reduce_matrix += size_reduce_matrix
            average = (value/(size_reduce_matrix*size_reduce_matrix))
            output_matrix.append(average)
            value = 0
        initial_first_aux_size_reduce_matrix += size_reduce_matrix  
        first_aux_size_reduce_matrix += size_reduce_matrix
        second_aux_size_reduce_matrix = size_reduce_matrix
        initial_second_aux_size_reduce_matrix = 0
           
    return output_matrix          
              
    
for convergence_file in convergence_list:
     if "V.cappi_top.nc4" in convergence_file: 
        fh = Dataset(convergence_path+"/"+convergence_file, mode='r')
        cappi = fh.variables['Band1'][:]
        np_cappi = np.array(cappi)
        reduce_cappi = reduce_matrix(np_cappi, 10)
        output_cappi_matrix.append(1)
        input_cappi_matrix.append(reduce_cappi)
        
        
for not_convergence_file in not_convergence_list:
     if "V.cappi_top.nc4" in not_convergence_file: 
        fh = Dataset(not_convergence_path+"/"+not_convergence_file, mode='r')
        cappi = fh.variables['Band1'][:]
        np_cappi = np.array(cappi)
        reduce_cappi = reduce_matrix(np_cappi, 10)
        output_cappi_matrix.append(0)
        input_cappi_matrix.append(reduce_cappi)
        
        
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_cappi_matrix, output_cappi_matrix, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Now let's make the ANN!
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

current_input_dim = len(input_cappi_matrix[0])
current_units = int(current_input_dim/2)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = current_units, kernel_initializer = 'uniform', activation = 'relu', input_dim = current_input_dim))

# Adding the second hidden layer
classifier.add(Dense(units = current_units, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)