#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml #Just for dataset

fashion_data = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')

raw_images = fashion_data.data
raw_labels = fashion_data.target.astype(int) 

'''CNN does not convert the image into flattened vectors to predict but rather keep the image as the same like a 2D grid pixels so the advantage of CNN 
compared to basic NN is it sees image as image not as numeric pixels values and also CNN has exposure to likelihood pixel values'''

#Doing the conventional ML code step extracting the images from dataset spilitting into training and test sets 

#Creating a separate class for splitting the test and train sets out of 60000 images into random 400 test and train each and normalise them using mean trick

class fashion:
    def __init__(self , images , labels , subset = 400):
       self.images = images[:subset].astype(np.float32)
       self.labels = labels[:subset]

       #Normalizing image using mean trick
       self.images = self.images /255 #Max pixel value = 255 complete white so dividing with max value in image returns value between [0,1]

       self.images = self.images.reshape(-1, 28, 28) #reshaping the flatened vector into 2D grid 

    def get_train_test_split(self):
        #to get a unified split of 50/50 test and train sets divide by 2

        half_size = len(self.images) //2 #Floor division inscase split is decimal
        shuffled_indices = np.random.permutation(len(self.images)) 

        '''this is because fashion MNIST has images in order that is first 1000 can be tshirts next 1000 can be pants and so on so if we split them 
        orderly into 400 out of 60000 then 200 for each (test and train) first train may contain only tshirts so while testing the model will
        fail to recognize pants so shuffling the 400 and splitting makes easier for testing'''

        x_shuffled = self.images[shuffled_indices]
        y_shuffled = self.labels[shuffled_indices]

        #The final 50/50 split

        x_train = x_shuffled[:half_size]
        x_test = x_shuffled[half_size:]
        y_train = y_shuffled[:half_size]
        y_test = y_shuffled[half_size:]

        return x_train, x_test, y_train, y_test

# Instantiating the data class
dataloader = fashion(raw_images, raw_labels, subset=400)

# Extracting training and testing sets
x_train, x_test, y_train, y_test = dataloader.get_train_test_split()

#Now going to the original math behind CNN

'''Consider you are exploring a cave and you have a flashlight with you , to see the carvings on the wall you project the light on the wall
similarly here the image is the carvings on the wall and the light we used to see the image is called a filter/kernel'''

Lfilter = np.array([
    [1,0,-1] ,
    [1,0,-1],
    [1,0,-1]]).astype(np.float32) #A light that is going to be projected on image .This is a random filter it will change when backprop comes
 #Converting type to float because at gradient descent updation the arrays will be subtracted to the original will be folat so changing this to float

#There will be a step called pooling after that we need to pass these through ouput predictions for that we need weights and biases like basic NN

np.random.seed(42) #For continous reproducible weights

''' our image is 28 x 28 after pooling it becomes 14 x 14 so for basic NN we need flatenned vector so these 14 x 14 becomes (1000 , 196)flattened vector
and our hidden layer has 64 neurons , our vector is (1000 ,196) and weight is (196 , 64) so first layer output is (1000 , 64) adding bias (1 , 64) for each 
row and then passed to ouput layer of 10 neurons (64 ,10) then (1000 , 64) x ( 64 , 10)  is (1000 , 10) and add bias (1,10) passed to softmax'''

w_hidden_layer = np.random.randn(196 ,64) * 0.01 #Lesser weights more stable the neuron
b_hidden_layer = np.zeros((1,64)) #bias at zero keeps shifting clean and it learn shifting only when needed

w_output_layer = np.random.randn(64 ,10)*0.01
b_output_layer = np.zeros((1,10))

def feature_matrix_math(images , Lfilter , padding = 1):

   '''Padding is the process of adding artificial borders externally to the image because the edge of the image is scanned 
   less compared to the center of the image so the kernel can't scan the image edges properly if we add a extra layer as a boundary to the image
   it overlapse over it and the edge is treated as the center image (In our cave analogy if the light is projected at the corner of the cave
   it immediately turns out failing to sacn edged properly)'''

   batch_size = len(images)
   img_height , img_width = images[0].shape 
   filter_size = len(Lfilter) #used in defining the size of output matrix (The feature matrix)

   #Calculating the padding over the image dynamically
   padded_height = img_height + (2 * padding)
   output_size = (padded_height - filter_size) + 1

   #preallocating the memory for fmatrix
   feature_matrix = np.zeros((batch_size ,output_size ,output_size))

   for x in range(batch_size):
       padded_img = np.pad(images[x] ,pad_width = padding , mode = 'constant' , constant_values = 0)

       '''Now we need to convolute the image with kernel that is by projecting the kernel over the image dynamically and dot product the exact 
       element of kernel with Padded_Image or vice versa it looks like sum(kernel[i][j] x Padded_Image[i][j]) where i and j are respective rows and columns'''

       for i in range(output_size):
           for j in range(output_size):
               patch = padded_img [ i : i + filter_size , j: j + filter_size]
               feature_matrix[x , i , j] = np.sum(patch * Lfilter) #The original Convolution

   return feature_matrix


#Now if there are any -ve values inside fmatrix we need to make them zero because color cannot have zero value
def ReLU(matrix):
    return np.maximum(0,matrix)



#Now doing the pooling this is because there are parts of image that makes the image dull so to make it look highly compresses we do this

def pooling(features , pool_size = 2 , stride =2):
    #stride is the amount of projection of dynamic movement of light over the image

    batch_size , height , width = features.shape
     #After pooling the matrix becomes half in size that is from 28 x 28 to 14 x 14
    output_height = height // stride 
    output_width = width // stride

    pooled_matrix = np.zeros((batch_size , output_height , output_width))

    #the core mechanics of pooling
    for x in range(batch_size):
        for i in range(output_height):
            for j in range(output_width):
                #Boundaries for stride of 2 
                row_start = i * stride
                row_end = row_start  +pool_size
                col_start = j * stride 
                col_end = col_start + pool_size

                #Extracting the patch again for finding max in 2 x 2 distinct blocks of activated_fmatrix

                patch = features[x , row_start : row_end , col_start : col_end]

                pooled_matrix[x, i, j] = np.max(patch)

    return pooled_matrix


#Intiating the pipeline to dense neural layer 
fmatrix = feature_matrix_math(x_train ,Lfilter , padding =1)
activated_fmatrix = ReLU(fmatrix)
pooled_matrix = pooling(activated_fmatrix ,pool_size =2 , stride =2)

#flatten the 1000 x 14 x 14 matrix into (1000 , 196) vector

input_vector = pooled_matrix.reshape(len(pooled_matrix) ,-1)

#fist hidden layer scores
z1 = input_vector @ w_hidden_layer + b_hidden_layer
z1 = ReLU(z1)

z2 = z1 @ w_output_layer + b_output_layer

#Defining the softmax function
def softmax(z): #softmax(z2) = e^z2(i) / sumof(e^z2(j)) where j = z2(1),z2(2),z2(3),...,z2(10) and i is z2 current neuron value
    shifted_z = z - np.max(z ,axis=1 , keepdims=True) #axis =1 because our row has images and column has the class score if we keep axis =1 it will just operate for each column sidewise
    exp_z = np.exp(shifted_z)
    return exp_z /np.sum(exp_z , axis =1 , keepdims =True) #This keepdims changes (z,) to (z,1) so that in future it wont create mismatch dimensions error

#one hot encoding the output softmax values into vectors
def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y] #Eye creates a zero matrix  with diagnals as 1 (num_classes =10 because there are numbers from 0-9)

#Calculating loss using cross entropy
def cross_entropy(predictions , y_true):
    epsilon = 1e-12
    #initialized epsilon because if pred = 0 then log(0) is undefined so we add a small number epsilon which is 1e-12 = (10)^-12 = 0.0000...12
    entropy = -np.sum(y_true * np.log(predictions + epsilon), axis=1)
    return np.mean(entropy)


#Now going with backprop from dense layer - pooled matrix - kernel/filter reinitialization 

def dense_layer_backprop(predictions , y_true , z1 , a1 ,input_vector , w_output_layer , w_hidden_layer):
    # m is our batch size (1000)
    m = predictions.shape[0]

    # Difference between prediction and truth
    dz2 = predictions - y_true # Shape is (10, 1000)

    # Gradients for output weights and biases
    dw2 = (1 / m) * (a1.T @ dz2)# Shape is (64 ,10)
    db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)  # Shape is (1 ,10)

    # Pass error back through output weights and apply the ReLU derivative switch
    # ReLU derivative: 1 if z1 > 0, else 0
    dz1 = (dz2 @ w_output_layer.T) * (z1 > 0)# Shape is (1000 ,64)

    # Gradients for hidden weights and biases
    dw1 = (1 / m) * ( input_vector.T @ dz1)# Shape is (196 ,64)
    db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)# Shape is (1 ,64)

    #converting the flatenned error (blame vector) vector to blame matrix of original pool matrix size
    d_flattened = dz1 @ w_hidden_layer.T               
    return dw1, db1, dw2, db2, d_flattened

#Now doing pooling backprop
def pooling_backprop(d_pooled_input , original_activated_features , pool_size =2 , stride =2):
    #d_pooled_input is gradient matrix flattened to (1000 , 14 ,14)
    #original_activated_features is the actived_fmatrix at first we had with (1000 ,28 ,28)

    batch_size , height , width = original_activated_features.shape
    #pre allocate a de-pooling matrix 
    dx_pool = np.zeros_like(original_activated_features)
    output_height = height // stride
    output_width = width // stride

    for x in range(batch_size):
        for i in range(output_height):
            for j in range(output_width):
                row_start = i * stride
                row_end = row_start + pool_size
                col_start = j * stride
                col_end = col_start + pool_size

                #getting the original patch from activated_fmatrix
                patch = original_activated_features[x, row_start : row_end , col_start : col_end]
                #storing the max from the 2x2 window to replace that window with respective blame in d_pooled_input [ x,i,j]
                max_eval = np.max(patch)
                mask = (patch == max_eval) #Storing true for that place 
                #Now doing the substituion
                dx_pool[x, row_start:row_end, col_start:col_end] += mask * d_pooled_input[x, i, j]
    return dx_pool

#Now the most hardest part the backprop for convolution part
def convolution_backprop(dx_conv ,original_images , filter_size = 3, padding =1):
    #dX_conv are gradients coming back out of ReLU (1000, 28, 28)
    #original_images are the initial training images x_train (1000, 28, 28)
    batch_size, height, width = dX_conv.shape
    #Pre allocating the d_filter matrix 
    d_filter = np.zeros((filter_size , filter_size))

    for x in range(batch_size):
        padded_img = np.pad(original_images[x] ,pad_width = padding , mode ='constant' ,constant_values =0)
        for i in range(height):
            for j in range(width):
                patch = padded_img[ i : i + filter_size , j : j + filter_size]

                # Multiply the patch values by the incoming gradient scalar at this position
                # Accumulate the result across the entire batch
                d_filter += patch * dx_conv[x, i, j]

    return d_filter / batch_size #Average gradient over the batch          


#Training the model

learning_rate = 0.25
epochs = 50

print("Training is started")

for epoch in range(epochs):
    #forward pass
    #feature extraction (Conv + ReLU + Pool)
    fmatrix = feature_matrix_math(x_train, Lfilter, padding=1)
    activated_fmatrix = ReLU(fmatrix)
    pooled_matrix = pooling(activated_fmatrix, pool_size=2, stride=2)

    #flattening
    input_vector = pooled_matrix.reshape(len(pooled_matrix), -1)

    # Dense Hidden Layer
    z1 = input_vector @ w_hidden_layer + b_hidden_layer
    a1 = ReLU(z1)

    # Output Layer & Activation
    z2 = a1 @ w_output_layer + b_output_layer
    predictions = softmax(z2)

    # Compute Performance Metrics
    y_true_encoded = one_hot_encode(y_train, num_classes=10)
    loss = cross_entropy(predictions, y_true_encoded)

    # Calculate Accuracy for tracking
    pred_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(pred_labels == y_train) * 100

    print(f"Epoch {epoch+1}/{epochs} = Loss: {loss:.4f} and Accuracy: {accuracy:.2f}%")

    #Backward pass
    # Classification Layers Backprop
    dw1, db1, dw2, db2, d_flattened = dense_layer_backprop(
        predictions=predictions,
        y_true=y_true_encoded,
        z1=z1,
        a1=a1,
        input_vector=input_vector,
        w_output_layer=w_output_layer,
        w_hidden_layer=w_hidden_layer
    )

    # Un-pooling Layer Backprop
    d_pooled_reshaped = d_flattened.reshape(len(d_flattened), 14, 14)
    dX_pool = pooling_backprop(d_pooled_reshaped, activated_fmatrix)

    # ReLU Layer Backprop
    dX_conv = dX_pool * (fmatrix > 0)

    # Filter Backprop
    dLfilter = convolution_backprop(dX_conv, x_train, filter_size=3, padding=1)

    #Gradient descent
    Lfilter         -= learning_rate * dLfilter
    w_hidden_layer  -= learning_rate * dw1
    b_hidden_layer  -= learning_rate * db1
    w_output_layer  -= learning_rate * dw2
    b_output_layer  -= learning_rate * db2

print("Training is done ") 



#Testing the model
print("Testing is started")

# Run the test images through the forward pass pipeline
fmatrix_test = feature_matrix_math(x_test, Lfilter, padding=1)
activated_fmatrix_test = ReLU(fmatrix_test)
pooled_matrix_test = pooling(activated_fmatrix_test, pool_size=2, stride=2)

input_vector_test = pooled_matrix_test.reshape(len(pooled_matrix_test), -1)

z1_test = input_vector_test @ w_hidden_layer + b_hidden_layer
a1_test = ReLU(z1_test)

z2_test = a1_test @ w_output_layer + b_output_layer
test_predictions = softmax(z2_test)

# Calculate testing accuracy against y_test
test_pred_labels = np.argmax(test_predictions, axis=1)
test_accuracy = np.mean(test_pred_labels == y_test) * 100

print(f"Test Set Accuracy: {test_accuracy:.2f}%")








# In[ ]:




