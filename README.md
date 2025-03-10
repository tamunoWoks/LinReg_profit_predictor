# Linear Regression: Predicting Restaurant Profits with Linear Regression
This is a machine learning project that implements univariate linear regression to predict the profits of a restaurant franchise based on city population. This project demonstrates key concepts of supervised learning, cost function optimization, and gradient descent.

# Outline
- [ 1 - Packages ](#1)
- [ 2 - Linear regression with one variable ](#2)
  - [ 2.1 Problem Statement](#2.1)
  - [ 2.2  Dataset](#2.2)
  - [ 2.3 Refresher on linear regression](#2.3)
  - [ 2.4  Compute Cost](#2.4)
    - [ Exercise 1](#ex01)
  - [ 2.5 Gradient descent ](#2.5)
    - [ Exercise 2](#ex02)
  - [ 2.6 Learning parameters using batch gradient descent ](#2.6)

<a name="1"></a>
## 1 - Packages 

First, let's run the cell below to import all the packages that you will need during this assignment.
- [numpy](www.numpy.org) is the fundamental package for working with matrices in Python.
- [matplotlib](http://matplotlib.org) is a famous library to plot graphs in Python.
- ``utils.py`` contains helper functions.

```python
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
%matplotlib inline
```
## 2 -  Problem Statement
Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.
- We would like to expand your business to cities that may give your restaurant higher profits.
- The chain already has restaurants in various cities and you have data for profits and populations from the cities.
- You also have data on cities that are candidates for a new restaurant. 
    - For these cities, you have the city population.
    
Use the data to help identify which cities may potentially give the business higher profits?

## 3 - Dataset
We will start by loading the dataset for this task. 
- The `load_data()` function shown below loads the data into variables `x_train` and `y_train`
  - `x_train` is the population of a city
  - `y_train` is the profit of a restaurant in that city. A negative value for profit indicates a loss.   
  - Both `X_train` and `y_train` are numpy arrays.
```python
# load the dataset
x_train, y_train = load_data()
```
### View the variables
Before starting on any task, it is useful to get more familiar with your dataset.  
- A good place to start is to just print out each variable and see what it contains.
The code below prints the variable `x_train` and the type of the variable.
```python
# print x_train
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5])
```
**Output:**  
```
Type of x_train: <class 'numpy.ndarray'>  
First five elements of x_train are:  
 [6.1101 5.5277 8.5186 7.0032 5.8598]
```
 
`x_train` is a numpy array that contains decimal values that are all greater than zero.
- These values represent the city population times 10,000
- For example, 6.1101 means that the population for that city is 61,101

Now, let's print `y_train`
```python
# print y_train
print("Type of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5])
```
**Output:**  
```
Type of y_train: <class 'numpy.ndarray'>  
First five elements of y_train are:  
 [17.592   9.1302 13.662  11.854   6.8233]
```
 
Similarly, `y_train` is a numpy array that has decimal values, some negative, some positive.
- These represent your restaurant's average monthly profits in each city, in units of \$10,000.
  - For example, 17.592 represents \$175,920 in average monthly profits for that city.
  - -2.6807 represents -\$26,807 in average monthly loss for that city.

### Check the dimensions of your variables
Another useful way to get familiar with your data is to view its dimensions.  
Let's print the shape of `x_train` and `y_train` and see how many training examples you have in your dataset.
```python
print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))
```
**Output:**  
```
The shape of x_train is: (97,)
The shape of y_train is:  (97,)
Number of training examples (m): 97
```
The city population array has 97 data points, and the monthly average profits also has 97 data points. These are NumPy 1D arrays.
### Visualize the data
It is often useful to understand the data by visualizing it. 
- For this dataset, you can use a scatter plot to visualize the data, since it has only two properties to plot (profit and population). 
- Many other problems that you will encounter in real life have more than two properties (for example, population, average household income, monthly profits, monthly sales).When you have more than two properties, you can still use a scatter plot to see the relationship between each pair of properties.
```python
# Create a scatter plot of the data. To change the markers to red "x",
# we used the 'marker' and 'c' parameters
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()
```
![](https://github.com/tamunoWoks/LinReg_profit_predictor/blob/main/images/profit_vs_population.png)  
Our goal is to build a linear regression model to fit this data.
- With this model, we can then input a new city's population, and have the model estimate your restaurant's potential monthly profits for that city.
 
<a name="4"></a>
## 4 - Refresher on linear regression
Let us fit the linear regression parameters $(w,b)$ to our dataset.
- The model function for linear regression, which is a function that maps from `x` (city population) to `y` (your restaurant's monthly profit for that city) is represented as 
    $$f_{w,b}(x) = wx + b$$ 
- To train a linear regression model, you want to find the best $(w,b)$ parameters that fit your dataset.  
    - To compare how one choice of $(w,b)$ is better or worse than another choice, you can evaluate it with a cost function $J(w,b)$
      - $J$ is a function of $(w,b)$. That is, the value of the cost $J(w,b)$ depends on the value of $(w,b)$.
    - The choice of $(w,b)$ that fits your data the best is the one that has the smallest cost $J(w,b)$.
- To find the values $(w,b)$ that gets the smallest possible cost $J(w,b)$, you can use a method called **gradient descent**. 
  - With each step of gradient descent, your parameters $(w,b)$ come closer to the optimal values that will achieve the lowest cost $J(w,b)$.
- The trained linear regression model can then take the input feature $x$ (city population) and output a prediction $f_{w,b}(x)$ (predicted monthly profit for a restaurant in that city).

<a name="5"></a>
## 5 - Compute Cost
Gradient descent involves repeated steps to adjust the value of your parameter $(w,b)$ to gradually get a smaller and smaller cost $J(w,b)$.
- At each step of gradient descent, it will be helpful for you to monitor your progress by computing the cost $J(w,b)$ as $(w,b)$ gets updated. 
- In this section, you will implement a function to calculate $J(w,b)$ so that you can check the progress of your gradient descent implementation.
#### Cost function
As you may recall from the lecture, for one variable, the cost function for linear regression $J(w,b)$ is defined as
$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$ 
- You can think of $f_{w,b}(x^{(i)})$ as the model's prediction of your restaurant's profit, as opposed to $y^{(i)}$, which is the actual profit that is recorded in the data.
- $m$ is the number of training examples in the dataset
#### Model prediction
- For linear regression with one variable, the prediction of the model $f_{w,b}$ for an example $x^{(i)}$ is representented as:
$$ f_{w,b}(x^{(i)}) = wx^{(i)} + b$$
This is the equation for a line, with an intercept $b$ and a slope $w$
#### Implementation
Please complete the `compute_cost()` function below to compute the cost $J(w,b)$.
```python
def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    
    # Variable to keep track of sum of cost from each example
    cost_sum = 0
    
    # Loop over training examples
    for i in range(m):
        # code to get the prediction f_wb for the ith example
        f_wb = w * x[i] + b
        # code to get the cost associated with the ith example
        cost = (f_wb - y[i]) ** 2
        # Add to sum of cost for each example
        cost_sum = cost_sum + cost
        
    # Get the total cost as the sum divided by (2*m)
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost
```
We can check if your implementation was correct by running the following test code:
```python
# Compute cost with some initial values for paramaters w, b
initial_w = 2
initial_b = 1

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')

# Public tests
from public_tests import *
compute_cost_test(compute_cost)
```
**Output:**  
<class 'numpy.float64'>  
Cost at initial w: 75.203  
All tests passed!
<a name="6"></a>
## 6 - Gradient descent 

In this section, we will implement the gradient for parameters $w, b$ for linear regression. 

We will implement a function called `compute_gradient` which calculates $\frac{\partial J(w)}{\partial w}$, $\frac{\partial J(w)}{\partial b}$ 
```python
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]
    
    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0
    
    # Loop over examples
    for i in range(m):
        # Code to get  prediction f_wb for the ith example
        f_wb = w * x[i] + b
        
        # Code to get gradient for w from the ith example
        dj_dw_i = (f_wb - y[i]) * x[i]
        
        # Code to get gradient for b from the ith example
        dj_db_i = f_wb - y[i]
        
        # Update dj_db 
        dj_db += dj_db_i
        
        # Update dj_dw 
        dj_dw += dj_dw_i
        
    # Divide both dj_dw and dj_db by m
    dj_dw = dj_dw / m
    dj_db = dj_db / m
        
    return dj_dw, dj_db
```
```python
# Compute and display gradient with w initialized to zeroes
initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

compute_gradient_test(compute_gradient)
```
**Output:**
```
Gradient at initial w, b (zeros): -65.32884974555672 -5.83913505154639
Using X with shape (4, 1)
All tests passed!
```
Now let's run the gradient descent algorithm implemented above on our dataset.
```python
# Compute and display cost and gradient with non-zero w
test_w = 0.2
test_b = 0.2
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)

print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)
```
**Output:**  
```
Gradient at test w, b: -47.41610118114435 -4.007175051546391
```
