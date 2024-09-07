# Tag-in-Gridworld-with-Many-Taggers

## How to run this code 
To run the tasks main.py file is used. There are 4 experiments (or tasks) 
* Task 1 - Changing and Visualizing the Environment
* Task 2 - Running value iteration algorithm and visualzing initial, intermediate and final policies
* Task 3 - Running temporal difference learning algorithm and visualzing initial, intermediate and final policies
* Task 4 - Comparing the performance of the two algorithms 
### Changing and Visualizing the Environment 
Arguments : 
* Grid size - n - default = 15 
* Number of taggers - k - default = 2
* Experiment - exp - visualize
* Number of steps to sumulate - steps - default = 5

<br>

Example :
  ``` python main.py --n 15 --k 2 --exp visualize ```

### Running value iteration algorithm and visualzing initial, intermediate and final policies 
Arguments : 
* Grid size - n - default = 15 
* Number of taggers - k - default = 2
* Experiment - exp - vi
* Parameter gamma for value iteration - gamma - default = 0.99
* Theta threshold for stopping value iteration - theta - default = 1e-8

<br>

Example :
<br>
  ``` python main.py --exp vi --gamma 0.99 --theta 1e-8 ```
  <br> 
  or
  <br>
  ``` python main.py --exp vi ```
  
### Running temporal difference learning algorithm and visualzing initial, intermediate and final policies 
Arguments : 
* Grid size - n - default = 15 
* Number of taggers - k - default = 2
* Experiment - exp - td
* Parameter gamma for Td algorithm - gamma - default = 0.99
* Number of episodes to train Td algorithm on - episodes - default = 500 
* Parameter alpha for Td algorithm - alpha - default = 0.5
* Epsilon for exploration-exploitation in Td algorithm - epsilon - default = 0.1
* Parameter lambda for td algorihthm - lambda - default = 0.9

<br>

Example :
<br>
  ``` python main.py --exp td --gamma 0.99 --episodes 500 --alpha 0.5 --epsilon 0.1 --lambda 0.9 ```
  <br> 
  or
  <br>
  ``` python main.py --exp td ```

  ### Comparing the performance of the two algorithms 
  Arguments : 
* Grid size - n - default = 15 
* Number of taggers - k - default = 2
* Experiment - exp - compare
* Number of episodes to plot average rewards vs episode plot for - numepisodes - default = 1000

**Note: all the default parameters for vi and td are used here, however, they can be changed by passing in the arguments at this step**

<br>

Example :
<br>
  ``` python main.py --exp compare --numepisodes 1000 ```
  <br> 
  or
  <br>
  ``` python main.py --exp compare ```

## Bonus - Max Grid Size (n) and Number of taggers (k) values for the algorithms 
For Value Iteration algorithm, max n = 30 , k = 4 for 5 minutes time-limit.
<br> 
``` python main.py --n 30 --k 4 --exp vi ```
<br>

For Temporal Difference Learning algorithm, max n = 90 , k = 4 for 5 minutes time-limit.
<br> 
``` python main.py --n 90 --k 4 --exp td ```

  
