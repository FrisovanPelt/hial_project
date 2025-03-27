# HIAL PROJECT

Our implementation of the HIAL Final Project. Sadly, we were not able to get a success rate above 0 regardless of repeated attempts. The problem likely lies in the implementation of the AWAC method, but could also be tied to the feature function. 

## Key Files

### pref_learn
Here we make the videos, construct the trajectories, run it through aprel for querying and produce weights. Also the location of the features function.

### policy_learn
This file is primarily for preparing the demo files and environment for usage with the AWAC method

### awac
Full implementation of the AWAC method, with changes to the replay buffer and the run functions. There is also an added evaluate policy function. 

### core 
This is a helper file for AWAC

### policy test
For testing policies
