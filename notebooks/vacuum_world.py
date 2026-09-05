# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: tf-metal
#     language: python
#     name: python3
# ---

# %%
# %run bootstrap.ipynb

# %% [markdown]
# # THE VACUUM WORLD   
#
# In this notebook, we will be discussing **the structure of agents** through an example of the **vacuum agent**. The job of AI is to design an **agent program** that implements the agent function: the mapping from percepts to actions. We assume this program will run on some sort of computing device with physical sensors and actuators: we call this the **architecture**:
#
# <h3 align="center">agent = architecture + program</h3>

# %% [markdown]
# Before moving on, please review [<b>agents.ipynb</b>](https://github.com/aimacode/aima-python/blob/master/agents.ipynb)

# %% [markdown]
# ## CONTENTS
#
# * Agent
# * Random Agent Program
# * Table-Driven Agent Program
# * Simple Reflex Agent Program
# * Model-Based Reflex Agent Program
# * Goal-Based Agent Program
# * Utility-Based Agent Program
# * Learning Agent

# %% [markdown]
# ## AGENT PROGRAMS
#
# An agent program takes the current percept as input from the sensors and returns an action to the actuators. There is a difference between an agent program and an agent function: an agent program takes the current percept as input whereas an agent function takes the entire percept history.
#
# The agent program takes just the current percept as input because nothing more is available from the environment; if the agent's actions depend on the entire percept sequence, the agent will have to remember the percept.
#
# We'll discuss the following agent programs here with the help of the vacuum world example:
#
# * Random Agent Program
# * Table-Driven Agent Program
# * Simple Reflex Agent Program
# * Model-Based Reflex Agent Program
# * Goal-Based Agent Program
# * Utility-Based Agent Program

# %% [markdown]
# ## Random Agent Program
#
# A random agent program, as the name suggests, chooses an action at random, without taking into account the percepts.   
# Here, we will demonstrate a random vacuum agent for a trivial vacuum environment, that is, the two-state environment.

# %% [markdown]
# Let's begin by importing all the functions from the agents module:

# %%
from aima.agents import *
from aima.notebook_utils import psource

# %% [markdown]
# Let us first see how we define the TrivialVacuumEnvironment. Run the next cell to see how abstract class TrivialVacuumEnvironment is defined in agents module:

# %%
psource(TrivialVacuumEnvironment)


# %%
class Vacuum2D(GraphicEnvironment):
    '''Create graphic environment that allows for a 2D grid'''
    def __init__(self):
        super().__init__(2, 2)
        self.colors.update({
            'Dirt': (139, 69, 19), # Brown color for dirt
            'VacuumAgent2D': (0, 0, 255) # Blue color for agent
        })
        self.locations = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.status = {
            loc: random.choice(['Clean', 'Dirty']) for loc in self.locations
        }
        for loc, state in self.status.items():
            if state == 'Dirty':
                self.add_thing(Dirt(), location=loc)
        
    def percept(self, agent):
        '''Return a list of things that are in the agent's location'''
        things = self.list_things_at(agent.location)
        loc = copy.deepcopy(list(agent.location))
        if agent.direction.direction == Direction.R:
            loc[0] += 1
        elif agent.direction.direction == Direction.L:
            loc[0] -= 1
        elif agent.direction.direction == Direction.D:
            loc[1] += 1
        elif agent.direction.direction == Direction.U:
            loc[1] -= 1
        if not self.is_inbounds(loc):
            things.append(Bump())
        return things
    
    def execute_action(self, agent, action):
        agent_name = str(agent)[1:-1]
        if action == 'TurnRight':
            print('{} decided to {} at location: {}'.format(agent_name, action, agent.location))
            agent.turn(Direction.R)
        elif action == 'TurnLeft':
            print('{} decided to {} at location: {}'.format(agent_name, action, agent.location))
            agent.turn(Direction.L)
        elif action == 'MoveForward':
            percepts = self.percept(agent)
            if any(isinstance(p, Bump) for p in percepts):
                print('{} decided to move to the {} but bumped at location: {}'.format(agent_name, agent.direction.direction, agent.location))
                agent.moveforward(success=False)
            else:
                print('{} decided to move to the {} at location: {}'.format(agent_name, agent.direction.direction, agent.location))
                agent.moveforward(success=True)
        elif action == 'Suck':
            print('{} decided to {} at location: {}'.format(agent_name, action, agent.location))
            items = self.list_things_at(agent.location)
            if len(items) > 0 and agent.suck(items[0]):
                print('{} sucked {} at location: {}'.format(agent_name, str(items[0])[1:-1], agent.location))
                self.delete_thing(items[0])
                self.status[agent.location] = 'Clean'
        elif action == 'NoOp':
            print('{} decided to {} at location: {}'.format(agent_name, action, agent.location))

    def is_done(self):
        '''Done when all agents are dead or all tiles are clean'''
        all_clean = not any(isinstance(thing, Dirt) for thing in self.things)
        dead_agents = not any(agent.is_alive() for agent in self.agents)
        return dead_agents or all_clean


class VacuumAgent2D(Agent):
    location = [0, 1]
    direction = Direction("down")

    def moveforward(self, success=True):
        '''moveforward possible only if success is true'''
        if not success:
            return

        x, y = self.location
        if self.direction.direction == Direction.R:
            x += 1
        elif self.direction.direction == Direction.L:
            x -= 1
        elif self.direction.direction == Direction.D:
            y += 1
        elif self.direction.direction == Direction.U:
            y -= 1
        self.location = (x, y)

    def turn(self, d):
        self.direction = self.direction + d
        
    def suck(self, thing):
        return isinstance(thing, Dirt)


# %%
# 2x2 Vacuum Environment
vacuum_env = Vacuum2D()
print("Initial state of the Environment: {}.".format(vacuum_env.status))

# Create the random agent
random_agent = VacuumAgent2D(RandomAgentProgram(['TurnRight', 'TurnLeft', 'MoveForward', 'Suck']))

# Add random agent to the environment
vacuum_env.add_thing(random_agent)
print("RandomVacuumAgent is located at {}.".format(random_agent.location))

# %%
# Running the environment
vacuum_env.run()

print("Final state of the Environment: {}.".format(vacuum_env.status))
print("RandomVacuumAgent is located at {}.".format(random_agent.location))

# %% [markdown]
# ## TABLE-DRIVEN AGENT PROGRAM
#
# A table-driven agent program keeps track of the percept sequence and then uses it to index into a table of actions to decide what to do. The table represents explicitly the agent function that the agent program embodies.  
# In the two-state vacuum world, the table would consist of all the possible states of the agent.

# %%
# Remove random agent from environment
vacuum_env.delete_thing(random_agent)

# %%
table = {
    ('Dirty',): 'Suck',
    ('Clean', Direction.D): 'MoveForward',
    ('Clean', Direction.R): 'MoveForward',
    ('Clean', Direction.U): 'MoveForward',
    ('Clean', Direction.L): 'MoveForward',
    ('Bump', Direction.D): 'TurnLeft',
    ('Bump', Direction.R): 'TurnLeft',
    ('Bump', Direction.U): 'TurnLeft',
    ('Bump', Direction.L): 'TurnLeft'
}


# %%
# Create a table-driven agent program for the 2x2 environment
def TableDrivenAgentProgram(table):
    percept_sequence = []
    def program(percept):
        nonlocal percept_sequence
        
        percept_sequence.append(percept)
        if any(isinstance(p, Bump) for p in percept):
            percept_type = 'Bump'
        elif any(isinstance(p, Dirt) for p in percept):
            percept_type = 'Dirty'
        else:
            percept_type = 'Clean'

        if not hasattr(program, 'direction'):
            program.direction = Direction("down")

        if percept_type == 'Dirty':
            key = (percept_type,)
        else:
            key = (percept_type, program.direction.direction)

        action = table.get(key, 'MoveForward')
        if action == 'TurnLeft':
            program.direction = program.direction + Direction.L
        return action
    return program


# %%
# 2x2 Vacuum Environment
vacuum_env = Vacuum2D()
print("Initial state of the Environment: {}.".format(vacuum_env.status))

# Create a table-driven agent
table_driven_agent = VacuumAgent2D(program=TableDrivenAgentProgram(table=table))

# Add the table-driven agent to the environment
vacuum_env.add_thing(table_driven_agent)
print("TableDrivenVacuumAgent is located at {}.".format(table_driven_agent.location))

# %%
# Run the environment
vacuum_env.run()

print("Final state of the Environment: {}.".format(vacuum_env.status))
print("TableDrivenVacuumAgent is located at {}.".format(table_driven_agent.location))

# %% [markdown]
# ## SIMPLE REFLEX AGENT PROGRAM
#
# A simple reflex agent program selects actions on the basis of the *current* percept, ignoring the rest of the percept history. These agents work on a **condition-action rule** (also called **situation-action rule**, **production** or **if-then rule**), which tells the agent the action to trigger when a particular situation is encountered.  
#
# The schematic diagram shown in **Figure 2.9** of the book will make this more clear:
#
# "![simple reflex agent](images/simple_reflex_agent.jpg)"

# %% [markdown]
# Let us now create a simple reflex agent for the environment.

# %%
# Delete the previously added table-driven agent
vacuum_env.delete_thing(table_driven_agent)


# %%
# We change the simpleReflexAgentProgram so that it doesn't make use of the Rule class
def SimpleReflexAgentProgram():
    """This agent takes action based solely on the percept. [Figure 2.10]"""
    def program(percept):
        """If there is dirt, suck, otherwise if there's a wall, turn right, otherwise move forward"""
        for p in percept:
            if isinstance(p, Dirt):
                return 'Suck'
        for p in percept:
            if isinstance(p, Bump):
                return 'TurnRight'      
        return 'MoveForward'
    return program


# %% [markdown]
# Now add the agent to the environment:

# %%
# 2x2 Vacuum Environment
vacuum_env = Vacuum2D()
print("Initial state of the Environment: {}.".format(vacuum_env.status))

# Create a simple reflex agent
simple_reflex_agent = VacuumAgent2D(SimpleReflexAgentProgram())

# Add the simple reflex agent to the environment
vacuum_env.add_thing(simple_reflex_agent)
print("SimpleReflexVacuumAgent is located at {}.".format(simple_reflex_agent.location))

# %%
# Run the environment
vacuum_env.run()

print("Final state of the Environment: {}.".format(vacuum_env.status))
print("SimpleReflexVacuumAgent is located at {}.".format(simple_reflex_agent.location))

# %% [markdown]
# ## MODEL-BASED REFLEX AGENT PROGRAM
#
# A model-based reflex agent maintains some sort of **internal state** that depends on the percept history and thereby reflects at least some of the unobserved aspects of the current state. In addition to this, it also requires a **model** of the world, that is, knowledge about "how the world works".
#
# The schematic diagram shown in **Figure 2.11** of the book will make this more clear:
# <img src="images/model_based_reflex_agent.jpg">

# %% [markdown]
# We will now create a model-based reflex agent for the environment:

# %%
# Delete the previously added simple reflex agent
vacuum_env.delete_thing(simple_reflex_agent)


# %%
def update_state(state, action, percept, model):
    '''Updates the current state based on the action of the agent and percept'''
    if action == 'TurnRight':
        state['direction'] = state['direction'] + Direction.R
    elif action == 'TurnLeft':
        state['direction'] = state['direction'] + Direction.L
    elif action == 'MoveForward':
        x, y = state['location']
        if state['direction'].direction == 'down':
            y += 1
        elif state['direction'].direction == 'up':
            y -= 1
        elif state['direction'].direction == 'left':
            x -= 1
        elif state['direction'].direction == 'right':
            x += 1
        state['location'] = (x, y)
        
    if any(isinstance(p, Dirt) for p in percept): # if current tile has dirt
        model[state['location']] = 'Dirty' # mark dirty
    else:
        model[state['location']] = 'Clean' # otherwise mark clean
    return state

def ModelBasedReflexAgentProgram():
    last_action = None # what agent did last
    state = { # initial state
        'location': (0, 1),
        'direction': Direction("down"),
    }
    model = { # all tiles are unknown at start so assume they are dirty
                (0, 0): 'Dirty',
                (0, 1): 'Dirty',
                (1, 0): 'Dirty',
                (1, 1): 'Dirty',
            }
    
    def program(percept):
        nonlocal last_action, state, model
        
        state = update_state(state, last_action, percept, model) # get current state
        curr_loc = state['location'] # get current location
        if model[state['location']] == 'Dirty': # check if current location is actually dirty and suck if it is
            last_action = 'Suck'
            return last_action
        else:
            model[state['location']] = 'Clean' # otherwise set current location to clean
        
        dirty_tiles = [loc for loc, status in model.items() if status == 'Dirty'] # get remaining dirty tiles
        if len(dirty_tiles) == 0: # check if all tiles are clean and do nothing if so
            last_action = 'NoOp'
            return last_action
        
        nearest_dirty_tile = min(dirty_tiles, key=lambda loc: abs(curr_loc[0] - loc[0]) + abs(curr_loc[1] - loc[1]))
        tx, ty = nearest_dirty_tile # target location
        cx, cy = curr_loc # current location
        if ty > cy:
            dirty_tile_dir = 'down'
        elif ty < cy:
            dirty_tile_dir = 'up'
        elif tx > cx:
            dirty_tile_dir = 'right'
        elif tx < cx:
            dirty_tile_dir = 'left'
        else:
            dirty_tile_dir = state['direction'].direction
        
        if state['direction'].direction == dirty_tile_dir: # move forward if facing a dirty tile otherwise turn until facing dirty tile
            last_action = 'MoveForward'
        else:
            last_action = 'TurnRight'
        return last_action
    return program


# %%
# 2x2 Vacuum Environment
vacuum_env = Vacuum2D()
print("Initial state of the Environment: {}.".format(vacuum_env.status))

# Create a model-based reflex agent
model_based_reflex_agent = VacuumAgent2D(ModelBasedReflexAgentProgram())

# Add the simple reflex agent to the environment
vacuum_env.add_thing(model_based_reflex_agent, (0, 1)) # specify location so inital state matches
print("ModelBasedReflexAgent is located at {}.".format(model_based_reflex_agent.location))

# %%
# Run the environment
vacuum_env.run()

print("State of the Environment: {}.".format(vacuum_env.status))
print("ModelBasedVacuumAgent is located at {}.".format(model_based_reflex_agent.location))

# %% [markdown]
# ## GOAL-BASED AGENT PROGRAM
#
# A goal-based agent needs some sort of **goal** information that describes situations that are desirable, apart from the current state description.
#
# **Figure 2.13** of the book shows a model-based, goal-based agent:
# <img src="images/model_goal_based_agent.jpg">
#
# **Search** (Chapters 3 to 5) and **Planning** (Chapters 10 to 11) are the subfields of AI devoted to finding action sequences that achieve the agent's goals.
#
# ## UTILITY-BASED AGENT PROGRAM
#
# A utility-based agent maximizes its **utility** using the agent's **utility function**, which is essentially an internalization of the agent's performance measure.
#
# **Figure 2.14** of the book shows a model-based, utility-based agent:
# <img src="images/model_utility_based_agent.jpg">

# %% [markdown]
# ## LEARNING AGENT
#
# Learning allows the agent to operate in initially unknown environments and to become more competent than its initial knowledge alone might allow. Here, we will breifly introduce the main ideas of learning agents.  
#
# A learning agent can be divided into four conceptual components. The **learning element** is responsible for making improvements. It uses the feedback from the **critic** on how the agent is doing and determines how the performance element should be modified to do better in the future. The **performance element** is responsible for selecting external actions for the agent: it takes in percepts and decides on actions. The critic tells the learning element how well the agent is doing with respect to a fixed performance standard. It is necesaary because the percepts themselves provide no indication of the agent's success. The last component of the learning agent is the **problem generator**. It is responsible for suggesting actions that will lead to new and informative experiences.  
#
# **Figure 2.15** of the book sums up the components and their working:  
# <img src="images/general_learning_agent.jpg">
