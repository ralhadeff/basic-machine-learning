'''
Deep Q-learning network
'''

import numpy as np
import random
import torch

class Qlearn():
    
    def __init__(self, neurons, memory_size, gamma, temperature, batch_size=100, lr=0.001):
        '''
        Q-learning network ('brain')
            User provides a list for the number of neurons (at least 3 elements for input, hidden and output)
            User also provides the memory size (number of events to remember)
            The gamma (decay) parameter
            The temperature parameter (for softmax output)
            Optional: batch_size for how many events per batch (using experience replay)
            And optionally the learning rate (lr)
        '''
        self.network = Network(neurons[0],neurons[-1],neurons[1:-1])
        # optimizer for backprop
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=lr)
        # memory of events
        self.memory = LimitedList(memory_size)
        self.gamma = gamma
        self.temperature = temperature
        # check consistency
        if (batch_size>memory_size):
            print('your batch_size was larger than the memory_size and was set to be equal to it')
            batch_size = memory_size
        self.batch_size = batch_size
        # add one dimension for sample number
        self.previous_state = torch.zeros([1,neurons[0]])
        self.previous_action = None
        self.previous_reward = 0
        self.n_states = neurons[0]
        
    def get_action(self,state):
        '''Suggest an action to take given the current state'''
        # transform state into a torch variable, with no gradients
        with torch.no_grad():
            state = torch.autograd.Variable(state)
        # multiply by temperature to make softmax output more extreme
        p = torch.nn.functional.softmax(self.network(state*self.temperature),dim=1)
        # draw randomly based on probabilities
        a = p.multinomial(1)
        # extract index from torch Variable
        return int(a[0,0])
    
    def train(self, states, states_prime, rewards, actions):
        '''Train the network'''
        # get Q values for all states (and all actions)
        q = self.network(states)
        # select only the q values of the actions that were actually played
        q_taken = q.gather(1,actions.unsqueeze(1)).squeeze(1)
        # select the maximum of the q values for the states after the actions
        # detach to avoid the max being used in the gradient later
        q_prime = self.network(states_prime).detach().max(1)[0]
        # the expected Q values
        expected = rewards + self.gamma*q_prime
        # loss with respect to expectation
        loss = torch.nn.functional.smooth_l1_loss(q_taken,expected)
        # do SGD
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def terminal_update(self,reward,state):
        '''Update at the end of a session'''
        # update the last turn
        a = self.update(reward,state)
        # last turn iterates back to itself
        self.update(0,state)
        # set previous to 0 for the next turn ('start fresh')
        self.previous_state = torch.zeros([1,self.n_states])
        self.previous_action = None
        self.previous_reward = 0
        return a
    
    def update(self,reward,state):
        '''Update the model with a reward and a new state'''
        # convert what is needed into Tensors
        new_state = torch.Tensor(state).unsqueeze(0)
        if (self.previous_action is not None):
            a = torch.LongTensor([self.previous_action])
            r = torch.Tensor([self.previous_reward])
            s = self.previous_state
            # add transition memory
            self.memory.append((s,new_state,r,a))
        # select next action 
        action = self.get_action(new_state)
        # train network only if there are enough events to be used
        if len(self.memory)>self.batch_size:
            self.train(*sample_events(self.memory,self.batch_size))  
        # update previous
        self.previous_action = action
        self.previous_state = new_state
        self.previous_reward = reward
        return action
    
    def save(self, file_name):
        torch.save({'network': self.network.state_dict()}, file_name)
    
    def load(self, file_name):
        self.network.load_state_dict(torch.load(file_name)['network'])
        
class Network(torch.nn.Module):
    
    def __init__(self, n_input, n_output, n_hidden):
        '''
        Deep Q-learning network, should be used internally in Qlearn object
            User specifies the number of neurons in the input, output and hidden layers (a list)
        '''
        super().__init__()
        # neurons in input layer
        self.n_input = n_input
        # neurons in output layer (also actions)
        self.n_output = n_output
        # check that n_hidden is a list
        if (type(n_hidden) is int):
            n_hidden = [n_hidden]
        # list of all layers in the network
        self.layers = []
        self.input_layer = torch.nn.Linear(n_input,n_hidden[0])
        self.layers.append(self.input_layer)
        for i in range(len(n_hidden[1:])):
            n_in = n_hidden[i]
            n_out = n_hidden[i+1]
            self.layers.append(torch.nn.Linear(n_in,n_out))
        # final output layer
        self.layers.append(torch.nn.Linear(n_hidden[-1],n_output))
            
    def forward(self,X):
        '''Feed data forward through network'''
        # ReLU for all layers except the last one
        for layer in self.layers[:-1]:
            # all hidden layer
            X = torch.nn.functional.relu(layer(X))
        # output layer, return q values with no activation
        return self.layers[-1](X)

# for the experience replay memory unit
class LimitedList(list):
    
    def __init__(self,max_size):
        '''Modified list object that is limited in size to max_size'''
        super().__init__()
        self.max_size = max_size
    
    def append(self,x):
        super().append(x)
        if len(self)>self.max_size:
            del self[0]

    def extend(self,iterable):
        super().extend(iterable)
        if len(self)>self.max_size:
            excess = len(self)-self.max_size
            del self[:excess]

    def insert(self,i, x):
        super().insert(i,x)
        if len(self)>self.max_size:
            del self[0]
            
def sample_events(events,n):
    '''Randomly sample n events from the list of events'''
    samples = random.sample(events,n)
    # reshape so that each list is one feature rather that one event
    features = zip(*samples)
    # transform to torch Variable
    return map(lambda x: torch.autograd.Variable(torch.cat(x,0)) ,features)

if (__name__ == '__main__'):
    print('This module is not intended to run by iself')
