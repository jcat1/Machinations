# Imports.
import numpy as np
import numpy.random as npr
import random
import pygame as pg


from SwingyMonkey import SwingyMonkey

INF = 1000
class Learner(object):

    def __init__(self, actions, epsilon, eta, gamma):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.actions = actions
        self.epsilon = epsilon
        self.eta = eta
        self.gamma = gamma

        self.__init_state()
        self.init_Q()

    def __init_state(self):
        # define quantization here:
        # States:
        '''
        { 'score': <current score>,
          'tree': { 'dist': <pixels to next tree trunk>,
                    'top':  <screen height of top of tree trunk gap>,
                    'bot':  <screen height of bottom of tree trunk gap> },
          'monkey': { 'vel': <current monkey y-axis speed in pixels per iteration>,
                      'top': <screen height of top of monkey>,
                      'bot': <screen height of bottom of monkey> }}'''
        #dictify = lambda lst: {k : v for v,k in enumerate(lst)}
        
        # list of coarsened bins by upper bounds, least to greatest
        tree_dist =  [100, 200, 300, 450, INF]
        tree_top = [200, 300, INF]
        monkey_top = [50, 100, 150, 200, 250, 300, 350, INF]
        monkey_vel = [-20, -10, 0, 10, 20, 40, INF]

        self.coarse_bounds = [tree_dist, tree_top, monkey_top, monkey_vel]
        
        # state, action dimensions
        self.q_dims = [len(k) for k in self.coarse_bounds] + [2]
    
    def init_Q(self):
        self.Q = np.zeros(self.q_dims)

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def state_transformation(self, state):
        def categorize_ub(n, upper_bounds):
            for i, ub in enumerate(upper_bounds):
                if n<= ub:
                    return i

        if state == None:
            state_list = np.zeros(len(self.coarse_bounds))
        else:
            state_list = [state['tree']['dist']]
            state_list.append(state['tree']['top'])
            state_list.append(state['monkey']['top'])
            state_list.append(state['monkey']['vel'])

        assert (len(self.coarse_bounds) == len(state_list))
        
        coarsened_state = []
        for i, s in enumerate(state_list):
            coarsened_state.append(categorize_ub(s, self.coarse_bounds[i]))

        return coarsened_state

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        if self.last_action == None:
            self.last_action = random.choice(self.actions)
        if self.last_reward == None:
            self.last_reward = 0

        # First, update the Q-table
        # change lists to tuples in order to index into the q-table
        new_state = tuple(self.state_transformation(state))
        old_state = tuple(self.state_transformation(self.last_state))

        self.Q[old_state + (self.last_action,)] += self.eta*(self.last_reward 
            + self.gamma * max(self.Q[new_state + (0,)], self.Q[new_state + (1,)]) 
            - self.Q[old_state + (self.last_action,)])

        # Next, pick an action
        # epsilon greedy
        if (random.uniform(0,1) < self.epsilon):
            new_action = random.choice(self.actions)
        else:
            new_action = np.argmax(self.Q[new_state])

        self.last_action = new_action
        self.last_state  = state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)
        if ii> 1:
            if swing.score > max(hist[:-1]):
                print("new max:", swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner(actions=[0,1], epsilon=.05, eta=.2, gamma=.85)

    # Empty list to save history.
    hist = []

    # Run games. 
    run_games(agent, hist, 1000, 10)

    # etas= [.5,.2,.1,.05,.2,.2,.2]
    # gammas = [.5]*5 + [.85,.99]
    # epsilons = [.05]*4 + [0.1,.05,.05]

    # max_scores = []
    # for i in range(len(etas)):
    #     agent = Learner(actions=[0,1], epsilon=epsilons[i], eta=etas[i], gamma=gammas[i])
    #     hist = []
    #     run_games(agent, hist, 1000, 10)
    #     max_scores.append(max(hist))
    

    # Save history. 
    np.save('hist-finer-vel',np.array(hist))

