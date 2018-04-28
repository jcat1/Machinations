# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import random

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self, actions, epsilon, eta, gamma):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        
        self.actions = actions
        self.epsilon = epsilon
        self.eta = eta
        self.gamma = gamma
        self.Q = np.zeros((4,len(actions)))

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    # def learnQ(self, state, action, reward, ):
        # self.Q.get((state, action), 0.0)

    def state_transformation(self, state):
        if state == None:
            state_list = np.zeros(4)
        else:
            state_list = [state['monkey']['vel']]
            state_list.append(state['monkey']['top'])
            state_list.append(state['tree']['dist'])
            state_list.append(state['tree']['top'])
        # states_list.append()
        return state_list


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
        # if self.last_state == None:
        #     self.last_state = { 'score': 0,
        #          'tree': { 'dist': 215-swing.monkey_right,
        #                    'top': self.screen_height-next_tree['y'],
        #                    'bot': self.screen_height-next_tree['y']-self.tree_gap},
        #          'monkey': { 'vel': self.vel,
        #                      'top': self.screen_height - self.monkey_loc + self.monkey_img.get_height()/2,
        #                      'bot': self.screen_height - self.monkey_loc - self.monkey_img.get_height()/2}} 
        if self.last_reward == None:
            self.last_reward = 0

        last_state = self.state_transformation(self.last_state)

        last_value = np.multiply(self.Q[:,self.last_action], last_state)

        q0 = np.multiply(self.Q[:,0], self.state_transformation(state))
        q1 = np.multiply(self.Q[:,1], self.state_transformation(state))
        new_values = [q0,q1]
        
        new_action = np.argmax(new_values)
        new_state  = state


        test = self.Q[:,self.last_action].shape
        test1 = (self.eta*(self.last_reward + (self.gamma*new_values[new_action])-last_value)).shape
        test2 = np.multiply(self.eta*(self.last_reward + (self.gamma*new_values[new_action])-last_value), last_state).shape
        print(test,test1,test2)
        self.Q[:,self.last_action] = self.Q[:,self.last_action] + np.multiply(self.eta*(self.last_reward + (self.gamma*new_values[new_action])-last_value), last_state)
        # self.Q[:,self.last_action] = self.Q[:,self.last_action] - self.eta*(self.Q[:,self.last_action] - (self.last_reward+self.gamma*self.Q[:,new_action]))

        self.last_action = new_action
        self.last_state  = new_state

        # print(sself.Q)
        return new_action

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

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner(actions=[0,1], epsilon=.05, eta=.05, gamma=.05)

    # Empty list to save history.
    hist = []

    # Run games. 
    run_games(agent, hist, 20, 10)

    # Save history. 
    np.save('hist',np.array(hist))


