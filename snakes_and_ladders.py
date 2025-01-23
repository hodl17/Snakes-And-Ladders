# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:31:53 2025

@author: devli
"""
import random
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import numpy as np

class Dice:
    def __init__(self):
        self.roll = self.roll()
        
    def roll(self):
        possible_rolls = range(1,7)
        return random.choice(possible_rolls)
    
class Board:
    def __init__(self):
        self.plain_board = self.plain_board()
        self.board = self.make_board()
        
    def plain_board(self):
        return [
                1,2,3,4,5,6,7,8,9,10,
                11,12,13,14,15,16,17,18,19,20,
                21,22,23,24,25,26,27,28,29,30,
                31,32,33,34,35,36,37,38,39,40,
                41,42,43,44,45,46,47,48,49,50,
                51,52,53,54,55,56,57,58,59,60,
                61,62,63,64,65,66,67,68,69,70,
                71,72,73,74,75,76,77,78,79,80,
                81,82,83,84,85,86,87,88,89,90,
                91,92,93,94,95,96,97,98,99,100
               ]
        
    def ladders(self):
        return {9:27, 18:37, 25:54, 28:51, 56:64, 68:88, 76:97, 79:100}
    
    def snakes(self):
        return {16:7, 59:17, 63:19, 67:30, 87:24, 93:69, 95:75, 99:77}
    
    def make_board(self):
        clean_board = self.plain_board
        new_board = [self.ladders().get(n, n) for n in clean_board]
        new_board = [self.snakes().get(n, n) for n in new_board]
        
        return new_board

class Game:
    def __init__(self, board):
        self.board = board
        self.player_square = 0
        self.move_count = 0
        self.snake_count = 0
        self.ladder_count = 0
    
    def play(self, log=False):
        
        while self.player_square<100:
            self.move_count+=1
            dice = Dice()
            squares_to_move = dice.roll
            new_square = self.player_square + squares_to_move
            
            if log:
                print('Rolled {} move to square {}'.format(squares_to_move, new_square))
            
            if new_square >= 100:
                self.player_square = new_square
                
                if log:
                    print('Reached square {} - Game Over!'.format(new_square))
            
            else:
                actual_new_square = self.board[new_square-1]
                if actual_new_square < new_square:
                    self.snake_count+=1
                    
                    if log:
                        print('Slithery sneaking snake - slide back to {}'.format(actual_new_square))
                
                if actual_new_square > new_square:
                    self.ladder_count+=1
                    
                    if log:
                        print('Lovely leaning ladder - climb up to {}'.format(actual_new_square))
                
                self.player_square = actual_new_square
                
                if self.player_square == 100:
                    if log:
                        print('Reached square {} - Game Over!'.format(self.player_square))


class Sim:
    def __init__(self, ntrials):
        self.ntrials = ntrials
        self.board = Board().board
        self.moves_tracker = []
        self.snakes_tracker = []
        self.ladder_tracker = []
        
    def run(self):
        for i in range(self.ntrials):
            game = Game(self.board)
            game.play()
            
            self.moves_tracker.append(game.move_count)
            self.snakes_tracker.append(game.snake_count)
            self.ladder_tracker.append(game.ladder_count)
    
    def expected_moves(self):
        return statistics.mean(self.moves_tracker)
    
    def median_moves(self):
        return statistics.median(self.moves_tracker)
    
    def plot_move_distribution(self):
        sns.displot(data=self.moves_tracker, kind='kde')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.show()
        
    def plot_move_cdf(self):
        sns.displot(data=self.moves_tracker, kind='ecdf')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.show()
        

class MarkovAnalysis:
    def __init__(self):
        self.board = Board()
        self.transition_matrix = self.make_transition_matrix()

    def make_transition_matrix(self):
        n = 101
        transition_matrix = np.zeros(shape=(n,n))
        for i in range(n):
            row = np.zeros(n)
                
            if i not in self.board.ladders() or i not in self.board.snakes():
                # we are starting from a square that is not a ladder or snake
                for j in range(i+1, i+7):
                    j = self.board.ladders().get(j, j)
                    j = self.board.snakes().get(j, j)
                
                    while j > 100:
                        j-=1
                    row[j]+=1/6
            
                transition_matrix[i] = row
        
        # remove the rows and columns with ladders and snakes as you can never be there
        indices_to_remove = list(self.board.ladders().keys()) + list(self.board.snakes().keys())
        transition_matrix = np.delete(transition_matrix, indices_to_remove, axis=0)
        transition_matrix = np.delete(transition_matrix, indices_to_remove, axis=1)
                
        return transition_matrix
    
    def expected_visis_per_square(self):
        # drop the last row and column relating to square 100 (winning the game)
        sub_matrix = np.identity(self.transition_matrix.shape[0]-1) - self.transition_matrix[:-1, :-1]
        inverted = np.linalg.inv(sub_matrix)
        return inverted[0]
    
    def expected_moves(self):
        return np.sum(self.expected_visis_per_square())
    
    def display_gamestate(self, n=1):
        vector = np.zeros(self.transition_matrix.shape[0])
        vector[0]=1
        
        for i in range(n):
            vector = np.matmul(vector, self.transition_matrix)
        
        # to visualize in heatmap we need to add back the removed indices and make the board 10x10
        indices_to_add_back = sorted(list(self.board.ladders().keys()) + list(self.board.snakes().keys()))
        for ind in indices_to_add_back:
            vector = np.insert(vector, ind, 0)
        vector = vector[1:]
        vector = np.flip(vector)
        vector_2d = np.reshape(vector, (-1, 10))
        
        # need to flip every 2nd row to mimic board view
        for i in [0,3,5,7,9]:
            vector_2d[i] = np.flip(vector_2d[i])

        plt.imshow(vector_2d, cmap='Greys', interpolation='none', origin='upper')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.show()
        
markov = MarkovAnalysis()

sim = Sim(100000)
sim.run()

        