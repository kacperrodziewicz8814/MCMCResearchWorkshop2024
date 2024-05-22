import pandas as pd
import numpy as np
import math
import random
import os
from datetime import datetime

from initial_structure import SARWalk
from points_io import save_points_as_pdb
from utils import distance, random_versor, move_in_lattice

class DataProcessor:
    def __init__(self, path_to_file, chromosome):
        self.df = pd.read_csv(path_to_file, sep='\t')
        print('Preparing data...')
        self.prep_df()
        self.select_chromosome(chromosome)
        self.set_scale()
        self.remove_duplicates()
    
    def prep_df(self):
        self.df.columns = ['chrom1', 'coord1', 'chrom2', 'coord2'] # Naming columns

        self.df = self.df[~self.df.apply(lambda x: 'x' in x.values or 'X' in x.values, axis=1)] # Deleting rows containing 'x' or 'X'
        
        # Conversion to numeric
        for col in ['chrom1', 'coord1', 'chrom2', 'coord2']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def select_chromosome(self, n):
        chromosomes = self.df['chrom1'].unique()
        if not (min(chromosomes) <= n <= max(chromosomes)):
            raise ValueError(f'Chromosome number should be int value from {min(chromosomes)} to {max(chromosomes)}')
        
        self.df = self.df[(self.df['chrom1'] == n) & (self.df['chrom2'] == n)]
    
    def set_scale(self, scale_factor=1e6):
        self.df['bead1'] = (self.df['coord1'] // scale_factor).astype(int)
        self.df['bead2'] = (self.df['coord2'] // scale_factor).astype(int)

    def remove_duplicates(self):
        self.df = self.df.drop_duplicates(subset=['bead1', 'bead2'], keep=False)
        self.df['sorted_bead_pair'] = self.df.apply(lambda x: tuple(sorted((x['bead1'], x['bead2']))), axis=1)
        self.df = self.df.drop_duplicates(subset=['sorted_bead_pair'], keep=False)
        self.df.drop('sorted_bead_pair', axis=1, inplace=True)
        
        self.df =  self.df[self.df['bead1'] != self.df['bead2']]
    
    def get_number_of_beads(self):
        return max(max(self.df['bead1']), max(self.df['bead2']))

class Model:
    def __init__(self, path_to_file, chromosome, d, p, f, theta1, a, b, d2, mu1, mu2, move_range):
        self.chromosome = chromosome
        data_processor = DataProcessor(path_to_file, self.chromosome)
        self.data = data_processor.df
        
        self.number_of_beads = data_processor.get_number_of_beads()
        self.structure = SARWalk(self.number_of_beads, 1)

        print('Initilizing model...')
        self.initialize_parameters(d, p, f, theta1, a, b, d2, mu1, mu2)

        self.contact_matrix = self.create_contact_matrix()
        self.theta_matrix = self.create_theta_matrix()

        self.move_range = move_range

    def initialize_parameters(self, d, p, f, theta1, a, b, d2, mu1, mu2):
        self.parameters = {
            'd': d,
            'p': p,
            'f': f,
            'theta1': theta1,
            'a': a,
            'b': b,
            'd2': d2,
            'mu1': mu1,
            'mu2': mu2
        }

    def create_contact_matrix(self):
        contact_matrix = np.zeros((self.number_of_beads, self.number_of_beads), dtype=int)
        for _, row in self.data.iterrows():
            bead1 = int(row['bead1']) - 1  # Substract 1 to index from 0
            bead2 = int(row['bead2']) - 1
            contact_matrix[bead1, bead2] = 1
            contact_matrix[bead2, bead1] = 1  # Ensure symmetry
        
        return contact_matrix

    def calculate_theta_ij(self, i, j):
        theta_ij = 0
        
        # Iterating d2 distance around i, j
        for xp in range(i - self.parameters['d2'], i + self.parameters['d2'] + 1):
            for yp in range(j - self.parameters['d2'], j + self.parameters['d2'] + 1):
                if 0 <= xp < self.contact_matrix.shape[0] and 0 <= yp < self.contact_matrix.shape[1]:
                    if self.contact_matrix[xp, yp] == 1:
                        # Calculating gauss_value
                        gauss_value = math.exp(-((xp - i)**2 + (yp - j)**2) / self.parameters['mu2'])
                        theta_ij += gauss_value

        # Outcome should be <= 1
        theta_ij = min(theta_ij, 1)
        
        return theta_ij

    def create_theta_matrix(self):
        theta_matrix = np.zeros((self.number_of_beads, self.number_of_beads), dtype=float)

        for i in range(self.number_of_beads):
            for j in range(i):
                theta_matrix[i][j] = self.calculate_theta_ij(i, j)
                theta_matrix[j][i] = theta_matrix[i][j]
        
        return theta_matrix

    def calculate_loss(self):
        d1 = self.parameters['d'] / pow(self.parameters['theta1'], 1/3)
        L = 0.0

        for i in range(self.number_of_beads):
            for j in range(i):
                dist = distance(self.structure.coordinates[i], self.structure.coordinates[j])  # Calculating distance d(i, j)
                if self.contact_matrix[i, j] >= 1 or self.theta_matrix[i, j] == 1:
                    L += pow((dist - self.parameters['d']), 2) / pow(self.parameters['d'], 2)
                elif self.theta_matrix[i, j] > self.parameters['theta1']:
                    delta = self.parameters['d'] / pow(min(1, self.theta_matrix[i][j]), 1/3)
                    L += self.parameters['a'] * (1 - math.exp(-(pow(dist - delta, 2) / self.parameters['mu1'])))
                else:
                    L += self.parameters['b'] * (1 - 1 / (1 + math.exp(-(dist - (d1 - self.parameters['p'])) / self.parameters['f'])))
        return L

    def calculate_loss_for_bead(self, index):
        d1 = self.parameters['d'] / pow(self.parameters['theta1'], 1/3)
        L = 0.0

        for j in range(self.number_of_beads):
            if j != index:
                dist = distance(self.structure.coordinates[index], self.structure.coordinates[j])  # Calculating distance d(i, j)
                if self.contact_matrix[index, j] >= 1 or self.theta_matrix[index, j] == 1:
                    L += pow(dist - self.parameters['d'], 2) / pow(self.parameters['d'], 2)
                elif self.theta_matrix[index, j] > self.parameters['theta1']:
                    delta = self.parameters['d'] / pow(min(1, self.theta_matrix[index][j]), 1/3)
                    L += self.parameters['a'] * (1 - math.exp(-(pow(dist - delta, 2) / self.parameters['mu1'])))
                else:
                    L += self.parameters['b'] * (1 - 1 / (1 + math.exp(-(dist - (d1 - self.parameters['p'])) / self.parameters['f'])))

        return L

    def is_position_unique(self, position, current_index):
        # Check all cooridiantes except 'current_index'
        for index, pos in enumerate(self.structure.coordinates):
            if index != current_index:
                # Check if new position is free
                if np.allclose(pos, position):
                    return False
        return True

    def propose_move(self, curr_temp, trails=10):
        index = random.randint(0, self.number_of_beads - 1) # Randomly select a bead
        initial_loss = self.calculate_loss_for_bead(index)

        coordinate = self.structure.coordinates[index].copy()
        
        best_move = coordinate
        best_loss = initial_loss

        for _ in range(trails):
            random_move = self.move_range * random_versor()
            new_position = coordinate + random_move
            self.structure.coordinates[index] = new_position
            new_loss = self.calculate_loss_for_bead(index)
            
            if new_loss < best_loss:
                best_move = new_position
                best_loss = new_loss

            self.structure.coordinates[index] = coordinate

        acceptance_proba = min(math.exp(-(initial_loss - new_loss) / curr_temp), 1)
        accepted = False

        # Update the position if the chosen position is unique
        if self.is_position_unique(best_move, index) and random.random() < acceptance_proba:
            accepted = True
            self.structure.coordinates[index] = best_move

        return accepted
    
    def propose_move_lattice(self, curr_temp):
        # Another approch to proposing a move
        index = random.randint(0, self.number_of_beads - 1) # Randomly select a bead
        initial_loss = self.calculate_loss_for_bead(index)

        initial_position = self.structure.coordinates[index].copy()
        new_position = initial_position + move_in_lattice()
        self.structure.coordinates[index] = new_position

        new_loss = self.calculate_loss_for_bead(index)
        loss_change = new_loss - initial_loss

        if loss_change <= 0:
            return True
        
        acceptance_proba = math.exp(-loss_change/curr_temp)
        accepted = True

        if random.random() > acceptance_proba or not self.is_position_unique(new_position, index): # move not accepted
            self.structure.coordinates[index] = initial_position
            accepted = False
        
        return accepted
    
    def simulated_annealing(self, initial_temp, cooling_rate, i_max, iter_per_temp):
            print('Starting simulated annealing.')
            curr_temp = initial_temp
            consecutive_failures = 0

            for i in range(i_max):
                consecutive_failures += 1
                curr_temp = (cooling_rate ** i) * curr_temp
                print(f'Current temp: {curr_temp}')

                for j in range(iter_per_temp * self.number_of_beads):
                    accepted_counter = 0
                    
                    accepted = self.propose_move(curr_temp)
                    if accepted:
                        accepted_counter += 1
                    
                    if accepted_counter > 8 * self.number_of_beads:
                        consecutive_failures = 0
                        print('Acceptance threshold achieved. Reseting consecutive failures counter.')
                        break

                    if j % 500 == 0:
                        print(f'Current loss: {self.calculate_loss()}')

                if consecutive_failures == 3:
                    print('Three consecutive failures. Annealing process stopped.')
                    break
    
    def save_output(self):
        print('Saving structure.')
        now = datetime.now()
        dt_string = now.strftime("%d-%m_%H:%M")
        directory = f'structures/structure_{dt_string}'
        os.mkdir(directory)
        save_points_as_pdb(self.structure.coordinates, f'{directory}/structure.pdb')

        with open(f'{directory}/structure_details.txt', 'w') as f:
            f.write(f'Generated structure for chromosome: {self.chromosome}\n')
            f.write(f'Move range: {self.move_range}')
            f.write('\nModel parameters:\n')
            
            for key, value in self.parameters.items():
                f.write(f'{key}: {value}\n')


if __name__ == '__main__':
    path_to_file = 'Data/GSM1173493_cell-1.txt'
    chromosome = 13
    d = 8
    p = 1
    f = 0.1
    theta1 = 0.7
    a = 1
    b = 1
    d2 = 120
    mu1 = 20
    mu2 = 2
    move_range = 4

    model = Model(path_to_file, chromosome, d, p, f, theta1, a, b, d2, mu1, mu2, move_range)
    model.simulated_annealing(10, 0.9, 100, 100)
    model.save_output()