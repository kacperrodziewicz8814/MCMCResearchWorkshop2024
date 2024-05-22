import numpy as np
import matplotlib.pyplot as plt

class SARWalk:
    """
    Class for generating 3D self avoiding random walk
    which will serve as an initial structure for the model.
    """
    def __init__(self, number_of_steps, step_size):
        self.number_of_steps = number_of_steps
        self.step_size = step_size
        print('Initializing structure...')
        self.walk = self.generate_walk()
        self.coordinates = self.get_walk_coordinates()
        print('Structure initialized.')

    def generate_walk(self):
        position = np.array([0, 0, 0])  # Starting position
        walk = [position]

        potential_moves = [
            np.array(move) for move in [
                [self.step_size, 0, 0], [-self.step_size, 0, 0],
                [0, self.step_size, 0], [0, -self.step_size, 0],
                [0, 0, self.step_size], [0, 0, -self.step_size]
            ]
        ]
        steps_taken = 1

        while steps_taken < self.number_of_steps:
            np.random.shuffle(potential_moves)  # Shuffle potential moves
            move_made = False
            for move in potential_moves:
                new_position = position + move
                if not any(np.array_equal(new_position, pos) for pos in walk):  # Check if position was visited
                    position = new_position
                    walk.append(new_position)
                    steps_taken += 1
                    move_made = True
                    break

            if not move_made:
                # If no valid move is found, backtrack to previous position
                walk.pop()
                position = walk[-1]
                steps_taken -= 1

        return np.array(walk)

    def get_walk_coordinates(self):
        x_saw, y_saw, z_saw = zip(*self.walk)

        return np.column_stack((x_saw, y_saw, z_saw)).astype(float)
    
    def plot_walk(self):
        x_saw, y_saw, z_saw = zip(*self.walk)

        fig_saw = plt.figure(figsize=(10, 10))
        ax_saw = fig_saw.add_subplot(111, projection='3d')
        ax_saw.plot(x_saw, y_saw, z_saw, c='blue')

        ax_saw.set_xlabel('X Axis')
        ax_saw.set_ylabel('Y Axis')
        ax_saw.set_zlabel('Z Axis')

        ax_saw.set_xlim(min(x_saw) - self.step_size, max(x_saw) + self.step_size)
        ax_saw.set_ylim(min(y_saw) - self.step_size, max(y_saw) + self.step_size)
        ax_saw.set_zlim(min(z_saw) - self.step_size, max(z_saw) + self.step_size)

        plt.show()

    def get_number_of_beads(self):
        return len(self.coordinates)

if __name__ == '__main__':
    sarwalk = SARWalk(120, 5)
    sarwalk.plot_walk()