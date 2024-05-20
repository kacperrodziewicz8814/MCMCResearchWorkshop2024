import numpy as np
import matplotlib.pyplot as plt

def load_scHiC_data(file_path):
    """
    Load scHi-C data from a text file.
    The file format is assumed to be: chrom1 coord1 chrom2 coord2
    """
    data = np.loadtxt(file_path, dtype=str, skiprows=1)
    chrom1 = data[:, 0]
    coord1 = data[:, 1].astype(int)
    chrom2 = data[:, 2]
    coord2 = data[:, 3].astype(int)
    return chrom1, coord1, chrom2, coord2

def create_contact_matrix(chrom1, coord1, chrom2, coord2, resolution, chrom_num):
    """
    Create a contact matrix from scHi-C data.
    """
    max_coord = max(np.max(coord1), np.max(coord2))
    num_bins = int(max_coord / resolution) + 1
    contact_matrix = np.zeros((num_bins, num_bins))
    for i in range(len(chrom1)):
        if(chrom1[i]==chrom_num and chrom2[i]==chrom_num):
            bin1 = int(coord1[i] / resolution)
            bin2 = int(coord2[i] / resolution)
            contact_matrix[bin1, bin2] += 1
    return contact_matrix


def plot_heatmap(contact_matrix):
    """
    Plot a heatmap of the contact matrix.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(contact_matrix, cmap='Reds', interpolation='nearest')
    plt.colorbar()
    plt.title('scHi-C Contact Matrix')
    plt.xlabel('Bin Index')
    plt.ylabel('Bin Index')

    plt.show()


if __name__ == "__main__":
    # File path to the scHi-C data
    file_path = '../GSE48262_Th1_bgl_pool.txt'

    # Parameters


   # resolution = 20000  # Adjust resolution as needed

    # Load scHi-C data
    chrom1, coord1, chrom2, coord2 = load_scHiC_data(file_path)
    # Create contact matrix

    contact_matrix = create_contact_matrix(chrom1, coord1, chrom2, coord2, 2000000,"2")

    # Plot heatmap
    plot_heatmap(contact_matrix)
