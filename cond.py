import numpy as np

if __name__ == "__main__":
    # Load matrices from files
    A1 = np.loadtxt('./input/A1.txt')
    A2 = np.loadtxt('./input/A2.txt')
    A3 = np.load('./input/A3.npy')

    # Compute the 2-norm condition number for each matrix
    cond_A1 = np.linalg.cond(A1, p=2)
    cond_A2 = np.linalg.cond(A2, p=2)
    cond_A3 = np.linalg.cond(A3, p=2)
    
    # Print out the 2-norm condition numbers
    print("2-norm cond of A1: ", cond_A1)
    print("2-norm cond of A2: ", cond_A2)
    print("2-norm cond of A3: ", cond_A3)
