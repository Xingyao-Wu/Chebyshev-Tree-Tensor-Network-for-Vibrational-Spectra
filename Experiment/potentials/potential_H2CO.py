import numpy as np

def get_w0():
    # get frequency
    w0 = np.array([1202.31, 1267.91, 1536.03, 1823.05, 2876.45, 2930.65])
    return w0

def get_k30():
    # get cubic force constant
    datastr = """
    1, 1, 3, 58.8
    1, 1, 4, 39.9
    1, 1, 5, -43.1
    2, 2, 3, -72.6
    2, 2, 4, 20.9
    2, 2, 5, -13.9
    2, 3, 6, 17.4
    2, 4, 6, 28.4
    2, 5, 6, -5.2
    3, 3, 3, 10.4
    3, 3, 4, 70.6
    3, 3, 5, -26.2
    3, 4, 4, -44.1
    3, 4, 5, -20.5
    3, 5, 5, -13.1
    3, 6, 6, -58.6
    4, 4, 4, 98.7
    4, 4, 5, 9.0
    4, 5, 5, -21.5
    4, 6, 6, -61.2
    5, 5, 5, -223.0
    5, 6, 6, -711.0
    """
    k30 = np.zeros((6,6,6), dtype=np.float64)
    lines = datastr.strip().split('\n')
    for line in lines:
        i, j, k, value = map(float, line.split(','))
        k30[int(i)-1, int(j)-1, int(k)-1] = value
    return k30

def get_k40():
    # get quartic force constants k40
    datastr = """
    1, 1, 1, 1, 6.7
    1, 1, 2, 2, -9.7
    1, 1, 2, 6, 5.9
    1, 1, 3, 3, -3.2
    1, 1, 3, 4, -1.6
    1, 1, 3, 5, 1.1
    1, 1, 4, 4, -1.0
    1, 1, 4, 5, -0.9
    1, 1, 5, 5, 0.9
    1, 1, 6, 6, -7.3
    2, 2, 2, 2, 1.7
    2, 2, 2, 6, 0
    2, 2, 3, 3, 1.0
    2, 2, 3, 4, -5.3
    2, 2, 3, 5, 3.0
    2, 2, 4, 4, -1.4
    2, 2, 4, 5, -3.1
    2, 2, 5, 5, -2.4
    2, 2, 6, 6, 0
    2, 3, 3, 6, 1.5
    2, 3, 4, 6, 1.8
    2, 3, 5, 6, -3.6
    2, 4, 4, 6, 3.8
    2, 4, 5, 6, 0.6
    2, 5, 5, 6, 2.8
    2, 6, 6, 6, -1
    3, 3, 3, 3, 1.6
    3, 3, 3, 4, 1.8
    3, 3, 3, 5, 0.1
    3, 3, 4, 4, 4.8
    3, 3, 4, 5, -1.3
    3, 3, 5, 5, -0.6
    3, 3, 6, 6, -6.6
    3, 4, 4, 4, -5.4
    3, 4, 4, 5, -1.2
    3, 4, 5, 5, 0.7
    3, 4, 6, 6, -6.5
    3, 5, 5, 5, 2
    3, 5, 6, 6, 17.0
    4, 4, 4, 4, 7.0
    4, 4, 4, 5, 2.4
    4, 4, 5, 5, 0.9
    4, 4, 6, 6, -3.3
    4, 5, 5, 5, 3
    4, 5, 6, 6, 16
    5, 5, 5, 5, 22
    5, 5, 6, 6, 145
    6, 6, 6, 6, 25.1
    """
    k40 = np.zeros((6,6,6,6), dtype=np.float64)
    lines = datastr.strip().split('\n')
    for line in lines:
        i, j, k, l, value = map(float, line.split(','))
        k40[int(i)-1, int(j)-1, int(k)-1, int(l)-1] = value
    return k40
    
def get_potential_energy_H2CO(alpha=1000):
    """
    Potential of water molecule (H2O), Ref: 
        [1] Chemical Physics 54, 365 (1981)
        [2] Chemical Physics 300 (2004) 41-51
    Input:
        alpha: scaling factor, default=1000
    """
    
    w0 = get_w0()
    print("w0, harmonic constants:", np.count_nonzero(w0))
    w = w0 / alpha
    sqrtw = 1/np.sqrt(w)
    
    k30 = get_k30()
    print("k30, non-zero terms:", np.count_nonzero(k30))
    k3 = np.einsum('ijk,i,j,k->ijk', k30, sqrtw, sqrtw, sqrtw) / alpha

    k40 = get_k40()
    print("k40, non-zero terms:", np.count_nonzero(k40))
    k4 = np.einsum('ijkl,i,j,k,l->ijkl', k40, sqrtw, sqrtw, sqrtw, sqrtw) / alpha

    return w, k3, k4
