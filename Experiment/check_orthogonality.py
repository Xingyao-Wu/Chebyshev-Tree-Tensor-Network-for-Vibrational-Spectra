from pytreenet.ttns import TreeTensorNetworkState

import os
import numpy as np
def check_orthogonality(path: str, num_states: int):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    old_eigenstates_path = [os.path.join(current_dir,'..', f'{path}/lobpcg_state_{i}') for i in range(num_states)]
    old_eigenstates = [TreeTensorNetworkState.load(path) for path in old_eigenstates_path]
    ortho_check = np.zeros((num_states,num_states), dtype=np.float64)

    for i in range(num_states):
        for j in range(i):
            ovl = old_eigenstates[i].scalar_product(old_eigenstates[j])
            ortho_check[i,j] = ovl.real
            ortho_check[j,i] = ovl.imag
            if np.abs(ovl.real) >0.05:
                print(f"Warning: Overlap between state {i} and state {j} is {ovl.real}, which is greater than 0.05")
    print("Max overlap: ", np.max(np.abs(ortho_check)), np.argmax(np.abs(ortho_check))//num_states, np.argmax(np.abs(ortho_check))%num_states)        
    return ortho_check

def main():
    print(os.getcwd())
    # path = "./Results_final/test_harmonic/threetree/max_bond_dim_5/states"
    path = "./Results_test_3/ch3cn/mps/max_bond_dim_8/states"
    num_states = 84
    ortho_check = check_orthogonality(path, num_states)
    import matplotlib.pyplot as plt
    plt.imshow((ortho_check), cmap='viridis')
    plt.colorbar()
    plt.savefig(f'{path}/../check_ortho_logpcg.png')
    # plt.savefig(f'{path}/../check_ortho_ii.png')
    plt.show()
    
if __name__ == "__main__":
    main()
