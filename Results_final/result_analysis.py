import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pstats

SCRIPT_DIR = Path(__file__).resolve().parent

def collect_energy_tables(base_dir=None, output_file='collected_energies.csv'):
    """
    Collect energy tables from mps and threetree subfolders for bond dimensions 8 and 12.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing mps and threetree folders (default: current directory)
    output_file : str
        Output CSV filename for collected energies
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing collected energies with columns:
        ['Energy_Level', 'MPS_Bond8', 'MPS_Bond12', 'ThreeTree_Bond8', 'ThreeTree_Bond12']
    """
    base_path = Path(base_dir) if base_dir is not None else SCRIPT_DIR
    
    # Paths to the energy files
    files_lobpcg = {
        # 'MPS_LOBPCG_8': base_path / 'mps' / 'max_bond_dim_8' / 'lobpcg_energies.csv',
        # 'ThreeTree_LOBPCG_8': base_path / 'threetree' / 'max_bond_dim_8' / 'lobpcg_energies.csv',
        # 'LeafOnly_LOBPCG_8': base_path / 'leafonly' / 'max_bond_dim_8' / 'lobpcg_energies.csv',
        'MPS_LOBPCG_12': base_path / 'mps' / 'max_bond_dim_12' / 'lobpcg_energies.csv',     
        'ThreeTree_LOBPCG_12': base_path / 'threetree' / 'max_bond_dim_12' / 'lobpcg_energies.csv',       
        'LeafOnly_LOBPCG_12': base_path / 'leafonly' / 'max_bond_dim_12' / 'lobpcg_energies.csv',
        'Fork3_LOBPCG_12': base_path / 'fork3' / 'max_bond_dim_12' / 'lobpcg_energies.csv',
        'Fork4_LOBPCG_12': base_path / 'fork4' / 'max_bond_dim_12' / 'lobpcg_energies.csv',
        'Fork6_LOBPCG_12': base_path / 'fork6' / 'max_bond_dim_12' / 'lobpcg_energies.csv',
        'T3NS_LOBPCG_12': base_path / 't3ns' / 'max_bond_dim_12' / 'lobpcg_energies.csv',
    }
    
    files_ii = {
        'MPS_II_20': base_path / 'mps' / 'max_bond_dim_12' / 'inverse_iteration_energies_20.csv',
        'ThreeTree_II_20': base_path / 'threetree' / 'max_bond_dim_12' / 'inverse_iteration_energies_20.csv',
        'LeafOnly_II_20': base_path / 'leafonly' / 'max_bond_dim_12' / 'inverse_iteration_energies_20.csv',
        'Fork3_II_20': base_path / 'fork3' / 'max_bond_dim_12' / 'inverse_iteration_energies_20.csv',
        'Fork4_II_20': base_path / 'fork4' / 'max_bond_dim_12' / 'inverse_iteration_energies_20.csv',
        'Fork6_II_20': base_path / 'fork6' / 'max_bond_dim_12' / 'inverse_iteration_energies_20.csv',
        'T3NS_II_20': base_path / 't3ns' / 'max_bond_dim_12' / 'inverse_iteration_energies_20.csv',
    }
    # Initialize raw results dictionary (variable lengths)
    results_raw = {**{key: [] for key in files_lobpcg.keys()}, **{key: [] for key in files_ii.keys()}}
    # Read each energy file and extract converged energies (last column)
    for key, filepath in files_lobpcg.items():
        if filepath.exists():
            # Read CSV without headers and extract last column (converged energies)
            data = pd.read_csv(filepath, header=None)
            converged_energies = data.iloc[:, -1].values  # Last column
            results_raw[key] = np.sort(converged_energies)
            # print(f"Loaded {len(converged_energies)} energy levels from {key}")
        else:
            print(f"Warning: File not found: {filepath}")
            results_raw[key] = []
    for key, filepath in files_ii.items():
        if filepath.exists():
            # Read CSV without headers and extract last column (converged energies)
            data = np.loadtxt(filepath, delimiter=",",dtype=complex)
            converged_energies = data[:, -1].real # Last column
            results_raw[key] = np.sort(converged_energies)
            # print(f"Loaded {len(converged_energies)} energy levels from {key}")
        else:
            print(f"Warning: File not found: {filepath}")
            results_raw[key] = []
    # Create energy level indices and pad columns to equal length
    lengths = [len(v) for v in results_raw.values()]
    if not any(lengths):
        raise FileNotFoundError("No energy files found. Check the mps/threetree directories.")
    max_levels = max(lengths)
    results_padded = {'Energy_Level': list(range(max_levels))}
    for key, values in results_raw.items():
        arr = np.asarray(values, dtype=float)
        arr = arr*1e3
        arr[1:] = arr[1:] - arr[0]
        padded = np.full(max_levels, np.nan)
        if len(arr) > 0:
            padded[: len(arr)] = arr
        results_padded[key] = padded

    ref_path = base_path / '..' / '..' / 'Experiment' / 'ch3cn_ref.csv'
    results_padded['Ref'] = pd.read_csv(ref_path.resolve())['Energy_Ref'].dropna().values
    
    # Create DataFrame
    df = pd.DataFrame(results_padded)
    
    # Save to CSV
    output_path = base_path / output_file
    df.to_csv(output_path, index=False)
    print(f"Collected energies saved to: {output_path}")
    
    return df

def analyze_mean_errors(base_dir=None, ref_file=None):
    """
    Analyze mean errors for different tree structures and bond dimensions.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing the energy data
    ref_file : str
        Path to reference energies CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with analysis results
    """
    base_path = Path(base_dir) if base_dir is not None else SCRIPT_DIR
    
    # Collect energies
    collected_df = collect_energy_tables(base_path)
    
    # Calculate statistics for each method
    methods = ['MPS_LOBPCG_8', 'ThreeTree_LOBPCG_8', 'LeafOnly_LOBPCG_8', 
               'MPS_LOBPCG_12', 'ThreeTree_LOBPCG_12', 'LeafOnly_LOBPCG_12', 
               'Fork3_LOBPCG_12', 'Fork4_LOBPCG_12', 'Fork6_LOBPCG_12', 'T3NS_LOBPCG_12',
               'MPS_II_20', 'ThreeTree_II_20', 'LeafOnly_II_20',
               'Fork3_II_20', 'Fork4_II_20', 'Fork6_II_20', 'T3NS_II_20']
    analysis_results = []
    
    for method in methods:
        if method in collected_df.columns and len(collected_df[method]) > 0:
            calc_energies = collected_df[method].values
            min_len = min(len(calc_energies), len(collected_df['Ref']))
            treetype = method.split('_')[0].lower()
            opm  = method.split('_')[1].lower()
            max_bond_dim = int(method.split('_')[2])
            if opm == 'lobpcg':
                profile_file = base_path / f"{treetype}/max_bond_dim_{12}/profile_output.prof"
            elif opm == 'ii':
                profile_file = base_path / f"{treetype}/max_bond_dim_{12}/profile_output_ii_20.prof"
            else:
                runtime = 0
                raise Warning("Runtime cannot be extracted!")

            stat = pstats.Stats(str(profile_file))
            runtime =  stat.total_tt
            if min_len > 0:
                errors = calc_energies[:min_len] - collected_df['Ref'][:min_len]
                
                result = {
                    'Method': opm,
                    'Tree_Type': method.split('_')[0],
                    'Bond_Dimension': int(method.split('_')[2]),
                    'Num_States': min_len,
                    'Mean_Error': np.mean(errors),
                    'Median_Error': np.median(errors),
                    'Max_Error': np.max(errors),
                    'Min_Error': np.min(errors),
                    'Std_Error': np.std(errors),
                    'Mean_Relative_Error': np.mean(errors / collected_df['Ref'][:min_len]) * 100,  # in %
                    'Runtime': runtime
                }
                analysis_results.append(result)
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame(analysis_results)
    runtime_df = get_runtime_analysis(base_dir)
    # Save analysis results
    analysis_file = base_path / 'error_analysis.csv'
    print("Saving analysis results to:", analysis_file)
    analysis_df.to_csv(analysis_file, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("ERROR ANALYSIS SUMMARY")
    print("="*70)
    
    for _, row in analysis_df.iterrows():
        print(f"\n{row['Method']}:")
        print(f"  Tree Type: {row['Tree_Type']}")
        print(f"  Bond Dimension: {row['Bond_Dimension']}")
        print(f"  Number of States: {row['Num_States']}")
        print(f"  Mean Error: {row['Mean_Error']:.6e}")
        print(f"  Median Error: {row['Median_Error']:.6e}")
        print(f"  Mean Relative Error: {row['Mean_Relative_Error']:.4f}%")
        print(f"  Std Error: {row['Std_Error']:.6e}")
    
    # Compare tree types
    print(f"\n{'='*70}")
    print("COMPARISON BY TREE TYPE AND BOND DIMENSION")
    print("="*70)
    
    print(f"\nAnalysis results saved to: {analysis_file}")
    return analysis_df
    
def plot_all_energies(base_dir=None):
    base_path = Path(base_dir) if base_dir is not None else SCRIPT_DIR
    data_all = collect_energy_tables(base_path)
    plt.figure(figsize=(12, 8))
    colors = plt.get_cmap("tab20").colors
    plt.plot(data_all["Energy_Level"], abs(data_all["MPS_LOBPCG_12"]-data_all["Ref"]), 'o', color=colors[1], markersize=6, linewidth=1)
    plt.plot(data_all["Energy_Level"], abs(data_all["ThreeTree_LOBPCG_12"]-data_all["Ref"]), 's', color=colors[3], markersize=6, linewidth=1)
    plt.plot(data_all["Energy_Level"], abs(data_all["LeafOnly_LOBPCG_12"]-data_all["Ref"]), 'd', color=colors[5], markersize=6, linewidth=1)
    plt.plot(data_all["Energy_Level"], abs(data_all["Fork3_LOBPCG_12"]-data_all["Ref"]), '^', color=colors[7], markersize=6, linewidth=1)
    plt.plot(data_all["Energy_Level"], abs(data_all["Fork4_LOBPCG_12"]-data_all["Ref"]), 'v', color=colors[9], markersize=6, linewidth=1)
    plt.plot(data_all["Energy_Level"], abs(data_all["Fork6_LOBPCG_12"]-data_all["Ref"]), '*', color=colors[11], markersize=6, linewidth=1)
    plt.plot(data_all["Energy_Level"], abs(data_all["T3NS_LOBPCG_12"]-data_all["Ref"]), 'x', color=colors[13], markersize=6, linewidth=1)
    
    plt.plot(data_all["Energy_Level"], abs(data_all["MPS_II_20"]-data_all["Ref"]), 'o', color=colors[0], label='MPS', markersize=6, linewidth=1)
    plt.plot(data_all["Energy_Level"], abs(data_all["ThreeTree_II_20"]-data_all["Ref"]), 's', color=colors[2], label='T Tree', markersize=6, linewidth=1)
    plt.plot(data_all["Energy_Level"], abs(data_all["LeafOnly_II_20"]-data_all["Ref"]), 'd', color=colors[4], label='Leaf Only', markersize=6, linewidth=1)
    plt.plot(data_all["Energy_Level"], abs(data_all["Fork3_II_20"]-data_all["Ref"]), '^', color=colors[6], label='Fork-3', markersize=6, linewidth=1)
    plt.plot(data_all["Energy_Level"], abs(data_all["Fork4_II_20"]-data_all["Ref"]), 'v', color=colors[8], label='Fork-4', markersize=6, linewidth=1)
    plt.plot(data_all["Energy_Level"], abs(data_all["Fork6_II_20"]-data_all["Ref"]), '*', color=colors[10], label='Fork-6', markersize=6, linewidth=1)
    plt.plot(data_all["Energy_Level"], abs(data_all["T3NS_II_20"]-data_all["Ref"]), 'x', color=colors[12], label='Full T3NS', markersize=6, linewidth=1)
    plt.plot(data_all["Energy_Level"], np.ones(len(data_all["Energy_Level"])), '--', color='black', label='Tolerance', markersize=4, linewidth=1)
    plt.xlabel('Energy Level', fontsize=18)
    plt.ylabel('Energy Error (cm-1)', fontsize=18)
    plt.yscale('log')
    # plt.title('All Energies', fontsize=14)
    plt.legend(fontsize=18, loc='lower left',ncol=4)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(base_path / 'all_energies.pdf', bbox_inches='tight')
    plt.show()
    
def table2latex(base_dir=None):
    base_path = Path(base_dir) if base_dir is not None else SCRIPT_DIR
    data_all = pd.read_csv(base_path / 'collected_energies.csv', header=0);print(data_all.columns)
    colunm_methods = ["MPS_LOBPCG_12","ThreeTree_LOBPCG_12","LeafOnly_LOBPCG_12",
                      "Fork3_LOBPCG_12","Fork4_LOBPCG_12","Fork6_LOBPCG_12","T3NS_LOBPCG_12",
                      "MPS_II_20","ThreeTree_II_20","LeafOnly_II_20",
                      "Fork3_II_20","Fork4_II_20","Fork6_II_20","T3NS_II_20"]
    columns = ["Energy_Level","Ref"]+colunm_methods
    data_all = data_all[columns]
    # print(data_all)

    for col in colunm_methods:
        data_all[col] = data_all[col]-data_all["Ref"]
        print(f"{col}: mean error: {data_all[col].mean():.3f}")
        print(f"{col}: std error: {data_all[col].std():.3f}")
    # latex = data_all.to_latex(index=False,float_format="%.3f")   # <-- now it returns a string
    cols = data_all.columns[:-7]
    data_latex = pd.concat([data_all.iloc[:, :2], data_all.iloc[:, -7:]], axis=1).round(3).to_latex(index=False, float_format="%.3f")
    # data_latex =  data_all.iloc[:,-7:].round(3).to_latex(index=False, float_format="%.3f")
    print("LaTeX table is :", data_latex)

    # find column with minimum absolute value (row-wise)
    winners = data_all[cols].round(3).abs().idxmin(axis=1)

    # count how many times each column wins
    counts = winners.value_counts()
    print(winners)
    print(counts)

    row_min = data_all[cols].round(3).abs().min(axis=1)

    # 2. Create a boolean DataFrame where True = column has the row's min value
    is_min = data_all[cols].round(3).abs().eq(row_min, axis=0)

    # 3. Count how many rows each column is the minimum (ties counted for each)
    min_counts = is_min.sum()

    print(min_counts)

def get_runtime(treetype: str, method: str, max_bond_dim: int, base_dir=None):
    base_path = Path(base_dir) if base_dir is not None else SCRIPT_DIR
    if method == 'lobpcg':
        profile_file = base_path / f"{treetype}/max_bond_dim_{max_bond_dim}/profile_output.prof"
    elif method == 'ii':
        profile_file = base_path / f"{treetype}/max_bond_dim_{max_bond_dim}/profile_output_ii_20.prof"
    else:
        raise ValueError("Method must be 'lobpcg' or 'ii'")
    stat = pstats.Stats(str(profile_file))
    runtime =  stat.total_tt
    # print(f"Total runtime for {treetype} with max bond dim {max_bond_dim} using {method}: {runtime:.2f} seconds")
    return runtime

def get_runtime_analysis(base_dir=None):
    base_path = Path(base_dir) if base_dir is not None else SCRIPT_DIR
    treetypes = ['mps', 'threetree', 'leafonly', 'fork3', 'fork4', 'fork6', 't3ns']
    methods = ['lobpcg', 'ii']
    max_bond_dim = 12
    runtime_results = []
    
    for treetype in treetypes:
        for method in methods:
            runtime = get_runtime(treetype, method, max_bond_dim, base_path)
            runtime_results.append({
                'Tree_Type': treetype,
                'Method': method,
                'Max_Bond_Dim': max_bond_dim,
                'Runtime_sec': runtime
            })
    
    runtime_df = pd.DataFrame(runtime_results)
    print("\nRuntime Analysis Results:")
    print(runtime_df)
    return runtime_df

if __name__ == "__main__":

    print("\nAnalyzing mean errors...")
    analysis = analyze_mean_errors()
    # print(analysis)
    # table = analysis.set_index(["Method", "Tree_Type"]).sort_index()
    # table = table.reindex(['lobpcg', 'ii'], level='Method')
    # table = table.reindex(['MPS', 'ThreeTree', 'LeafOnly', 'Fork3', 'Fork4', 'Fork6', 'T3NS'], level='Tree_Type')
    # table_2 = table[['Bond_Dimension', 'Mean_Error', 'Runtime']]
    # print(table_2.round(3).to_latex(float_format="%.3f"))

    
    # print("Collecting all energies...")
    # collect_all_energies()
    
    # print("Plotting all energies...")
    # plot_all_energies()
    
    # print("Generating LaTeX table...")
    table2latex()

    # get_runtime_analysis()

