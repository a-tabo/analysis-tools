import ase
from ase import Atoms
from ase.io import read
import numpy as np
from collections import defaultdict, Counter
import statistics

#constants
cutoff = 5.0
bond_threshold_h = 1.2 # C-H bond threshold
bond_threshold_o = 1.6 # O bond threshold (with C or S)
polymer_num_atoms = 1810 # number of atoms in polymer

def classify_carbons(atoms, bond_threshold):
	carbon_classification = {}
	for i, atom in enumerate(atoms):
		if atom.symbol == 'C':
			hydrogen_count = sum(
				1 for j, other_atom in enumerate(atoms)
				if other_atom.symbol == 'H' and atoms.get_distance(i, j, mic=True) < bond_threshold
			)
			carbon_classification[i] = 'Cal' if hydrogen_count >= 2 else 'Car'
	return carbon_classification

def classify_oxygens(atoms, bond_threshold):
	oxygen_classification = {}
	for i, atom in enumerate(atoms):
		if atom.symbol == 'O':
			bonded_to_s = False
			bonded_to_c = 0
			total_bonds = 0
			
			for j, other_atom in enumerate(atoms):
				if other_atom.symbol in ['S', 'C']:
					distance = atoms.get_distance(i, j, mic=True)
					if distance < bond_threshold:
						if other_atom.symbol == 'S':
							bonded_to_s = True
						elif other_atom.symbol == 'C':
							bonded_to_c += 1
						total_bonds += 1
						
			if bonded_to_s:
				oxygen_classification[i] = 'Os'
			elif bonded_to_c == 1 and total_bonds == 1:
				oxygen_classification[i] = 'Oc'
	return oxygen_classification

def analyze_region(atoms, li_indices, carbon_classification, oxygen_classification, polymer_indices, ec_dmc_indices, cutoff):
	polymer_counter = Counter()
	ec_dmc_counter = Counter()
	
	for li_index in li_indices:
		distances = atoms.get_distances(li_index, range(len(atoms)), mic=True)
		close_atoms = np.where((distances < cutoff) & (distances > 0))[0]
		
		for j in close_atoms:
			atom_symbol = atoms[j].symbol
			atom_type = (
				carbon_classification[j]
				if atom_symbol == 'C' else
				oxygen_classification.get(j, atom_symbol)
			)
			if j in polymer_indices:
				polymer_counter[atom_type] += 1
			elif j in ec_dmc_indices:
				ec_dmc_counter[atom_type] += 1
				
	return polymer_counter, ec_dmc_counter
	
def compute_avg_std(trajectory, li_indices, carbon_classification, oxygen_classification, polymer_indices, ec_dmc_indices, cutoff):
	avg_std_polymer_counts = defaultdict(lambda: [0, []])
	avg_std_ec_dmc_counts = defaultdict(lambda: [0, []])
	
	for frame_number, atoms in enumerate(trajectory):
		polymer_counter, ec_dmc_counter = analyze_region(atoms, li_indices, carbon_classification, oxygen_classification, polymer_indices, ec_dmc_indices, cutoff)
		
		num_li_atoms = len(li_indices)
		for atom_type, count in polymer_counter.items():
			avg_count_per_li = count / num_li_atoms
			avg_std_polymer_counts[atom_type][0] += avg_count_per_li
			avg_std_polymer_counts[atom_type][1].append(avg_count_per_li)
			
		for atom_type, count in ec_dmc_counter.items():
			avg_count_per_li = count / num_li_atoms
			avg_std_ec_dmc_counts[atom_type][0] += avg_count_per_li
			avg_std_ec_dmc_counts[atom_type][1].append(avg_count_per_li)
		
		result_polymer = {
			atom: (
				round(total / len(trajectory), 2),
				round(statistics.stdev(counts), 2) if len(counts) > 1 else 0.0
			)
			for atom, (total, counts) in avg_std_polymer_counts.items()
		}

		result_ec_dmc = {
			atom: (
				round(total / len(trajectory), 2),
				round(statistics.stdev(counts), 2) if len(counts) > 1 else 0.0
			)
			for atom, (total, counts) in avg_std_ec_dmc_counts.items()
		}
		
	return result_polymer, result_ec_dmc

def write_results(file_name, results):
	with open(file_name, 'w') as f:
		for atom_type, (avg, stddev) in sorted(results.items()):
			f.write(f"{atom_type} {avg} {stddev}\n")

#read trajectory and define li_indices
trajectory = read('aspi-1-5x-n-5_lco-104_ec-dmc_nvt-2_pfp-d3_polymer_ec_dmc.xyz', index='1000:1100')
li_indices = [i for i, atom in enumerate(trajectory[0]) if atom.symbol == 'Li']

#compute classification based on the first frame_number
carbon_classification = classify_carbons(trajectory[0], bond_threshold_h)
oxygen_classification = classify_oxygens(trajectory[0], bond_threshold_o)

#define polymer and ec/dmc indices
polymer_indices = list(range(polymer_num_atoms))
ec_dmc_indices = list(range(polymer_num_atoms, len(trajectory[0])))

#compute average distances with standard deviation
avg_std_polymer_counts, avg_std_ec_dmc_counts = compute_avg_std(
	trajectory, li_indices, carbon_classification, oxygen_classification, polymer_indices, ec_dmc_indices, cutoff
)

#write results to files
write_results('Li-5A_polymer.txt', avg_std_polymer_counts)
write_results('Li-5A_ec-dmc.txt', avg_std_ec_dmc_counts)

