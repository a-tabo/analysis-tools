import ase
from ase.io import read

min_dist = 1.5
max_dist = 2.5

def find_bonded_atoms(li_indices, atoms, min_dist, max_dist):
	bonded_atoms = set()
	for li_index in li_indices:
		distances = atoms.get_distances(li_index, range(len(atoms)), mic=True)
		bonded = {
			(index, atoms[index].symbol)
			for index, distance in enumerate(distances)
			if min_dist <= distance <= max_dist and atoms[index].symbol in {'O', 'F'}	
		}
		bonded_atoms.update(bonded)
	return bonded_atoms
	
def calculate_changes_ratio(current_bonded, previous_bonded, num_li_atoms):
	added_bonds = current_bonded - previous_bonded
	removed_bonds = previous_bonded - current_bonded
	num_changes = len(added_bonds) + len(removed_bonds)
	ratio = round(num_changes / num_li_atoms, 2) if num_li_atoms > 0 else 0.0
	return num_changes, ratio

def analyze(trajectory, min_dist, max_dist):
	previous_bonded = set()
	results = []
	
	for time_step, atoms in enumerate(trajectory):
		li_indices = [idx for idx, atom in enumerate(atoms) if atom.symbol == 'Li']
		num_li_atoms = len(li_indices)
		
		current_bonded = find_bonded_atoms(li_indices, atoms, min_dist, max_dist)
		
		if time_step > 0:
			num_changes, ratio = calculate_changes_ratio(current_bonded, previous_bonded, num_li_atoms)
		else:
			num_changes, ratio = 0, 0.0
	
		results.append((time_step, num_li_atoms, num_changes, ratio))
		previous_bonded = current_bonded
		
	return results

def write_results(file, results):
	with open(file, 'w') as f:
		f.write('Time step\tLi atoms\tChanged atoms\tRatio\n')
		for time_step, num_li_atoms, num_changes, ratio in results:
			f.write(f'{time_step}\t{num_li_atoms}\t{num_changes}\t{ratio}\n')

#read trajectory file
trajectory = read('aspi-1-5x-n-5_lco-104_ec-dmc_nvt-2_pfp-d3_polymer_ec_dmc.xyz', index=':100')

#compute li coordinate change
li_coord_chg = analyze(trajectory, min_dist, max_dist)

#write results
write_results('li-coord-chg.txt', li_coord_chg)