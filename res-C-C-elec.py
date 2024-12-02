import os
import subprocess
import numpy as np
import pandas as pd
import ase
from ase.io import read, write


class CarbonCarbonAnalyzer:
	def __init__(self, input_template, residue_numbers, output_prefix, n_files, time_step=0.04):	
		self.input_template = input_template
		self.residue_numbers = residue_numbers
		self.output_prefix = output_prefix
		self.n_files = n_files
		self.time_step = time_step
		self.all_c_c_dist = []
		self.all_c_c_count = []
	
	def moving_average(data, window_size):
		return np.convolve(data, np.ones(window_size), 'valid')/window_size
	
	def convert_pdb_to_xyz(self, pdb_file, xyz_file):
		command = f'obabel -ipdb {pdb_file} -oxyz -O {xyz_file}'
		subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		
	def extract_residues(self, pdb_file, selected_residues, output_file):
		columns = ['ATOM', 'atom_number', 'atom_name', 'residue_name', 'residue_number',
					'x', 'y', 'z', 'unknown', 'unknown2', 'atom_type']
		df = pd.read_csv(pdb_file, sep=r'\s+', skiprows=1, skipfooter=2, names=columns, engine='python')
		sel_atoms = df[df['residue_number'].isin(selected_residues)]
		sel_res = (sel_atoms['atom_number'].astype(int) - 1).tolist()
		
		atoms = read(output_file)
		res_atoms = atoms[sel_res]
		write(output_file, res_atoms, format='xyz')
		
		return sel_res, df

	def find_carbon_atom(self, atoms, df, sel_res, residue_number, atom_symbol, neighbor_symbol, neighbor_count_thr, dist_thr):
		for atom_idx, atom in enumerate(atoms):
			if df.iloc[sel_res[atom_idx]]['residue_number'] == residue_number and atom.symbol == atom_symbol:
				neighbor_count = 0
				for neighbor_idx, neighbor in enumerate(atoms):
					if neighbor.symbol == neighbor_symbol:
						dist = atoms.get_distance(atom_idx, neighbor_idx)
						if dist <= dist_thr:
							neighbor_count += 1
				if neighbor_count == neighbor_count_thr:
					return atom_idx
		return None
 
	def analyze(self):
		for i in self.n_files:
			pdb_file = self.input_template.format(i)
			xyz_file = f"{self.output_prefix}_{i}.xyz"
			
			#convert PDB to XYZ
			self.convert_pdb_to_xyz(pdb_file, xyz_file)
		
			#extract selected residues
			sel_res, df = self.extract_residues(pdb_file, self.residue_numbers, xyz_file)
		
			#read XYZ file
			atoms = read(xyz_file)

			#find the target C atoms
			carbon_13 = self.find_carbon_atom(atoms, df, sel_res, 13, "C", "O", 2, 1.5)
			carbon_114 = self.find_carbon_atom(atoms, df, sel_res, 114, "C", "N", 3, 1.5)
			
			#calculate C-C distance
			c_c_dist_thr = 4.0
			c_c_count = 0
			if carbon_13 is not None and carbon_114 is not None:
				c_c_dist = round(atoms.get_distance(carbon_13, carbon_114), 2)
				if c_c_dist < c_c_dist_thr:
					c_c_count = 1
				else:
					c_c_dist = 0
			self.all_c_c_dist.append(c_c_dist)
			self.all_c_c_count.append(c_c_count)
			
			#save individual XYZ file
			write(xyz_file, atoms)

	def save_results(self, output_file, window_size = 10):
		moving_avg_c_c_dist = CarbonCarbonAnalyzer.moving_average(self.all_c_c_dist, window_size)
		moving_avg_c_c_count = CarbonCarbonAnalyzer.moving_average(self.all_c_c_count, window_size)
		
		with open(output_file, 'w') as file:
			file.write("Time_Step\tC-C_Bond\tC-C_Count\n")
			time_step = self.time_step
			for i in range(len(self.all_c_c_dist)):
				file.write(f"{round(time_step, 2)}\t{self.all_c_c_dist[i]}\t{self.all_c_c_count[i]}\n")
				time_step += self.time_step

	def save_combined_xyz(self, output_file):
		all_structures = []
		for i in self.n_files:
			filename = f"{self.output_prefix}_{i}.xyz"
			structure = read(filename)
			all_structures.append(structure)
		write(output_file, all_structures, format='xyz')

#example usage
input_template = '4yl4-wt-ph7-holo_{}.pdb'
residue_numbers = [13, 114]
output_prefix = '4yl4-wt-ph7-holo_res_13_114_C-C'
n_files = range(1, 101)

analyzer = CarbonCarbonAnalyzer(input_template, residue_numbers, output_prefix, n_files)
analyzer.analyze()
analyzer.save_results(f"{output_prefix}.txt")
analyzer.save_combined_xyz(f"{output_prefix}_all.xyz")
