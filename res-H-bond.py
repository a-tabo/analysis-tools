import os
import subprocess
import numpy as np
import pandas as pd
import ase
from ase.io import read, write


class HBondAnalyzer:
	def __init__(self, input_template, residue_numbers, output_prefix, n_files, time_step=0.04):	
		self.input_template = input_template
		self.residue_numbers = residue_numbers
		self.output_prefix = output_prefix
		self.n_files = n_files
		self.time_step = time_step
		self.all_H_ave_dist = []
		self.all_H_dist = []
		self.all_H_count = []
		
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
		
	def compute_hbonds(self, atoms, idxO, idxN, thresholds):
		H_dist = []
		H_count = 0
		
		for i in idxO:
			for j in idxN:
				O_N_dist = round(atoms.get_distance(i, j), 2)
				if O_N_dist < thresholds['O_N']:
					for k, atom in enumerate(atoms):
						if atom.symbol == 'H':
							H_N_dist = round(atoms.get_distance(j, k), 2)
							if H_N_dist < thresholds['H_N']:
								O_H_N_ang = round(atoms.get_angle(i, k, j), 1)
								if O_H_N_ang > thresholds['O_H_N']:
									H_dist.append(O_N_dist)
									H_count += 1
								
		avg_H_dist = round(sum(H_dist) / len(H_dist), 2) if H_dist else 0
		return avg_H_dist, H_dist, H_count
		
	def analyze(self):
		thresholds = {'H_N': 1.20, 'O_N': 3.50, 'O_H_N': 90.0}
		
		for i in self.n_files:
			pdb_file = self.input_template.format(i)
			xyz_file = f"{self.output_prefix}_{i}.xyz"
			
			#convert PDB to XYZ
			self.convert_pdb_to_xyz(pdb_file, xyz_file)
		
			#extract selected residues
			sel_res, df = self.extract_residues(pdb_file, self.residue_numbers, xyz_file)
		
			#read XYZ file and filter residues
			atoms = read(xyz_file)
			atom_residues = df.iloc[sel_res]['residue_number'].values
			idxO = [atom.index for atom, res_num in zip(atoms, atom_residues) if atom.symbol == 'O' and res_num in self.residue_numbers[:4]]
			idxN = [atom.index for atom, res_num in zip(atoms, atom_residues) if atom.symbol == 'N' and res_num == self.residue_numbers[-1]]
		
			#compute H-bond metrics
			avg_H_dist, H_dist, H_count = self.compute_hbonds(atoms, idxO, idxN, thresholds)
			self.all_H_ave_dist.append(avg_H_dist)
			self.all_H_dist.append(H_dist)
			self.all_H_count.append(H_count)
			
			#save individual XYZ file
			write(xyz_file, atoms)
		
	def save_results(self, output_file, window_size = 10):
		moving_avg_H_ave_dist = HBondAnalyzer.moving_average(self.all_H_ave_dist, window_size)
		moving_avg_H_count = HBondAnalyzer.moving_average(self.all_H_count, window_size)
		
		with open(output_file, 'w') as file:
			file.write("Time_Step\tAverage_H_Bond\tH_Count\n")
			time_step = self.time_step
			for i in range(len(self.all_H_ave_dist)):
				file.write(f"{round(time_step, 2)}\t{self.all_H_ave_dist[i]}\t{self.all_H_count[i]}\n")
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
residue_numbers = [13, 14, 15, 16, 114]
output_prefix = '4yl4-wt-ph7-holo_res_13-16_114_H-bond'
n_files = range(1, 101)

analyzer = HBondAnalyzer(input_template, residue_numbers, output_prefix, n_files)
analyzer.analyze()
analyzer.save_results(f"{output_prefix}.txt")
analyzer.save_combined_xyz(f"{output_prefix}_all.xyz")
