import mdtraj as md
import numpy as np
import pandas as pd

class Calc_Features:
    """
    Take input PDB and calc features needed for NDPredict.
    """
    def __init__(self, pdb, asns=None):
        """
        Parameters
        ----------
        pdb : str
            Input PDB file path.
        asns : list of ints
            List of residue numbers to select which ASN residues to calculate
            features for. If deafult None, all ASN residues found will be used.
        """
        self.pdb = pdb
        # Load the PDB file using MDTraj
        self.traj = md.load(pdb)

        # find all ASN residues in the topology
        if asns is None:
            asn_indices = self.traj.topology.select('resname ASN')
            # Get the residue indices of ASN residues
            self.asns = [self.traj.topology.atom(index).residue.index for index in asn_indices]
            print(self.asns) # TODO: test this
        else:
            # user defined residue indices of the Asn residues
            self.asns = asns

    def get_adjacent_residues(self, residue_index):
        # Get the residues adjacent to the Asn residue
        residue_after = self.traj.topology.residue(residue_index + 1)
        residue_before = self.traj.topology.residue(residue_index - 1)
        aa_after = residue_after.name
        aa_before = residue_before.name
        return aa_before, aa_after

    def calc_halflife(self, table="data/N-D_halftimes_GlyXxxAsnYyyGly.csv"):
        """
        Using the Xxx-ASN-Yyy peptide table, calc halflife of ASN.
        The table will have the Xxx residue on the first column and
        the Yyy residue on the first row.

        Parameters
        ----------
        table : str
            Path to peptide halflife table in csv format.

        Returns
        -------
        halflives : array of shape (n_asns, 1)
            Calculated halflife of each ASN residue in self.asns.
        """
        halflife_df = pd.read_csv(table)
        halflife_df

    def calc_attack_distance(self):
        """
        Calculate nucleophilic attack distance.
        """
        # TODO: catch error for wrong residue indexing (can't find CG atom)
        asn_atom = self.traj.topology.select(f'residue {self.asns} and name CG')
        cterm_atom = self.traj.topology.select(f'residue {self.asns + 1} and name N')
        distances = md.compute_distances(self.traj, np.array([asn_atom, cterm_atom]).reshape(1,2))
        attack_distance = distances.mean()
        # convert from nm to Angstroms
        attack_distance *= 10

    def calc_bfactors(self):
        # Calculate normalized B-factors 
        # (TODO: do this if experimental b_factors are not present)
        b_factors = traj.topology.select('all')  # Select all atoms
        print(traj.xyz.shape)
        average_b_factor = traj.xyz[:, b_factors, 2].mean()
        std_b_factor = traj.xyz[:, b_factors, 2].std()
        asn_atoms = traj.topology.select(f'residue {residue_index}')
        normalized_b_factors = (traj.xyz[:, asn_atoms, 2] - average_b_factor) / std_b_factor

        print(normalized_b_factors)

    def calc_dssp(self):
        # Calculate secondary structure using DSSP
        dssp = md.compute_dssp(traj, simplified=True)
        secondary_structure = dssp[:, residue_index]

    def calc_psa_sasa(self):
        # Calculate solvent accessibilities (SASA) using MDTraj
        sasa_residue = md.shrake_rupley(traj, mode='residue')[0][:, residue_index]
        sasa_sidechain = md.shrake_rupley(traj, mode='atom')[0][:, traj.top.select(f'sidechain and residue {residue_index}')]

        # Calculate percentage solvent accessibilities
        psa = sasa_residue.mean()
        pssa = sasa_sidechain.mean()

    def calc_dihedrals(self):
        # Calculate backbone torsion angles (Phi and Psi)
        phi = md.compute_phi(traj)[0][:, residue_index]
        psi = md.compute_psi(traj)[0][:, residue_index]

        # Calculate side chain torsion angles (Chi1 and Chi2)
        chi1 = md.compute_chi1(traj)[0][:, residue_index]
        chi2 = md.compute_chi2(traj)[0][:, residue_index]

    def calc_deamidation_binary(self):
        """
        Calculate deamidation feature (0 or 1)
        """
        # for now, just use 0, since not needed, eventually account for this in csv_proc
        #deamidation = int(traj.topology.atom(residue_index).residue.name == 'ASN')
        self.deamidation = 0

    def construct_feat_array(self):
        """
        Construct the final feature array.
        """
        feature_array = np.array([
            ['PDB', 'Residue #', 'AA following Asn', 'attack_distance', 'Half_life', 'norm_B_factor_C',
            'norm_B_factor_CA', 'norm_B_factor_CB', 'norm_B_factor_CG', 'secondary_structure', 'PSA',
            'PSSA', 'Psi', 'Phi', 'Chi1', 'Chi2', 'Deamidation'],
            ['your_pdb_file.pdb', residue_index, aa_following, attack_distance, '', normalized_b_factors[0],
            normalized_b_factors[1], normalized_b_factors[2], normalized_b_factors[3], secondary_structure,
            psa, pssa, psi, phi, chi1, chi2, deamidation]
        ])

        # Print the feature array
        for row in feature_array:
            print('\t'.join(map(str, row)))


if __name__ == "__main__":
    Calc_Features('pdb/11bg_leap.pdb', asns=[17])