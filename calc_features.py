import mdtraj as md
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

class Calc_Features:
    """
    Take input PDB and calc features needed for NDPredict.
    """
    def __init__(self, pdb, asns=None, chainid="A"):
        """
        Parameters
        ----------
        pdb : str
            Input PDB file path.
        asns : list of ints
            List of residue numbers to select which ASN residues to calculate
            features for. If default None, all ASN residues found will be used.
            If providing a list, use true residue indexing from 1, this will get
            corrected to index from 0 to be consistent with MDTraj.
        chainid : str
            Which chain of the PDB file to use.
        """
        self.pdb = pdb
        # Load the PDB file using MDTraj
        self.traj = md.load(pdb)

        self.chainid = chainid

        # find all ASN residues in the topology
        if asns is None:
            asn_indices = self.traj.topology.select('resname ASN')
            # Get the residue indices of ASN residues
            self.asns = [self.traj.topology.atom(index).residue.index for index in asn_indices]
            print(self.asns) # TODO: test this
        else:
            # user defined residue indices of the Asn residues
            self.asns = np.subtract(asns, 1)

    @staticmethod
    def aa_string_formatter(aa):
        """
        Parameters
        ----------
        aa : str
            Three letter AA string, can have any capitalizations.
        
        Returns
        -------
        aa : str
            Standardized formatting needed for the Xxx-ASN-Yyy indexing table.
            e.g. asn --> Asn
        """
        # convert to all lower case
        aa = aa.lower()
        # capitalize the first letter
        aa = aa.capitalize()
        return aa

    def get_adjacent_residues(self, asn):
        """
        Get the residues adjacent to the Asn residue.

        Parameters
        ----------
        asn : int
            ASN residue index (from 0).

        Returns
        -------
        aa_before : int
            Residue name before indexed ASN.
        aa_after : int
            Residue name after indexed ASN.
        """
        residue_after = self.traj.topology.residue(asn + 1)
        residue_before = self.traj.topology.residue(asn - 1)
        # names are in all uppercase, convert to capitalized format
        # to be compatible with Xxx-ASN-Yyy lookup table
        aa_after = self.aa_string_formatter(residue_after.name)
        aa_before = self.aa_string_formatter(residue_before.name)
        return aa_before, aa_after

    def calc_halflife(self, asn, table="data/N-D_halftimes_GlyXxxAsnYyyGly.csv"):
        """
        Using the Xxx-ASN-Yyy peptide table, calc halflife of ASN.
        The table will have the Xxx residue on the first column and
        the Yyy residue on the first row.

        Parameters
        ----------
        asn : int
            ASN residue index (from 0).
        table : str
            Path to peptide halflife table in csv format.

        Returns
        -------
        halflife : float 
            Calculated halflife of indexed ASN residue.
        """
        # read in Xxx-ASN-Yyy halflife table
        halflife_df = pd.read_csv(table, index_col=0)
        # get adjacent residue names
        aa_before, aa_after = self.get_adjacent_residues(asn)
        # index halflife value where row is Xxx and col is Yyy
        halflife = halflife_df.loc[aa_before, aa_after]
        return halflife

    def calc_attack_distance(self, asn):
        """
        Calculate nucleophilic attack distance.

        Parameters
        ----------
        asn : int
            ASN residue index (from 0).

        Returns
        -------
        attack_distance : float
        """
        # TODO: catch error for wrong residue indexing (can't find CG atom)
        asn_atom = self.traj.topology.select(f'resid {asn} and name CG')
        cterm_atom = self.traj.topology.select(f'resid {asn + 1} and name N')
        distances = md.compute_distances(self.traj, np.array([asn_atom, cterm_atom]).reshape(1,2))
        attack_distance = distances.mean()
        # convert from nm to Angstroms
        attack_distance *= 10
        return attack_distance

    def calc_bfactors(self, asn):
        """
        Parameters
        ----------
        asn : int
            ASN residue index (from 0).

        Returns
        -------
        b_factors : 1darray of floats
            TODO: normalized array of b-factors for C, CA, CB, CG
                  also need to better validate with training set PDB.
        """
        # Calculate normalized B-factors 
        # (TODO: do this if experimental b_factors are not present)
        # # Get the indices of atoms in the Asn residue
        # residue_atoms = traj.topology.select(f'residue {residue_index}')
        # # Get the atomic fluctuations (variance) for each atom in the trajectory
        # fluctuations = np.var(traj.xyz[:, residue_atoms], axis=0)
        # # Calculate the average and standard deviation of atomic fluctuations for all atoms
        # average_fluctuation = np.mean(fluctuations)
        # std_fluctuation = np.std(fluctuations)
        # # Normalize the atomic fluctuations to obtain the normalized B-factors
        # normalized_b_factors = (fluctuations - average_fluctuation) / std_fluctuation

        # extract experimental b-factors from PDB file
        # Create a PDBParser object
        parser = PDBParser()

        # Parse the PDB file
        structure = parser.get_structure('pdb', self.pdb)

        # Get the first model (assuming single model)
        model = structure[0]

        # Get the chain and residue of interest
        # TODO: add something here to loop through all chains and find residue index
        chain = model[self.chainid]
        # note biopython will index from 1 but mdtraj from 0
        residue = chain[int(asn + 1)]

        # Define the atom names of interest
        atom_names = ['C', 'CA', 'CB', 'CG']

        # Collect B-factors of all atoms in the PDB structure
        all_b_factors = []
        for atom in structure.get_atoms():
            all_b_factors.append(atom.get_bfactor())

        # Calculate the average and standard deviation of all B-factors
        average_b_factor = np.mean(all_b_factors)
        std_b_factor = np.std(all_b_factors)

        # Get the B-factor and normalize for each atom in the residue
        normalized_b_factors = {}
        for atom in residue:
            if atom.name in atom_names:
                b_factor = atom.get_bfactor()
                z_score = (b_factor - average_b_factor) / std_b_factor
                normalized_b_factors[atom.name] = z_score
        
        return normalized_b_factors

    def calc_dssp(self, asn):
        """ 
        Calculate secondary structure using DSSP.

        Parameters
        ----------
        asn : int
            ASN residue index (from 0).

        Returns
        -------
        secondary_structure : int
            ASN local secondary structure: 
            alpha helix = 1, beta sheet = 2, coil = 3, and turn = 4.
        """
        dssp = md.compute_dssp(self.traj, simplified=True)
        # returns 'H', 'S', 'C' for helix, strand, coil
        secondary_structure = dssp[:, asn]
        # need to convert to numerical 1-4
        # TODO: for now forgetting about turns (considered coil)
        #       eventually could not use the simplified dssp output to get turns.
        mapping = {'H':1, 'S':2, 'C':3}
        secondary_structure = mapping[secondary_structure[0]]
        return secondary_structure

    def calc_psa_sasa(self, asn):
        """
        Calculate solvent accessibilities (SASA) using MDTraj.

        Parameters
        ----------
        asn : int
            ASN residue index (from 0).

        Returns
        -------
        psa : float
            PSA: Percent Solvent Accessibility (of the entire residue).
        pssa : float
            PSSA: Percent Sidechain Solvent Accessibility (of the sidechain).
        """
        # calc residue and sidechain sasa
        sasa_residue = md.shrake_rupley(self.traj, mode='residue')[:, asn]
        sasa_sidechain = md.shrake_rupley(
            self.traj, mode='atom')[:, self.traj.top.select(f'sidechain and resid {asn}')]

        # Calculate percentage solvent accessibilities
        psa = sasa_residue.sum()
        pssa = sasa_sidechain.sum()

        # TODO: need to make consistent with training set data calc
        # SASA conversion: nm^2 --> A^2
        return psa * 100, pssa * 100

    def calc_dihedrals(self, asn):
        """
        Calculate psi, phi, chi1, chi2 torsion angles.

        Parameters
        ----------
        asn : int
            ASN residue index (from 0).

        Returns
        -------
        psi, phi, chi1, chi2 : floats
            TODO: make array?
        """
        # TODO: prob don't need calc for entire protein, could select for ASN first?
        #       or just calc for protein once, then index each angle
        # Calculate backbone torsion angles (Phi and Psi)
        phi = md.compute_phi(self.traj)[1][:, asn-1]
        psi = md.compute_psi(self.traj)[1][:, asn]

        # Calculate side chain torsion angles (Chi1 and Chi2)
        # TODO: the chi calcs are not the same as training set
        chi1 = md.compute_chi1(self.traj)[1][:, asn]
        chi2 = md.compute_chi2(self.traj)[1][:, asn]

        # convert from radians to degrees
        return psi*(180/np.pi), phi*(180/np.pi), chi1*(180/np.pi), chi2*(180/np.pi)

    def calc_deamidation_binary(self):
        """
        Calculate deamidation feature (0 or 1)
        """
        # for now, just use 0, since not needed, eventually account for this in csv_proc
        #deamidation = int(traj.topology.atom(residue_index).residue.name == 'ASN')
        deamidation = 0
        return deamidation

    def construct_feat_array(self):
        """
        Construct the final feature array.
        """
        # TODO: make empty df or array

        # loop each ASN and calc each feature
        for asn in self.asns:
            print(self.calc_halflife(asn))
            #print(self.calc_attack_distance(asn))
            #print(self.calc_bfactors(asn))
            #print(self.calc_dssp(asn))
            #print(self.calc_psa_sasa(asn))
            #print(self.calc_dihedrals(asn))


        # feature_array = np.array([
        #     ['PDB', 'Residue #', 'AA following Asn', 'attack_distance', 'Half_life', 'norm_B_factor_C',
        #     'norm_B_factor_CA', 'norm_B_factor_CB', 'norm_B_factor_CG', 'secondary_structure', 'PSA',
        #     'PSSA', 'Psi', 'Phi', 'Chi1', 'Chi2', 'Deamidation'],
        #     ['your_pdb_file.pdb', residue_index, aa_following, attack_distance, '', normalized_b_factors[0],
        #     normalized_b_factors[1], normalized_b_factors[2], normalized_b_factors[3], secondary_structure,
        #     psa, pssa, psi, phi, chi1, chi2, deamidation]
        # ])

        # # Print the feature array
        # for row in feature_array:
        #     print('\t'.join(map(str, row)))


if __name__ == "__main__":
    cf = Calc_Features('pdb/11bg_leap.pdb', asns=[17])
    cf.construct_feat_array()