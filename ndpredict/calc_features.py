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
            self.asns = np.unique(self.asns)
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
        # TODO: add rules for when Xxx and Yyy have residues not in table
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
        C, CA, CB, CG : floats
            Normalized b-factors for C, CA, CB, CG
            TODO: need to better validate with training set PDB.
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
        
        # return in correct order for feature table
        return normalized_b_factors["C"], normalized_b_factors["CA"], \
               normalized_b_factors["CB"], normalized_b_factors["CG"]

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
        mapping = {'H':1, 'E':2, 'C':3}
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

    @staticmethod
    def find_chi_angles(asn_idxs, mdtraj_angle_calc):
        """
        Parameters
        ----------
        asn_idxs : list
            Atom indicies of the ASN residue of interest.
        mdtraj_angle_calc : tuple of arrays
            (all_idxs : 2darray, all_angles : 1darray)
            all_idxs: Array of n_angle rows and 4 cols for each atom incides.
            all_angles: Array of n_angles.

        Returns
        -------
        asn_angle : float
            Single angle value that we are interested in indexing from
            the larger arrays.
        """
        # unpack the mdtraj angle calc results
        all_idxs, all_angles = mdtraj_angle_calc

        # go through each set of 4 atom indices of each angle calc
        for i, idxs in enumerate(all_idxs):
            # check if any of the the indices match the asn of interest
            if any(x in asn_idxs for x in idxs):
                #print("asn_idxs: ", asn_idxs)
                #print("idxs: ", idxs)
                asn_angle = all_angles[0, i]
                #print("asn_angle: ", asn_angle * (180/np.pi))
        
        # this should be working but it doesn't seem to match the training set
        return asn_angle

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
        """
        # TODO: prob don't need calc for entire protein, could select for ASN first?
        #       or just calc for protein once, then index each angle
        
        # atom indices from the ASN residue of interest from the trajectory
        sidechain_selection = self.traj.topology.select(f'resid {asn} and not type H and not backbone')
        #print("selection: ", sidechain_selection)

        # Calculate backbone torsion angles (Phi and Psi)
        phi = float(md.compute_phi(self.traj)[1][:,asn-1])
        psi = float(md.compute_psi(self.traj)[1][:,asn])

        # Calculate side chain torsion angles (Chi1 and Chi2)
        # TODO: the chi calcs are not the same as training set
        chi1 = self.find_chi_angles(sidechain_selection, md.compute_chi1(self.traj))
        chi2 = self.find_chi_angles(sidechain_selection, md.compute_chi2(self.traj))

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
        Main public method of Calc_Features class constructing the feature array.

        Returns
        -------
        feature_array : df of shape (len(self.asns), len(feature_names))
        """
        feature_names = ['PDB', 'Residue #', 'AA following Asn', 'attack_distance', 'Half_life', 'norm_B_factor_C', 'norm_B_factor_CA', 'norm_B_factor_CB', 'norm_B_factor_CG', 'secondary_structure', 'PSA', 'PSSA', 'Psi', 'Phi', 'Chi1', 'Chi2', 'Deamidation']
        # feature array to be filled in
        feature_array = np.zeros((len(self.asns), len(feature_names)), dtype=object)

        # loop each ASN and calc each feature to fill in feature_array
        for i, asn in enumerate(self.asns):
            # PDB
            feature_array[i,0] = self.pdb
            # resid to PDB resnum
            feature_array[i,1] = asn + 1
            # AA following
            feature_array[i,2] = self.get_adjacent_residues(asn)[1].upper()
            # attack distance
            feature_array[i,3] = self.calc_attack_distance(asn)
            # calc halflife
            feature_array[i,4] = self.calc_halflife(asn)
            # b factors of C, CA, CB, CG
            feature_array[i,5:9] = self.calc_bfactors(asn)
            # DSSP prediction
            feature_array[i,9] = self.calc_dssp(asn)
            # SASA
            feature_array[i,10:12] = self.calc_psa_sasa(asn)
            # dihedrals
            feature_array[i,12:16] = self.calc_dihedrals(asn)
            # deamidation (TODO: not needed)
            feature_array[i, 16] = self.calc_deamidation_binary()

        # return the feature array
        return pd.DataFrame(feature_array, columns=feature_names)


if __name__ == "__main__":
    #cf = Calc_Features('pdb/1hk0_leap.pdb', asns=None, chainid="X")
    #cf = Calc_Features('pdb/1gb1_leap.pdb', asns=None, chainid="A")
    cf = Calc_Features('pdb/2m3t_leap.pdb', asns=None, chainid="A")
    fa = cf.construct_feat_array()
    fa.to_csv("2m3t_features.csv", index=False)