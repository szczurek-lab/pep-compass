"""Molecular fingerprint utilities for optimization policies."""

# -*- coding: utf-8 -*-
#
# Fingerprints
#

import itertools
from collections import defaultdict

import numpy as np
from mapchiral.mapchiral import get_shingles
from mhfp.encoder import MHFPEncoder
from rdkit import Chem
from rdkit.Chem import rdmolops, rdMHFPFingerprint
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdmolops import GetDistanceMatrix

class MAP4Calculator:

    def __init__(self, dimensions=1024, radius=2, is_counted=False):
        """
        MAP4 calculator class
        """
        self.dimensions = dimensions
        self.radius = radius
        self.is_counted = is_counted
        self.encoder = MHFPEncoder()

    def calculate(self, mol):
        """Calculates the atom pair minhashed fingerprint
        Arguments:
            mol -- rdkit mol object
        Returns:
            tmap VectorUint -- minhashed fingerprint
        """

        atom_env_pairs = self._calculate(mol)
        return self._fold(atom_env_pairs)

    def calculate_many(self, mols):
        """ Calculates the atom pair minhashed fingerprint
        Arguments:
            mols -- list of mols
        Returns:
            list of tmap VectorUint -- minhashed fingerprints list
        """

        atom_env_pairs_list = [self._calculate(mol) for mol in mols]
        return [self._fold(pairs) for pairs in atom_env_pairs_list]

    def _calculate(self, mol):
        return self._all_pairs(mol, self._get_atom_envs(mol))

    def _fold(self, pairs):
        fp_hash = self.encoder.hash(set(pairs))
        return self.encoder.fold(fp_hash, self.dimensions)

    def _get_atom_envs(self, mol):
        atoms_env = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()

            for radius in range(1, self.radius + 1):
                if idx not in atoms_env:
                    atoms_env[idx] = []

                atoms_env[idx].append(MAP4Calculator._find_env(mol, idx, radius))

        return atoms_env

    @classmethod
    def _find_env(cls, mol, idx, radius):
        env = rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius, idx)
        atom_map = {}

        submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)

        if idx in atom_map:
            smiles = Chem.MolToSmiles(submol, rootedAtAtom=atom_map[idx], canonical=True, isomericSmiles=False)
            return smiles

        return ''

    def _all_pairs(self, mol, atoms_env):
        atom_pairs = []
        distance_matrix = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        shingle_dict = defaultdict(int)

        for idx1, idx2 in itertools.combinations(range(num_atoms), 2):
            dist = str(int(distance_matrix[idx1][idx2]))

            for i in range(self.radius):
                env_a = atoms_env[idx1][i]
                env_b = atoms_env[idx2][i]

                ordered = sorted([env_a, env_b])

                shingle = '{}|{}|{}'.format(ordered[0], dist, ordered[1])

                if self.is_counted:
                    shingle_dict[shingle] += 1
                    shingle += '|' + str(shingle_dict[shingle])

                atom_pairs.append(shingle.encode('utf-8'))

        return list(set(atom_pairs))


class MAP4CCalculator:

    def __init__(self, dimensions=1024, radius=2):
        """
        MAP4C calculator class
        """
        self.dimensions = dimensions
        self.radius = radius
        self._encoder = rdMHFPFingerprint.MHFPEncoder()

    def _fold(self, hash_values, length):
        folded = np.zeros(length, dtype=np.uint8)

        if len(hash_values) == 0:
            return folded

        folded[hash_values % length] = 1

        return folded

    def calculate(self, mol):
        shingles = get_shingles(mol, self.radius)
        fingerprint = np.asarray(self._encoder.FromStringArray(shingles), dtype=np.uint32)
        folded_fingerprint = self._fold(fingerprint, self.dimensions)

        return folded_fingerprint

    def calculate_many(self, mols):
        """ Calculates the atom pair minhashed fingerprint
        Arguments:
            mols -- list of mols
        Returns:
            list of tmap VectorUint -- minhashed fingerprints list
        """
        return np.asarray([self.calculate(mol) for mol in mols])


class Map4Fingerprint:
    """
    A class for computing the MAP4 fingerprints.

    """

    def __init__(self, input_type='helm', dimensions=4096, radius=1, chiral=True, HELM_parser='mobius', HELM_extra_library_filename=None):
        """
        Constructs a new instance of the MAP4 fingerprint class.

        Parameters
        ----------
        input_type : str, optional (default='helm_rdkit')
            The input format for the polymers. It can be one of 'fasta', 
            'helm_rdkit', 'helm', or 'smiles'.
        dimensions : int, optional (default=4096)
            The length of the MAP4 fingerprint.
        radius : int, optional (default=1)
            The maximum radius of the subgraphs used for computing 
            the fingerprints.
        chiral : bool, optional (default=True)
            If True, use the chiral version of the MAP4 fingerprint (MAPC).
        is_folded : bool, optional (default=True)
            If True, the fingerprint will be folded.
        HELM_parser : str, optional (default='mobius')
            The HELM parser to be used. It can be 'mobius' or 'rdkit'. 
            When using 'rdkit' parser, only D or L standard amino acids
            are supported with disulfide bridges for macrocyclic polymers.
            Using the (slower) internal 'mobius' HELM parser, all kind of
            scaffolds and monomers can be supported. Monomers library can 
            be easily extended by providing an extra HELM Library file using 
            the `HELM_extra_library_filename` parameter.
        HELM_extra_library_filename : str, optional (default=None)
            The path to a HELM Library file containing extra monomers.
            Extra monomers will be added to the internal monomers library. 
            Internal monomers can be overriden by providing a monomer with
            the same MonomerID.

        Notes
        -----
        When using HELM format as input, the fingerprint of a same molecule 
        can differ depending on the HELM parser used. The fingerprint of a 
        molecule in FASTA format is guaranteed to be the same if the same 
        molecule in HELM format uses `rdkit` for the parsing, but won't be 
        necessarily with the internal HELM parser.

        Citations
        ---------
        * One chiral fingerprint to find them all. 
        Markus Orsi and Jean-Louis Reymond. 2024.
        https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00849-6
        * One molecular fingerprint to rule them all: drugs, biomolecules, and the metabolome.
        Alice Capecchi, Daniel Probst, and Jean-Louis Reymond. 2020. 
        https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00445-4

        """
        msg_error = 'Format (%s) not handled. Please use FASTA, HELM or SMILES format.'
        assert input_type.lower() in ['fasta', 'helm', 'smiles'], msg_error

        self._dimensions = dimensions
        self._radius = radius
        self._chiral = chiral
        self._input_type = input_type.lower()
        self._HELM_parser = HELM_parser.lower()
        self._HELM_extra_library_filename = HELM_extra_library_filename

    def __call__(self, polymers):
        """
        Transform input polymers into MAP4 or MAP4C (if chiral set to True) fingerprints.
        
        Parameters
        ----------
        polymers : list, tuple, ndarray or str
            A list of polymers or a single polymer to be transformed.

        Returns
        -------
        fps : ndarray
            The MAP4 or MAP4C fingerprints of the input polymers.

        """
        return self.transform(polymers)

    def transform(self, polymers):
        """
        Transforms the input polymers into MAP4 or MAP4C (if chiral set to True) fingerprints.

        Parameters
        ----------
        polymers : str, list, tuple, or numpy.ndarray
            A list of polymers or a single polymer to be transformed.

        Returns
        -------
        fingerprint : numpy.ndarray
            The MAP4 or MAP4C fingerprints of the input polymers.

        """
        if not isinstance(polymers, (list, tuple, np.ndarray)):
            polymers = [polymers]

        # We define the method here, otherwise we cannot pickle Map4Fingerprint object
        if self._chiral:
            map4calc = MAP4CCalculator(dimensions=self._dimensions, radius=self._radius)
        else:
            map4calc = MAP4Calculator(dimensions=self._dimensions, radius=self._radius)

        try:
            if self._input_type == 'fasta':
                mols = [Chem.rdmolfiles.MolFromFASTA(c) for c in polymers]
            elif self._input_type == 'helm' and self._HELM_parser == 'rdkit':
                mols = [Chem.rdmolfiles.MolFromHELM(c) for c in polymers]
            elif self._input_type == 'helm' and self._HELM_parser == 'mobius':
                mols = MolFromHELM(polymers, self._HELM_extra_library_filename)
            else:
                mols = [Chem.rdmolfiles.MolFromSmiles(c) for c in polymers]
        except AttributeError:
            print('Error: there are issues with the input polymers')
            print(polymers)

        fps = map4calc.calculate_many(mols)

        fps = np.asarray(fps)

        return fps


class MHFingerprint:
    """
    A class for computing the MHFingerprint.

    """

    def __init__(self, input_type='helm', dimensions=4096, radius=3, rings=True, kekulize=True, 
                 HELM_parser='mobius', HELM_extra_library_filename=None):
        """
        Constructs a new instance of the MHFingerprint class.

        Parameters
        ----------
        input_type : str, default 'helm_rdkit'
            Input format of the polymers. The options are 'fasta', 
            'helm_rdkit', 'helm', or 'smiles'.
        dimensions : int, default 4096
            The number of bits in the resulting binary feature vector.
        radius : int, default 3
            The radius parameter of the Morgan fingerprint.
        rings : bool, default True
            Whether to use information about the presence of rings 
            in the molecular structures.
        kekulize : bool, default True
            Whether to kekulize the structures before generating 
            the fingerprints.
        HELM_parser : str, optional (default='mobius')
            The HELM parser to be used. It can be 'mobius' or 'rdkit'. 
            When using 'rdkit' parser, only D or L standard amino acids
            are supported with disulfide bridges for macrocyclic polymers.
            Using the (slower) internal 'mobius' HELM parser, all kind of
            scaffolds and monomers can be supported. Monomers library can 
            be easily extended by providing an extra HELM Library file using 
            the `HELM_extra_library_filename` parameter.
        HELM_extra_library_filename : str, optional (default=None)
            The path to a HELM Library file containing extra monomers.
            Extra monomers will be added to the internal monomers library. 
            Internal monomers can be overriden by providing a monomer with
            the same MonomerID.

        Notes
        -----
        When using HELM format as input, the fingerprint of a same molecule 
        can differ depending on the HELM parser used. The fingerprint of a 
        molecule in FASTA format is guaranteed to be the same if the same 
        molecule in HELM format uses `rdkit` for the parsing, but won't be 
        necessarily with the internal HELM parser.

        """
        msg_error = 'Format (%s) not handled. Please use FASTA, HELM or SMILES format.'
        assert input_type.lower() in ['fasta', 'helm', 'smiles'], msg_error

        self._dimensions = dimensions
        self._radius = radius
        self._rings = rings
        self._kekulize = kekulize
        self._encoder = MHFPEncoder()
        self._input_type = input_type.lower()
        self._HELM_parser = HELM_parser.lower()
        self._HELM_extra_library_filename = HELM_extra_library_filename

    def __call__(self, polymers):
        """
        Transform input polymers into MHFingerprints.
        
        Parameters
        ----------
        polymers : list, tuple, ndarray or str
            A list of polymers or a single polymer to be transformed.

        Returns
        -------
        fps : ndarray
            The MHFingerprint for the input polymers.

        """
        return self.transform(polymers)

    def transform(self, polymers):
        """
        Transforms the input polymers into MHFingerprints.

        Parameters
        ----------
        polymers : str, list or numpy.ndarray
            A list of polymers or a single polymer to be transformed.

        Returns
        -------
        fingerprint : numpy.ndarray
           The MHFingerprint for the input polymers.

        """
        if not isinstance(polymers, (list, tuple, np.ndarray)):
            polymers = [polymers]

        try:
            if self._input_type == 'fasta':
                mols = [Chem.rdmolfiles.MolFromFASTA(c) for c in polymers]
            elif self._input_type == 'helm' and self._HELM_parser == 'rdkit':
                mols = [Chem.rdmolfiles.MolFromHELM(c) for c in polymers]
            elif self._input_type == 'helm' and self._HELM_parser == 'mobius':
                mols = MolFromHELM(polymers, self._HELM_extra_library_filename)
            else:
                mols = [Chem.rdmolfiles.MolFromSmiles(c) for c in polymers]
        except AttributeError:
            print('Error: there are issues with the input molecules')
            print(polymers)

        fps = [self._encoder.fold(self._encoder.encode_mol(m, radius=self._radius, rings=self._rings, kekulize=self._kekulize), length=self._dimensions) for m in mols]
        fps = np.asarray(fps)

        return fps


class MorganFingerprint:
    """
    A class for computing Morgan Fingerprints.

    """

    def __init__(self, input_type='helm', dimensions=4096, radius=2, 
                 HELM_parser='mobius', HELM_extra_library_filename=None):
        """
        Constructs a new instance of the MorganFingerprint class.

        Parameters
        ----------
        input_type : str, default : 'helm'
            The type of input polymers. It can be either 'fasta', 'helm', or 'smiles'.
        dimensions : int, default : 4096
            The length of the fingerprint vector.
        radius : int, default : 2
            The radius of the Morgan fingerprint.
        HELM_parser : str, optional (default='mobius')
            The HELM parser to be used. It can be 'mobius' or 'rdkit'. 
            When using 'rdkit' parser, only D or L standard amino acids
            are supported with disulfide bridges for macrocyclic polymers.
            Using the (slower) internal 'mobius' HELM parser, all kind of
            scaffolds and monomers can be supported. Monomers library can 
            be easily extended by providing an extra HELM Library file using 
            the `HELM_extra_library_filename` parameter.
        HELM_extra_library_filename : str, optional (default=None)
            The path to a HELM Library file containing extra monomers.
            Extra monomers will be added to the internal monomers library. 
            Internal monomers can be overriden by providing a monomer with
            the same MonomerID.

        Notes
        -----
        When using HELM format as input, the fingerprint of a same molecule 
        can differ depending on the HELM parser used. The fingerprint of a 
        molecule in FASTA format is guaranteed to be the same if the same 
        molecule in HELM format uses `rdkit` for the parsing, but won't be 
        necessarily with the internal HELM parser.

        """
        msg_error = 'Format (%s) not handled. Please use FASTA, HELM or SMILES format.'
        assert input_type.lower() in ['fasta', 'helm', 'smiles'], msg_error

        self._radius = radius
        self._dimensions = dimensions
        self._input_type = input_type.lower()
        self._HELM_parser = HELM_parser.lower()
        self._HELM_extra_library_filename = HELM_extra_library_filename
    
    def __call__(self, polymers):
        """
        Transform input polymers into Morgan fingerprints.
        
        Parameters
        ----------
        polymers : list, tuple, ndarray or str
            A list of polymers or a single polymer to be transformed.

        Returns
        -------
        fps : ndarray
            The Morgan fingerprints of the input polymers as a 2D numpy array.

        """
        return self.transform(polymers)

    def transform(self, polymers):
        """
        Transform input polymers into Morgan fingerprints.

        Parameters
        ----------
        polymers : list, tuple, ndarray or str
            A list of polymers or a single polymer to be transformed.

        Returns
        -------
        fps : ndarray
            The Morgan fingerprints of the input polymers as a 2D numpy array.

        """
        if not isinstance(polymers, (list, tuple, np.ndarray)):
            polymers = [polymers]

        try:
            if self._input_type == 'fasta':
                mols = [Chem.rdmolfiles.MolFromFASTA(c) for c in polymers]
            elif self._input_type == 'helm' and self._HELM_parser == 'rdkit':
                mols = [Chem.rdmolfiles.MolFromHELM(c) for c in polymers]
            elif self._input_type == 'helm' and self._HELM_parser == 'mobius':
                mols = MolFromHELM(polymers, self._HELM_extra_library_filename)
            else:
                mols = [Chem.rdmolfiles.MolFromSmiles(c) for c in polymers]
        except AttributeError:
            print('Error: there are issues with the input molecules')
            print(polymers)

        fpg = rdFingerprintGenerator.GetMorganGenerator(includeChirality=True, radius=self._radius, fpSize=self._dimensions)
        fps = [fpg.GetFingerprintAsNumPy(m) for m in mols]
        fps = np.asarray(fps)

        return fps
