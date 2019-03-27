class Protein():
    
    def __init__(self, code, n, approx):
        self.code = code
        self.n = n 
        self.source_sos = '0'*n
        self.source_eos = 'Z'*n
        
        self.target_sos = (-3.5, -3.5)
        self.target_eos = (3.5, 3.5)
        
        self.peptide = self._get_peptide(code)
        
        self.aa_seq = self.peptide.get_sequence()
        self.ngrams = self._split_ngrams(self.aa_seq)
        
        self.source = self._get_sequence()
        self.target = self._get_torsions(approx)[:-1]
        self.label  = self._get_torsions(approx)[1:]
        
        self.tkz_source = None
        self.tkz_target = None
        self.tkz_label  = None
        
        # MAY NEED TO PAD THESE TOO
        self.secondary_structure = self._get_secondary_structure()
        self.residue_names = self._get_residue_names()
        self.residue_ids = self._get_residue_ids()
    
    def _get_peptide(self, code):
        pdbl = Bio.PDB.PDBList(verbose=False)
        pdb_filename = pdbl.retrieve_pdb_file(code, pdir="pdb")
        
        structure = Bio.PDB.MMCIFParser(QUIET=True).get_structure(code, pdb_filename)
        peptides = Bio.PDB.CaPPBuilder().build_peptides(structure)
        
        return peptides[0] # Only want single chain per protein
    
    def _split_ngrams(self, seq):
        """
        'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
        """
        #a, b, c = zip(*[iter(seq)]*self.n), zip(*[iter(seq[1:])]*self.n), zip(*[iter(seq[2:])]*self.n)
        str_ngrams = []
        for ngrams in [zip(*[iter(seq[i:])]*self.n) for i in range(self.n)]:
            x = []
            for ngram in ngrams:
                x.append("".join(ngram))
            str_ngrams.append(x)
        return str_ngrams
    
    def _get_sequence(self):
        sequence = [self.source_sos] + self.ngrams[0]
        
#         for i in range(len(self.ngrams[0])):
#             for ngrams in self.ngrams:
#                 try:
#                     sequence.append(ngrams[i])
#                 except:
#                     pass
        return sequence
    
    def _get_torsions(self, approx):
        
        phi_psi = [self.target_sos] + self.peptide.get_phi_psi_list() + [self.target_eos]
        phi_psi = np.array(phi_psi, dtype=float)
        
        phi_psi = np.nan_to_num(phi_psi) # Replace None values
        phi_psi = np.around(phi_psi, approx) # Discretize angles

        return phi_psi
    
    def _get_secondary_structure(self):
        parser = Bio.PDB.MMCIFParser(QUIET = True)
        structure = parser.get_structure(self.code, "pdb/"+self.code.lower()+".cif")

        model = structure[0]
        try:
            dssp = Bio.PDB.DSSP(model, "pdb/"+self.code.lower()+".cif")
            ss = [dssp[dssp.keys()[i]][2].encode('utf-8') for i in range(len(dssp.keys()))]
            return ss
        except:
            ss = ['X'.encode('utf-8') for i in range(len(self.aa_seq))]

            return ss
    
    def _get_residue_names(self):
        residue_names = [Bio.PDB.Polypeptide.one_to_three(aa).encode('utf-8') for aa in list(self.aa_seq)]
        return residue_names
        
    def _get_residue_ids(self):
        return [aa.encode('utf-8') for aa in list(self.aa_seq)]
