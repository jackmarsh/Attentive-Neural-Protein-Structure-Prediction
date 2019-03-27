class PreProcessor():
    
    def __init__(self, filename, N, n, approx):
        self.pdb_codes = self._get_pdb_codes(filename)
        self.N = N # Sample size
        self.n = n # ngram size
        self.approx = approx
        
        self.source_vocab = ['0'*n]
        self.target_vocab = [np.array((-3.5, -3.5))]
        
        self.proteins = self._get_proteins()
        
        self.source_vocab_dict, self.rev_source_vocab_dict = self._create_vocab(self.source_vocab)
        self.target_vocab_dict, self.rev_target_vocab_dict = self._create_vocab(self.target_vocab)
        
        self.max_source_len = 0
        self.max_target_len = 0
        
        self._tokenize()
        self._pad()
        
        train_proteins, test_proteins, val_proteins = self._split_dataset(0.6, 0.8)
        
        self._create_data_record(os.path.join(TFR_DIR,'train.tfrecord'), train_proteins)
        self._create_data_record(os.path.join(TFR_DIR,'test.tfrecord'), test_proteins)
        self._create_data_record(os.path.join(TFR_DIR,'val.tfrecord'), val_proteins)
        
    
    def _get_pdb_codes(self, filename):
        """
        returns: list of pdb codes
        """
        with open(filename, "r") as f:
            pdb_codes = f.readline().replace(',','').lower().split()
            return pdb_codes
    
    def _get_proteins(self):
        proteins = []
        
        for code in self.pdb_codes[:self.N]:
            p = Protein(code, self.n, self.approx)
            
            self.source_vocab += [n for ngram in p.ngrams for n in ngram]
            self.target_vocab += [torsion for torsion in p.target]
            
            proteins.append(p)
        
        self.source_vocab += ['Z'*self.n]    
        self.target_vocab += [np.array((3.5, 3.5))]
        return proteins
    
    def _create_vocab(self, vocab):
        
        vocab = sorted(list(set([tuple(v) for v in vocab])))
        token_index = dict([(token, i+1) for i, token in enumerate(vocab)])
        
        reverse_token_index = dict((i, (*token, np.pi)) for token, i in token_index.items())

        return token_index, reverse_token_index
    
    def _tokenize(self):
        """
        """
        for protein in self.proteins:
            protein.tkz_source = [self.source_vocab_dict[tuple(elem)] for elem in protein.source]
            protein.tkz_target = [self.target_vocab_dict[tuple(elem)] for elem in protein.target]
            protein.tkz_label = [self.target_vocab_dict[tuple(elem)] for elem in protein.label]
            
            # Track the input & output lengths
            if self.max_source_len < len(protein.tkz_source):
                self.max_source_len = len(protein.tkz_source)
            if self.max_target_len < len(protein.tkz_target):
                self.max_target_len = len(protein.tkz_target)
    
    def _pad(self):
        """
        """
        for protein in self.proteins:
            protein.tkz_source += [0 for i in range(self.max_source_len-len(protein.tkz_source))]
            protein.tkz_target += [0 for i in range(self.max_target_len-len(protein.tkz_target))]
            protein.tkz_label  += [0 for i in range(self.max_target_len-len(protein.tkz_label))]
            protein.residue_ids += [b'X' for i in range(self.max_target_len-len(protein.residue_ids))]
            protein.residue_names += [b'XXX' for i in range(self.max_target_len-len(protein.residue_names))]
            protein.secondary_structure  += [b'X' for i in range(self.max_target_len-len(protein.secondary_structure))]
    
    def _split_dataset(self, test_split, val_split):

        shuffle(self.proteins)
        
        test_split = int(len(self.proteins)*test_split)
        val_split = int(len(self.proteins)*val_split)
        
        train = self.proteins[:test_split]
        test = self.proteins[test_split:val_split]
        val = self.proteins[val_split:]
    
        return train, test, val
    
    def _create_data_record(self, out_filename, proteins):
        
        def _int64_list_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        def _bytes_list_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        
        with tf.python_io.TFRecordWriter(out_filename) as tfwriter:
            for protein in proteins:
                
                feature = {
                    'code'   : _bytes_feature(protein.code.encode('utf-8')),
                    'source' : _int64_list_feature(protein.tkz_source),
                    'target' : _int64_list_feature(protein.tkz_target),
                    'label'  : _int64_list_feature(protein.tkz_label),
                    'residue_ids' : _bytes_list_feature(protein.residue_ids),
                    'residue_names' : _bytes_list_feature(protein.residue_names),
                    'secondary_structure' : _bytes_list_feature(protein.secondary_structure)
                    
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                
                tfwriter.write(example.SerializeToString())
            
