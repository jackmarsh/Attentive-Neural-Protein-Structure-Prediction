class AminoVec():
    """
    Creates the amino acid embedding
    uses the Protein class
    
    """
    def __init__(self, proteins, size, window, epochs, force_build=False):
        # Word2Vec word embedding
        self.n = proteins[0].n
        
        self.corpus = [["0"*self.n]+ngram+["Z"*self.n] for protein in proteins for ngram in protein.ngrams]
        self.source = [protein.source+["Z"*self.n] for protein in proteins]
        
        self.size = size
        self.window = window
        self.epochs = epochs
        
        self.filename = '-'.join(str(i) for i in [self.n, self.size, self.window, self.epochs])
        
        if force_build:
            self.embedding = self._build_embedding()
        else:
            try:
                self.embedding = self.get_embedding()
            except:
                self.embedding = self._build_embedding()
    
    
    def get_embedding(self):
      
        X_wordvec = np.load(os.path.join(EMB_DIR, self.filename+".npy"))
        
        print("Retreived embedding...")
        print("n: %d - size: %d - window: %d - epochs: %d" 
              % (self.n, self.size, self.window, self.epochs))
        
        return X_wordvec
      
    def _build_embedding(self):
        print("Building embedding...")
        print("n: %d - size: %d - window: %d - epochs: %d" 
              % (self.n, self.size, self.window, self.epochs))
        
        model = Word2Vec(self.corpus,
                         size=self.size, # Dense vector size
                         window=self.window,
                         min_count=1, # Discount words with freq < 2
                         workers=10)
        model.train(self.source, total_words=len(preproc.source_vocab_dict), epochs=self.epochs)
        
        self.vocab = list(sorted(model.wv.vocab.keys())) 
        X_wordvec = model.wv[self.vocab]
        
        np.save(os.path.join(EMB_DIR, self.filename+".npy"), X_wordvec)
        print("Saved embedding")
        
        print("Creating metadata...")
        self._create_metadata("features.csv", self.filename+".tsv")
        print("Saved metadata")
        
        return X_wordvec
    
    def _create_metadata(self, feature_file, metadata_file):
        # Create feature dictionary
        features = {'0' : [0.0 for i in range(243)], 'Z' : [1.0 for i in range(243)]}
        with open(os.path.join(EMB_DIR, feature_file), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader, None)
            for row in reader:
                features[row[0]] = list(map(float, row[1:]))

        # Write metadata
        with open(os.path.join(EMB_DIR, metadata_file), 'w') as mf:
            header = "Ngram"+"\t{}"*243 + "\n"
            mf.write(header.format(*[str(label) for label in range(243)]))
            for ngram in self.vocab:
                line = ''.join(ngram)+"\t{}"*243 + "\n"
                mf.write(line.format(*[str(np.mean(value)) for value in zip(*[features[aa] for aa in ngram])]))
                
