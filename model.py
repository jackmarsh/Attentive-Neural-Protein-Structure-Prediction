class Model():
    
    def __init__(self, model_dir, params):
        
        self.model_dir = model_dir
        self.eval_dir = os.path.join(self.model_dir, "eval")
        self.train_writer = tf.summary.FileWriter(self.model_dir)
        self.eval_writer  = tf.summary.FileWriter(self.eval_dir)
        
        
        self.src_vocab_size = params['src_vocab_size']
        self.tgt_vocab_size = params['tgt_vocab_size']
        self.encoder_length = params['encoder_length']
        self.decoder_length = params['decoder_length']
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        self.embed_dim = params['embed_dim']
        self.num_units = params['num_units']
        self.enc_layers = params['enc_layers']
        self.att_layers = params['att_layers']
        self.dec_layers = params['dec_layers']
        self.num_gpus = params['num_gpus']
        self.dropout = params['dropout']
        self.l1_reg = params['l1_reg']
        self.l2_reg = params['l2_reg']
        self.attentive = params['attentive']
        self.beam_search = params['beam_search']
        self.beam_width = params['beam_width']
        self.wordvec = params['wordvec']
        self.r_angle_dict = params['r_angle_dict']
        self.max_gradient_norm = params['max_gradient_norm']
        
        self._build_train_graph()
        self._build_infer_graph()
        
        self.saver = tf.train.Saver()
        
    def _build_train_graph(self):
        tf.reset_default_graph()
        
        with tf.variable_scope("encoding") as encoding_scope:
            self.encoder_inputs = tf.placeholder(tf.int32, shape=(self.encoder_length, self.batch_size), name="encoder_inputs")
            W = tf.Variable(tf.constant(0.0, shape=[self.src_vocab_size, self.embed_dim]),trainable=False, name="embedding")
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.src_vocab_size, self.embed_dim])
            embedding_encoder = W.assign(self.embedding_placeholder)
            enc_embedded_inputs = tf.nn.embedding_lookup(embedding_encoder, self.encoder_inputs)
            
            self.source_length = tf.reduce_sum(tf.sign(self.encoder_inputs), axis=0)
            self.source_length = tf.cast(self.source_length, tf.int32)
            
            cells_fw, cells_bw = [], []
    
            if self.num_gpus > 0:
                for i in range(self.enc_layers):
                    cells_fw.append(tf.contrib.rnn.DeviceWrapper(
                                        tf.nn.rnn_cell.DropoutWrapper(
                                            tf.nn.rnn_cell.LSTMCell(self.embed_dim),
                                            input_keep_prob=self.dropout),
                                        "/gpu:%d" % (self.enc_layers % self.num_gpus)))
                    cells_bw.append(tf.contrib.rnn.DeviceWrapper(
                                        tf.nn.rnn_cell.DropoutWrapper(
                                            tf.nn.rnn_cell.LSTMCell(self.embed_dim),
                                            input_keep_prob=self.dropout),
                                        "/gpu:%d" % (self.enc_layers % self.num_gpus)))
            else:
                for i in range(self.enc_layers):
                    cells_fw.append(tf.nn.rnn_cell.ResidualWrapper(
                                        tf.nn.rnn_cell.DropoutWrapper(
                                            tf.nn.rnn_cell.LSTMCell(self.embed_dim),
                                            input_keep_prob=self.dropout)))
                    cells_bw.append(tf.nn.rnn_cell.ResidualWrapper(
                                        tf.nn.rnn_cell.DropoutWrapper(
                                            tf.nn.rnn_cell.LSTMCell(self.embed_dim),
                                            input_keep_prob=self.dropout)))
                    
            stacked_fw = tf.contrib.rnn.MultiRNNCell(cells_fw)
            stacked_bw = tf.contrib.rnn.MultiRNNCell(cells_bw)


            bi_outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                                                cell_fw=stacked_fw,
                                                cell_bw=stacked_bw,
                                                inputs=enc_embedded_inputs,
                                                dtype=tf.float32,
                                                time_major=True)
            
            
            # Get state of final layer
            encoder_states = []
            for i in range(self.enc_layers):
                if isinstance(fw_state[i], tf.nn.rnn_cell.LSTMStateTuple):
                    encoder_state_c = tf.concat((fw_state[i].c, bw_state[i].c), 1, name='bidirectional_concat_c')
                    encoder_state_h = tf.concat((fw_state[i].h, bw_state[i].h), 1, name='bidirectional_concat_h')
                    encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
                elif isinstance(fw_state[i], tf.Tensor):
                    encoder_state = tf.concat((fw_state[i], bw_state[i]), 1, name='bidirectional_concat')
                encoder_states.append(encoder_state)
            
            self.encoder_output = tf.concat(bi_outputs, -1)
            self.encoder_state = encoder_states[-1]
        
        with tf.variable_scope("decoding") as decoding_scope:    
            self.decoder_inputs = tf.placeholder(tf.int32, shape=(self.decoder_length, self.batch_size), name="decoder_inputs")
            self.decoder_lengths = tf.placeholder(tf.int32, shape=(self.batch_size), name="decoder_length")
            self.embedding_decoder = tf.get_variable("embedding_decoder", [self.tgt_vocab_size, self.embed_dim])
            
            dec_embedded_inputs = tf.nn.embedding_lookup(self.embedding_decoder, self.decoder_inputs)

            target_length = tf.reduce_sum(tf.sign(self.decoder_inputs), axis=0)
            target_length = tf.cast(target_length, tf.int32)

            self.projection_layer = tf.layers.Dense(self.tgt_vocab_size, use_bias=False)

            # DECODER CELL
            self.decoder_cell = tf.nn.rnn_cell.LSTMCell(2*self.num_units)
            self.stacked_decoder = tf.contrib.rnn.MultiRNNCell([self.decoder_cell for _ in range(self.dec_layers)])

            helper = tf.contrib.seq2seq.TrainingHelper(dec_embedded_inputs, self.decoder_lengths, time_major=True)

            # ATTENTION
            if self.attentive:
                self.encoder_output = tf.layers.batch_normalization(self.encoder_output)

                self.encoder_output = tf.transpose(self.encoder_output, [1, 0, 2])
                
                # ATTENTION (TRAINING)
                # Shares weights with inference
                with tf.variable_scope('shared_attention_mechanism') as self.shared_attention_mechanism: 
                    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                                                num_units=self.num_units,
                                                memory=self.encoder_output,
                                                memory_sequence_length=self.source_length)

                cells = []
                for i in range(self.att_layers):
                    self.attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                                                self.stacked_decoder,
                                                attention_mechanism,
                                                attention_layer_size=self.num_units,
                                                alignment_history=True)
                    cells.append(self.attention_cell)
                
                self.attention_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                
                self.initial_state = self.attention_cell.zero_state(self.batch_size, tf.float32)
                
                decoder = tf.contrib.seq2seq.BasicDecoder(
                                            self.attention_cell,
                                            helper,
                                            self.initial_state,
                                            output_layer=self.projection_layer) # maybe get rid of proj layer

            else:
                self.initial_state = self.encoder_state
                decoder = tf.contrib.seq2seq.BasicDecoder(
                                            self.stacked_decoder,
                                            helper,
                                            self.initial_state,
                                            output_layer=self.projection_layer) # maybe get rid of proj layer
            
            # DECODER
            with tf.variable_scope('shared_decoder') as self.shared_decoder:
                decoder_outputs, decoder_states, _ = tf.contrib.seq2seq.dynamic_decode(decoder)


        with tf.name_scope("optimization"):
            # LOSS
            self.logits = decoder_outputs.rnn_output

            self.target_labels = tf.placeholder(tf.int32, shape=(self.batch_size, self.decoder_length))

            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_labels, logits=self.logits) # Switch to sampled for speed

            loss_mask = tf.sequence_mask(target_length, self.decoder_length)
            crossent = crossent * tf.to_float(loss_mask)
            self.loss = tf.reduce_mean(crossent)

            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            # Apply regularisation to wieghts 
            params = tf.trainable_variables()
            
            l1_l2_reg = tf.contrib.layers.l1_l2_regularizer(scale_l1=self.l1_reg, scale_l2=self.l2_reg)
            
            l1_l2_reg_pen = tf.contrib.layers.apply_regularization(l1_l2_reg, params)
            
            self.loss = self.loss + l1_l2_reg_pen # TODO: do we apply to biases too?
            
            gradients = tf.gradients(self.loss, params)
            # CLIP GRADIENTS
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

            # OPTIMIZER
            optimiser = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimiser.apply_gradients(zip(clipped_gradients, params),
                                                      global_step=self.global_step)

        with tf.name_scope("summaries"):
            self.loss_summary = tf.placeholder(tf.float32, shape=())
            self.acc_summary = tf.placeholder(tf.float32, shape=())
            
            tf.summary.scalar("loss", self.loss_summary)
            tf.summary.scalar("accuracy", self.acc_summary)
    
    def _build_infer_graph(self):
        with tf.name_scope("inference"):
            
            maximum_iterations = tf.round(tf.reduce_max(self.decoder_length))

            if self.beam_search:
                
                # Beam Search
                # Replicate encoder infos beam_width times
                tiled_encoder_output = tf.contrib.seq2seq.tile_batch(self.encoder_output,
                                                                     multiplier=self.beam_width)
                tiled_source_length = tf.contrib.seq2seq.tile_batch(self.source_length,
                                                                    multiplier=self.beam_width)
                tiled_encoder_state = tf.contrib.seq2seq.tile_batch(self.encoder_state,
                                                                    multiplier=self.beam_width)
                
                                                                    
                # ATTENTION (PREDICTING)
                with tf.variable_scope(self.shared_attention_mechanism, reuse=True):
                    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                                                num_units=self.num_units,
                                                memory=tiled_encoder_output,
                                                memory_sequence_length=tiled_source_length)
                
                cells = []
                for i in range(self.att_layers):
                    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                                                self.stacked_decoder,
                                                attention_mechanism,
                                                attention_layer_size=self.num_units,
                                                alignment_history=True)
                    cells.append(decoder_cell)
                
                attention_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                
                #### I THINK I CAN STACK ATTENTION TOO
                
                tiled_initial_state = attention_cell.zero_state(self.batch_size * self.beam_width, tf.float32)

                # Define a beam-search decoder
                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                                            cell=attention_cell,
                                            embedding=self.embedding_decoder,
                                            start_tokens=tf.fill([self.batch_size], 1),
                                            end_token=self.tgt_vocab_size-1, # TODO: is this right
                                            initial_state=tiled_initial_state,
                                            beam_width=self.beam_width,
                                            output_layer=self.projection_layer,
                                            length_penalty_weight=0.0)

                # Dynamic decoding
                with tf.variable_scope(self.shared_decoder, reuse=True):
                    outputs, states, _ = tf.contrib.seq2seq.dynamic_decode(
                                            inference_decoder,
                                            maximum_iterations=maximum_iterations)
                
                self.translation = outputs.predicted_ids[:, :, 0]
                c_state = states.cell_state[0] # TODO: is this the right layer to use
                self.attention_matrix = tf.transpose(c_state.alignment_history.stack(), [1,0,2])[:, :, 0]
            else:
                #Inference
                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                                        self.embedding_decoder,
                                        tf.fill([self.batch_size], 1),
                                        self.tgt_vocab_size) # need to check eos tag

                inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                                        self.attention_cell,
                                        inference_helper,
                                        self.initial_state,
                                        output_layer=self.projection_layer)

                outputs, states, _ = tf.contrib.seq2seq.dynamic_decode(
                                        inference_decoder,
                                        maximum_iterations=maximum_iterations)

                self.translation = outputs.sample_id
                c_state = states.cell_state[0] # TODO: is this the right layer to use
                self.attention_matrix = tf.transpose(c_state.alignment_history.stack(), [1,0,2])
        
    def input_fn(self, filenames):
        # TODO: put this in a name scope and fix graph thing
        def extract_fn(data_record):
            features = {
                'code'  : tf.FixedLenFeature([], tf.string),
                'source': tf.VarLenFeature(tf.int64),
                'target': tf.VarLenFeature(tf.int64),
                'label' : tf.VarLenFeature(tf.int64),
                'residue_ids' : tf.VarLenFeature(tf.string),
                'residue_names' : tf.VarLenFeature(tf.string),
                'secondary_structure' : tf.VarLenFeature(tf.string),
                
            }

            sample = tf.parse_single_example(data_record, features)

            sample['source'] = tf.sparse.to_dense(sample['source'])
            sample['target'] = tf.sparse.to_dense(sample['target'])
            sample['label']  = tf.sparse.to_dense(sample['label'])
            sample['residue_ids'] = tf.sparse.to_dense(sample['residue_ids'], default_value='X')
            sample['residue_names'] = tf.sparse.to_dense(sample['residue_names'], default_value='XXX')
            sample['secondary_structure']  = tf.sparse.to_dense(sample['secondary_structure'], default_value='X')
            
            return sample

        def trans_fn(data_record):

            # Transpose the source and target for efficiency
            data_record['source'] = tf.transpose(data_record['source'])
            data_record['target'] = tf.transpose(data_record['target'])

            return data_record

        # Initialize all tfrecord paths
        dataset = tf.data.TFRecordDataset(filenames)

        dataset = dataset.map(extract_fn) \
                         .shuffle(buffer_size=2048) \
                         .batch(self.batch_size, drop_remainder=True) \
                         .map(trans_fn)

        iterator = dataset.make_one_shot_iterator()
        next_elem = iterator.get_next()
        
        return next_elem
        
    def train(self, sess, data):
        
        start_time = time.time()
        
        summary = tf.summary.merge_all()
        
        losses, accs = [], []
        fetches = {
            'train_op' : self.train_op,
            'loss' : self.loss,
            'logits' : self.logits
        }
        try:
            while True:
                
                record = sess.run(data)
                
                feed_dict = {
                    self.encoder_inputs:   record['source'],
                    self.decoder_inputs:   record['target'],
                    self.target_labels:    record['label'],
                    self.decoder_lengths:  np.ones((self.batch_size), dtype=int) * self.decoder_length,
                    self.embedding_placeholder: self.wordvec
                }

                result = sess.run(fetches, feed_dict=feed_dict)
                
                acc = np.mean(result['logits'].argmax(axis=-1) == record['label']) # TODO: in the graph instead
                losses.append(result['loss'])
                accs.append(acc)
        except tf.errors.OutOfRangeError:
            pass
        
        loss_value = np.mean(losses)
        acc_value  = np.mean(accs)
        
        # Logging
        global_step = tf.train.global_step(sess, self.global_step)
        self.train_writer.add_graph(sess.graph, global_step=global_step)
        
        summary_feed = {
            self.loss_summary : loss_value,
            self.acc_summary : acc_value
        }
        s = sess.run(summary, feed_dict=summary_feed)
        self.train_writer.add_summary(s, global_step)
        
        self.saver.save(sess, os.path.join(self.model_dir,"model.ckpt"), global_step)
        
        end_time = time.time() - start_time
        print(" %f - Loss: %f - Accuracy: %f - Training" % (end_time, loss_value, acc_value))
        
    def evaluate(self, sess, data):
        
        start_time = time.time()
        
        summary = tf.summary.merge_all()
        
        losses, accs = [], []
        fetches = {
            'loss' : self.loss,
            'logits' : self.logits
        }
        try:
            while True:
                record = sess.run(data)
                
                feed_dict = {
                    self.encoder_inputs:   record['source'],
                    self.decoder_inputs:   record['target'],
                    self.target_labels:    record['label'],
                    self.decoder_lengths:  np.ones((self.batch_size), dtype=int) * self.decoder_length,
                    self.embedding_placeholder: self.wordvec
                }

                result = sess.run(fetches, feed_dict=feed_dict)
                
                acc = np.mean(result['logits'].argmax(axis=-1) == record['label']) # TODO: in the graph instead
                losses.append(result['loss'])
                accs.append(acc)
        except tf.errors.OutOfRangeError:
            pass
        
        loss_value = np.mean(losses)
        acc_value  = np.mean(accs)
        
        # Logging
        global_step = tf.train.global_step(sess, self.global_step)
        self.eval_writer.add_graph(sess.graph, global_step=global_step)
        
        summary_feed = {
            self.loss_summary : loss_value,
            self.acc_summary : acc_value
        }
        s = sess.run(summary, feed_dict=summary_feed)
        self.eval_writer.add_summary(s, global_step)

        end_time = time.time() - start_time
        print(" %f - Loss: %f - Accuracy: %f - Validation" % (end_time, loss_value, acc_value))
    
    def infer(self, sess, data):
        
        fetches = {
            'translation' : self.translation,
            'attention' : self.attention_matrix
        }
        predictions = []
        try:
            while True:
                record = sess.run(data)
                feed_dict = {
                    self.encoder_inputs: record['source'],
                    self.decoder_inputs: record['target'], # TODO: Shouldnt need to put this in tbh
                    self.decoder_lengths:  np.ones((self.batch_size), dtype=int) * self.decoder_length,
                    self.embedding_placeholder: self.wordvec
                }

                result = sess.run(fetches, feed_dict=feed_dict)
                
                for i in range(self.batch_size):
                    prediction = {
                        'code' : record['code'][i].decode('utf-8'),
                        'label' : record['label'][i], # TODO: while we figure out pnerf
                        'translation' : result['translation'][i],
                        'attention' : result['attention'][i],
                        'residue_ids' : [res_id.decode('utf-8') for res_id in record['residue_ids'][i]],
                        'residue_names' : [res_name.decode('utf-8') for res_name in record['residue_names'][i]],
                        'secondary_structure' : [ss.decode('utf-8') for ss in record['secondary_structure'][i]],
                    }
                    predictions.append(prediction)
        
        except tf.errors.OutOfRangeError:
            pass    
        
        return predictions
    
    def project_embeddings(self, sess, metadata_file):
        embedding = tf.Variable(self.wordvec, name="embedding")    
        sess.run(embedding.initializer)
        
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embedding.name
        embedding_conf.metadata_path = "../../"+os.path.join(EMB_DIR,metadata_file + ".tsv")
        projector.visualize_embeddings(self.train_writer, config)
        
        self.saver.save(sess, os.path.join(self.model_dir,"model.ckpt"), 0)
        
    def initialise(self, sess):
        # TODO: Handle this error properly
        # if states is none it wont have all_model_checkpoint_paths
        try:
            states = tf.train.get_checkpoint_state(self.model_dir)
            checkpoint_paths = states.all_model_checkpoint_paths
            self.saver.recover_last_checkpoints(checkpoint_paths)
            self.saver.restore(sess, self.saver.last_checkpoints[-1])
        except:
            sess.run(tf.global_variables_initializer())
        
        # TODO: check if below even does anything
        global_step = tf.train.global_step(sess, self.global_step)
        self.train_writer.add_graph(sess.graph, global_step=global_step)
        self.eval_writer.add_graph(sess.graph, global_step=global_step)
    
    
