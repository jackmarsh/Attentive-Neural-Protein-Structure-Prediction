class Visualiser():
    
    def __init__(self, n=2):
        self.n = n
    
    def show_structure(self, struct_id):
        structure = Bio.PDB.MMCIFParser(QUIET = True).get_structure(struct_id, "pdb/"+struct_id.lower()+".cif")

        return nv.show_biopython(structure)
    
    def plot_ramachandran(self, angles, dims=2):
        # TO DO: Make it work for both single seq and batch of seqs
        #        Get phis psis in two lists
        if dims==2:
            for phi_psi in phis_psis:
                plt.scatter(*zip(*phi_psi), s=1)

            plt.ylim(-np.pi, np.pi)
            plt.xlim(-np.pi,np.pi)
            plt.show()
        else:
            all_phis, all_psis = [], []
            
            #  NEED TO SPLIT PHIS AND PSIS INTO TWO LISTS HERE
            
            hist, xedges, yedges = np.histogram2d(all_phis, all_psis, bins=bins, range=[[-3.2, 3.2], [-3.2, 3.2]])

            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(111, projection='3d')

            # Construct arrays for the anchor positions of the 16 bars.
            xpos, ypos = np.meshgrid(xedges[:-1] + 1/bins, yedges[:-1] + 1/bins, indexing="ij")
            xpos = xpos.ravel()
            ypos = ypos.ravel()
            zpos = 0

            # Construct arrays with the dimensions for the 16 bars.
            dx = dy = (2/bins) * np.ones_like(zpos)
            dz = hist.ravel()

            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

            plt.show()
    
    def plot_attention(self, attention_map, input_tags = None, output_tags = None):
        """ plot_attention(alignments[:, i, :], input_tags, output_tags)"""
        
        
        try:
            # Remove padding if possible
            index = np.where(attention_map==0)

            attention_maps = np.hsplit(attention_map, [index[1][0]])
            attention_maps = np.vsplit(attention_maps[0], [index[1][0]*self.n]) 
            # make n more robust - config?

            attention_map = attention_maps[0]

            
        except:
            pass
        
        inp_len = attention_map.shape[1]
        out_len = attention_map.shape[0]
        # Plot the attention_map
        plt.clf()
        f = plt.figure(figsize=(20, 10))
        ax = f.add_subplot(1, 1, 1)

        # Add image
        i = ax.imshow(attention_map.T, interpolation='nearest', cmap='Blues')

        # Add colorbar
        cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
        cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
        cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)

        # Add labels
        ax.set_yticks(range(inp_len))
        if output_tags != None:
            ax.set_yticklabels(output_tags[::self.n][:inp_len], rotation=45)
            for index, label in enumerate(ax.yaxis.get_ticklabels()):
                if index % 1 != 0:
                    label.set_visible(False)
        else:
            for index, label in enumerate(ax.yaxis.get_ticklabels()):
                if index % 10 != 0:
                    label.set_visible(False)

        ax.set_xticks(range(out_len))
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        if input_tags != None:
            ax.set_xticklabels(input_tags[:out_len], rotation=45)
            for index, label in enumerate(ax.xaxis.get_ticklabels()):
                if index % 1 != 0:
                    label.set_visible(False)
        else:
            for index, label in enumerate(ax.xaxis.get_ticklabels()):
                if index % 10 != 0:
                    label.set_visible(False)

        ax.set_xlabel('Output Sequence')
        ax.set_ylabel('Input Sequence')

        # add grid and legend
        ax.grid()

        plt.show()
