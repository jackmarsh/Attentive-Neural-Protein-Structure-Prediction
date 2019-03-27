class PDBWriter():
    """
    THINK IVE FUCKED SOMETHING UP IN HERE
    """
    def string_builder(self, i, atom_name, res_name, res_id, res_number, coord):
        """
        Return line of pdb file
        """
        line = 'ATOM' + ' '*2                                 # “ATOM” 1-4
        line += ' '*(5-len(str(i+1))) + str(i+1)              # Atom Serial Number 7-11
        line += ' '*2 + atom_name + ' '*(3-len(atom_name))    # Atom Name 13-16
        line += ' '                                           # Alternate Location Indicator 17
        line += res_name                                      # Residue name 18-20
        line += ' ' + res_id                                  # Chain ID 22
        line += ' '*(4-len(str(res_number))) + str(res_number)# Residue Sequence number 23-26
        line += ' '*4                                         # Code for insertions of residues 27
        line += ' '*(8-len(str(coord[0]))) + str(coord[0])    # X coord 31-38
        line += ' '*(8-len(str(coord[1]))) + str(coord[1])    # Y coord 39-46
        line += ' '*(8-len(str(coord[2]))) + str(coord[2])    # Z coord 47-54
        line += ' '*2 + '1.00'                                # Occupancy 55-60
        line += ' ' + '30.00'                                 # Temperature factor 61-66
        line += ' '*5 + ' '*5                                 # Segment identifier 73-76
        line += ' ' + atom_name[0]                            # Element Symbol 77-78
        line += '\n'
        
        return line
        
    def write_to_pdb(self, filename, residue_names, residue_ids, residue_coords):
        '''
        Write out calculated coords to PDB
        '''
        
        atom_names = ['N', 'CA', 'C']*(len(residue_names))
        with open(filename, 'w') as f:
            atom_coords = []
            for atom_coord in residue_coords:
                atom_coords.append([np.around(c, 3) for c in atom_coord])

            res_number = 0
            for i, coord in enumerate(atom_coords):

                if i % 3 == 0:
                    res_number += 1
                    
                if residue_names[res_number-1] == "XXX" or residue_ids[res_number-1] == "X":
                    break
                f.write(self.string_builder(i, atom_names[i], residue_names[res_number-1],
                                            residue_ids[res_number-1], res_number, coord))
            f.write("END")
                    
