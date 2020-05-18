import ase
import os
from ase.db import connect
from ase.visualize import view

db_name = 'wf_engy.db'

db = connect(db_name)
target_path = '/home/lxt/code/vasprun.xml/'
total_frame_cnt = 0
total_file_cnt =0
if __name__ == '__main__':
    # atoms = ase.io.read('/home/lxt/code/vasprun.xml/Mo2C(001)/Co/1xCo/vasprun.xml', index=':')
    # print(len(atoms))
    for base_path,folder_list,file_list in os.walk(target_path):
        for f in file_list:
            total_file_cnt += 1
            vsp_path = base_path + '/' + f
            atoms = ase.io.read(vsp_path, index=':')
            total_frame_cnt += len(atoms)
            for at in atoms:
                # print(at.get_potential_energy())
                db.write(at, data={'wf_en': float(at.get_potential_energy())})
    print(total_file_cnt)
    print(total_frame_cnt)
    # view(atoms)
