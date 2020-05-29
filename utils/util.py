import os
import pickle
import numpy as np

def load_pickle_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def write_pickle_file(pkl_path, data_dict):
    with open(pkl_path, 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=2)

        
def save_to_obj(verts, faces, path):
    """
    Save the mesh into .obj file.
    Parameter:
    ---------
    path: Path to save.
    """

    with open(path, 'w') as fp:
        fp.write('g\n')
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
        fp.write('s off\n')


def load_obj(obj_file):
    with open(obj_file, 'r') as fp:
        verts = []
        faces = []
        vts = []
        vns = []
        faces_vts = []
        faces_vns = []

        for line in fp:
            line = line.rstrip()
            line_splits = line.split()
            if len(line_splits) == 0:
                continue
            prefix = line_splits[0]

            if prefix == 'v':
                verts.append(np.array([line_splits[1], line_splits[2], line_splits[3]], dtype=np.float32))

            elif prefix == 'vn':
                vns.append(np.array([line_splits[1], line_splits[2], line_splits[3]], dtype=np.float32))

            elif prefix == 'vt':
                vts.append(np.array([line_splits[1], line_splits[2]], dtype=np.float32))

            elif prefix == 'f':
                f = []
                f_vt = []
                f_vn = []
                for p_str in line_splits[1:4]:
                    p_split = p_str.split('/')
                    f.append(p_split[0])
                    f_vt.append(p_split[1])
                    f_vn.append(p_split[2])

                faces.append(np.array(f, dtype=np.int32) - 1)
                faces_vts.append(np.array(f_vt, dtype=np.int32) - 1)
                faces_vns.append(np.array(f_vn, dtype=np.int32) - 1)

            else:
                #raise ValueError(prefix)
                continue

        obj_dict = {
            'vertices': np.array(verts, dtype=np.float32),
            'faces': np.array(faces, dtype=np.int32),
            'vts': np.array(vts, dtype=np.float32),
            'vns': np.array(vns, dtype=np.float32),
            'faces_vts': np.array(faces_vts, dtype=np.int32),
            'faces_vns': np.array(faces_vns, dtype=np.int32)
        }

        return obj_dict


def normalize_vertex(vertex):
    '''
    normalize vertex to -1 1, input vertex type is numpy array
    '''
    norm_vertex = 2 * (vertex - np.min(vertex)) / (np.max(vertex) - np.min(vertex)) + (-1)
    return norm_vertex


def get_obj_vertex_open3d(obj_file):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh_vertices = np.asarray(mesh.vertices)
    return mesh_vertices  

    
def get_obj_vertex_ali(file):
    '''
    get obj vertex, some obj file can not be loaded through open3d or trimesh
    '''
    with open(file, 'r') as f:
        vertex_group = []
        part_vertex = []
        last_fisrt = ''

        lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            split_line = line.split(' ')
            curr_first = split_line[0]
            if curr_first != last_fisrt:
                if part_vertex != []: vertex_group += part_vertex
                part_vertex = []
            if 'v' == curr_first:
                try:
                    vertex = [float(split_line[-3]), float(split_line[-2]), float(split_line[-1])]
                except:
                    continue
                    pdb.set_trace()
                    vertex = [float(split_line[-3]), float(split_line[-2]), float(split_line[-1])]
                part_vertex.append(vertex)

            last_fisrt = curr_first

        # remove shadow vertex
        if len(vertex_group[-1]) == 4 and len(vertex_group) != 0:
            vertex_group.pop()
    
    return np.array(vertex_group)


def replace_and_save_obj(input_vertex, raw_obj_file, save_file):
    all_input_vertex = []
    for i, input_part_vertex in enumerate(input_vertex):
        all_input_vertex.append(input_part_vertex)
     
    valid_lines = []
    with open(raw_obj_file, 'r') as f:

        v_id = 0
        lines = f.readlines()
        try:
            for i, line in enumerate(lines):
                split_line = line.split(' ')
                curr_first = split_line[0]

                # check if shadow exist in line
                for element in split_line:
                    if 'shadow' in element:
                        break
                if 'shadow' in split_line[-1] or 'Shadow' in split_line[-1] or 'SHADOW' in split_line[-1]:
                    break

                if 'v' == curr_first:
                    curr_vertex = all_input_vertex[v_id]
                    curr_line = 'v  ' + str(curr_vertex[0]) + ' ' + str(curr_vertex[1]) + ' ' + str(curr_vertex[2]) + '\n'
                    v_id += 1
                else:
                    curr_line = line
                valid_lines.append(curr_line)
            f.close()
        except:
            print(raw_obj_file)
    
    if len(valid_lines) < 50:
        return None
    
    with open(save_file, 'w') as f:
        for valid_line in valid_lines:
            f.write(valid_line)
        f.close()