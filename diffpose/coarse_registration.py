import numpy as np
import cv2
import trimesh
import vedo
from stl import  mesh
import SimpleITK as sitk
import sys
import torch
sys.path.append('..')
from torchvision import transforms
from src.femur_models import PretrainedUNet
from src.metrics import jaccard,dice
def triangulate_faces(mesh):
    """
    Triangulate the faces of a given mesh.
    
    :param mesh: A trimesh.Trimesh object representing the mesh to be triangulated.
    :return: A new trimesh.Trimesh object with triangulated faces.
    """
    new_faces = []
    new_vertices = mesh.vertices.tolist()
    
    for face in mesh.faces:
        polygon = mesh.vertices[face]
        if len(face) == 3:
            new_faces.append(face)
        else:
            tri_faces = trimesh.triangles.triangulate_polygon(polygon)
            for tri in tri_faces:
                indices = []
                for vert in tri:
                    if list(vert) in new_vertices:
                        indices.append(new_vertices.index(list(vert)))
                    else:
                        new_vertices.append(list(vert))
                        indices.append(len(new_vertices) - 1)
                new_faces.append(indices)
                
    triangulated_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    return triangulated_mesh

def compute_covariance_matrix(points, center):
    cov_matrix = np.zeros((3, 3))
    num_points = len(points)

    for point in points:
        p = point - center
        cov_matrix += np.outer(p, p)

    cov_matrix /= float(num_points - 1)
    return cov_matrix

def rebuild_mesh_registration_by_pca(src_points, dst_points):
    src_points = np.array(src_points)
    dst_points = np.array(dst_points)
    
    src_center = np.mean(src_points, axis=0)
    dst_center = np.mean(dst_points, axis=0)

    src_cov = compute_covariance_matrix(src_points, src_center)
    dst_cov = compute_covariance_matrix(dst_points, dst_center)

    src_eigenvalues, src_eigenvectors = np.linalg.eigh(src_cov)
    dst_eigenvalues, dst_eigenvectors = np.linalg.eigh(dst_cov)

    mirror = [
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1]
    ]

    transforms = []
    for i in range(4):
        _dst_vector = dst_eigenvectors * mirror[i]

        src_translocation = np.eye(4)
        src_translocation[:3, :3] = src_eigenvectors
        src_translocation[:3, 3] = src_center

        dst_translocation = np.eye(4)
        dst_translocation[:3, :3] = _dst_vector
        dst_translocation[:3, 3] = dst_center

        curr_transform = np.dot(dst_translocation, np.linalg.inv(src_translocation))
        transforms.append(curr_transform)
    
    return transforms

def projection_rebuild(img, origin, config):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return  None
    
    w_step = [(config[9 + i] - config[i]) / img.shape[1] for i in range(3)]
    h_step = [(config[3 + i] - config[i]) / img.shape[0] for i in range(3)]
    
    contours = sorted(contours, key=lambda x: len(x), reverse=True)
    polygon = contours[0]
    
    contour_3d = []
    for curr_pt in polygon:
        pt = [
            config[0] + curr_pt[0][0] * w_step[0] + curr_pt[0][1] * h_step[0],
            config[1] + curr_pt[0][0] * w_step[1] + curr_pt[0][1] * h_step[1],
            config[2] + curr_pt[0][0] * w_step[2] + curr_pt[0][1] * h_step[2]
        ]
        if not contour_3d or np.linalg.norm(np.array(pt) - np.array(contour_3d[-1])) > 1:
            if len(contour_3d) < 2 or not np.allclose(np.cross(np.array(pt) - np.array(contour_3d[-2]), np.array(pt) - np.array(contour_3d[-1])), 0):
                contour_3d.append(pt)

    vertices = [origin] + contour_3d
    faces = [[0, i + 1, i + 2] for i in range(len(contour_3d) - 1)]
    faces.append([0, 1, len(contour_3d)])
    # # 计算三角形面
    # contour_3d_xy = np.array([[p[0], p[1]] for p in contour_3d])
    # tri = Delaunay(contour_3d_xy)
    # for simplex in tri.simplices:
    #     faces.append([simplex[0] + 1, simplex[1] + 1, simplex[2] + 1])
    
    mesh=trimesh.Trimesh(vertices=vertices, faces=faces)
    triangulated_mesh = triangulate_faces(mesh)
    #triangulated_mesh = triangulated_mesh.subdivide_to_size(max_edge=0.5)
    # 确保存闭性
    if not triangulated_mesh.is_volume:
        triangulated_mesh.fix_normals()
        triangulated_mesh.fill_holes()
        triangulated_mesh = triangulated_mesh.convex_hull
    return triangulated_mesh

def biplane_projection_intersection(ap_img, rl_img, ap_origin, rl_origin, ap_config, rl_config):
    if ap_img.dtype != np.uint8 or rl_img.dtype != np.uint8:
        return None
    mesh_ap = projection_rebuild(ap_img, ap_origin, ap_config)
    
    if mesh_ap is None:
        print("Failed to rebuild AP mesh.")
        return None
    mesh_rl = projection_rebuild(rl_img, rl_origin, rl_config)
    if mesh_rl is None:
        print("Failed to rebuild RL mesh.")
        return None
    #print(mesh_ap.vertices)
     # Convert trimesh objects to vedo Mesh objects
    ap_mesh_vedo = vedo.Mesh([mesh_ap.vertices, mesh_ap.faces])
    rl_mesh_vedo = vedo.Mesh([mesh_rl.vertices, mesh_rl.faces])
    # 计算交集
    try:
        intersection_vedo = ap_mesh_vedo.boolean('intersect', rl_mesh_vedo)
        if intersection_vedo is None:
            print("No intersection found.")
            return None

        intersection_mesh = trimesh.Trimesh(vertices=intersection_vedo.vertices, faces=intersection_vedo.cells)
    except Exception as e:
        print(f"Error during intersection calculation: {e}")
        return None
    intersection_mesh = triangulate_faces(intersection_mesh)
    intersection_mesh.export('interaction.stl', file_type='stl')
    #print(intersection_mesh.vertices)
    return intersection_mesh

#先测试CBCT的数据，输入的数据是nunpy类型
def get_segment_image(xray_ap_path, xray_rl_path,model,device):
    xray_ap_np=np.load(xray_ap_path).astype(np.float32)
    xray_ap_np=np.expand_dims(xray_ap_np,axis=0) #shape [1,1024,1024]     
    print(xray_ap_np.shape)
    xray_rl_np=np.load(xray_rl_path).astype(np.float32)
    xray_rl_np=np.expand_dims(xray_rl_np,axis=0) #shape [1,1024,1024]
    xray_ap_np_8bit = xray_ap_np - np.min(xray_ap_np)
    xray_ap_np_8bit = xray_ap_np_8bit / np.max(xray_ap_np_8bit)
    xray_ap_tensor=torch.from_numpy(xray_ap_np_8bit)-0.5
    xray_rl_np_8bit = xray_rl_np - np.min(xray_rl_np)
    xray_rl_np_8bit = xray_rl_np_8bit / np.max(xray_rl_np_8bit)
    xray_rl_np_tensor=torch.from_numpy(xray_rl_np_8bit)-0.5
    resize = transforms.Resize([512,512])
    xray_ap_resize=resize(xray_ap_tensor) #shape [1,512,512]
    xray_rl_resize=resize(xray_rl_np_tensor) #shape [1,512,512]
    with torch.no_grad():
        xray_ap_resize=torch.stack([xray_ap_resize])
        #print(xray_ap_resize.shape)
        xray_ap_resize=xray_ap_resize.to(device)
        out=model(xray_ap_resize)
        #print(out.shape)
        softmax=torch.nn.functional.log_softmax(out,dim=1)
        out=torch.argmax(softmax,dim=1)
        xray_ap=xray_ap_resize[0].to('cpu')
        out=out[0].to('cpu')
        
        xray_rl_resize=torch.stack([xray_rl_resize])
        xray_rl_resize=xray_rl_resize.to(device)
        out_rl=model(xray_rl_resize)
        softmax_rl=torch.nn.functional.log_softmax(out_rl,dim=1)
        out_rl=torch.argmax(softmax_rl,dim=1)
        xray_rl=xray_rl_resize[0].to('cpu')
        out_rl=out_rl[0].to('cpu')
    return xray_ap,out,xray_rl,out_rl