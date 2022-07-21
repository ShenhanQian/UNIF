import logging
from tkinter import W

import numpy as np
import matplotlib
import cv2 as cv
import torch
import torch.nn.functional as F
from skimage import measure
import trimesh
import pyrender
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras, DirectionalLights, Materials

from utils.mise import MISE
from utils.sdf_renderer import sphere_tracing, phong_shading
from utils.geometry import gradient


class Visualizer3D(object):
    """Visualizer3D of 3D implicit representations."""
    def __init__(self, resolution_mc, resolution_render, mc_value, gradient_direction, uniform_grid=True, connected=False, cam_position=[0., 0., 2.]):
        super().__init__()

        self.resolution_mc = resolution_mc
        self.resolution_render = resolution_render
        self.mc_value = mc_value
        self.gradient_direction = gradient_direction
        self.uniform_grid = uniform_grid
        self.connected = connected
        self.cam_position = cam_position

    @torch.no_grad()
    def get_grid_pred(self, decoder, priors_batch=None, pts=None, mise=True):
        if len(priors_batch) > 0:
            B = priors_batch[list(priors_batch.keys())[0]].shape[0]
        else:
            if pts is not None:
                B = pts.shape[0]
            else:
                B = 1
        
        if mise:  # selective point inference
            # set up grid
            if pts is not None:
                bbox_min = pts.reshape(-1, 3).min(0)[0]
                bbox_max = pts.reshape(-1, 3).max(0)[0]
            else:
                bbox_min = torch.zeros(3).cuda() - 1
                bbox_max = torch.zeros(3).cuda() + 1

            box_size = (bbox_max - bbox_min) * 1.1
            box_center = (bbox_min + bbox_max) / 2
            box_min = box_center - box_size / 2

            # Evaluating points
            grid_pred_batch = []
            levels = 4
            init_resolution = self.resolution_mc >> levels
            assert init_resolution<<levels == self.resolution_mc
            for i in range(B):
                priors = dict(map(lambda x: (x[0], x[1][i:i+1]), priors_batch.items()))

                mesh_extractor = MISE(init_resolution, levels, self.mc_value)
                points = mesh_extractor.query()
                while points.shape[0] != 0:
                    # Query points
                    pointsf = torch.tensor(points).float().cuda()
                    # Normalize to bounding box
                    pointsf = pointsf / mesh_extractor.resolution * box_size + box_min
                    # Evaluate model and update
                    values = []
                    for i, pnts in enumerate(torch.split(pointsf, 10000, dim=0)):  # to avoid OOM
                        values_block = decoder(pnts.cuda().unsqueeze(0), **priors)
                        values.append(values_block.detach().cpu().numpy())
                    values = np.concatenate(values, axis=1).reshape(-1).astype(np.float64)

                    mesh_extractor.update(points, values)
                    points = mesh_extractor.query()

                value_grid = mesh_extractor.to_dense()
                grid_pred_batch.append(torch.tensor(value_grid).unsqueeze(0))
            grid_pred = torch.cat(grid_pred_batch, dim=0).unsqueeze(-1)  # (B, X, Y, Z, 1)

            x = torch.linspace(-box_size[0]/2,box_size[0]/2, self.resolution_mc+1) + box_center[0].item()
            y = torch.linspace(-box_size[1]/2,box_size[1]/2, self.resolution_mc+1) + box_center[1].item()
            z = torch.linspace(-box_size[2]/2,box_size[2]/2, self.resolution_mc+1) + box_center[2].item()
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            grid_pts = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1), zz.unsqueeze(-1)], dim=-1
                ).float().unsqueeze(0).repeat_interleave(B, dim=0)  # (B, X, Y, Z, 3)
            return grid_pts.numpy(), grid_pred.numpy()
        else:  # dense point inference
            # Generating grid
            if self.uniform_grid:
                grid = self.get_grid_uniform()
            else:
                grid = self.get_grid(pts.reshape(-1, pts.shape[-1]))
            grid_pts = grid['grid_pts'].unsqueeze(0).repeat(B, 1, 1)  # (Batch, Num_pts, 3)

            # Evaluating points
            grid_pred = []
            for i, pnts in enumerate(torch.split(grid_pts, 100000, dim=1)):
                grid_pred_block = decoder(pnts.cuda(), **priors_batch)
                grid_pred.append(grid_pred_block.detach().cpu().numpy())
            grid_pred = np.concatenate(grid_pred, axis=1)

            # Collecting results
            grid_pred = grid_pred.reshape(B, grid['xyz'][0].shape[0], grid['xyz'][1].shape[0], grid['xyz'][2].shape[0], 1)  # (B, X, Y, Z, 1)
            grid_pts = grid_pts.reshape(B, grid['xyz'][0].shape[0], grid['xyz'][1].shape[0], grid['xyz'][2].shape[0], 3).detach().cpu().numpy()  # (B, X, Y, Z, 3)
            
            return grid_pts, grid_pred
    
    @torch.no_grad()
    def color_mesh_by_part(self, decoder, meshes, priors_batch=None):
        meshes_colored = []
        for idx, mesh in enumerate(meshes):
            if mesh is None:
                meshes_colored.append(None)
            else:
                pts = torch.tensor(mesh.vertices).unsqueeze(0).float()
                priors = dict(map(lambda x: (x[0], x[1][idx:idx+1]), priors_batch.items()))
                part_index = []
                for i, pts_block in enumerate(torch.split(pts, 10000, dim=1)):
                    _, part_index_block, _, _ = decoder(pts_block.cuda(), return_parts=True, **priors)
                    part_index.append(part_index_block.detach().cpu())
                part_index = torch.cat(part_index, dim=1).squeeze(0).squeeze(-1).numpy()

                mesh_colored = mesh.copy()
                mesh_colored.visual.vertex_colors = (self.query_color(part_index) * 255).astype(np.uint8)
                
                meshes_colored.append(mesh_colored)
        return meshes_colored
    
    def query_color(self, index):
        cmap = matplotlib.cm.get_cmap('tab20')
        index %= 20
        return cmap(index/19.)

    def get_mesh_from_grid(self, grid_pts, grid_pred):
        if (not (np.min(grid_pred) > self.mc_value or np.max(grid_pred) < self.mc_value)):
            # Marching cubes
            verts, faces, vertex_normals, values = measure.marching_cubes(
                volume=grid_pred.squeeze(-1),
                level=self.mc_value,
                spacing=(
                    grid_pts[1, 0, 0, 0] - grid_pts[0, 0, 0, 0],
                    grid_pts[0, 1, 0, 1] - grid_pts[0, 0, 0, 1],
                    grid_pts[0, 0, 1, 2] - grid_pts[0, 0, 0, 2],),
                gradient_direction=self.gradient_direction,
            )
            verts = verts + grid_pts[None, 0, 0, 0]

            # Constructing Trimesh
            # mesh = trimesh.Trimesh(verts, faces, vertex_normals=vertex_normals, vertex_colors=values)
            mesh = trimesh.Trimesh(verts, faces)
            if self.connected:
                connected_comp = mesh.split(only_watertight=False)
                max_area = 0
                max_comp = None
                for comp in connected_comp:
                    if comp.area > max_area:
                        max_area = comp.area
                        max_comp = comp
                mesh = max_comp
            
            mesh.visual.vertex_colors = (128, 128, 128)
            return mesh
        else:
            return None

    def get_mesh_of_coord_system(self, Jtr, Btr, Brot):
        scene = trimesh.scene.Scene()

        transfm = np.concatenate([np.pad(Brot, ((0,0), (0,1), (0,0))), np.pad(Btr[:, :, None], ((0,0), (0,1), (0,0)), constant_values=1)], axis=2)
        for t in transfm:
            scene.add_geometry(trimesh.creation.axis(transform=t, axis_radius=0.006, axis_length=0.06, origin_size=0., origin_color=None))

        for t in Jtr:
            scene.add_geometry(trimesh.primitives.Sphere(radius=0.015, center=t))
        return self.as_mesh(scene)
    
    def as_mesh(self, scene_or_mesh):
        """
        Convert a possible scene to a mesh.

        If conversion occurs, the returned mesh has only vertex and face data.
        """
        if isinstance(scene_or_mesh, trimesh.Scene):
            if len(scene_or_mesh.geometry) == 0:
                mesh = None  # empty scene
            else:
                # we lose texture information here
                mesh = trimesh.util.concatenate(
                    tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces, vertex_colors=g.visual.vertex_colors)
                        for g in scene_or_mesh.geometry.values()))
        else:
            assert(isinstance(mesh, trimesh.Trimesh))
            mesh = scene_or_mesh
        return mesh

    def render_mesh(self, mesh, pointlight=False):
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene()
        scene.add(mesh)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        cam_x, cam_y, cam_z = self.cam_position
        camera_pose = np.array([
            [1., 0., 0., cam_x],
            [0., 1., 0., cam_y],
            [0., 0., 1., cam_z],
            [0., 0., 0., 1.],
         ])
        scene.add(camera, pose=camera_pose)

        if pointlight:
            light = pyrender.PointLight(color=np.ones(3), intensity=23.0)
            light_pose = camera_pose.copy()
            light_pose[1, 3] += 1.
            scene.add(light, pose=light_pose)
        else:
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
            scene.add(light)

        r = pyrender.OffscreenRenderer(self.resolution_render, self.resolution_render)
        color, depth = r.render(scene)
        return color
    
    @torch.no_grad()
    def render_sdf(self, sdf, priors=None, colored=True, device='cuda'):
        cam_x, cam_y, cam_z = self.cam_position
        R, T = look_at_view_transform(
            # dist=(cam_z,),
            # elev=(0.,),
            # azim=(0.,),
            degrees=False,
            eye=((cam_x, cam_y, cam_z),),
            at=((0, cam_y, 0),),
            device=device,
        )
        cameras = PerspectiveCameras(
            focal_length=np.sqrt(3),
            R=R, T=T, 
            device=device,
        )

        y = torch.linspace(1., -1., self.resolution_render, device=device)
        x = torch.linspace(1., -1., self.resolution_render, device=device)
        y, x = torch.meshgrid(y, x, indexing='ij')
        z = torch.ones_like(y)
        positions = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
        positions = cameras.unproject_points(positions, world_coordinates=True)
        
        # directions = positions - cameras.get_camera_center().reshape(-1, 1, 3)
        # directions = F.normalize(directions, dim=-1)

        # Sphere tracing
        pnts_list = []
        cvgs_list = []
        for i, pnts in enumerate(torch.split(positions, 10000, dim=-2)):  # to avoid OOM
            dirs = pnts - cameras.get_camera_center()
            dirs = F.normalize(dirs, dim=-1)

            pnts, converged = sphere_tracing(
                signed_distance_function=sdf, 
                positions=pnts, 
                directions=dirs, 
                num_iterations=100, 
                convergence_threshold=1e-3,
                filter_threshold=1.0,
                priors=priors,
            )

            pnts_list.append(pnts)
            cvgs_list.append(converged)
        
        positions = torch.cat(pnts_list, dim=-2)
        converged = torch.cat(cvgs_list, dim=-2)
        positions = torch.where(converged, positions, torch.zeros_like(positions))

        # Normals computation with autograd
        with torch.enable_grad():
            positions.requires_grad_()
            normals = []
            part_index = []
            for i, pnts in enumerate(torch.split(positions, 10000, dim=-2)):  # to avoid OOM
                pred_i, part_i, _, _ = sdf(pnts.unsqueeze(0), return_parts=True, **priors)
                normals_i = gradient(pnts, pred_i)

                normals.append(normals_i.detach().squeeze(0))
                part_index.append(part_i.detach().squeeze(0))
        part_index = torch.cat(part_index, dim=-2)
        normals = torch.cat(normals, dim=-2)
        normals = F.normalize(normals, dim=-1)

        # Shading (silhouette) ------------------------------------------------
        # renderings = converged

        # Shading (normals) ---------------------------------------------------
        # renderings = normals * 0.5 + 0.5
        # renderings = torch.where(converged, renderings, torch.ones_like(renderings))

        # Shading (color) -----------------------------------------------------
        lights = DirectionalLights(
            ambient_color=torch.full((1, 3), 0.3),
            diffuse_color=torch.full((1, 3), 0.9),
            specular_color=((0.01, 0.01, 0.01),),
            direction=((0., 0., 1.),),
            device=device,
        )
        materials = Materials(device=device)
        if colored:
            textures = torch.tensor(self.query_color(part_index.squeeze(-1).cpu().numpy()))[..., :3].to(device)
        else:
            textures = torch.ones_like(positions)

        renderings = phong_shading(positions, normals, textures, cameras, lights, materials)
        renderings = torch.where(converged, renderings, torch.ones_like(renderings))
        renderings = torch.min(renderings, torch.ones_like(renderings))
        # ---------------------------------------------------------------------

        renderings = renderings.reshape(self.resolution_render, self.resolution_render, -1)
        renderings = (renderings * 255).cpu().numpy().astype(np.uint8)
        return renderings

    def get_grid(self, points):
        eps = 0.
        input_min = torch.min(points, dim=0)[0].squeeze().cpu().numpy()
        input_max = torch.max(points, dim=0)[0].squeeze().cpu().numpy()
        bounding_box = input_max - input_min
        shortest_axis = np.argmin(bounding_box)
        if (shortest_axis == 0):
            x = np.linspace(input_min[shortest_axis] - eps,
                            input_max[shortest_axis] + eps, self.resolution_mc)
            length = np.max(x) - np.min(x)
            y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
            z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        elif (shortest_axis == 1):
            y = np.linspace(input_min[shortest_axis] - eps,
                            input_max[shortest_axis] + eps, self.resolution_mc)
            length = np.max(y) - np.min(y)
            x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
            z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        elif (shortest_axis == 2):
            z = np.linspace(input_min[shortest_axis] - eps,
                            input_max[shortest_axis] + eps, self.resolution_mc)
            length = np.max(z) - np.min(z)
            x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
            y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

        xx, yy, zz = torch.meshgrid(
                torch.tensor(x), 
                torch.tensor(y), 
                torch.tensor(z), indexing='ij')
        grid_points = torch.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T.float()
    
        return {"grid_pts": grid_points,
                "shortest_axis_length": length,
                "xyz": [x, y, z],
                "shortest_axis_index": shortest_axis}

    def get_grid_uniform(self):
        x = np.linspace(-1.2,1.2, self.resolution_mc)
        y = x
        z = x

        xx, yy, zz = torch.meshgrid(
            torch.tensor(x), 
            torch.tensor(y), 
            torch.tensor(z), indexing='ij')
        grid_points = torch.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T.float()

        return {"grid_pts": grid_points,
                "shortest_axis_length": 2.4,
                "xyz": [x, y, z],
                "shortest_axis_index": 0}


class Visualizer2D(object):
    """Visualizer3D of 2D implicit representations."""
    def __init__(self, surface_value, resolution):
        super().__init__()
        
        self.resolution = resolution
        self.surface_value = surface_value

    def scatter_pts(self, pts, labels=None):
        color_out = 128
        color_in = 255
        color_bg = 0

        img = np.zeros((self.resolution, self.resolution), np.uint8) + color_bg

        if labels is not None:
            for pt, label in zip(pts, labels):
                color = color_in if label else color_out
                pt_img = tuple(((pt + 0.5) * self.resolution).astype(np.int64))
                cv.circle(img, pt_img, radius=0, color=color, thickness=-1)
        else:
            for pt in pts:
                pt_img = tuple(((pt + 0.5) * self.resolution).astype(np.int64))
                cv.circle(img, pt_img, radius=0, color=color_in, thickness=-1)
        return img
    
    def get_grid_pred(self, decoder, priors=None, points={}):
        # Generating grid
        grid = self.get_grid(points.reshape(-1, points.shape[-1]))
        
        if len(priors) > 0:
            B = priors[list(priors.keys())[0]].shape[0]
        else:
            if points is not None:
                B = points.shape[0]
            else:
                B = 1
        grid_pts = grid['grid_pts'].unsqueeze(0).repeat(B, 1, 1)

        # Evaluating points
        grid_pred = []
        with torch.no_grad():
            for i, pnts in enumerate(torch.split(grid_pts, 100000, dim=1)):
                grid_pred_block = decoder(pnts.cuda(), **priors)
                grid_pred.append(grid_pred_block.detach().cpu().numpy())
        grid_pred = np.concatenate(grid_pred, axis=1)

        # Collecting results
        grid_pred = grid_pred.reshape(B, grid['xy'][0].shape[0], grid['xy'][1].shape[0], 1)
        grid_pts = grid_pts.reshape(B, grid['xy'][0].shape[0], grid['xy'][1].shape[0], 2).detach().cpu().numpy()
        
        return grid_pts, grid_pred
    
    def get_grid(self, points):
        x = torch.linspace(-0.5, 0.5, self.resolution)
        y = torch.linspace(-0.5, 0.5, self.resolution)

        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid_points = torch.vstack([xx.flatten(), yy.flatten()]).T.float()
    
        return {"grid_pts": grid_points,
                "xy": [x, y]}


if __name__ == '__main__':
    from config.pipeline.default import get_cfg_defaults
    from model import get_model
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file('config/cape-scan-subject-cloth_unif.py')
    cfg.freeze()
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    model = get_model(cfg.MODEL.name)(**cfg.MODEL.kwargs).cuda()

    vis = Visualizer3D(**cfg.VISUALIZER.kwargs)

    x = torch.rand([100, 3]).cuda()
    x = model(x)

    vis.plot_surface(model)



