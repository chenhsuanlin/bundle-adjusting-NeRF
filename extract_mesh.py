"""Extracts a 3D mesh from a pretrained model using marching cubes."""

import importlib
import sys

import mcubes
import numpy as np
import options
import torch
from tqdm import tqdm
import trimesh


opt_cmd = options.parse_arguments(sys.argv[1:])
opt = options.set(opt_cmd=opt_cmd)

with torch.cuda.device(opt.device), torch.no_grad():

  model = importlib.import_module("model.{}".format(opt.model))
  m = model.Model(opt)

  m.load_dataset(opt, eval_split="test")
  m.build_networks(opt)

  m.restore_checkpoint(opt)
  N = 256
  # The coordinate range might change from model to model.
  t = np.linspace(-1.2, 1.2, N + 1)

  query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
  # sh = query_pts.shape
  flat = query_pts.reshape([-1, 3])

  chunk = 1024
  densities = []
  for i in tqdm(range(0, flat.shape[0], chunk)):
    points = torch.from_numpy(np.expand_dims(flat[i:i + chunk],
                                             axis=0)).to(opt.device)
    ray = points
    points_3D_samples = points
    ray_unit_samples = None
    # Dummy ray to comply with interface, not used.
    ray_unit = torch.zeros([1, 1024, 3], dtype=torch.float32).to(opt.device)
    _, density_samples = m.graph.nerf.forward(
        opt, points_3D_samples, ray_unit=ray_unit, mode=None)

    densities.append(density_samples.detach().cpu().numpy())
  densities = np.concatenate(densities, axis=1)
  densities = np.reshape(densities, list(query_pts.shape[:-1]) + [-1])

  sigma = np.maximum(densities[..., -1], 0.)

  threshold = 25.
  vertices, triangles = mcubes.marching_cubes(sigma, threshold)

  mesh = trimesh.Trimesh(vertices / N - .5, triangles)
  mesh.show()
