from typing import Optional, List

import numpy as np
import trimesh
from meshly import Mesh, MeshUtils
from pydantic import Field


# --- Definir TexturedMesh ---
class TexturedMesh(Mesh):
    texture_coords: np.ndarray = Field(..., description="Texture coordinates")
    normals: Optional[np.ndarray] = Field(None, description="Vertex normals")
    material_name: str = Field("default", description="Material name")
    tags: List[str] = Field(default_factory=list, description="Tags for the mesh")

    # Métodos de optimización
    def optimize_vertex_cache(self):
        MeshUtils.optimize_vertex_cache(self)

    def optimize_overdraw(self):
        MeshUtils.optimize_overdraw(self)

    def optimize_vertex_fetch(self):
        MeshUtils.optimize_vertex_fetch(self)

    def simplify(self, target_ratio: float = 1.0):
        MeshUtils.simplify(self, target_ratio=target_ratio)

    # Encode / Decode / Save / Load
    def encode(self):
        return MeshUtils.encode(self)

    @classmethod
    def decode(cls, encoded_mesh):
        return MeshUtils.decode(cls, encoded_mesh)

    def save_to_zip(self, path: str):
        MeshUtils.save_to_zip(self, path)

    @classmethod
    def load_from_zip(cls, path: str):
        return MeshUtils.load_from_zip(cls, path)


# --- Cargar mesh desde el ZIP original ---
zip_path = "textured_cube.zip"  # reemplaza con la ruta de tu ZIP
mesh = TexturedMesh.load_from_zip(zip_path)

# Verificar los arrays
print("Vertices shape:", mesh.vertices.shape)
print("Indices shape:", mesh.indices.shape)
print("Normals shape:", mesh.normals.shape)
print("Texture coords shape:", mesh.texture_coords.shape)

# --- Crear mesh de trimesh con color simple ---
tri_mesh = trimesh.Trimesh(
    vertices=mesh.vertices,
    faces=mesh.indices.reshape(-1, 3),
    vertex_normals=mesh.normals,
    process=False
)

# Asignar color simple a todas las caras (RGBA)
tri_mesh.visual.face_colors = [200, 200, 250, 255]

# Mostrar mesh en 3D
tri_mesh.show()
