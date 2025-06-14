# โมดูลสำหรับเรนเดอร์ 3D mesh ด้วย PyTorch3D
import torch
import torch.nn.functional as F
from scipy.io import loadmat
import numpy as np
import config

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    PointLights, HardPhongShader, SoftSilhouetteShader, Materials, Textures
)
from pytorch3d.io import load_objs_as_meshes
from utils import perspective_proj_withz  # ฟังก์ชันช่วยโปรเจคต์จุด 3D

# สร้างคลาส Renderer ที่สืบทอดมาจาก torch.nn.Module
class Renderer(torch.nn.Module):
    def __init__(self, image_size, device):
        super(Renderer, self).__init__()

        self.image_size = image_size  # กำหนดขนาดภาพที่จะเรนเดอร์

        # กำหนดกล้องให้หันไปที่วัตถุ โดยวางกล้องห่างจากวัตถุ 2.7 หน่วย มุม 0 องศา
        R, T = look_at_view_transform(2.7, 0, 0, device=device) 
        self.cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)  # กำหนดกล้องแบบ perspective

        # กำหนดสีของ mesh จาก config แล้ว normalize ให้อยู่ในช่วง 0-1
        self.mesh_color = torch.FloatTensor(config.MESH_COLOR).to(device)[None, None, :] / 255.0

        # ตั้งค่าพารามิเตอร์สำหรับ blend ใน silhouette rendering (ค่าเล็ก ๆ เพื่อความนุ่มนวล)
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        # ตั้งค่าการ rasterization สำหรับ silhouette
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,  # คำนวณค่า blur radius
            faces_per_pixel=100,  # จำนวน faces ที่ตรวจสอบต่อ pixel (มากเพื่อความละเอียด silhouette)
        )

        # สร้าง renderer สำหรับ silhouette (ใช้ SoftSilhouetteShader เพื่อความนุ่มนวล)
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

        # ตั้งค่าการ rasterization สำหรับการเรนเดอร์สี (ไม่มี blur, เฟรมเดียวต่อ pixel)
        raster_settings_color = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        
        # กำหนดแสงแบบ point light อยู่หน้ากล้อง
        lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

        # สร้าง renderer สำหรับภาพสี ใช้ HardPhongShader สำหรับ shading แบบ Phong
        self.color_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings_color
            ),
            shader=HardPhongShader(
                device=device, 
                cameras=self.cameras,
                lights=lights,
            )
        )

    # ฟังก์ชัน forward สำหรับเรนเดอร์ mesh ที่รับ vertices, points, faces เข้ามา
    # render_texture ถ้า True จะเรนเดอร์ภาพสีด้วย
    def forward(self, vertices, points, faces, render_texture=False):
        # สร้าง texture สีของ mesh โดยให้สีเท่ากันทั้งหมด (mesh_color) มี shape (1, V, 3)
        tex = torch.ones_like(vertices) * self.mesh_color 
        textures = Textures(verts_rgb=tex)  # สร้าง texture object

        # สร้าง Meshes object จาก vertices, faces, และ textures
        mesh = Meshes(verts=vertices, faces=faces, textures=textures)

        # เรนเดอร์ silhouette (ภาพขอบเงา) แล้วดึง channel alpha (-1) ออกมาเพิ่มมิติ channel
        sil_images = self.silhouette_renderer(mesh)[..., -1].unsqueeze(1)

        # สร้าง tensor ขนาดหน้าจอที่เท่ากับ image_size สำหรับแต่ละ batch
        screen_size = torch.ones(vertices.shape[0], 2).to(vertices.device) * self.image_size

        # โปรเจคต์จุด 3D 'points' ไปยังหน้าจอ pixel coordinates (กลับแกน y,x)
        proj_points = self.cameras.transform_points_screen(points, screen_size)[:, :, [1, 0]]

        # ถ้าขอเรนเดอร์สี (texture)
        if render_texture:
            # เรนเดอร์ภาพสี mesh และปรับแกนให้อยู่ในรูปแบบ (batch, channels, height, width)
            color_image = self.color_renderer(mesh).permute(0, 3, 1, 2)[:, :3, :, :]
            return sil_images, proj_points, color_image
        else:
            # คืน silhouette และจุดที่โปรเจคต์แล้ว
            return sil_images, proj_points
