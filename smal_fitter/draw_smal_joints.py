import numpy as np
import cv2
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import config


# คลาสสำหรับวาดตำแหน่งข้อต่อ (joints) บนภาพสัตว์ที่เป็นโมเดล SMAL
class SMALJointDrawer:

    # ฟังก์ชันวาด joint บน tensor image ที่เป็น torch.Tensor
    @staticmethod
    def draw_joints(image, landmarks, visible=None):
        # แปลง tensor image จากรูปแบบ [batch, channels, height, width] -> [batch, height, width, channels]
        image_np = np.transpose(image.cpu().data.numpy(), (0, 2, 3, 1))
        # แปลง landmarks เป็น numpy array
        landmarks_np = landmarks.cpu().data.numpy()

        # ถ้ามีข้อมูล visibility (ว่าจุด joint ไหนมองเห็นหรือไม่) ให้แปลงเป็น numpy ด้วย
        if visible is not None:
            visible_np = visible.cpu().data.numpy()
        else:
            visible_np = visible  # อาจเป็น None

        # เรียกฟังก์ชันวาด joint บน numpy image
        return_stack = SMALJointDrawer.draw_joints_np(
            image_np, landmarks_np, visible_np
        )

        # แปลงภาพกลับเป็น tensor ในรูปแบบเดิม [batch, channels, height, width]
        return torch.FloatTensor(np.transpose(return_stack, (0, 3, 1, 2)))

    # ฟังก์ชันวาด joint บนภาพ numpy array
    @staticmethod
    def draw_joints_np(image_np, landmarks_np, visible_np=None):
        # แปลงภาพจาก float [0,1] เป็น uint8 [0,255]
        image_np = (image_np * 255.0).astype(np.uint8)

        # bs = batch size (จำนวนภาพในชุดข้อมูล)
        # nj = จำนวน joint ต่อภาพ
        bs, nj, _ = landmarks_np.shape

        # ถ้าไม่กำหนด visibility ให้ถือว่าทุก joint มองเห็นหมด (ค่า True)
        if visible_np is None:
            visible_np = np.ones((bs, nj), dtype=bool)

        # รายการเก็บภาพที่วาด joint แล้ว
        return_images = []

        # วนลูปใน batch ทีละภาพ
        for image_sgl, landmarks_sgl, visible_sgl in zip(
            image_np, landmarks_np, visible_np
        ):
            image_sgl = image_sgl.copy()  # สำเนาภาพเพื่อไม่ให้แก้ไขภาพต้นฉบับ
            inv_ctr = 0  # ตัวนับ joint ที่ไม่มองเห็น

            # วนลูปแต่ละ joint ในภาพนั้น
            for joint_id, ((y_co, x_co), vis) in enumerate(
                zip(landmarks_sgl, visible_sgl)
            ):
                # ดึงสีของ marker ตาม joint id จาก config
                color = np.array(config.MARKER_COLORS)[joint_id]
                # ดึงชนิด marker จาก config เช่น cv2.MARKER_CROSS, MARKER_TILTED_CROSS เป็นต้น
                marker_type = np.array(config.MARKER_TYPE)[joint_id]

                # ถ้าจุด joint นั้นไม่มองเห็น (vis == False)
                if not vis:
                    # กำหนดตำแหน่ง marker นอกภาพ (หรือมุมซ้ายบน) เพื่อแสดงว่า invisible
                    x_co, y_co = inv_ctr * 10, 0
                    inv_ctr += 1

                # วาด marker บนภาพด้วย OpenCV
                cv2.drawMarker(
                    image_sgl,
                    (x_co, y_co),  # ตำแหน่งจุด
                    (int(color[0]), int(color[1]), int(color[2])),  # สี RGB
                    marker_type,  # รูปแบบ marker
                    8,  # ขนาด marker
                    thickness=3,  # ความหนาเส้น marker
                )

            # เก็บภาพที่วาด joint เสร็จแล้วลงใน list
            return_images.append(image_sgl)

        # รวมภาพใน batch กลับเป็น array 4 มิติ [batch, height, width, channels]
        return_stack = np.stack(return_images, 0)
        # normalize ให้ค่าระหว่าง 0-1
        return_stack = return_stack / 255.0

        return return_stack
