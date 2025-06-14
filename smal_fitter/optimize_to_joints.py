# เพิ่ม path ของ root directory เพื่อให้สามารถ import modules ได้
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

# ไลบรารีพื้นฐาน
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import pickle as pkl
import torch
import imageio
import trimesh

# ไลบรารีช่วยโหลด dataset
from data_loader import load_badja_sequence, load_stanford_sequence

# ไลบรารีช่วยแสดง progress bar
from tqdm import trange

# config สำหรับพารามิเตอร์ต่างๆ
import config

# คลาสสำหรับเซฟภาพและไฟล์ผลลัพธ์จากการฟิตโมเดล
class ImageExporter():
    def __init__(self, output_dir, filenames):
        # สร้างโฟลเดอร์สำหรับ export ไฟล์ในแต่ละภาพ
        self.output_dirs = self.generate_output_folders(output_dir, filenames)
        self.stage_id = 0
        self.epoch_name = 0

    def generate_output_folders(self, root_directory, filename_batch):
        # สร้างโฟลเดอร์ย่อยตามชื่อไฟล์ภาพ
        if not os.path.exists(root_directory):
            os.mkdir(root_directory)

        output_dirs = [] 
        for filename in filename_batch:
            filename_path = os.path.join(root_directory, os.path.splitext(filename)[0])
            output_dirs.append(filename_path)
            os.mkdir(filename_path)
        
        return output_dirs

    def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces):
        # บันทึกภาพ Visualization ของผลลัพธ์
        imageio.imsave(os.path.join(self.output_dirs[global_id], "st{0}_ep{1}.png".format(self.stage_id, self.epoch_name)), collage_np)

        # บันทึกพารามิเตอร์ของโมเดลในรูปแบบ .pkl
        with open(os.path.join(self.output_dirs[global_id], "st{0}_ep{1}.pkl".format(self.stage_id, self.epoch_name)), 'wb') as f:
            pkl.dump(img_parameters, f)

        # แปลง vertices และ faces เป็น mesh และเซฟเป็น .ply
        vertices = vertices[batch_id].cpu().numpy()
        mesh = trimesh.Trimesh(vertices = vertices, faces = faces, process = False)
        mesh.export(os.path.join(self.output_dirs[global_id], "st{0}_ep{1}.ply".format(self.stage_id, self.epoch_name)))

# ฟังก์ชันหลัก
def main():
    # กำหนดให้ใช้ GPU ที่ต้องการ
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS

    # สร้างโฟลเดอร์ผลลัพธ์หากยังไม่มี
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # ตรวจสอบว่าใช้ CUDA ได้หรือไม่

    # แยกชื่อ dataset และชื่อ sequence
    dataset, name = config.SEQUENCE_OR_IMAGE_NAME.split(":")

    # โหลดข้อมูลตาม dataset ที่ระบุ
    if dataset == "badja":
        data, filenames = load_badja_sequence(
            config.BADJA_PATH, name, 
            config.CROP_SIZE, image_range=config.IMAGE_RANGE)
    else:
        data, filenames = load_stanford_sequence(
            config.STANFORD_EXTRA_PATH, name,
            config.CROP_SIZE
        )

    dataset_size = len(filenames)
    print ("Dataset size: {0}".format(dataset_size))

    # ตรวจสอบว่ามีการกำหนด shape family ที่เหมาะสมหรือไม่
    assert config.SHAPE_FAMILY >= 0, "Shape family should be greater than 0"

    use_unity_prior = config.SHAPE_FAMILY == 1 and not config.FORCE_SMAL_PRIOR

    # แจ้งเตือนหากใช้ limb scaling โดยไม่มี Unity prior
    if not use_unity_prior and not config.ALLOW_LIMB_SCALING:
        print("WARNING: Limb scaling is only recommended for the new Unity prior. TODO: add a regularizer to constrain scale parameters.")
        config.ALLOW_LIMB_SCALING = False

    # สร้าง object สำหรับ export รูปภาพและไฟล์
    image_exporter = ImageExporter(config.OUTPUT_DIR, filenames)

    # สร้างโมเดล SMALFitter
    model = SMALFitter(device, data, config.WINDOW_SIZE, config.SHAPE_FAMILY, use_unity_prior)

    # เริ่มกระบวนการ optimize โดยแบ่งเป็น stages ตาม config
    for stage_id, weights in enumerate(np.array(config.OPT_WEIGHTS).T):
        opt_weight = weights[:6]      # น้ำหนัก loss แต่ละประเภท
        w_temp = weights[6]           # น้ำหนักของ temporal loss
        epochs = int(weights[7])      # จำนวนรอบที่ฝึกฝน
        lr = weights[8]               # learning rate

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

        if stage_id == 0:
            # เฟสแรก: ฟิตเฉพาะ torso
            model.joint_rotations.requires_grad = False
            model.betas.requires_grad = False
            model.log_beta_scales.requires_grad = False
            target_visibility = model.target_visibility.clone()
            model.target_visibility *= 0
            model.target_visibility[:, config.TORSO_JOINTS] = target_visibility[:, config.TORSO_JOINTS]
        else:
            # เฟสต่อไป: ฟิตข้อต่อทั้งหมด
            model.joint_rotations.requires_grad = True
            model.betas.requires_grad = True
            if config.ALLOW_LIMB_SCALING:
                model.log_beta_scales.requires_grad = True
            model.target_visibility = data[-1].clone()

        # วนลูป epochs
        t = trange(epochs, leave=True)
        for epoch_id in t:
            image_exporter.stage_id = stage_id
            image_exporter.epoch_name = str(epoch_id)

            acc_loss = 0
            optimizer.zero_grad()

            # วนลูปทีละ batch
            for j in range(0, dataset_size, config.WINDOW_SIZE):
                batch_range = list(range(j, min(dataset_size, j + config.WINDOW_SIZE)))
                loss, losses = model(batch_range, opt_weight, stage_id)
                acc_loss += loss.mean()

            # คำนวณ temporal loss
            joint_loss, global_loss, trans_loss = model.get_temporal(w_temp)

            # แสดงผลรวม loss
            desc = "EPOCH: Optimizing Stage: {}\t Epoch: {}, Loss: {:.2f}, Temporal: ({}, {}, {})".format(
                stage_id, epoch_id, 
                acc_loss.data, joint_loss.data, 
                global_loss.data, trans_loss.data)
            t.set_description(desc)
            t.refresh()

            # รวม loss ทั้งหมดแล้ว backward
            acc_loss = acc_loss + joint_loss + global_loss + trans_loss
            acc_loss.backward()
            optimizer.step()

            # แสดงผลภาพทุก ๆ config.VIS_FREQUENCY รอบ
            if epoch_id % config.VIS_FREQUENCY == 0:
                model.generate_visualization(image_exporter)

    # แสดงภาพขั้นสุดท้ายหลังการฝึกฝนเสร็จ
    image_exporter.stage_id = 10
    image_exporter.epoch_name = str(0)
    model.generate_visualization(image_exporter)

# เรียกฟังก์ชัน main เมื่อรันไฟล์นี้
if __name__ == '__main__':
    main()
