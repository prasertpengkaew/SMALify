####
### ffmpeg -framerate 50 -i %04d.png -pix_fmt yuv420p rs_dog.gif
###
# คำสั่ง ffmpeg ด้านบนใช้สร้างไฟล์ gif จากรูปภาพ .png ที่ชื่อเรียงลำดับ 0000.png, 0001.png, ... ที่ความเร็ว 50 เฟรมต่อวินาที

import sys, os
sys.path.append(os.path.dirname(sys.path[0]))  # เพิ่ม path ของโฟลเดอร์หลักใน sys.path เพื่อให้ import module ภายนอกได้

import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from smal_fitter import SMALFitter  # โมดูลสำหรับฟิตโมเดล SMAL

import torch
import imageio  # สำหรับอ่านเขียนไฟล์ภาพ
import config   # ไฟล์ config ตั้งค่าต่างๆ

from data_loader import load_badja_sequence, load_stanford_sequence  # ฟังก์ชันโหลดข้อมูลภาพและ annotation
import time
import pickle as pkl  # สำหรับบันทึกข้อมูลแบบ pickle

# คลาสสำหรับส่งออกภาพและข้อมูล mesh ต่างๆ
class ImageExporter():
    def __init__(self, output_dir):
        self.output_dir = output_dir  # กำหนดโฟลเดอร์ที่จะบันทึกไฟล์

    # ฟังก์ชันบันทึกภาพและไฟล์พารามิเตอร์ต่างๆ
    def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces):
        # บันทึกภาพในรูปแบบ .png ตามลำดับ global_id เช่น 0001.png, 0002.png
        imageio.imsave(os.path.join(self.output_dir, "{0:04}.png".format(global_id)), collage_np)

        # บันทึกพารามิเตอร์ภาพในรูปแบบ pickle (.pkl)
        with open(os.path.join(self.output_dir, "{0:04}.pkl".format(global_id)), 'wb') as f:
            pkl.dump(img_parameters, f)

def main():
    # กำหนดโฟลเดอร์สำหรับบันทึกผลลัพธ์ โดยมีโครงสร้างตามชื่อ checkpoint และ epoch
    OUTPUT_DIR = os.path.join("exported", config.CHECKPOINT_NAME, config.EPOCH_NAME)

    # ตั้งค่า GPU ที่จะใช้ ตาม config
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS

    # เลือก device ให้เป็น GPU ถ้ามี ไม่เช่นนั้นเป็น CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # อ่านข้อมูล dataset และ sequence/image name จาก config
    dataset, name = config.SEQUENCE_OR_IMAGE_NAME.split(":")

    # โหลดข้อมูล sequence ตามชื่อ dataset และ sequence
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

    # สร้างโฟลเดอร์เก็บผลลัพธ์ถ้ายังไม่มี
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # เช็คว่าต้องใช้ Unity prior หรือไม่
    use_unity_prior = config.SHAPE_FAMILY == 1 and not config.FORCE_SMAL_PRIOR

    # ถ้าไม่ได้ใช้ Unity prior และปิด limb scaling ให้แจ้งเตือน
    if not use_unity_prior and not config.ALLOW_LIMB_SCALING:
        print("WARNING: Limb scaling is only recommended for the new Unity prior. TODO: add a regularizer to constrain scale parameters.")
        config.ALLOW_LIMB_SCALING = False

    # สร้างออบเจกต์สำหรับ export ภาพ
    image_exporter = ImageExporter(OUTPUT_DIR)

    # สร้างโมเดล SMALFitter โดยส่ง device, ข้อมูล, window size, shape family, และ prior ที่ใช้
    model = SMALFitter(device, data, config.WINDOW_SIZE, config.SHAPE_FAMILY, use_unity_prior)

    # โหลดโมเดลจาก checkpoint และ epoch ที่กำหนดใน config
    model.load_checkpoint(os.path.join("checkpoints", config.CHECKPOINT_NAME), config.EPOCH_NAME)

    # เรียกสร้างภาพ visualizations (ภาพสุดท้าย) และบันทึกผ่าน ImageExporter
    model.generate_visualization(image_exporter) # Final stage

if __name__ == '__main__':
    main()
