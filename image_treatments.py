# -*- coding: utf-8 -*-
"""
Pixel-by-pixel georeferencing using DEM
Based on OpenDroneMap techniques
"""

from PIL import Image
import numpy as np
import math
from dotenv import load_dotenv
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_gcps, from_bounds
from rasterio.control import GroundControlPoint
from pyproj import Transformer
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
import cv2
import rasterio.windows
from pathlib import Path
from datetime import datetime
from dateutil import parser
import subprocess
import json
from osgeo import gdal, osr



class ImageDrone:
    """Georeference DJI drone images with RTK precision"""
    
    def __init__(self, image_path, exiftool_path, DEM_path=None, epsg_code=32738,
                 sensor_width_mm=13.2, sensor_height_mm=8.8, focal_length_mm=8.8
                 ):
        
        self.image_path = image_path
        self.exiftool_path = exiftool_path
        self.DEM_path = DEM_path
        self.epsg_code = epsg_code
        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm
        self.focal_length_mm = focal_length_mm

        
        # Metadata from XMP
        self.lat = None
        self.lon = None
        self.altitude_absolute = None
        self.altitude_relative = None
        self.yaw_gimbal = None
        self.pitch_gimbal = None
        self.roll_gimbal = None
        self.yaw_drone = None
        self.pitch_drone = None
        self.roll_drone = None
        self.dewarpflag = None
        self.k1 = self.k2 = self.k3 = 0.0
        self.p1 = self.p2 = 0.0
        self.date_taken = None
        
        # Lever arm
        self.lever_x = 0
        # self.lever_y = 0.036
        # self.lever_z = -0.192
        self.lever_y = 0
        self.lever_z = 0
         
         
        # Calibrated intrinsics
        self.fx_calib = None
        self.fy_calib = None
        self.cx_calib = None
        self.cy_calib = None
        
        # Image data
        self.image_loaded = None
        self.height_image_loaded = None
        self.width_image_loaded = None
        self.bands = None
        
        # Calculated data
        self.ground_elevation = None
        self.height_above_ground = None
        self.corners_camera = None
        self.center_image_camera = None
        self.rotation_matrix = None
        self.corners_world = None
        self.transform = None
        self.center_image_x = None
        self.center_image_y = None
        
        # Camera intrinsics
        self.K = None
        self.K_inv = None
    def extract_metadata(self):
        print(f"--- Processing {os.path.basename(self.image_path)} ---")

        exif_data = subprocess.check_output([
            self.exiftool_path,
            "-json",
            "-G",
            "-n",
            "-EXIF:all",
            "-XMP:all",
            "-MakerNotes:all",
            "-Composite:all",
            self.image_path
        ])

        meta = json.loads(exif_data)[0]

        def get(*keys):
            for k in keys:
                if k in meta:
                    return meta[k]
            return None
        
        def getf(*keys, default=0.0):
            for k in keys:
                if k in meta and meta[k] not in [None, ""]:
                    try:
                        return float(meta[k])
                    except:
                        pass
            return default

        # --- Position ---
        self.lat = get("Composite:GPSLatitude")
        self.lon = get("Composite:GPSLongitude")
        self.altitude_absolute = getf("MakerNotes:AbsoluteAltitude")
        self.altitude_relative = getf("MakerNotes:RelativeAltitude")

        # --- Gimbal angles ---
        self.yaw_gimbal   = math.radians(getf("XMP:GimbalYawDegree"))
        self.pitch_gimbal = math.radians(getf("XMP:GimbalPitchDegree"))
        self.roll_gimbal  = math.radians(getf("XMP:GimbalRollDegree"))

        # --- Drone angles ---
        self.yaw_drone   = math.radians(getf("XMP:FlightYawDegree"))
        self.pitch_drone = math.radians(getf("XMP:FlightPitchDegree"))
        self.roll_drone  = math.radians(getf("XMP:FlightRollDegree"))

        # --- Distortion ---
        self.dewarpflag = getf("XMP:DewarpFlag")

        dewarp = meta.get("XMP:DewarpData")
        if dewarp:
            try:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", dewarp)
                nums = list(map(float, nums))

                if len(nums) > 9:
                    nums = nums[-9:]

                self.fx_calib, self.fy_calib, _, _, self.k1, self.k2, self.p1, self.p2, self.k3 = nums
            except Exception as e:
                print(f"[WARN] DewarpData parse error: {e}")

        # --- Optical center ---
        self.cx_calib = getf("XMP:CalibratedOpticalCenterX")
        self.cy_calib = getf("XMP:CalibratedOpticalCenterY")

        # --- Date ---
        self.date_taken = get("EXIF:DateTimeOriginal") or get("XMP:CreateDate")

        print(f"Lat={self.lat:.8f}, Lon={self.lon:.8f}")
        print(f"Gimbal YPR: {math.degrees(self.yaw_gimbal):.2f}, {math.degrees(self.pitch_gimbal):.2f}, {math.degrees(self.roll_gimbal):.2f}")
        print(f"Drone YPR: {math.degrees(self.yaw_drone):.2f}, {math.degrees(self.pitch_drone):.2f}, {math.degrees(self.roll_drone):.2f}")
        print(f"Dewarp: {self.dewarpflag}")
        print(f"Dewarp values : k1 {self.k1},k2 {self.k2}, k3 {self.k3}, p1 {self.p1}, p2 {self.p2}")

        return True
   
    def load_image(self):
        """Load image with OpenCV"""
        self.image_loaded = cv2.imread(self.image_path)
        if self.image_loaded is None:
            print("[ERR] Failed to load image")
            return False

        self.image_rgb = cv2.cvtColor(self.image_loaded, cv2.COLOR_BGR2RGB)

        if len(self.image_rgb.shape) == 3:
            self.height_image_loaded, self.width_image_loaded, self.bands = self.image_rgb.shape
        else:
            self.height_image_loaded, self.width_image_loaded = self.image_rgb.shape
            self.bands = 1
            self.image_rgb = self.image_rgb[:, :, np.newaxis]

        print(f"Image: {self.width_image_loaded} x {self.height_image_loaded} px, {self.bands} bands")
        return True
    
    def correction_distortion(self):
        """Apply distortion correction"""
        if self.fx_calib and self.fy_calib and self.cx_calib and self.cy_calib:
            fx, fy, cx, cy = self.fx_calib, self.fy_calib, self.cx_calib, self.cy_calib
        else:
            fx = self.focal_length_mm * self.width_image_loaded / self.sensor_width_mm
            fy = self.focal_length_mm * self.height_image_loaded / self.sensor_height_mm
            cx = self.width_image_loaded / 2.0
            cy = self.height_image_loaded / 2.0
        print(f"fx: {fx:.2f}",f"fy: {fy:.2f}" )
    
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)
        D = np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)
        
        h, w = self.image_rgb.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=1)
        
        image_undistorted = cv2.undistort(self.image_rgb, K, D, None, new_K)
   
        self.image_undistorted = image_undistorted
        self.new_K = new_K  
        try:
            self.K_inv = np.linalg.inv(new_K)
        except Exception:
            self.K_inv = None
    
        self.height_image_undistorted, self.width_image_undistorted = image_undistorted.shape[:2]
        return True
    
    def calculate_flight_height(self):
        """Calculate height above ground using DEM"""
        if not self.DEM_path:
            self.ground_elevation = None
            self.height_above_ground = abs(self.altitude_relative)
            print(f"[INFO] No DEM → Height: {self.height_above_ground:.2f} m")
            return True

        samples = []
        try:
            with rasterio.open(self.DEM_path) as src:
                dem_crs = src.crs.to_string()
                transform = Transformer.from_crs(f"EPSG:{self.epsg_code}", dem_crs, always_xy=True)
                
                transformer_wgs84 = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg_code}", always_xy=True)
                center_x, center_y = transformer_wgs84.transform(self.lon, self.lat)
                
                try:
                    x_dem, y_dem = transform.transform(center_x, center_y)
                    row, col = src.index(x_dem, y_dem)
                    row = max(0, min(row, src.height - 1))
                    col = max(0, min(col, src.width - 1))
                    window = rasterio.windows.Window(col, row, 1, 1)
                    value = float(src.read(1, window=window)[0, 0])
                    samples.append(value)
                except Exception as e:
                    print(f"[WARN] Could not sample DEM: {e}")
     
        except Exception as e:
            print(f"[ERR] DEM reading failed: {e}")
            self.ground_elevation = None
            self.height_above_ground = abs(self.altitude_relative)
            return True

        if len(samples) == 0:
            self.ground_elevation = None
            self.height_above_ground = abs(self.altitude_relative)
            return True

        self.ground_elevation = float(np.mean(samples))
        self.height_above_ground = abs(self.altitude_absolute - self.ground_elevation)
        print(f"[INFO] Ground elevation: {self.ground_elevation:.2f} m")
        print(f"[INFO] Height above ground: {self.height_above_ground:.2f} m")
        return True
   
    # def calculate_camera_geometry(self):
    #     """Calculate camera geometry in normalized coordinates"""
    #     if self.fx_calib and self.fy_calib:
    #         fx = self.fx_calib
    #         fy = self.fy_calib
    #     else:
    #         fx = self.focal_length_mm * self.width_image_undistorted / self.sensor_width_mm
    #         fy = self.focal_length_mm * self.height_image_undistorted / self.sensor_height_mm

    #     half_w_norm = (self.width_image_undistorted / 2.0) / fx
    #     half_h_norm = (self.height_image_undistorted / 2.0) / fy

    #     self.corners_camera = np.array([
    #         [-half_w_norm,  half_h_norm, -1.0],
    #         [ half_w_norm,  half_h_norm, -1.0],
    #         [ half_w_norm, -half_h_norm, -1.0],
    #         [-half_w_norm, -half_h_norm, -1.0]
    #     ])
    #     self.center_image_camera = np.array([0.0, 0.0, -1.0])
    #     return True

    def calculate_camera_geometry(self):
        """Calcule les rayons des coins avec divergence perspective"""
        
        # Calculer le FOV en radians
        if self.fx_calib and self.fy_calib:
            fx = self.fx_calib
            fy = self.fy_calib
        else:
            fx = self.focal_length_mm * self.width_image_undistorted / self.sensor_width_mm
            fy = self.focal_length_mm * self.height_image_undistorted / self.sensor_height_mm
        
        # FOV = 2 * arctan(sensor / (2 * focal))
        FOV_h = 2 * np.arctan(self.width_image_undistorted / (2 * fx))
        FOV_v = 2 * np.arctan(self.height_image_undistorted / (2 * fy))
        
        # Créer des rayons qui DIVERGENT (comme code 1)
        self.corners_camera = np.array([
            [-np.tan(FOV_v/2), np.tan(FOV_h/2), 1.0],   # Haut-gauche
            [ np.tan(FOV_v/2), np.tan(FOV_h/2), 1.0],   # Haut-droit
            [ np.tan(FOV_v/2), -np.tan(FOV_h/2), 1.0],  # Bas-droit
            [-np.tan(FOV_v/2), -np.tan(FOV_h/2), 1.0]   # Bas-gauche
        ])
        
        # NORMALISER les rayons (garder juste la direction)
        for i in range(len(self.corners_camera)):
            norm = np.linalg.norm(self.corners_camera[i])
            self.corners_camera[i] = self.corners_camera[i] / norm
        
        self.center_image_camera = np.array([0.0, 0.0, 1.0])
        return True

    def calculate_rotation_matrix(self,yaw, pitch, roll):

        pitch += ( math.radians(90))
  
        Rz = np.array([
            [math.cos(yaw), math.sin(yaw), 0],
            [- math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]])
        Ry = np.array([
            [math.cos(pitch), 0, - math.sin(pitch)],
            [0, 1, 0],
            [math.sin(pitch), 0, math.cos(pitch)]])
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll), math.sin(roll)],
            [0, - math.sin(roll), math.cos(roll)]])
        rotation_matrix = Rz.dot(Ry).dot(Rx)
        return rotation_matrix

    def ray_dem_intersection(self, pixel_x, pixel_y, dem_dataset, transformer_to_dem):
        """Calculate ray-DEM intersection for a given pixel"""
        if self.K_inv is None:
            return None
        
        Rotation_camera = self.calculate_rotation_matrix(
            self.yaw_gimbal, self.pitch_gimbal, self.roll_gimbal
        )
        
        Rotation_drone = self.calculate_rotation_matrix(
            self.yaw_drone, self.pitch_drone, self.roll_drone
        )
        
        # Calcul du rayon
        pixel_source = np.array([pixel_x, pixel_y, 1.0])
        ray_camera = self.K_inv @ pixel_source
        ray_world_raw = Rotation_camera @ ray_camera

        # if ray_world_raw[2] > 0:
        #     ray_world_raw[2] = -ray_world_raw[2]  
        #     print(f"[FIX] Ray inversé: {ray_world_raw}")


        ray_world = ray_world_raw / np.linalg.norm(ray_world_raw)  
        
        # Transformation GPS
        transformer_wgs84 = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg_code}", always_xy=True)
        gps_x, gps_y = transformer_wgs84.transform(self.lon, self.lat)
        gps_z = self.altitude_absolute
        
        # Lever arm
        lever_drone = np.array([self.lever_x, self.lever_y, self.lever_z])
        lever_world = Rotation_drone @ lever_drone
        
        camera_x = gps_x - lever_world[0]
        camera_y = gps_y - lever_world[1]
        camera_z = gps_z - lever_world[2]
        
        # Check ray valid
        if ray_world[2] <= 0:
            return None
        
        # Calcul trajectory
        if self.ground_elevation is not None:
            ground_estimate = self.ground_elevation
        else:
            ground_estimate = self.altitude_absolute - abs(self.altitude_relative)
        
        altitude_diff = camera_z - ground_estimate
        trajectory_ground = abs(altitude_diff / ray_world[2])  
  
        step_size = 0.5
        num_steps = int(trajectory_ground / step_size) + 50
        
        best_intersection = None
        min_diff = float('inf')
        
        for i in range(num_steps):
            trajectory = step_size * i
            
            point_x = camera_x + trajectory * ray_world[0]
            point_y = camera_y + trajectory * ray_world[1]  
            point_z = camera_z - trajectory * ray_world[2]
            
            try:
                dem_x, dem_y = transformer_to_dem.transform(point_x, point_y)
                row, col = dem_dataset.index(dem_x, dem_y)

                row_floor = int(np.floor(row))
                col_floor = int(np.floor(col))
                row_frac = row - row_floor
                col_frac = col - col_floor
                
                if 0 <= row_floor < dem_dataset.height-1 and 0 <= col_floor < dem_dataset.width-1:
                    z11 = float(dem_dataset.read(1, window=rasterio.windows.Window(col_floor,   row_floor,   1, 1))[0, 0])
                    z21 = float(dem_dataset.read(1, window=rasterio.windows.Window(col_floor+1, row_floor,   1, 1))[0, 0])
                    z12 = float(dem_dataset.read(1, window=rasterio.windows.Window(col_floor,   row_floor+1, 1, 1))[0, 0])
                    z22 = float(dem_dataset.read(1, window=rasterio.windows.Window(col_floor+1, row_floor+1, 1, 1))[0, 0])

                    dem_elevation = (
                        z11 * (1-col_frac)*(1-row_frac) +
                        z21 * col_frac*(1-row_frac) +
                        z12 * (1-col_frac)*row_frac +
                        z22 * col_frac*row_frac
                    )
                elif 0 <= row_floor < dem_dataset.height and 0 <= col_floor < dem_dataset.width:
                    dem_elevation = float(dem_dataset.read(1, window=rasterio.windows.Window(col_floor, row_floor, 1, 1))[0, 0])
                else:
                    continue
                
                diff = abs(point_z - dem_elevation)
                
                if diff < 0.1:
                    return (point_x, point_y, dem_elevation)
                
                if diff < min_diff:
                    min_diff = diff
                    best_intersection = (point_x, point_y, dem_elevation)
                
                if point_z < dem_elevation:
                    break
                    
            except Exception:
                continue
        
        return best_intersection


    def georeference_with_dem_precise(self, output_path, subsample=100):
        """
        Precise georeferencing with DEM intersection for each control point
        subsample = number of pixels needed for each GCP
        """
        if not self.DEM_path:
            print("[ERR] DEM required")
            return False
        
        print(f"\n[INFO] Precise Georeferencing with DEM")
        print(f"[INFO] Subsample: {subsample}")
        
        with rasterio.open(self.DEM_path) as dem_dataset:
            dem_crs = dem_dataset.crs.to_string()
            transformer_to_dem = Transformer.from_crs(
                f"EPSG:{self.epsg_code}", 
                dem_crs, 
                always_xy=True
            )
            
            gcps = []
            total_pixels = (self.height_image_undistorted // subsample) * (self.width_image_undistorted // subsample)
            processed = 0
            
            print("[INFO] Computing GCPs...")
                       
            for row in range(0, self.height_image_undistorted, subsample):
                for col in range(0, self.width_image_undistorted, subsample):
                    processed += 1
                    if processed % 100 == 0:
                        progress = (processed / total_pixels) * 100
                        print(f"[INFO] Progress: {progress:.1f}%", end='\r')
                    
                    intersection = self.ray_dem_intersection(
                        col, row, dem_dataset, transformer_to_dem
                    )
                    
                    if intersection is not None:
                        x_world, y_world, z_world = intersection
                        gcps.append({
                            'pixel': (col, row),
                            'world': (x_world, y_world, z_world)
                        })
            
            print(f"\n[INFO] {len(gcps)} GCPs generated")
            
            if len(gcps) < 4:
                print(f"[ERR] Not enough GCPs ({len(gcps)})")
                return False
            
            rasterio_gcps = [
                GroundControlPoint(
                    row=gcp['pixel'][1], 
                    col=gcp['pixel'][0],
                    x=gcp['world'][0],
                    y=gcp['world'][1],
                    z=gcp['world'][2]
                )
                for gcp in gcps
            ]

            # Créer un dataset en mémoire avec les GCPs
            driver = gdal.GetDriverByName('MEM')
            mem_ds = driver.Create(
                '',
                self.width_image_undistorted,
                self.height_image_undistorted,
                self.bands,
                gdal.GDT_Byte
            )
            
            # Écrire l'image
            image_to_write = np.rot90(self.image_undistorted, k=2)
            image_to_write = np.fliplr(image_to_write)
            
            for i in range(self.bands):
                mem_ds.GetRasterBand(i + 1).WriteArray(image_to_write[:, :, i])
            
            # Ajouter les GCPs
            gcp_list = [
                gdal.GCP(
                    gcp['world'][0],  # x
                    gcp['world'][1],  # y
                    gcp['world'][2],  # z
                    gcp['pixel'][0],  # pixel
                    gcp['pixel'][1]   # line
                )
                for gcp in gcps
            ]
            
            # Définir la projection
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(self.epsg_code)
            
            mem_ds.SetGCPs(gcp_list, srs.ExportToWkt())
            
            # Warper avec transformation polynomiale ou TPS
            warp_options = gdal.WarpOptions(
                dstSRS=f'EPSG:{self.epsg_code}',
                transformerOptions=['METHOD=GCP_TPS'],  # Thin Plate Spline!
                resampleAlg=gdal.GRA_Bilinear,
                format='GTiff',
                creationOptions=['COMPRESS=LZW']
            )
            
            gdal.Warp(output_path, mem_ds, options=warp_options)
            
            mem_ds = None
            print(f"[OK] GeoTIFF created with TPS: {output_path}")
            return True
            
            # transform = from_gcps(rasterio_gcps)
            
            # # Fix image orientation
            # # image_to_save = self.image_undistorted
            # image_to_save = np.rot90(self.image_undistorted, k=2)
            # image_to_save = np.fliplr(image_to_save)
            # # image_to_save = np.flipud(image_to_save)
            

            # with rasterio.open(
            #     output_path,
            #     'w',
            #     driver='GTiff',
            #     height=self.height_image_undistorted,
            #     width=self.width_image_undistorted,
            #     count=self.bands,
            #     dtype=image_to_save.dtype,
            #     crs=CRS.from_epsg(self.epsg_code),
            #     transform=transform,
            #     compress='lzw'
            # ) as dst:
            #     for i in range(self.bands):
            #         dst.write(image_to_save[:, :, i], i + 1)
         
            # print(f"[OK] GeoTIFF created: {output_path}")
            # return True


    def save_geotiff(self, output_path):
        """Save georeferenced image (fast method)"""
        if self.transform is None:
            return False

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=self.height_image_undistorted,
            width=self.width_image_undistorted,
            count=self.bands,
            dtype=self.image_to_save.dtype,
            crs=CRS.from_epsg(self.epsg_code),
            transform=self.transform,
            compress='lzw'
        ) as dst:
            for i in range(self.bands):
                dst.write(self.image_to_save[:, :, i], i + 1)

        print(f"[OK] GeoTIFF created: {output_path}")
        return True
    
    def crop_geotiff_center_75_percent(self, input_path, output_path):
        """
        Crop the central 75% of a georeferenced GeoTIFF
        Preserves georeference information correctly
        """
        with rasterio.open(input_path) as src:
            h, w = src.height, src.width
            new_w, new_h = int(w * 0.75), int(h * 0.75)
            
            start_col = (w - new_w) // 2
            start_row = (h - new_h) // 2
            window = rasterio.windows.Window(start_col, start_row, new_w, new_h)
            cropped_data = src.read(window=window)
            transform = src.window_transform(window)
        
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=new_h,
                width=new_w,
                count=src.count,
                dtype=cropped_data.dtype,
                crs=src.crs,
                transform=transform,
                compress='lzw'
            ) as dst:
                dst.write(cropped_data)
        
        print(f"[OK] Cropped GeoTIFF: {output_path}")
        print(f"[INFO] Original: {w}x{h} → Cropped: {new_w}x{new_h}")
        return True
    

if __name__ == "__main__":

    load_dotenv()

    image_folder = Path(os.getenv("IMAGE_FOLDER"))
    DEM_path = Path(os.getenv("DEM_PATH"))
    exiftool_path = Path(os.getenv("Exiftool_path"))

    print("Image folder :", image_folder)

    output_folder = image_folder / "georef_precise_fluxes"
    os.makedirs(output_folder, exist_ok=True)


    # image_name = input("Image Name :").strip()
    # image_path = image_folder / image_name

    images = sorted(image_folder.glob("*.JPG"))

    if not images:
        raise RuntimeError("No JPG images found")

    print("Images found:")
    for i, img in enumerate(images):
        print(f"{i} → {img.name}")

    idx = int(input("Choose image number: "))
    image_path = images[idx]
    image_name = image_path.name


    if not image_path.exists():
        print(f"[ERR] Image {image_name} not found in {image_folder}")
        exit(1)


    drone_image = ImageDrone(
        str(image_path),
        exiftool_path,
        DEM_path=DEM_path,
        epsg_code=32738
    )

    print("\n=== STEP 1: PREPARATION ===")

    drone_image.extract_metadata()
    drone_image.load_image()

    if drone_image.dewarpflag == 0:
        print("[INFO] Applying lens distortion correction")
        drone_image.correction_distortion()
    else:
        drone_image.image_undistorted = drone_image.image_loaded
        drone_image.height_image_undistorted = drone_image.height_image_loaded
        drone_image.width_image_undistorted = drone_image.width_image_loaded
        print("[INFO] Image already dewarped by DJI : skipping distortion correction")

    drone_image.calculate_camera_geometry()
    drone_image.calculate_flight_height()

    print("\n=== STEP 2: PRECISE GEOREFERENCING ===")

    output_name = image_path.stem + "_PRECISE_FULL.tif"
    output_path = output_folder / output_name

    success = drone_image.georeference_with_dem_precise(
        str(output_path),
        subsample=200
    )

    if success:
        print("\n=== STEP 3: CROP CENTER 75% ===")

        output_cropped = image_path.stem + "_PRECISE_CROPPED_FLUX_2.tif"
        output_path_cropped = output_folder / output_cropped

        drone_image.crop_geotiff_center_75_percent(
            str(output_path),
            str(output_path_cropped)
        )

        print("\n✓ SUCCESS!")
        print(f"  Full GeoTIFF: {output_path}")
        print(f"  Cropped GeoTIFF: {output_path_cropped}")

    else:
        print("\n✗ FAILED")

        
