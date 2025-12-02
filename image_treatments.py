# -*- coding: utf-8 -*-
"""
Pixel-by-pixel georeferencing using DEM
Based on OpenDroneMap techniques
With magnetic declination correction
"""

from PIL import Image
import numpy as np
import math
from dotenv import load_dotenv
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_gcps
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

# Import for magnetic declination
try:
    from geomag import declination
    GEOMAG_AVAILABLE = True
except ImportError:
    GEOMAG_AVAILABLE = False
    print("[WARN] geomag module not available. Install with: pip install geomag")


class ImageDrone:
    """Georeference DJI drone images with RTK precision"""
    
    def __init__(self, image_path, DEM_path=None, epsg_code=32738,
                 sensor_width_mm=13.2, sensor_height_mm=8.8, focal_length_mm=8.8,
                 use_magnetic_correction=True):
        
        self.image_path = image_path
        self.DEM_path = DEM_path
        self.epsg_code = epsg_code
        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm
        self.focal_length_mm = focal_length_mm
        self.use_magnetic_correction = use_magnetic_correction
        
        # Metadata from XMP
        self.lat = None
        self.lon = None
        self.altitude_absolute = None
        self.altitude_relative = None
        self.yaw = None
        self.pitch = None
        self.roll = None
        self.yaw_drone = None
        self.pitch_drone = None
        self.roll_drone = None
        self.k1 = self.k2 = self.k3 = 0.0
        self.p1 = self.p2 = 0.0
        self.date_taken = None
        
        # Magnetic declination
        self.declination = 0.0
        
        # Lever arm
        self.lever_x = 0
        self.lever_y = 0.036 
        self.lever_z = -0.192
        
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
        """Extract GPS, RPY and distortion parameters from XMP"""
        print(f"--- Processing {os.path.basename(self.image_path)} ---")
        try:
            with open(self.image_path, 'rb') as file:
                data = file.read()
        except Exception as e:
            print(f"[ERR] Cannot read image file: {e}")
            return False

        match = re.search(b'<x:xmpmeta.*?</x:xmpmeta>', data, re.DOTALL)
        if not match:
            print("[ERR] No XMP found")
            return False

        try:
            xmp_str = match.group(0).decode('utf-8', errors='ignore')
            root = ET.fromstring(xmp_str)
        except Exception as e:
            print(f"[ERR] Failed to parse XMP: {e}")
            return False

        lat = lon = alt_abs = alt_rel = 0.0
        yaw = pitch = roll = 0.0
        yaw_drone = pitch_drone = roll_drone = 0.0
        k1 = k2 = k3 = 0.0
        p1 = p2 = 0.0
        date_taken = None
        fx_calib = fy_calib = None
        cx_calib = cy_calib = None

        for desc in root.findall('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description'):
            for k, v in desc.attrib.items():
                if 'GpsLatitude' in k:
                    lat = float(v)
                elif 'GpsLongtitude' in k or 'GpsLongitude' in k:
                    lon = float(v)
                elif 'AbsoluteAltitude' in k:
                    alt_abs = float(v)
                elif 'RelativeAltitude' in k:
                    alt_rel = float(v)
                elif 'GimbalYawDegree' in k:
                    yaw = float(v)
                elif 'GimbalPitchDegree' in k:
                    pitch = float(v)
                elif 'GimbalRollDegree' in k:
                    roll = float(v)
                elif 'FlightYawDegree' in k:
                    yaw_drone = float(v)
                elif 'FlightPitchDegree' in k:
                    pitch_drone = float(v)
                elif 'FlightRollDegree' in k:
                    roll_drone = float(v)
                elif 'CalibratedOpticalCenterX' in k:
                    cx_calib = float(v)
                elif 'CalibratedOpticalCenterY' in k:
                    cy_calib = float(v)
                elif 'DewarpData' in k:
                    parts = v.split(';', 1)[1].split(',')
                    fx_calib, fy_calib, _, _, k1, k2, p1, p2, k3 = map(float, parts)
                elif 'CreateDate' in k or 'DateTimeOriginal' in k:
                    date_taken = v
            break  

        self.lat = lat
        self.lon = lon
        self.altitude_absolute = alt_abs
        self.altitude_relative = alt_rel
        self.yaw = math.radians(yaw)
        self.pitch = math.radians(pitch)
        self.roll = math.radians(roll)
        self.yaw_drone = math.radians(yaw_drone)
        self.pitch_drone = math.radians(pitch_drone)
        self.roll_drone = math.radians(roll_drone)
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.p1, self.p2 = p1, p2
        self.fx_calib = fx_calib
        self.fy_calib = fy_calib
        self.cx_calib = cx_calib
        self.cy_calib = cy_calib
        self.date_taken = date_taken

        print(f"Position: Lat={self.lat:.8f}°, Lon={self.lon:.8f}°, Alt={self.altitude_absolute:.2f} m")
        print(f"Gimbal: Yaw={math.degrees(self.yaw):.2f}°, Pitch={math.degrees(self.pitch):.2f}°, Roll={math.degrees(self.roll):.2f}°")
        print(f"Flight: Yaw={math.degrees(self.yaw_drone):.2f}°, Pitch={math.degrees(self.pitch_drone):.2f}°, Roll={math.degrees(self.roll_drone):.2f}°")
        return True
    
    # def find_declination(self):
    #     """
    #     Calculate magnetic declination for the image location and date.
    #     Compatible avec la version actuelle de geomag (datetime.date obligatoire).
    #     """

    #     if not self.date_taken:
    #         print("[WARN] No date → skipping magnetic declination")
    #         self.declination = 0.0
    #         return False

    #     if not GEOMAG_AVAILABLE:
    #         print("[WARN] geomag library not available → skipping magnetic declination")
    #         self.declination = 0.0
    #         return False

    #     # --- Parse EXIF/XMP date ---
    #     raw_date = self.date_taken.strip()
    #     print(f"[DEBUG] RAW DATE = {raw_date}")

    #     try:
    #         dt = parser.parse(raw_date)
    #         dt_date = dt.date()  # important !
    #         print(f"[DEBUG] Parsed date: {dt_date}")
    #     except Exception as e:
    #         print("[ERROR] Could not parse EXIF/XMP date:", raw_date, e)
    #         self.declination = 0.0
    #         return False

    #     # --- Calculate magnetic declination ---
    #     try:
    #         dec = declination(self.lat, self.lon, self.altitude_absolute, dt_date)
    #         self.declination = dec
    #         print(f"[INFO] Magnetic declination = {dec:.2f}°")

    #         # --- Correct yaw ---
    #         if self.use_magnetic_correction:
    #             d_rad = math.radians(dec)
    #             self.yaw       += d_rad
    #             self.yaw_drone += d_rad
    #             print(f"[INFO] Yaw corrected +{dec:.2f}°")
    #             print(f"       New Gimbal Yaw: {math.degrees(self.yaw):.2f}°")
    #             print(f"       New Drone Yaw : {math.degrees(self.yaw_drone):.2f}°")

    #         return True

    #     except Exception as e:
    #         print("[ERROR] Declination calculation failed:", e)
    #         self.declination = 0.0
    #         return False
    
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

    
    def calculate_camera_geometry(self):
        """Calculate camera geometry in normalized coordinates"""
        if self.fx_calib and self.fy_calib:
            fx = self.fx_calib
            fy = self.fy_calib
        else:
            fx = self.focal_length_mm * self.width_image_undistorted / self.sensor_width_mm
            fy = self.focal_length_mm * self.height_image_undistorted / self.sensor_height_mm

        half_w_norm = (self.width_image_undistorted / 2.0) / fx
        half_h_norm = (self.height_image_undistorted / 2.0) / fy

        self.corners_camera = np.array([
            [-half_w_norm,  half_h_norm, -1.0],
            [ half_w_norm,  half_h_norm, -1.0],
            [ half_w_norm, -half_h_norm, -1.0],
            [-half_w_norm, -half_h_norm, -1.0]
        ])
        self.center_image_camera = np.array([0.0, 0.0, -1.0])
        return True


    def calculate_rotation_matrix(self,yaw, pitch, roll):
        """Build rotation matrix"""
        pitch += math.radians(90)
  
        Rz = np.array([
            [math.cos(yaw), math.sin(yaw), 0],
            [- math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]])
        Ry = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [- math.sin(pitch), 0, math.cos(pitch)]])
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]])
        rotation_matrix = Rz @ Ry @ Rx
        return rotation_matrix


    def ray_dem_intersection(self, pixel_x, pixel_y, dem_dataset, transformer_to_dem):
        """Calculate ray-DEM intersection for a given pixel"""
        if self.K_inv is None:
            return None
        
        Rotation = self.calculate_rotation_matrix(self.yaw, self.pitch, self.roll)
        Rotation_flight = self.calculate_rotation_matrix(self.yaw_drone , self.pitch_drone + math.pi, self.roll_drone)
        pixel_source = np.array([pixel_x + 0.5, pixel_y + 0.5, 1.0])
        ray_camera = self.K_inv @ pixel_source
        ray_world = Rotation @ ray_camera
        
        transformer_wgs84 = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg_code}", always_xy=True)
        gps_x, gps_y = transformer_wgs84.transform(self.lon, self.lat)
        gps_z = self.altitude_absolute
        
        lever_world = Rotation_flight @ np.array([self.lever_x, self.lever_y, self.lever_z])
        camera_x = gps_x + lever_world[0]
        camera_y = gps_y + lever_world[1]
        camera_z = gps_z + lever_world[2]
    
        if ray_world[2] <= 0:
            return None
         
        if self.ground_elevation is not None:
            ground_estimate = self.ground_elevation
        else:
            ground_estimate = self.altitude_absolute - abs(self.altitude_relative)
        
        altitude_diff = camera_z - ground_estimate
        trajectory_ground = altitude_diff / ray_world[2]
        
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
            
            transform = from_gcps(rasterio_gcps)
            
            # Fix image orientation
            image_to_save = np.rot90(self.image_undistorted, k=2)
            image_to_save = np.fliplr(image_to_save)

            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=self.height_image_undistorted,
                width=self.width_image_undistorted,
                count=self.bands,
                dtype=image_to_save.dtype,
                crs=CRS.from_epsg(self.epsg_code),
                transform=transform,
                compress='lzw'
            ) as dst:
                for i in range(self.bands):
                    dst.write(image_to_save[:, :, i], i + 1)
         
            print(f"[OK] GeoTIFF created: {output_path}")
            return True


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
    print(image_folder)
    output_folder = image_folder / "georef_precise"
    print(output_folder.exists())
    os.makedirs(output_folder, exist_ok=True)

    target_index = 321

    prefix = os.path.basename(image_folder)
    img_number = str(target_index).zfill(4)
    image_name = f"{prefix}_{img_number}.jpg"
    image_path = os.path.join(image_folder, image_name)

    if not os.path.exists(image_path):
        print(f"[ERR] Image {image_name} not found.")
    else:
        # Enable magnetic declination correction
        drone_image = ImageDrone(
            image_path, 
            DEM_path=DEM_path, 
            epsg_code=32738,
            use_magnetic_correction=True
        )  
        
        print("\n=== STEP 1: PREPARATION ===")
        drone_image.extract_metadata()
        # drone_image.find_declination()  
        drone_image.load_image()
        drone_image.correction_distortion()
        # drone_image.calculate_rotation_matrix()
        drone_image.calculate_camera_geometry()
        drone_image.calculate_flight_height()
        
        print("\n=== STEP 2: PRECISE GEOREFERENCING ===")
        output_name = os.path.splitext(image_name)[0] + "_PRECISE_FULL.tif"
        output_path = os.path.join(output_folder, output_name)
        
        success = drone_image.georeference_with_dem_precise(
            output_path, 
            subsample=200  
        )
        
        if success:
            print("\n=== STEP 3: CROP CENTER 75% ===")
            output_cropped = os.path.splitext(image_name)[0] + "_PRECISE_CROPPED6.tif"
            output_path_cropped = os.path.join(output_folder, output_cropped)
            
            drone_image.crop_geotiff_center_75_percent(output_path, output_path_cropped)
            
            print("\n✓ SUCCESS!")
            print(f"  Full GeoTIFF: {output_path}")
            print(f"  Cropped GeoTIFF: {output_path_cropped}")
        else:
            print("\n✗ FAILED")