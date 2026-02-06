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
import quaternion
import geomag

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
        self.declination = 0.0  # Default declination

        
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

        # --- Angles gimbal ---
        self.yaw_gimbal   = math.radians(getf("XMP:GimbalYawDegree"))
        self.pitch_gimbal = math.radians(getf("XMP:GimbalPitchDegree"))
        self.roll_gimbal  = math.radians(getf("XMP:GimbalRollDegree"))

        # --- Angles drone ---
        self.yaw_drone   = math.radians(getf("XMP:FlightYawDegree"))
        self.pitch_drone = math.radians(getf("XMP:FlightPitchDegree"))
        self.roll_drone  = math.radians(getf("XMP:FlightRollDegree"))

        # --- Dewarp / distorsion ---
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

        # --- Centre optique ---
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
   
    def get_projected_corners(self):
        """
        Calcule les coins projetés au sol selon la perspective réelle (avec correction magnétique si self.declination)
        Retourne une liste de 4 points [x, y, z] en coordonnées projetées (EPSG)
        """
        # FOV en radians
        FOVw = 2 * math.atan(self.sensor_width_mm / (2 * self.focal_length_mm))
        FOVh = 2 * math.atan(self.sensor_height_mm / (2 * self.focal_length_mm))

        # Vecteurs coins dans le repère caméra
        rays = [
            np.array([-math.tan(FOVh/2),  math.tan(FOVw/2), 1.0]),
            np.array([-math.tan(FOVh/2), -math.tan(FOVw/2), 1.0]),
            np.array([ math.tan(FOVh/2), -math.tan(FOVw/2), 1.0]),
            np.array([ math.tan(FOVh/2),  math.tan(FOVw/2), 1.0])
        ]
        rays = [v/np.linalg.norm(v) for v in rays]

        # Correction magnétique sur le yaw si self.declination est défini (en radians)
        yaw_corr = self.yaw_gimbal
        if hasattr(self, 'declination') and self.declination:
            yaw_corr = self.yaw_gimbal + self.declination
        print(f"Yaw gimbal (rad): {self.yaw_gimbal:.4f}, Declinaison (rad): {self.declination:.4f}, Yaw corrigé (rad): {yaw_corr:.4f} | (deg): {math.degrees(self.yaw_gimbal):.2f}, {math.degrees(self.declination):.2f}, {math.degrees(yaw_corr):.2f}")
        pitch_gimbal_corr = self.pitch_gimbal + math.radians(90)
        roll_gimbal_corr = self.roll_gimbal + math.radians(180)
        q = quaternion.from_euler_angles(yaw_corr, pitch_gimbal_corr, roll_gimbal_corr)
        q = q.normalized()
        # Appliquer la rotation à chaque rayon
        rays_world = [np.array((q * np.quaternion(0, *v) * q.inverse()).vec) for v in rays]

        # Origine caméra
        transformer_wgs84 = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg_code}", always_xy=True)
        gps_x, gps_y = transformer_wgs84.transform(self.lon, self.lat)
        gps_z = self.altitude_absolute
        lever = np.array([self.lever_x, self.lever_y, self.lever_z])
        camera_pos = np.array([gps_x-lever[0], gps_y-lever[1], gps_z-lever[2]])

        # Intersection avec le sol (z=ground_elevation ou z=0 si pas DEM)
        ground_z = self.ground_elevation if self.ground_elevation is not None else (self.altitude_absolute-abs(self.altitude_relative))
        corners_world = []
        for ray in rays_world:
            if ray[2] == 0:
                continue
            t = (ground_z - camera_pos[2]) / ray[2]
            pt = camera_pos + ray * t
            corners_world.append(pt)
        self.corners_world = np.array(corners_world)
        return self.corners_world

    def find_declination(self):
        """
        Calculate magnetic declination for the image location and date.
        Compatible avec la version actuelle de geomag (datetime.date obligatoire).
        """

        if not self.date_taken:
            print("[WARN] No date → skipping magnetic declination")
            self.declination = 0.0
            return False

        if not GEOMAG_AVAILABLE:
            print("[WARN] geomag library not available → skipping magnetic declination")
            self.declination = 0.0
            return False

        # --- Parse EXIF/XMP date ---
        raw_date = self.date_taken.strip()
        print(f"[DEBUG] RAW DATE = {raw_date}")

        try:
            dt = parser.parse(raw_date)
            dt_date = dt.date()  # important !
            print(f"[DEBUG] Parsed date: {dt_date}")
        except Exception as e:
            print("[ERROR] Could not parse EXIF/XMP date:", raw_date, e)
            self.declination = 0.0
            return False

        # --- Calculate magnetic declination ---
        try:
            dec = declination(self.lat, self.lon, self.altitude_absolute, dt_date)
            self.declination = dec
            print(f"[INFO] Magnetic declination = {dec:.2f}°")

            # --- Correct yaw ---
            if self.use_magnetic_correction:
                d_rad = math.radians(dec)
                self.yaw       += d_rad
                self.yaw_drone += d_rad
                print(f"[INFO] Yaw corrected +{dec:.2f}°")
                print(f"       New Gimbal Yaw: {math.degrees(self.yaw):.2f}°")
                print(f"       New Drone Yaw : {math.degrees(self.yaw_drone):.2f}°")

            return True

        except Exception as e:
            print("[ERROR] Declination calculation failed:", e)
            self.declination = 0.0
            return False

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
                    
                    # Utilisation directe des coins projetés pour les GCPs
                    corners = self.get_projected_corners()
                    for idx, pt in enumerate(corners):
                        gcps.append({
                            'pixel': (int([0, self.width_image_undistorted, self.width_image_undistorted, 0][idx]),
                                      int([0, 0, self.height_image_undistorted, self.height_image_undistorted][idx])),
                            'world': (pt[0], pt[1], pt[2])
                        })
                    break  # On ne fait qu'une fois pour les 4 coins
            
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
            # image_to_save = self.image_undistorted
            image_to_save = np.rot90(self.image_undistorted, k=2)
            image_to_save = np.fliplr(image_to_save)
            # image_to_save = np.flipud(image_to_save)
            

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
    exiftool_path = Path(os.getenv("Exiftool_path"))

    print("Image folder :", image_folder)

    output_folder = image_folder / "georef_precise_fluxes"
    os.makedirs(output_folder, exist_ok=True)

    # List available images (try both .JPG and .jpg)
    images = sorted(image_folder.glob("*.JPG")) + sorted(image_folder.glob("*.jpg"))
    images = list(dict.fromkeys(images))  # Remove duplicates
    
    if not images:
        print(f"[ERR] No JPG images found in {image_folder}")
        print(f"[INFO] Contents of {image_folder}:")
        if image_folder.exists():
            for item in image_folder.iterdir():
                print(f"  - {item.name}")
        else:
            print(f"[ERR] Folder does not exist!")
        exit(1)
    
    print("\n=== AVAILABLE IMAGES ===")
    for i, img in enumerate(images):
        print(f"{i} → {img.name}")
    
    idx = int(input("\nChoose image number: "))
    
    if idx < 0 or idx >= len(images):
        print(f"[ERR] Invalid index {idx}")
        exit(1)
    
    image_path = images[idx]

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

    # Rotation de la photo à 180°
    # drone_image.image_undistorted = np.rot90(drone_image.image_undistorted, 2)
    drone_image.get_projected_corners()
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

        output_cropped = image_path.stem + "_PRECISE_CROPPED_FLUX2.tif"
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