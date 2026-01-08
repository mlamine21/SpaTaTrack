import numpy as np
import os
import cv2
import glob
import pickle
import os
import time
import yaml

def calibrate_camera(checkerboard_size=(6, 4), square_size=40.0, from_images=False):
    """
    Calibrate camera using checkerboard pattern images.
    
    Args:
        checkerboard_size: Tuple of (columns, rows) internal corners in checkerboard
        square_size: Size of checkerboard squares in mm
    
    Returns:
        camera_matrix: Intrinsic camera matrix
        dist_coeffs: Distortion coefficients
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  # Convert to real-world measurements
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Create directory for calibration images if it doesn't exist
    if not os.path.exists('calibration_images'):
        os.makedirs('calibration_images')
    
    # Capture multiple images of the checkerboard
    print("Hold a checkerboard pattern in front of the camera.")
    print("Images will be catpured every second (collect 10-20 images).")
    print("Press 'q' to quit capturing and start calibration.")
    
    if from_images:
        print("Calibrating from existing images...")
        # Load images from the calibration_images directory
        image_files = glob.glob('calibration_images/*.jpg')
        if len(image_files) == 0:
            print("No calibration images found. Please capture images first.")
            return None, None
        img_count = 0
        for img_file in image_files:
            img = cv2.imread(img_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                img_count += 1
                print(f"Loaded {img_file} with corners detected.")
            else:
                print(f"Checkerboard not found in {img_file}.")
    else:

        cap = cv2.VideoCapture(0)
        img_count = 0

        # Time tracking for capturing images every second
        last_capture_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Display the frame
            cv2.imshow('Camera Calibration', frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            # elif key == ord('c'):
            current_time = time.time()
            # Capture an image every second
            if current_time - last_capture_time >= 1.0:
                # Find checkerboard corners
                ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
                
                if ret:
                    # Save the image
                    img_name = f"calibration_images/checkerboard_{img_count}.jpg"
                    cv2.imwrite(img_name, frame)
                    print(f"Saved {img_name}")
                    
                    # Draw and display the corners
                    cv2.drawChessboardCorners(frame, checkerboard_size, corners, ret)
                    cv2.imshow('Corners Detected', frame)
                    
                    
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    img_count += 1
                else:
                    print("Checkerboard not found. Try again.")
                last_capture_time = current_time
        
        cap.release()
        cv2.destroyAllWindows()
    
    if img_count == 0:
        print("No images captured. Calibration failed.")
        return None, None
    
    print(f"Calibrating camera using {img_count} images...")
    
    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    if ret:
        print("Calibration successful!")
        # Save the calibration results
        calibration_data = {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs
        }
        with open('calibration_data.pkl', 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print("Camera matrix:")
        print(camera_matrix)
        print("\nDistortion coefficients:")
        print(dist_coeffs)
        
        return camera_matrix, dist_coeffs
    else:
        print("Calibration failed.")
        return None, None

def load_calibration_data(to_yaml=True):
    """Load camera calibration data if it exists."""
    try:
        with open('calibration_data.pkl', 'rb') as f:
            data = pickle.load(f)
            
        return data['camera_matrix'], data['dist_coeffs']
    except FileNotFoundError:
        print("Calibration data not found. Run calibration first.")
        return None, None   


def test_calibration_quality():
    """Test the quality of camera calibration by calculating reprojection error."""
    camera_matrix, dist_coeffs = load_calibration_data()
    if camera_matrix is None or dist_coeffs is None:
        print("No calibration data available. Run calibration first.")
        return
    
    # Load the calibration images
    image_files = glob.glob('calibration_images/checkerboard_*.jpg')
    if len(image_files) == 0:
        print("No calibration images found.")
        return
    
    print(f"Testing calibration quality using {len(image_files)} images...")
    
    # Prepare object points
    checkerboard_size = (6, 4)  # Must match the size used during calibration
    square_size = 40.0  # Must match the size used during calibration
    
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size
    
    total_error = 0
    total_corners = 0
    
    for img_file in image_files:
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            # Calculate reprojection error
            _, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
            print(rvec, tvec)
            reprojected_points, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
            
            error = cv2.norm(corners, reprojected_points, cv2.NORM_L2) / len(reprojected_points)
            total_error += error
            total_corners += 1
            
            print(f"Image {img_file}: Reprojection error = {error:.4f} pixels")
    
    if total_corners > 0:
        mean_error = total_error / total_corners
        print(f"\nMean reprojection error: {mean_error:.4f} pixels")
        
        if mean_error < 0.5:
            print("Excellent calibration (error < 0.5 pixels)")
        elif mean_error < 1.0:
            print("Good calibration (error < 1.0 pixels)")
        else:
            print("Calibration may need improvement (error >= 1.0 pixels)")
    else:
        print("No valid checkerboard patterns found in the images.")

def test_undistortion():
    """Test calibration by undistorting a live camera feed."""
    camera_matrix, dist_coeffs = load_calibration_data()
    if camera_matrix is None or dist_coeffs is None:
        print("No calibration data available. Run calibration first.")
        return
    
    print("Starting undistortion test.")
    print("Press 'q' to quit.")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Undistort the frame
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # Crop the image based on ROI
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        # Display side by side
        if frame.shape[0] > 0 and frame.shape[1] > 0 and undistorted.shape[0] > 0 and undistorted.shape[1] > 0:
            # Resize for display if necessary
            display_width = 1280
            display_height = int(display_width * frame.shape[0] / frame.shape[1] / 2)
            
            frame_resized = cv2.resize(frame, (display_width//2, display_height))
            undistorted_resized = cv2.resize(undistorted, (display_width//2, display_height))
            
            # Create side-by-side display
            comparison = np.hstack((frame_resized, undistorted_resized))
            
            # Add labels
            cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "Undistorted", (display_width//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Undistortion Test', comparison)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # calibrate_camera(checkerboard_size=(6, 4), square_size=40.0, from_images=True)
    
    # print(load_calibration_data())
    test_calibration_quality()
    # test_undistortion()