
import cv2
import numpy as np
import os
from scipy.optimize import linear_sum_assignment

# =============================================================================
# 1. MOG (Background Subtraction from Task 1)
# =============================================================================
class MOG:
    def __init__(self, height, width, number_of_gaussians=3, background_thresh=0.6, lr=0.05):
        self.number_of_gaussians = number_of_gaussians
        self.background_thresh = background_thresh
        self.lr = lr
        self.height = height
        self.width = width
        
        # Initialize Means, Variances, Weights
        # We use float32 for speed
        self.mus = np.zeros((height, width, number_of_gaussians, 3), dtype=np.float32)
        self.sigmaSQs = np.full((height, width, number_of_gaussians), 36.0, dtype=np.float32)
        self.omegas = np.full((height, width, number_of_gaussians), 1.0/number_of_gaussians, dtype=np.float32)
        self.is_initialized = False

    def get_mask(self, img):
        frame = img.astype(np.float32)
        alpha = self.lr

        # Initialization Strategy for Short Video:
        # Set the initial Gaussians to the first frame's colors to avoid initial "ghosting"
        if not self.is_initialized:
            for k in range(self.number_of_gaussians):
                self.mus[:, :, k, :] = frame
            self.is_initialized = True

        # 1. Vectorized Distance Calculation
        diff = frame[:, :, np.newaxis, :] - self.mus
        dist_sq = np.sum(diff**2, axis=3)

        # 2. Check Matches
        matches = dist_sq < (6.25 * self.sigmaSQs)
        match_idx = np.argmax(matches, axis=2)
        has_match = np.any(matches, axis=2)

        # 3. Update Parameters
        self.omegas = (1 - alpha) * self.omegas
        rows, cols = np.indices((self.height, self.width))
        
        # Update matched weights
        self.omegas[rows[has_match], cols[has_match], match_idx[has_match]] += alpha

        # Update matched Means & Variances
        matched_indices = (rows[has_match], cols[has_match], match_idx[has_match])
        self.mus[matched_indices] = (1 - alpha) * self.mus[matched_indices] + alpha * frame[has_match]
        self.sigmaSQs[matched_indices] = (1 - alpha) * self.sigmaSQs[matched_indices] + alpha * dist_sq[matched_indices]

        # 4. Handle No Match (New pixel colors)
        no_match = ~has_match
        self.mus[no_match, -1] = frame[no_match]
        self.sigmaSQs[no_match, -1] = 100.0 # High variance for new objects
        self.omegas[no_match, -1] = alpha

        # 5. Normalize & Sort
        self.omegas /= np.sum(self.omegas, axis=2, keepdims=True)
        fitness = self.omegas / np.sqrt(self.sigmaSQs)
        sorted_indices = np.argsort(fitness, axis=2)[:, :, ::-1]

        self.omegas = np.take_along_axis(self.omegas, sorted_indices, axis=2)
        self.sigmaSQs = np.take_along_axis(self.sigmaSQs, sorted_indices, axis=2)
        self.mus = np.take_along_axis(self.mus, sorted_indices[:, :, :, np.newaxis], axis=2)

        # 6. Background Classification
        diff_sorted = frame[:, :, np.newaxis, :] - self.mus
        dist_sq_sorted = np.sum(diff_sorted**2, axis=3)
        matches_sorted = dist_sq_sorted < (6.25 * self.sigmaSQs)

        cum_weights = np.cumsum(self.omegas, axis=2)
        prev_cum = np.concatenate([np.zeros((self.height, self.width, 1)), cum_weights[:,:,:-1]], axis=2)
        is_bg_model = prev_cum < self.background_thresh

        # Pixel is background if it matches a background distribution
        is_bg_pixel = np.any(matches_sorted & is_bg_model, axis=2)
        
        # Return 0 for BG, 255 for FG
        return np.where(is_bg_pixel, 0, 255).astype(np.uint8)

# =============================================================================
# 2. Particle Filter (Tracking)
# =============================================================================
class ParticleFilter:
    def __init__(self, init_x, init_y, num_particles=100):
        self.num_particles = num_particles
        # State: [x, y, vx, vy]
        self.particles = np.zeros((num_particles, 4), dtype=np.float32)
        self.particles[:, 0] = init_x + np.random.normal(0, 10, num_particles)
        self.particles[:, 1] = init_y + np.random.normal(0, 10, num_particles)
        self.particles[:, 2] = np.random.normal(0, 2, num_particles)
        self.particles[:, 3] = np.random.normal(0, 2, num_particles)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self):
        # Constant Velocity Motion Model
        self.particles[:, 0] += self.particles[:, 2]
        self.particles[:, 1] += self.particles[:, 3]
        # Add process noise
        self.particles[:, 0] += np.random.normal(0, 2, self.num_particles)
        self.particles[:, 1] += np.random.normal(0, 2, self.num_particles)
        
        est_x = np.sum(self.particles[:, 0] * self.weights)
        est_y = np.sum(self.particles[:, 1] * self.weights)
        return est_x, est_y

    def update(self, measurement):
        # Weight particles based on distance to measurement
        dists = np.linalg.norm(self.particles[:, :2] - measurement, axis=1)
        sigma = 30.0
        likelihoods = np.exp(-(dists**2) / (2 * sigma**2))
        self.weights *= likelihoods
        self.weights += 1.e-300 # Avoid zero div
        self.weights /= np.sum(self.weights)
        self.resample()

    def resample(self):
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

# =============================================================================
# 3. Tracker Manager (Counting Logic)
# =============================================================================
class TrackerManager:
    def __init__(self, max_dist=100, max_skip=5, min_hits=3):
        self.tracks = [] 
        self.next_id = 1
        self.max_dist = max_dist
        self.max_skip = max_skip
        self.min_hits = min_hits
        self.confirmed_count = 0

    def update(self, detections):
        # Predict
        track_preds = []
        for track in self.tracks:
            pred_x, pred_y = track['pf'].predict()
            track_preds.append((pred_x, pred_y))

        # Associate (Hungarian Algorithm)
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        cost = np.zeros((num_tracks, num_dets))
        for t in range(num_tracks):
            for d in range(num_dets):
                # Distance between Track Prediction and Detection Centroid
                cost[t, d] = np.linalg.norm(np.array(track_preds[t]) - np.array(detections[d][:2]))

        row_inds, col_inds = linear_sum_assignment(cost)
        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_inds, col_inds):
            # Gating: Don't associate if distance is too large
            if cost[r, c] < self.max_dist:
                self.tracks[r]['pf'].update(detections[c][:2])
                self.tracks[r]['skipped'] = 0
                self.tracks[r]['hits'] += 1
                self.tracks[r]['bbox_size'] = detections[c][2] # Update box size
                
                # Logic: Only count unique people if they persist for min_hits frames
                if not self.tracks[r]['confirmed'] and self.tracks[r]['hits'] >= self.min_hits:
                    self.tracks[r]['confirmed'] = True
                    self.confirmed_count += 1
                
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # Handle Unassigned Tracks
        for i in range(num_tracks):
            if i not in assigned_tracks:
                self.tracks[i]['skipped'] += 1

        # Handle New Detections (Create new tracks)
        for i in range(num_dets):
            if i not in assigned_dets:
                centroid = detections[i][:2]
                bbox_size = detections[i][2]
                new_pf = ParticleFilter(centroid[0], centroid[1])
                self.tracks.append({
                    'pf': new_pf, 
                    'id': self.next_id, 
                    'skipped': 0, 
                    'hits': 1, 
                    'confirmed': False,
                    'bbox_size': bbox_size
                })
                self.next_id += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t['skipped'] <= self.max_skip]
        
        # Return only CONFIRMED tracks for visualization
        return [t for t in self.tracks if t['confirmed']]

# =============================================================================
# 4. Helper Functions
# =============================================================================
def merge_detections(detections, merge_threshold=40):
    # Merge overlapping bounding boxes (e.g. split body parts)
    if len(detections) == 0: return []
    merged = []
    used = [False] * len(detections)
    
    for i in range(len(detections)):
        if used[i]: continue
        current_cluster = [detections[i]]
        used[i] = True
        for j in range(i+1, len(detections)):
            if used[j]: continue
            dist = np.linalg.norm(np.array(detections[i][:2]) - np.array(detections[j][:2]))
            if dist < merge_threshold:
                current_cluster.append(detections[j])
                used[j] = True
        
        cluster_arr = np.array([d[:2] for d in current_cluster])
        avg_x = np.mean(cluster_arr[:, 0])
        avg_y = np.mean(cluster_arr[:, 1])
        # Use largest width/height in cluster
        max_w = max([d[2][0] for d in current_cluster])
        max_h = max([d[2][1] for d in current_cluster])
        merged.append((avg_x, avg_y, (max_w, max_h)))
        
    return merged

# =============================================================================
# 5. Main Execution
# =============================================================================
def main():
    mog = None
    tracker = TrackerManager(max_dist=100, max_skip=5, min_hits=3)
    
    # Kernel for noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    if not os.path.exists('output_task2'): os.makedirs('output_task2')

    for i in range(1, 16):
        filename = f'imgs/{i:04d}.jpg'
        if not os.path.exists(filename): continue
        img = cv2.imread(filename)
        h, w = img.shape[:2]

        # Initialize MOG with first frame dimensions
        if mog is None: 
            mog = MOG(h, w, number_of_gaussians=3, background_thresh=0.6, lr=0.05)

        # 1. Background Subtraction
        mask = mog.get_mask(img)
        
        # 2. Post-processing (Cleaning)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)

        # 3. Detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_detections = []
        
        for c in contours:
            area = cv2.contourArea(c)
            x, y, bw, bh = cv2.boundingRect(c)
            
            # --- Perspective Filter ---
            # Objects higher in the image (smaller y) are further away and appear smaller.
            # We relax the area threshold for top half (y < h/2)
            min_area = 300 if y < h/2 else 800
            
            if area < min_area: continue 
            
            # --- Aspect Ratio Filter ---
            # People are taller than wide (> 0.8). Cars are wider (< 0.8).
            ratio = bh / float(bw)
            if ratio > 0.8: 
                centroid = (x + bw/2, y + bh/2)
                raw_detections.append((centroid[0], centroid[1], (bw, bh)))

        # 4. Merge close detections
        final_detections = merge_detections(raw_detections)

        # 5. Tracking & Counting
        confirmed_tracks = tracker.update(final_detections)

        # 6. Visualization
        for t in confirmed_tracks:
            px, py = t['pf'].predict()
            bw, bh = t['bbox_size']
            
            # Draw Green Box for confirmed people
            top_left = (int(px - bw/2), int(py - bh/2))
            bottom_right = (int(px + bw/2), int(py + bh/2))
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(img, f"ID:{t['id']}", (int(px), int(py)-int(bh/2)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(img, f"Total Unique People: {tracker.confirmed_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imwrite(f'output_task2/final_{i:04d}.jpg', img)
        print(f"Frame {i}: Displaying Tracks {[t['id'] for t in confirmed_tracks]}")

    print(f"\nFinal Verified Count: {tracker.confirmed_count}")

if __name__ == "__main__":
    main()