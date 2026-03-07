"""
Simplified OC-SORT Tracker Implementation
Based on: https://github.com/noahcao/OC_SORT
Adapted for YOLOv8 integration
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class KalmanBoxTracker:
    """
    Kalman Filter tracker for bounding boxes in image space
    State: [x, y, s, r, vx, vy, vs]
    where (x,y) is center, s is scale/area, r is aspect ratio
    """
    count = 0
    
    def __init__(self, bbox, cls, conf, delta_t=3):
        """
        Initialize tracker with a bounding box
        bbox: [x1, y1, x2, y2]
        """
        # Define Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        # Measurement noise
        self.kf.R[2:,2:] *= 10.
        
        # Process noise
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cls = cls
        self.conf = conf
        
        # OC-SORT specific
        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t
        
    def update(self, bbox, cls, conf):
        """
        Update tracker with new detection
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Convert bbox to measurement space
        z = self._convert_bbox_to_z(bbox)
        
        # Store observation
        self.last_observation = np.array([float(z[0]), float(z[1]), float(z[2]), float(z[3]), float(conf)])
        self.observations[self.age] = self.last_observation
        
        # Kalman update
        self.kf.update(z)
        
        # Update velocity
        if self.velocity is None:
            self.velocity = np.array([0, 0])
        
        self.cls = cls
        self.conf = conf
        
    def predict(self):
        """
        Advance state and return predicted bounding box
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if(self.time_since_update>0):
            self.hit_streak = 0
            
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self):
        """
        Return current bounding box estimate
        """
        return self._convert_x_to_bbox(self.kf.x)
    
    def _convert_bbox_to_z(self, bbox):
        """
        Convert [x1,y1,x2,y2] to [x,y,s,r]
        where x,y is center, s is scale/area, r is aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x):
        """
        Convert [x,y,s,r] to [x1,y1,x2,y2]
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([
            x[0]-w/2.,
            x[1]-h/2.,
            x[0]+w/2.,
            x[1]+h/2.
        ]).reshape((1, 4))


class OCSort:
    """
    OC-SORT: Observation-Centric SORT
    """
    def __init__(self, det_thresh=0.3, max_age=30, min_hits=3, 
                 iou_threshold=0.3, delta_t=3, inertia=0.2):
        """
        Initialize OC-SORT tracker
        
        Args:
            det_thresh: Detection confidence threshold
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IOU threshold for matching
            delta_t: Time steps for observation-centric recovery
            inertia: Weight for velocity in prediction
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.inertia = inertia
        
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: numpy array of detections [x1, y1, x2, y2, conf, cls]
        
        Returns:
            tracks: numpy array of active tracks [x1, y1, x2, y2, track_id, cls, conf]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, trks, self.iou_threshold
        )
        
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(
                detections[m[0], :4],
                int(detections[m[0], 5]),
                detections[m[0], 4]
            )
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            if detections[i, 4] >= self.det_thresh:
                trk = KalmanBoxTracker(
                    detections[i, :4],
                    int(detections[i, 5]),
                    detections[i, 4],
                    delta_t=self.delta_t
                )
                self.trackers.append(trk)
        
        # Prepare output
        ret = []
        for trk in self.trackers:
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state()[0]
                ret.append(np.concatenate((
                    d,
                    [trk.id + 1],
                    [trk.cls],
                    [trk.conf]
                )).reshape(1, -1))
        
        # Remove dead trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))
    
    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Assign detections to tracked objects using Hungarian algorithm
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        # Compute IOU matrix
        iou_matrix = self._iou_batch(detections[:, :4], trackers[:, :4])
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = self._linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))
        
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def _iou_batch(self, bb_test, bb_gt):
        """
        Compute IOU between two sets of bounding boxes
        """
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        
        return o
    
    def _linear_assignment(self, cost_matrix):
        """
        Solve linear assignment problem
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return np.column_stack((row_ind, col_ind))