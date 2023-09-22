import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class PlayerTracking:
    def __init__(self):
        pass

    def reduce_jitter(self, boxes):
        """
        This method takes in a list of bounding boxes and applies Kalman filtering to reduce jitter in the boxes.
        """
        # Kalman filter initialization
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

        # Apply Kalman filter to each bounding box
        for i, box in enumerate(boxes):
            kf.correct(np.array([box[0], box[1], box[2], box[3]], np.float32))
            boxes[i] = tuple(kf.predict())
        return boxes

    def improve_contrast(self, frame):
            """
             This method takes in a frame and applies histogram equalization to improve the contrast of the players' uniforms.
            """
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(gray, gray)
            return gray

    def reduce_noise(self, frame, kernel_size=(3, 3)):
        """
        This method takes in a frame and applies a median blur to reduce noise.
        """
        return cv2.medianBlur(frame, kernel_size[0])

    def morphological_operation(self, frame, operation='dilation', kernel_size=(3, 3)):
        """
        This method takes in a frame and applies a morphological operation.
        """
        kernel = np.ones(kernel_size, np.uint8)
        if operation == 'dilation':
            return cv2.dilate(frame, kernel, iterations=1)
        elif operation == 'erosion':
            return cv2.erode(frame, kernel, iterations=1)
        elif operation == 'opening':
            return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        else:
            print("Invalid operation. Choose 'dilation', 'erosion', 'opening', or 'closing'.")

    def Grabcut(self):
        # Define the ROI
        x, y, w, h = cv2.selectROI(self)
        roi = self[y:y + h, x:x + w]
        # Create mask with grabcut
        mask = np.zeros(self.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (x, y, w, h)
        cv2.grabCut(self, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        # Multiply image with mask to obtain segmentation
        output_image = self * mask[:, :, np.newaxis]
        return output_image

    def draw_player_bounding_boxes(self, contours):
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(self, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return self

    def meanshift(self, x, y, width, height):
        # Define the region of interest (ROI) by specifying the initial x, y coordinates and width, height
        roi = self[y:y + height, x:x + width]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Set the termination criteria for the meanshift algorithm
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        while True:
            hsv = cv2.cvtColor(self, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, track_window = cv2.meanShift(dst, (x, y, width, height), term_crit)
            x, y, w, h = track_window
            cv2.rectangle(self, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Meanshift", self)
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
        cv2.destroyAllWindows()

    def kmeans_segmentation(self, n_clusters=2):
        # Convert frame to float32 and reshape for k-means input
        self = self.astype(np.float32)
        self = self.reshape((self.shape[0] * self.shape[1], 3))
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self)
        # Get cluster labels for each pixel
        labels = kmeans.labels_
        # Reshape labels to match frame shape
        labels = labels.reshape((self.shape[0], self.shape[1]))
        return labels

    def camshift_segmentation(self, cap, roi_box, np=None):
        hsv = cap.cvtColor(self, cap.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_blue = (125, 85, 85)
        upper_blue = (125, 125, 125)
        # Threshold the HSV image to get only blue colors
        mask = cap.inRange(hsv, lower_blue, upper_blue)
        # Bitwise-AND mask and original image
        res = cap.bitwise_and(self, self, mask=mask)
        # apply camshift on the result
        r, roi_box = cap.CamShift(res, roi_box, term_crit=6)
        pts = cap.boxPoints(r)
        pts = np.int0(pts)
        img2 = cap.polylines(self, [pts], True, 255, 2)
        cap.imshow("Camshift Segmentation", img2)
        cap.waitKey(0)
        return img2

    def morph_image(image):
        # Define kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Apply morphological operations to the image
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.dilate(image, kernel, iterations=1)
        return image

    def GMM_segmentation(frame):
        # Reshape the frame to a 2D array of (x, y, 3)
        frame_reshape = frame.reshape((-1, 3))
        # Define the GMM model with 2 components
        gmm = GaussianMixture(n_components=2)
        # Fit the GMM model to the frame
        gmm.fit(frame_reshape)
        # Get the labels for each pixel
        labels = gmm.predict(frame_reshape)
        # Reshape the labels to the original frame shape
        labels = labels.reshape(frame.shape[:2])
        # Create a mask with the labels
        mask = np.zeros_like(frame)
        mask[labels == 0] = (255, 255, 255)
        return mask

    class PlayerTracking:
        def __init__(self):
            pass

    def convert_to_bgr(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def create_binary_mask(self, frame):
        frame = cv2.fromarray(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.medianBlur(mask, 15)
        return mask

    def detect_players(self, frame):
        # Apply noise reduction
        frame = self.reduce_noise(frame)
        # Improve contrast on player uniforms
        frame = self.improve_contrast(frame)
        mask = self.create_binary_mask(frame)
        # Apply morphological operations
        mask = self.morphological_operation(mask)
        # Use Camshift to track players
        players, boxes = self.camshift_segmentation(frame, mask)
        # Apply jitter reduction
        players, boxes = self.reduce_jitter(players, boxes)
        print(players, boxes)
        return players, boxes
