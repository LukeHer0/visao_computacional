import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('./img/lab/bottle-front.jpeg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('./img/lab/bottle-right.jpeg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize AKAZE detector
akaze = cv2.AKAZE_create()

# Find keypoints and descriptors with AKAZE
keypoints1, descriptors1 = akaze.detectAndCompute(img1_gray, None)
keypoints2, descriptors2 = akaze.detectAndCompute(img2_gray, None)

# FLANN parameters (for AKAZE, you can use KDTree)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

# Create FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match descriptors
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
ratio_threshold = 0.75
for m, n in matches:
    if m.distance < ratio_threshold * n.distance:
        good_matches.append(m)

# Sort good matches by distance.
good_matches = sorted(good_matches, key=lambda x: x.distance)


def draw_matches_mpl(img1, keypoints1, img2, keypoints2, matches):
    """Draw matches using Matplotlib."""
    # Convert images to RGB for Matplotlib
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    # Combine images horizontally
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1
    combined_img[:h2, w1:] = img2

    plt.figure(figsize=(12, 6))
    plt.imshow(combined_img)
    plt.axis('off')

    # Draw lines connecting the matches
    offset = w1
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0] + offset), int(pt2[1])
        plt.plot([x1, x2], [y1, y2], 'r-', linewidth=0.5)
        plt.plot(x1, y1, 'bo', markersize=3)  # Mark keypoints
        plt.plot(x2, y2, 'bo', markersize=3)

    return plt


# Draw matches using the custom function
plt = draw_matches_mpl(img1, keypoints1, img2, keypoints2, good_matches)

# Show the image with matches using Matplotlib
plt.title('Matches')
plt.show()
