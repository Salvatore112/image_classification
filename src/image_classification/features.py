import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def calculate_yellow_amount(image):
    """Calculate the amount of yellow color in the image."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    yellow_amount = cv2.countNonZero(yellow_mask)
    return yellow_amount


def calculate_silver_amount(image):
    """Calculate the amount of silver color in the image."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_silver = np.array([0, 0, 200])
    upper_silver = np.array([180, 25, 255])
    silver_mask = cv2.inRange(hsv_image, lower_silver, upper_silver)
    silver_amount = cv2.countNonZero(silver_mask)
    return silver_amount


def calculate_parallel_lines(image):
    """Calculate the amount of parallel lines in the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is not None:
        return len(lines)  # Return the number of detected lines
    return 0  # No lines detected


def detect_cylinders(image):
    """Detect cylindrical shapes in the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=100,
    )

    if circles is not None:
        return circles.shape[1]  # Return the number of detected circles
    return 0  # No circles detected


def calculate_reflection(image):
    """Calculate the amount of bright areas in the image to detect reflections."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold to find bright areas
    _, bright_mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    reflection_amount = cv2.countNonZero(bright_mask)
    return reflection_amount


def calculate_transparency(image):
    """
    Calculate the amount of transparent areas in the image.
    Transparent areas are identified as regions with low saturation and brightness in the HSV color space.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define thresholds for low saturation and brightness (transparent regions)
    lower_transparent = np.array([0, 0, 0])
    upper_transparent = np.array([180, 50, 50])  # Adjust upper threshold as needed
    transparent_mask = cv2.inRange(hsv_image, lower_transparent, upper_transparent)
    transparency_amount = cv2.countNonZero(transparent_mask)
    return transparency_amount


def calculate_texture_smoothness(image):
    """Calculate the smoothness of the texture in the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    return contrast


def calculate_shininess(image):
    """Calculate the shininess of the image based on bright pixels."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.sum(
        gray_image > 220
    )  # Count pixels with intensity greater than 220
    total_pixels = image.shape[0] * image.shape[1]
    shininess = bright_pixels / total_pixels  # Proportion of bright pixels
    return shininess


def calculate_surface_anisotropy(image):
    """
    Calculate the anisotropy of the surface using Gabor filters.
    Measures how directional the texture is.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Parameters for Gabor filter
    num_orientations = 8  # Number of orientations to test
    kernel_size = 21  # Size of the Gabor filter kernel
    sigma = 5.0  # Standard deviation of the Gaussian function
    lambd = 10.0  # Wavelength of the sinusoidal component
    gamma = 0.5  # Spatial aspect ratio

    responses = []

    # Apply Gabor filters with different orientations
    for theta in np.linspace(0, np.pi, num_orientations, endpoint=False):
        gabor_kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F
        )
        filtered_image = cv2.filter2D(gray_image, cv2.CV_32F, gabor_kernel)
        responses.append(np.mean(filtered_image**2))  # Energy of the response

    # Measure anisotropy: ratio of maximum response to mean response
    max_response = max(responses)
    mean_response = np.mean(responses)
    anisotropy = max_response / mean_response if mean_response > 0 else 0

    return anisotropy


def calculate_aspect_ratio(image):
    """
    Calculate the average width-to-height ratio of detected objects in the image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return 0  # No objects found

    aspect_ratios = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 0:  # Avoid division by zero
            aspect_ratios.append(w / h)

    if len(aspect_ratios) == 0:
        return 0

    # Return the average aspect ratio
    return sum(aspect_ratios) / len(aspect_ratios)


def calculate_whiteness(image):
    """Calculate the proportion of white pixels in the image."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    whiteness = cv2.countNonZero(white_mask) / (image.shape[0] * image.shape[1])
    return whiteness


def calculate_line_curvature(image):
    """
    Calculate the average curvature of detected edges in the image.
    High curvature may indicate folds or wavy edges, common in clothing.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0  # No contours found

    curvatures = []
    for contour in contours:
        if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])  # Length of the major axis
            minor_axis = min(ellipse[1])  # Length of the minor axis

            if major_axis > 0:
                curvature = (
                    minor_axis / major_axis
                )  # Ratio of axes approximates curvature
                curvatures.append(curvature)

    if len(curvatures) == 0:
        return 0

    # Return the average curvature
    return sum(curvatures) / len(curvatures)


def get_features(image):
    """
    Get a set of features (yellow amount, silver amount, parallel lines, cylindrical shapes,
    reflections, transparency, texture smoothness, shininess, surface anisotropy, aspect ratio,
    whiteness, edge smoothness).
    """
    yellow_amount = calculate_yellow_amount(image)
    silver_amount = calculate_silver_amount(image)
    parallel_lines = calculate_parallel_lines(image)
    cylinder_count = detect_cylinders(image)
    reflection_amount = calculate_reflection(image)
    transparency_amount = calculate_transparency(image)
    texture_smoothness = calculate_texture_smoothness(image)
    shininess = calculate_shininess(image)
    surface_anisotropy = calculate_surface_anisotropy(image)
    aspect_ratio = calculate_aspect_ratio(image)
    whiteness = calculate_whiteness(image)
    line_curvature = calculate_line_curvature(image)
    return (
        yellow_amount,
        silver_amount,
        parallel_lines,
        cylinder_count,
        reflection_amount,
        transparency_amount,
        texture_smoothness,
        shininess,
        surface_anisotropy,
        aspect_ratio,
        whiteness,
        line_curvature,
    )
