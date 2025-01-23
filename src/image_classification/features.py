from abc import abstractmethod
from asyncio import Protocol
from typing import Any

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


class IFeature(Protocol):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> Any: ...

    @staticmethod
    @abstractmethod
    def name() -> str: ...


class AmountOfYellow(IFeature):
    def __call__(self, image: np.ndarray) -> Any:
        """Calculate the amount of yellow color in the image."""

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        return cv2.countNonZero(yellow_mask)

    @staticmethod
    def name() -> str:
        return "yellow_amount"


class AmountOfSilver(IFeature):
    def __call__(self, image: np.ndarray) -> Any:
        """Calculate the amount of silver color in the image."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_silver = np.array([0, 0, 200])
        upper_silver = np.array([180, 25, 255])
        silver_mask = cv2.inRange(hsv_image, lower_silver, upper_silver)

        return cv2.countNonZero(silver_mask)

    @staticmethod
    def name() -> str:
        return "silver_amount"


class AmountOfParallelLines(IFeature):
    def __call__(self, image: np.ndarray) -> Any:
        """Calculate the amount of parallel lines in the image."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        return len(lines) if lines is not None else 0

    @staticmethod
    def name() -> str:
        return "parallel_lines"


class AmountOfCylinders(IFeature):
    def __call__(self, image: np.ndarray) -> Any:
        """Detect cylindrical shapes in the image."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)
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

        return circles.shape[1] if circles is not None else 0

    @staticmethod
    def name() -> str:
        return "cylinder_count"


class AmountOfReflections(IFeature):
    def __call__(self, image: np.ndarray) -> Any:
        """Calculate the amount of bright areas in the image to detect reflections."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

        return cv2.countNonZero(bright_mask)

    @staticmethod
    def name() -> str:
        return "reflection_amount"


class AmountOfTransparency(IFeature):
    def __call__(self, image: np.ndarray) -> Any:
        """
        Calculate the amount of transparent areas in the image.
        Transparent areas are identified as regions with low saturation and brightness in the HSV color space.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_transparent = np.array([0, 0, 0])
        upper_transparent = np.array([180, 50, 50])  # Adjust upper threshold as needed
        transparent_mask = cv2.inRange(hsv_image, lower_transparent, upper_transparent)
        transparency_amount = cv2.countNonZero(transparent_mask)
        return transparency_amount

    @staticmethod
    def name() -> str:
        return "transparency_amount"


class AmountOfTextureSmoothness(IFeature):
    def __call__(self, image: np.ndarray) -> Any:
        """Calculate the smoothness of the texture in the image."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray_image, [1], [0], 256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, "contrast")[0, 0]
        return contrast

    @staticmethod
    def name() -> str:
        return "texture_smoothness"


class AmountOfTextureShininess(IFeature):
    def __call__(self, image: np.ndarray) -> Any:
        """Calculate the shininess of the image based on bright pixels."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_pixels = np.sum(
            gray_image > 220
        )  # Count pixels with intensity greater than 220
        total_pixels = image.shape[0] * image.shape[1]

        return bright_pixels / total_pixels  # Proportion of bright pixels

    @staticmethod
    def name() -> str:
        return "shininess"


class AmountOfSurfaceAnisotropy(IFeature):
    def __call__(self, image: np.ndarray) -> Any:
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
        for theta in np.linspace(0, np.pi, num_orientations, endpoint=False):
            gabor_kernel = cv2.getGaborKernel(
                (kernel_size, kernel_size),
                sigma,
                theta,
                lambd,
                gamma,
                0,
                ktype=cv2.CV_32F,
            )
            filtered_image = cv2.filter2D(gray_image, cv2.CV_32F, gabor_kernel)
            responses.append(np.mean(filtered_image**2))  # Energy of the response

        # Measure anisotropy: ratio of maximum response to mean response
        max_response = max(responses)
        mean_response = np.mean(responses)

        return max_response / mean_response if mean_response > 0 else 0

    @staticmethod
    def name() -> str:
        return "surface_anisotropy"


class AmountOfAspectRatio(IFeature):
    def __call__(self, image: np.ndarray) -> Any:
        """
        Calculate the average width-to-height ratio of detected objects in the image.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
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

        return sum(aspect_ratios) / len(aspect_ratios)

    @staticmethod
    def name() -> str:
        return "aspect_ratio"


class AmountOfWhiteness(IFeature):
    def __call__(self, image: np.ndarray) -> Any:
        """Calculate the proportion of white pixels in the image."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

        return cv2.countNonZero(white_mask) / (image.shape[0] * image.shape[1])

    @staticmethod
    def name() -> str:
        return "whiteness"


class AmountOfLineCurvature(IFeature):
    def __call__(self, image: np.ndarray) -> Any:
        """
        Calculate the average curvature of detected edges in the image.
        High curvature may indicate folds or wavy edges, common in clothing.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            return 0

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

        return sum(curvatures) / len(curvatures) if len(curvatures) > 0 else 0

    @staticmethod
    def name() -> str:
        return "line_curvature"
