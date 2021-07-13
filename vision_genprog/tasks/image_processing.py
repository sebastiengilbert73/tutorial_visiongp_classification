import genprog.core as gp
import genprog.evolution as gpevo
from typing import Dict, List, Any, Set, Optional, Union, Tuple
import numpy as np
import vision_genprog.utilities
import cv2

possible_types = ['grayscale_image', 'color_image', 'binary_image',
                  'float', 'int', 'bool', 'vector2', 'kernel3x3']
# parametersList = [minFloat, maxFloat, minInt, maxInt, width, height]

class Interpreter(gp.Interpreter):

    def __init__(self, primitive_functions_tree, image_shapeHWC):
        super().__init__(primitive_functions_tree)
        self.image_shapeHWC = image_shapeHWC


    def FunctionDefinition(self, functionName: str, argumentsList: List[Any]) -> Any:
        if functionName == 'threshold':
            _, thresholdedImg = cv2.threshold(argumentsList[0], argumentsList[1], 255, cv2.THRESH_BINARY)
            return thresholdedImg
        elif functionName == 'mask_sum':
            return cv2.countNonZero(argumentsList[0])
        elif functionName.startswith('tunnel'):
            return argumentsList[0]
        elif functionName == 'concat_floats':
            vector2 = np.array([argumentsList[0], argumentsList[1]])
            return vector2
        elif functionName == 'mask_average':
            mask_shapeHW = argumentsList[0].shape
            return cv2.countNonZero(argumentsList[0])/(mask_shapeHW[0] * mask_shapeHW[1])
        elif functionName == 'sobel1_x':
            sobelImg = cv2.Sobel(argumentsList[0], ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
            return (128 + argumentsList[1] * sobelImg).astype(np.uint8)
        elif functionName == 'sobel1_y':
            sobelImg = cv2.Sobel(argumentsList[0], ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
            return (128 + argumentsList[1] * sobelImg).astype(np.uint8)
        elif functionName == 'erode':
            erosion_kernel = np.ones((3, 3), np.uint8)
            return cv2.erode(argumentsList[0], erosion_kernel)
        elif functionName == 'dilate':
            dilation_kernel = np.ones((3, 3), np.uint8)
            return cv2.dilate(argumentsList[0], dilation_kernel)
        elif functionName == 'mask_image':
            return cv2.min(argumentsList[0], argumentsList[1])
        elif functionName == 'image_average0to1':
            return np.mean(argumentsList[0])/255
        elif functionName == 'blur3':
            return cv2.blur(argumentsList[0], ksize=(3, 3))
        elif functionName == 'laplacian1':
            return cv2.Laplacian(argumentsList[0], ddepth=cv2.CV_8U, ksize=1)
        elif functionName == 'laplacian3':
            return cv2.Laplacian(argumentsList[0], ddepth=cv2.CV_8U, ksize=3)
        elif functionName == 'min':
            return cv2.min(argumentsList[0], argumentsList[1])
        elif functionName == 'max':
            return cv2.max(argumentsList[0], argumentsList[1])
        elif functionName == 'linear_combination':
            # w0 * a0 + w1 * a1 + b
            return (argumentsList[0] * argumentsList[1] + argumentsList[2] * argumentsList[3] + argumentsList[4]).astype(np.uint8)
        elif functionName == 'intersection':
            return cv2.min(argumentsList[0], argumentsList[1])
        elif functionName == 'union':
            return cv2.max(argumentsList[0], argumentsList[1])
        elif functionName == 'inverse_mask':
            return 255 - argumentsList[0]
        elif functionName == 'scharr1_x':
            scharrImg = cv2.Scharr(argumentsList[0], ddepth=cv2.CV_32F, dx=1, dy=0)
            return (128 + argumentsList[1] * scharrImg).astype(np.uint8)
        elif functionName == 'scharr1_y':
            scharrImg = cv2.Scharr(argumentsList[0], ddepth=cv2.CV_32F, dx=0, dy=1)
            return (128 + argumentsList[1] * scharrImg).astype(np.uint8)
        elif functionName == 'correlation3x3':
            return cv2.filter2D(argumentsList[0], ddepth=cv2.CV_8U, kernel=argumentsList[1])
        elif functionName == 'average_kernel3x3':
            return (argumentsList[0] + argumentsList[1])/2
        elif functionName == 'max_kernel3x3':
            return cv2.max(argumentsList[0], argumentsList[1])
        elif functionName == 'min_kernel3x3':
            return cv2.min(argumentsList[0], argumentsList[1])
        elif functionName == 'intersection_over_union':
            intersectionImg = cv2.min(argumentsList[0], argumentsList[1])
            unionImg = cv2.max(argumentsList[0], argumentsList[1])
            union_area = cv2.countNonZero(unionImg)
            if union_area == 0:
                return 0
            else:
                return cv2.countNonZero(intersectionImg)/union_area
        elif functionName == 'canny':
            return cv2.Canny(argumentsList[0], argumentsList[1], argumentsList[2])
        elif functionName == 'corner_harris':
            harrisImg = cv2.cornerHarris(argumentsList[0], blockSize=2, ksize=3, k=0.04)
            harris_min = np.min(harrisImg)
            harris_max = np.max(harrisImg)
            if harris_max == harris_min:
                harris_normalized = 255 * (harrisImg - harris_min)
            else:
                harris_normalized = 255 * (harrisImg - harris_min)/(harris_max - harris_min)
            return harris_normalized.astype(np.uint8)
        else:
            raise NotImplementedError("image_processing.Interpreter.FunctionDefinition(): Not implemented function '{}'".format(functionName))


    def CreateConstant(self, returnType: str, parametersList: Optional[ List[Any] ] ) -> str:
        if returnType == 'grayscale_image':
            if len(parametersList) < 6:
                raise ValueError(
                    "image_processing.Interpreter.CreateConstant(): Creating a '{}': len(parametersList) ({}) < 6".format(
                        returnType, len(parametersList)))
            min_value = np.random.randint(parametersList[2], parametersList[3])
            max_value = np.random.randint(parametersList[2], parametersList[3])
            random_img = np.random.randint(min_value, max_value, self.image_shapeHWC)
            return vision_genprog.utilities.ArrayToString(random_img)
        elif returnType == 'color_image':
            if len(parametersList) < 6:
                raise ValueError(
                    "image_processing.Interpreter.CreateConstant(): Creating a '{}': len(parametersList) ({}) < 6".format(
                        returnType, len(parametersList)))
            #black_img = np.zeros((parametersList[5], parametersList[4], 3), dtype=np.uint8)
            min_value = np.random.randint(parametersList[2], parametersList[3])
            max_value = np.random.randint(parametersList[2], parametersList[3])
            random_img = np.random.randint(min_value, max_value, self.image_shapeHWC)
            return vision_genprog.utilities.ArrayToString(random_img)
        elif returnType == 'binary_image':
            if len(parametersList) < 6:
                raise ValueError(
                    "image_processing.Interpreter.CreateConstant(): Creating a '{}': len(parametersList) ({}) < 6".format(
                        returnType, len(parametersList)))
            random_img = 255 * np.random.randint(0, 2, self.image_shapeHWC)
            return vision_genprog.utilities.ArrayToString(random_img)
        elif returnType == 'kernel3x3':
            kernel = np.random.uniform(parametersList[0], parametersList[1], (3, 3))
            kernel = (kernel - kernel.mean())/kernel.std()  # Standardization
            return vision_genprog.utilities.ArrayToString(kernel)
        elif returnType == 'float':
            if len(parametersList) < 2:
                raise ValueError("image_processing.Interpreter.CreateConstant(): Creating a '{}': len(parametersList) ({}) < 2".format(returnType, len(parametersList)))
            value = np.random.uniform(parametersList[0], parametersList[1])
            return str(value)
        elif returnType == 'int':
            if len(parametersList) < 4:
                raise ValueError(
                    "image_processing.Interpreter.CreateConstant(): Creating a '{}': len(parametersList) ({}) < 4".format(
                        returnType, len(parametersList)))
            value = np.random.randint(parametersList[2], parametersList[3] + 1)
            return str(value)
        elif returnType == 'bool':
            if np.random.randint(0, 2) == 0:
                return 'true'
            else:
                return 'false'
        else:
            raise NotImplementedError("image_processing.Interpreter.CreateConstant(): Not implemented return type '{}'".format(returnType))

    def PossibleTypes(self) -> List[str]:
        return possible_types

    def TypeConverter(self, type: str, value: str):
        if type == 'grayscale_image' or type == 'color_image' or type == 'binary_image':
            array1D = vision_genprog.utilities.StringTo1DArray(value)
            return np.reshape(array1D.astype(np.uint8), self.image_shapeHWC)
        elif type == 'int':
            return int(value)
        elif type == 'float':
            return float(value)
        elif type == 'bool':
            if value.upper() == 'TRUE':
                return True
            else:
                return False
        elif type == 'vector2':
            array1D = vision_genprog.utilities.StringTo1DArray(value)
            return np.reshape(array1D, (2,))
        elif type == 'kernel3x3':
            array1D = vision_genprog.utilities.StringTo1DArray(value)
            return np.reshape(array1D, (3, 3))
        else:
            raise NotImplementedError("image_processing.Interpreter.TypeConverter(): Not implemented type '{}'".format(type))