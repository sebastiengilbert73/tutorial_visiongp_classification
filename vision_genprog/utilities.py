import base64
import numpy as np

def ArrayToString(array):
    # Cf. https://stackoverflow.com/questions/30167538/convert-a-numpy-ndarray-to-stringor-bytes-and-convert-it-back-to-numpy-ndarray
    return base64.binascii.b2a_base64(array.astype(float)).decode("ascii")


def StringTo1DArray(string_repr):
    # Cf. https://stackoverflow.com/questions/30167538/convert-a-numpy-ndarray-to-stringor-bytes-and-convert-it-back-to-numpy-ndarray
    return np.frombuffer(base64.binascii.a2b_base64(string_repr.encode("ascii"))).copy()  # To have a writable array
