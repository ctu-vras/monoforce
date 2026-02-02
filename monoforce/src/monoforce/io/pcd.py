"""
Read PCD pointclouds into Numpy arrays.

See the spec at https://pointclouds.org/documentation/tutorials/pcd_file_format.html .
"""

import io

from collections import namedtuple
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


PcdPointFieldType = namedtuple('PcdPointFieldType', ['type_name', 'size'])

# prefix to the names of dummy fields we add to get byte alignment correct. this needs to not
# clash with any actual field names
DUMMY_FIELD_PREFIX = 'padding'

PCD_DATATYPE_TO_NUMPY = {
    PcdPointFieldType('I', 1): np.dtype('int8'),
    PcdPointFieldType('I', 2): np.dtype('int16'),
    PcdPointFieldType('I', 4): np.dtype('int32'),
    PcdPointFieldType('U', 1): np.dtype('uint8'),
    PcdPointFieldType('U', 2): np.dtype('uint16'),
    PcdPointFieldType('U', 4): np.dtype('uint32'),
    PcdPointFieldType('F', 4): np.dtype('float32'),
    PcdPointFieldType('F', 8): np.dtype('float64'),
}


@dataclass
class PCDHeader:
    fields: Sequence[str] = field(default_factory=tuple)
    sizes: Sequence[int] = field(default_factory=tuple)
    types: Sequence[str] = field(default_factory=tuple)
    counts: Sequence[int] = field(default_factory=tuple)
    width: int = 0
    height: int = 0
    viewpoint: Sequence[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    data_type: str = ''
    data_offset: int = -1


def read_pcd_line(pcd_io: io.BytesIO, size=-1):
    """Read a line of text from the PCD header. Ignore comment lines."""

    line = pcd_io.readline(size).decode().rstrip()
    if line.startswith("#"):
        return read_pcd_line(pcd_io, max(-1, size - len(line) - 1))
    return line


def read_pcd_header(pcd: bytes) -> PCDHeader:
    """Read the textual header of a PCD file and return the parsed header structure.
    :param pcd: The bytes of the PCD file.
    :return: The parsed PCD header.
    """
    
    pcd_io = io.BytesIO(pcd)
    header = PCDHeader()
    
    line = read_pcd_line(pcd_io, 200)
    assert line == "VERSION .7"
    
    line = read_pcd_line(pcd_io, 1000)
    assert line.startswith("FIELDS")
    header.fields = tuple(line.split(" ")[1:])
    
    line = read_pcd_line(pcd_io, 100)
    assert line.startswith("SIZE")
    header.sizes = tuple(map(int, line.split(" ")[1:]))
    
    line = read_pcd_line(pcd_io, 100)
    assert line.startswith("TYPE")
    header.types = tuple(line.split(" ")[1:])
    
    line = read_pcd_line(pcd_io, 100)
    assert line.startswith("COUNT")
    header.counts = tuple(map(int, line.split(" ")[1:]))
    
    assert len(header.fields) == len(header.sizes) == len(header.types) == len(header.counts)

    line = read_pcd_line(pcd_io, 100)
    assert line.startswith("WIDTH")
    header.width = int(line.split(" ")[1])
    
    line = read_pcd_line(pcd_io, 100)
    assert line.startswith("HEIGHT")
    header.height = int(line.split(" ")[1])
    
    line = read_pcd_line(pcd_io, 100)
    assert line.startswith("VIEWPOINT")
    header.viewpoint = list(map(float, line.split(" ")[1:]))
    
    line = read_pcd_line(pcd_io, 100)
    assert line.startswith("POINTS")
    num_points = int(line.split(" ")[1])
    
    assert num_points == header.width * header.height
    
    line = read_pcd_line(pcd_io, 100)
    assert line.startswith("DATA")
    header.data_type = line.split(" ")[1]
    
    assert header.data_type in ('ascii', 'binary', 'binary_compressed')
    
    header.data_offset = pcd_io.tell()
    
    return header


def pcd_header_to_numpy_dtype(header: PCDHeader) -> Sequence[Tuple[str, np.dtype]]:
    """Convert a PCD header to a numpy record datatype."""

    offset = 0
    np_dtype_list = []
    for i in range(len(header.fields)):
        field_name = header.fields[i]
        field_size = header.sizes[i]
        field_type = header.types[i]
        field_count = header.counts[i]

        dtype = PCD_DATATYPE_TO_NUMPY[PcdPointFieldType(field_type, field_size)]
        if field_count != 1:
            dtype = np.dtype((dtype, field_count))

        np_dtype_list.append((field_name, dtype))
        offset += field_size * field_count

    return np_dtype_list


def pcd_to_numpy(pcd: bytes, squeeze=True, dummy_field_prefix=DUMMY_FIELD_PREFIX):
    """Convert PCD file content to a numpy recordarray. 

    Reshapes the returned array to have shape (height, width), even if the height is 1.
    """
    header = read_pcd_header(pcd)
    dtype = pcd_header_to_numpy_dtype(header)

    if header.data_type == 'binary':
        cloud_arr = np.frombuffer(pcd[header.data_offset:], dtype)
    else:
        raise NotImplementedError()

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[[fname for fname, _type in dtype if not (fname.startswith(dummy_field_prefix))]]

    if squeeze and header.height == 1:
        return np.reshape(cloud_arr, (header.width,))
    else:
        return np.reshape(cloud_arr, (header.height, header.width))
