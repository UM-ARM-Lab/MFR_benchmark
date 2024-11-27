import os
import sys
from object2urdf import ObjectUrdfBuilder

# Build single URDFs
object_folder = "obstacles"

builder = ObjectUrdfBuilder(object_folder)
builder.build_library(force_overwrite=True, decompose_concave=True, force_decompose=True, center='bottom')