from setuptools import setup, find_packages
import os

root_dir = os.path.dirname(os.path.realpath(__file__))

# Installation operation
setup(
    name="MFR_benchmark",
    author="fanyangr@umich.edu",
    version="0.0.1",
    description="Multi-finger Reorientation Benchmark featuring dexterous manipulation in ARMLab in NVIDIA IsaacGym.",
    keywords=["robotics"],
    include_package_data=True,
    # python_requires=">=3.6.*",
    packages=find_packages("."),
    zip_safe=False,
)

# EOF
