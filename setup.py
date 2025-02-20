from setuptools import setup, find_packages
setup(
    name="MAMBEV",
    version="0.0.2",
    # packages=[
    #     "petr",
    #     "detr",
    #     "mmcv_dep",
    #     "mambev",
    #     "detr3d",
    #     "dataset_converters"
    # ],
    # packages=find_packages(include=["projects*", "tools*"]),
    packages = find_packages(
        # All keyword arguments below are optional:
        where='.',  # '.' by default
        include=['projects*', 'tools*'],  # ['*'] by default
        exclude=['mypackage.tests'],  # empty by default
    ),

    # py_modules=["dataset_converters"],
    # package_dir={
    #     "dataset_converters":"tools",
    #     "petr":"projects/PETR",
    #     "detr":"projects/DETR_dep",
    #     "mmcv_dep":"projects",
    #     "mambev":"projects/MAMBEV",
    #     "detr3d":"projects/DETR3D",
    # }
)
