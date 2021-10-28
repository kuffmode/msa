from setuptools import setup, find_packages

DESCRIPTION = "Multi-perturbation Shapley value Analysis (MSA)"
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

base_packages = ["pandas~=1.3.3",
                 "typeguard~=2.13.0",
                 "joblib~=1.1.0",
                 "numpy~=1.20.3",
                 "setuptools~=58.0.4",
                 "tqdm~=4.62.3",
                 "ray~=1.7.0"]
setup(name="msa",
      version="0.0.1",
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      author='Kayson Fakhar',
      author_email='kayson.fakhar@gmail.com',
      url="https://github.com/kuffmode/msa",
      packages=find_packages(),
      classifiers=[
          "Programming Language :: Python :: 3.8",
          "License :: OSI Approved :: MIT License",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering"],
      python_requires='>=3.8',
      install_requires=base_packages,
      include_package_data=True)
