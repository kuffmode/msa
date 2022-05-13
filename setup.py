from setuptools import setup, find_packages

DESCRIPTION = "Multi-perturbation Shapley value Analysis (MSA)"
with open("docs/README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

base_packages = ["pandas~=1.3.3",
                 "typeguard~=2.13.0",
                 "joblib~=1.1.0",
                 "numpy~=1.20.3",
                 "setuptools~=58.0.4",
                 "tqdm~=4.62.3",
                 "ray~=1.7.0",
                 "ordered-set ~= 4.0.2",
                 "matplotlib ~= 3.4.3",
                 "seaborn ~= 0.11.2"
                 ]

test_packages = ["pytest~=6.2.5"]

setup(name="msapy",
      version="1.0.0",
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      author='Kayson Fakhar, Shrey Dixit',
      author_email='kayson.fakhar@gmail.com, shrey.akshaj@gmail.com',
      url="https://github.com/kuffmode/msa",
      packages=find_packages(),
      classifiers=[
          "Programming Language :: Python :: 3.9",
          "License :: OSI Approved :: MIT License",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering"],
      python_requires='>=3.8',
      install_requires=base_packages,
      include_package_data=True)
