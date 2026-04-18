![RandomVar Logo](media/logo.png)

### RandomVar: Symbolic random variables and statistics in Python

---
### Overview
***randomvar*** is a Python package that provides a comprehensive library for discrete and continuous random variables and statistical tools to generate insights from data. The objective is to provide a robust set of statistical tools that are easy to use and well documented.

NOTE: This is a work in progress. New features will be added. Once the package is fully complete, it will be published to *pip*.

### Getting Started
To get started, clone the repository to a desired location. Because this is not a published package (yet), install the dependencies:
```
pip install -r requirements.txt
```

### Running Tests
There is a unit test suite. To run:
```
cd /path/to/randomvar
pytest
```

### Examples
There are some examples using the random variables in `/src/examples`. These can be used as a reference for using the randomvar package. To run an example:
```
cd randomvar/src
py -m examples.coin_simulation
```