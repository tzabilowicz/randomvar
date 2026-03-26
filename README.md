![RandomVar Logo](docs/logo.png)

### RandomVar: A Symbolic Library for Random Variables

---
### Overview
***randomvar*** is a Python package that provides a comprehensive, symbolic, library for discrete and continuous random variables. The objective is to provide a robust set of statistical tools that are easy to use and well documented.

NOTE: This is a work in progress. New features will be added. Once the package is fully complete, it will be published to *pip*.

### Getting Started
To get started, clone the repository to a desired location. Because this is not a published package (yet), install the dependencies:
```
pip install -r requirements.txt
```

### Running Tests
There is a unit test suite. To run:
```
cd randomvar
pytest
```

### Examples
There are some examples using the random variables in `/src/examples`. These can be used as a reference for using the randomvar package. To run an example:
```
cd randomvar/src
py -m examples.coin_simulation
```

### Future Milestones/Features
- Complete DiscreteDistribution expectation and variance
- Finish continuous distribution library
- Add comprehensive LaTex style documentation
- Expand distribution library