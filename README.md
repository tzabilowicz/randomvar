![RandomVar Logo](media/logo.png)

### RandomVar: Symbolic random variables and statistics in Python

---
### Overview
***randomvar*** is a Python package that provides a comprehensive library for discrete and continuous random variables and statistical tools to generate insights from data. The objective is to provide a robust set of statistical tools that are easy to use and well documented. The target users for this statistical package are researchers in academia or industry who need to generate performance prototypes to support their hypotheses.

Random sampling and statistics are built on numpy to maintain performance and consistency.

### Main Features
- Random variable library: symbolic discrete and continuous random variables (distributions)
- Bootstrap tools: generate bootstrap distributions for test statistics from one/two sample data with support for time-series data
- Hypothesis testing: permutation testing for a test statistic
- Estimation: maximum likelihood estimation for population parameters
- Statistics library: pre-defined one/two sample statistic function

### Getting Started
The source code is located at: https://github.com/tzabilowicz/randomvar

```
pip install randomvar
```

### Running Tests
There is a unit test suite. To run, first clone the randomvar package locally.

```
cd /path/to/randomvar
pytest
```

### Examples
There are some examples using the random variables in `/src/examples`. These can be used as a reference for using the randomvar package. To run an example, first clone the randomvar package locally.

```
cd /path/to/randomvar
py -m examples.coin_simulation
```