# Dice Regression

Building on a different project of mine (located here: [https://github.com/DevinGent/probability-play](https://github.com/DevinGent/probability-play)), I will try to perform regression analysis on a set of data generated from rolling dice repeatedly and counting the number of sixes that appear.  In particular, I am curious how the number of rolls per test, and the number of tests, affects the likelihood of rolling about 16% sixes in a test.

This project is licensed under the terms of the MIT license.

## Content Overview

### `create_df.py`

In this script a dataframe is constructed where each row represents a particular experiment.  Each experiment consists of a number of tests.  For a given experiment, each test consists of rolling a six sided die n times (n is recorded in the column 'Rolls per Test').   

