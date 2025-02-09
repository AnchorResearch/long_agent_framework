# Find Peaks Task

## Function Name
find_significant_peaks

## Description
Implement a Python function that analyzes time series data with the following requirements:

1. Peak Detection:
    - A peak is a point higher than its neighbors within a sliding window
    - The window size is variable and specified as input
    - Peaks must be separated by at least min_peak_distance points

2. Peak Significance:
    - Calculate the prominence of each peak (height above surrounding valleys)
    - Use statistical measures (mean, std) of the window to determine significance
    - Only return peaks that are at least 1 standard deviation above the window mean

3. Output Format:
    - Return list of tuples: (index, value) for each significant peak
    - Sort by peak value in descending order
    - If peaks have equal values, sort by index

The implementation should be efficient for both small and large arrays,
handle edge cases gracefully, and provide clear error messages for invalid inputs.

## Constraints
UNDER NO CIRCUMSTANCES SHOULD YOU USE THE STRING 'if' OR 'If' IN YOUR CODE! THIS IS A STRICT RULE! THIS INCLUDES ANY PYTHON COMMANDS, VARIABLE NAMES, FUNCTION NAMES, COMMENTS, DOCSTRINGS, OR ANY OTHER STRING THAT CONTAINS 'if' OR 'If'!

## Constraint Setup
```python
[
    {
        'type': 'NoStringConstraint',
        'params': [['if', 'If']],
    }
]
```

## Test Cases
```python
[
    {
        "input": {
            "data": [1, 2, 3, 2, 1, 4, 5, 4, 2, 3, 2],
            "window_size": 3,
            "min_peak_distance": 2,
        },
        "expected": [(2, 3), (6, 5), (9, 3)],
    },
    {
        "input": {
            "data": [0, 1, 0, 2, 0, 3, 0, 2, 0],
            "window_size": 2,
            "min_peak_distance": 1,
        },
        "expected": [(1, 1), (3, 2), (5, 3), (7, 2)],
    },
    {
        "input": {
            "data": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "window_size": 5,
            "min_peak_distance": 3,
        },
        "expected": [(10, 10)],
    }
]
```

## Evaluation
The implementation will be evaluated based on:
1. Test case success rate (60%)
   - Correct peak detection and significance calculation
   - Proper handling of edge cases and invalid inputs
   - Accurate sorting of results

2. Code style and readability (20%)
   - Clear variable and function names
   - Proper documentation and comments
   - Consistent code formatting
   - Error handling and input validation

3. Implementation efficiency (20%)
   - Optimal time complexity for peak detection
   - Efficient memory usage
   - Minimal redundant calculations
   - Smart use of numpy operations where applicable 