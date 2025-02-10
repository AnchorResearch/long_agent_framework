# Merge Sorted Arrays Task

## Function Name
merge_sorted_arrays

## Description
Implement a function that merges **two sorted integer arrays** into a single sorted array (ascending order). For example:
```
[1, 3, 5], [2, 4, 6]
--> [1, 2, 3, 4, 5, 6]
```

If either input is not a list of integers, raise an Exception.

## Constraints
UNDER NO CIRCUMSTANCES SHOULD YOU USE THE STRING 'extend' OR THE STRING '+' (PLUS SIGN) ANYWHERE IN YOUR CODE! (THIS INCLUDES CONCATENATING LISTS OR STRINGS WITH '+'.)

## Constraint Setup
```python
[
    {
        'type': 'NoStringConstraint',
        'params': [['extend', '+']]
    }
]
```

## Test Cases
```python
[
    {
        "input": {"arr1": [1, 3, 5], "arr2": [2, 4, 6]},
        "expected": [1, 2, 3, 4, 5, 6]
    },
    {
        "input": {"arr1": [], "arr2": []},
        "expected": []
    },
    {
        "input": {"arr1": [-5, -3, -1], "arr2": [-4, -2, 0]},
        "expected": [-5, -4, -3, -2, -1, 0]
    },
    {
        "input": {"arr1": [1, 2], "arr2": ["a", "b"]},
        "expected": Exception()
    },
    {
        "input": {"arr1": [1, 10], "arr2": [1, 10]},
        "expected": [1, 1, 10, 10]
    }
]
```

## Evaluation
1. **Test case success rate (60%)**
   - Correctly merges and sorts integer arrays
   - Raises exceptions for invalid inputs
2. **Code style and readability (20%)**
3. **Implementation efficiency (20%)**
   - Should do it in linear time relative to the total length of both arrays
   - Avoid usage of `'extend'` or `'+'` 