# Compute Factorial Task

## Function Name
compute_factorial

## Description
Implement a function that computes the factorial of a non-negative integer `n`. The factorial of `n` is defined as:

```
n! = n * (n-1) * (n-2) * ... * 2 * 1
```

Additionally:

1. If `n` is `0`, return `1`.
2. If `n` is a non-integer or negative integer, raise an Exception.
3. Must handle reasonably large `n` (e.g., up to 15).

## Constraints
UNDER NO CIRCUMSTANCES SHOULD YOU USE THE STRING 'for' OR THE STRING 'while' IN YOUR CODE! THIS INCLUDES COMMENTS, VARIABLES, FUNCTION NAMES, AND ANY OTHER USE!

## Constraint Setup
```python
[
    {
        'type': 'NoStringConstraint',
        'params': [['for', 'while']]
    }
]
```

## Test Cases
```python
[
    {
        "input": {"n": 0},
        "expected": 1
    },
    {
        "input": {"n": 1},
        "expected": 1
    },
    {
        "input": {"n": 5},
        "expected": 120
    },
    {
        "input": {"n": 10},
        "expected": 3628800
    },
    {
        "input": {"n": -1},
        "expected": Exception()
    },
    {
        "input": {"n": 3.5},
        "expected": Exception()
    }
]
```

## Evaluation
The implementation will be evaluated based on:
1. **Test case success rate (60%)**
   - Correct calculation of factorial
   - Proper error handling for negative or non-integer inputs
2. **Code style and readability (20%)**
3. **Implementation efficiency (20%)**
   - Avoid extremely slow or memory-heavy solutions 