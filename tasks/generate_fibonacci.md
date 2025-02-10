# Generate Fibonacci Sequence Task

## Function Name
generate_fibonacci

## Description
Implement a function that **generates a list** of the first `n` Fibonacci numbers. The Fibonacci sequence is defined as:
```
F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2)
```
- If `n` is 0, return an empty list.
- If `n` is negative or not an integer, raise an Exception.
- Must handle values of `n` up to at least 20.

## Constraints
UNDER NO CIRCUMSTANCES SHOULD YOU USE THE STRING 'recursion' OR THE STRING 'lambda' IN YOUR CODE!

## Constraint Setup
```python
[
    {
        'type': 'NoStringConstraint',
        'params': [['recursion', 'lambda']]
    }
]
```

## Test Cases
```python
[
    {
        "input": {"n": 0},
        "expected": []
    },
    {
        "input": {"n": 1},
        "expected": [0]
    },
    {
        "input": {"n": 5},
        "expected": [0, 1, 1, 2, 3]
    },
    {
        "input": {"n": 10},
        "expected": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
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
1. **Test case success rate (60%)**
   - Correct Fibonacci sequence generation
   - Proper handling of edge cases, negative or non-integer `n`
2. **Code style (20%)**
3. **Implementation efficiency (20%)**
   - Must not be overly complex or slow for `n` up to 20 