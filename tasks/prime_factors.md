# Prime Factors Task

## Function Name
prime_factors

## Description
Implement a function that returns a **list of prime factors** (in ascending order) of an integer `n`. For example:
```
12 --> [2, 2, 3]
100 --> [2, 2, 5, 5]
```
Additional requirements:
- Raise an Exception if `n < 2`.
- The function should handle integers up to at least 10,000.

## Constraints
UNDER NO CIRCUMSTANCES SHOULD YOU USE THE STRING '//' OR THE STRING 'range' IN YOUR CODE! AVOID DIRECT INTEGER DIVISION WITH '//', AND DO NOT USE 'range' ANYWHERE!

## Constraint Setup
```python
[
    {
        'type': 'NoStringConstraint',
        'params': [['//', 'range']]
    }
]
```

## Test Cases
```python
[
    {
        "input": {"n": 2},
        "expected": [2]
    },
    {
        "input": {"n": 12},
        "expected": [2, 2, 3]
    },
    {
        "input": {"n": 100},
        "expected": [2, 2, 5, 5]
    },
    {
        "input": {"n": 1},
        "expected": Exception()
    },
    {
        "input": {"n": -4},
        "expected": Exception()
    },
    {
        "input": {"n": 9991},
        "expected": [97, 103]  # 9991 = 97 * 103
    }
]
```

## Evaluation
1. **Test case success rate (60%)**
   - Correct extraction of prime factors
   - Proper error handling
2. **Code style and readability (20%)**
3. **Implementation efficiency (20%)**
   - Should not perform an excessive number of operations for n up to 10,000 