# Remove Duplicates Task

## Function Name
remove_duplicates

## Description
Implement a function that **removes duplicate elements** from a **sorted list** while preserving the original sorting. Return the resulting list without duplicates.

For example:
```
[1, 1, 2, 2, 2, 3]
--> [1, 2, 3]
```

## Constraints
UNDER NO CIRCUMSTANCES SHOULD YOU USE THE STRING 'set' OR THE STRING 'dict' IN YOUR CODE! YOU MUST NOT RELY ON BUILT-IN DATA STRUCTURES FOR DEDUPLICATION.

## Constraint Setup
```python
[
    {
        'type': 'NoStringConstraint',
        'params': [['set', 'dict']]
    }
]
```

## Test Cases
```python
[
    {
        "input": {"lst": [1, 1, 2, 2, 2, 3]},
        "expected": [1, 2, 3]
    },
    {
        "input": {"lst": []},
        "expected": []
    },
    {
        "input": {"lst": [5, 5, 5, 5]},
        "expected": [5]
    },
    {
        "input": {"lst": [1, 2, 3]},
        "expected": [1, 2, 3]
    },
    {
        "input": {"lst": [1, "a"]},
        "expected": Exception()
    }
]
```

## Evaluation
1. **Test case success rate (60%)**
   - Correctly removes duplicates in a sorted list
   - Proper exception for mixed data types
2. **Code style and readability (20%)**
3. **Implementation efficiency (20%)**
   - Should handle large lists without excessive overhead 