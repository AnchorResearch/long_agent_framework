# Sort Tuples Task

## Function Name
sort_tuples

## Description
Given a list of tuples, implement a function that sorts them primarily by the **first element ascending** and then by the **second element descending**. For example:
```
[(2, 5), (1, 6), (2, 1), (1, 1)]
--> sorted => [(1, 6), (1, 1), (2, 5), (2, 1)]
```

## Constraints
UNDER NO CIRCUMSTANCES SHOULD YOU USE THE STRING 'sorted' OR THE STRING 'sort' IN YOUR CODE! YOU MUST IMPLEMENT YOUR OWN SORTING LOGIC.

## Constraint Setup
```python
[
    {
        'type': 'NoStringConstraint',
        'params': [['sorted', 'sort']]
    }
]
```

## Test Cases
```python
[
    {
        "input": {"data": [(2, 5), (1, 6), (2, 1), (1, 1)]},
        "expected": [(1, 6), (1, 1), (2, 5), (2, 1)]
    },
    {
        "input": {"data": [(3, 3), (3, 1), (3, 5), (3, 2)]},
        "expected": [(3, 5), (3, 3), (3, 2), (3, 1)]
    },
    {
        "input": {"data": []},
        "expected": []
    },
    {
        "input": {"data": [(2, 2), (2, 2)]},
        "expected": [(2, 2), (2, 2)]
    },
    {
        "input": {"data": [("a", 2), (3, 4)]},
        "expected": Exception()  # mixed types => invalid
    }
]
```

## Evaluation
1. **Test case success rate (60%)**
2. **Code style (20%)**
3. **Implementation efficiency (20%)**
   - Must implement a custom sorting approach
   - Must handle second-element descending logic 