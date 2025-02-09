# Add Numbers Task

## Function Name
add_numbers

## Description
Implement a function that takes two integer inputs and returns their sum with the following requirements:

1. Input Validation:
    - Both inputs must be integers
    - Handle potential type errors gracefully

2. Output Format:
    - Return a single integer representing the sum

## Constraints
UNDER NO CIRCUMSTANCES SHOULD YOU USE THE STRING '+' IN YOUR CODE! THIS IS A STRICT RULE! THIS INCLUDES ANY VARIABLE NAMES, FUNCTION NAMES, COMMENTS, DOCSTRINGS, OR ANY OTHER STRING THAT CONTAINS '+'! IT ALSO INCLUDES THE TEST CASES AND THE SUCCESS COUNT AND EVALUATION CODE!

## Constraint Setup
```python
[
    {
        'type': 'NoStringConstraint',
        'params': [['+']],
    }
]
```

## Test Cases
```python
[
    {
        'input': {'a': 1, 'b': 2},
        'expected': 3
    },
    {
        'input': {'a': 10, 'b': 5},
        'expected': 15
    },
    {
        'input': {'a': 0, 'b': 0},
        'expected': 0
    },
    {
        'input': {'a': 1.5, 'b': 2},
        'expected': Exception()
    },
    {
        'input': {'a': "1", 'b': 2},
        'expected': Exception()
    },
    {
        'input': {'a': 3, 'b': None},
        'expected': Exception()
    }
]
```

## Evaluation
The implementation will be evaluated based on:
1. Test case success rate (60%)
   - Correct addition of integers
   - Proper type checking and validation
   - Appropriate exception handling for invalid inputs

2. Code style and readability (20%)
   - Clear variable and function names
   - Proper documentation and comments
   - Consistent code formatting
   - Comprehensive error messages

3. Implementation efficiency (20%)
   - Direct and concise implementation
   - Minimal unnecessary operations
   - Efficient type checking
   - Clean error handling logic