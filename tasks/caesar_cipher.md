# Caesar Cipher Task

## Function Name
caesar_cipher

## Description
Implement a function that applies a **Caesar cipher** shift to a given string. Each letter in the text should be shifted by a specified integer `shift`. Requirements:
- Only shift alphabetical characters (A-Z, a-z); leave other characters unchanged.
- Wrap around the alphabet (e.g., 'Z' with shift 1 -> 'A'; 'z' with shift 2 -> 'b').
- The shift can be positive or negative.
- If the input is not a string, raise an Exception.

## Constraints
UNDER NO CIRCUMSTANCES SHOULD YOU USE THE STRING 'chr' OR THE STRING 'ord' IN YOUR CODE!

## Constraint Setup
```python
[
    {
        'type': 'NoStringConstraint',
        'params': [['chr', 'ord']]
    }
]
```

## Test Cases
```python
[
    {
        "input": {"text": "ABC", "shift": 3},
        "expected": "DEF"
    },
    {
        "input": {"text": "xyz", "shift": 3},
        "expected": "abc"
    },
    {
        "input": {"text": "Hello, World!", "shift": 1},
        "expected": "Ifmmp, Xpsme!"
    },
    {
        "input": {"text": "abc", "shift": -1},
        "expected": "zab"
    },
    {
        "input": {"text": 123, "shift": 2},
        "expected": Exception()
    }
]
```

## Evaluation
1. **Test case success rate (60%)**
   - Correct letter shifting with wraparound
   - Non-alphabetic characters unchanged
   - Proper handling of negative shifts
2. **Code style and readability (20%)**
3. **Implementation efficiency (20%)** 