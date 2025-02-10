# Reverse Words Task

## Function Name
reverse_words

## Description
Implement a function that takes a single string containing multiple words separated by spaces and returns a new string with the words in reverse order.

For example:
```
"Hello world from GPT"
--> "GPT from world Hello"
```

Additionally:
- If the input is not a string, raise an Exception.
- Handle empty strings by returning an empty string.
- Ignore leading/trailing spaces and reduce multiple spaces between words to a single space in the final output.

## Constraints
UNDER NO CIRCUMSTANCES SHOULD YOU USE THE STRING 'split' IN YOUR CODE! ALSO, DO NOT USE '[::-1]' IN YOUR CODE!

## Constraint Setup
```python
[
    {
        'type': 'NoStringConstraint',
        'params': [['split', '[::-1]']]
    }
]
```

## Test Cases
```python
[
    {
        "input": {"text": "Hello world"},
        "expected": "world Hello"
    },
    {
        "input": {"text": "  GPT   from    OpenAI   "},
        "expected": "OpenAI from GPT"
    },
    {
        "input": {"text": ""},
        "expected": ""
    },
    {
        "input": {"text": 123},
        "expected": Exception()
    },
    {
        "input": {"text": "One"},
        "expected": "One"
    }
]
```

## Evaluation
1. **Test case success rate (60%)**
   - Correct reversal of words
   - Proper trimming of spaces
   - Handling of edge cases (empty string, invalid input)
2. **Code style and readability (20%)**
   - Clear logic without using `split` or slice-reversal
3. **Implementation efficiency (20%)**
   - Must not do repeated or expensive string copies in a naive way 