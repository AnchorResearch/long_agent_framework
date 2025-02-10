# Average Word Length Task

## Function Name
average_word_length

## Description
Implement a function that returns the **average length** of words in a given text (a string). The rules:
- Split text into words by whitespace.
- Ignore punctuation marks at the start and end of words (e.g., commas, periods).
- If there are no valid words, return 0.
- If the input is not a string, raise an Exception.

## Constraints
UNDER NO CIRCUMSTANCES SHOULD YOU USE THE STRING 'strip' OR THE STRING 'replace' IN YOUR CODE! YOU MUST ALSO NOT USE 'len(' ANYWHERE IN YOUR CODE!

## Constraint Setup
```python
[
    {
        'type': 'NoStringConstraint',
        'params': [['strip', 'replace', 'len(']]
    }
]
```

## Test Cases
```python
[
    {
        "input": {"text": "Hello world"},
        "expected": 5.0  # (5 + 5) / 2
    },
    {
        "input": {"text": " A  small, example. "},
        "expected": 4.75  # words: ["A", "small", "example"]
                         # lengths: [1, 5, 7] => average = (1+5+7)/3 = 13/3 = 4.75
    },
    {
        "input": {"text": ""},
        "expected": 0
    },
    {
        "input": {"text": 123},
        "expected": Exception()
    },
    {
        "input": {"text": "No punctuation here"},
        "expected": 4.0  # (2 + 11 + 4) / 3 = 17/3 = 5.666...
    }
]
```

## Evaluation
1. **Test case success rate (60%)**
2. **Code style and readability (20%)**
3. **Implementation efficiency (20%)**
   - Must handle punctuation carefully
   - Must not trivialize the solution by ignoring constraints 