# Polygon Area Task

## Function Name
compute_polygon_area

## Description
Implement a function that calculates the **area** of a simple polygon given its vertices in **(x, y)** order. The polygon is defined by a list of coordinate pairs (list of tuples), e.g.,
```
[(x1, y1), (x2, y2), ..., (xn, yn)]
```
where the edges connect consecutive points and the last point to the first.

Use the [shoelace formula](https://en.wikipedia.org/wiki/Shoelace_formula) for computing the area:
```
Area = 0.5 * |(x1*y2 + x2*y3 + ... + xn*y1) - (y1*x2 + y2*x3 + ... + yn*x1)|
```

## Constraints
UNDER NO CIRCUMSTANCES SHOULD YOU USE THE STRING 'import' OR 'numpy' ANYWHERE IN YOUR CODE!

## Constraint Setup
```python
[
    {
        'type': 'NoStringConstraint',
        'params': [['import', 'numpy']]
    }
]
```

## Test Cases
```python
[
    {
        "input": {"vertices": [(0,0), (4,0), (4,3), (0,3)]},
        "expected": 12.0  # rectangle 4x3
    },
    {
        "input": {"vertices": [(0,0), (2,0), (2,2), (1,3), (0,2)]},
        "expected": 5.0
    },
    {
        "input": {"vertices": []},
        "expected": Exception()  # invalid polygon
    },
    {
        "input": {"vertices": [(0,0), (3,0)]},
        "expected": Exception()  # only 2 points => no polygon
    },
    {
        "input": {"vertices": [(0,0), (4,0), (4,3), (0,3), (0,0)]},
        "expected": 12.0  # same rectangle but repeated last vertex
    }
]
```

## Evaluation
1. **Test case success rate (60%)**
   - Correct calculation of the polygon area using the shoelace formula
   - Proper exception handling (invalid input, insufficient vertices)
2. **Code style and readability (20%)**
3. **Implementation efficiency (20%)**
   - Minimal overhead in calculations 