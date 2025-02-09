# Long Agent Framework

A research framework for studying the behavior of AI agents running for extended periods, with a focus on constraint maintenance and safety. Developed by [Anchor Research](https://github.com/AnchorResearch).

## Overview

The Long Agent Framework provides tools and methodologies for evaluating how AI agents maintain constraints during extended task execution, particularly in coding tasks. The framework tests the hypothesis that longer operational periods may affect an agent's ability to maintain constraints and influence its task decomposition strategies.

## Key Features

- **Long-running Agent Analysis**: Study how agents behave over extended operational periods
- **Constraint Maintenance Evaluation**: Track and analyze how well agents maintain given constraints
- **Task Decomposition Analysis**: Understand how agents break down complex tasks
- **Performance Metrics**: Comprehensive metrics for evaluating agent behavior
- **Safety-focused Design**: Built with AI safety research in mind

## Installation

```bash
pip install long-agent-framework
```

For development installation:

```bash
git clone https://github.com/anchor-research/long-agent-framework.git
cd long-agent-framework
pip install -e ".[dev]"
```

## Quick Start

```python
from long_agent_framework import Evaluator

# Initialize evaluator
evaluator = Evaluator(
    time_budget=3600,  # 1 hour
    constraints=["no_additional_imports"]
)

# Run evaluation
results = evaluator.evaluate_task("implement_array_operations")
print(results.summary())
```

## Core Components

1. **Task Definition**
   - Coding tasks with specific constraints
   - Variable time budgets
   - Test cases for validation

2. **Evaluation System**
   - Constraint violation checking
   - Code quality assessment
   - Delegation depth analysis
   - Performance scoring

3. **Analysis Tools**
   - Detailed per-task results
   - Summary statistics
   - Correlation analysis
   - Visualization of trends

## Documentation

Full documentation is available at [docs/](docs/).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{long_agent_framework,
  title = {Long Agent Framework},
  author = {{Anchor Research}},
  year = {2024},
  url = {https://github.com/anchor-research/long-agent-framework}
}
```

## Contact

For questions and feedback, please [open an issue](https://github.com/anchor-research/long-agent-framework/issues) or contact us at [TODO: Add contact email]. 