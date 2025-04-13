# Contributing to Raster Features

Thank you for considering contributing to Raster Features! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

There are many ways to contribute to Raster Features:

1. **Report bugs**: Submit issues for any bugs you encounter
2. **Suggest features**: Propose new features or improvements
3. **Improve documentation**: Help make the documentation more clear and comprehensive
4. **Submit code**: Implement new features or fix bugs
5. **Review pull requests**: Help review and test other contributions

## Development Process

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/raster-features.git
   cd raster-features
   ```
3. Create a virtual environment and install development dependencies:
   ```bash
   conda create -n raster-features-dev python=3.9
   conda activate raster-features-dev
   conda install -c conda-forge gdal rasterio
   pip install -e ".[dev]"
   ```

### Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests to ensure your changes don't break existing functionality:
   ```bash
   pytest
   ```
4. Update documentation if necessary
5. Commit your changes with a descriptive commit message:
   ```bash
   git commit -m "Add feature X" -m "Detailed description of the changes"
   ```

### Pull Request Process

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Submit a pull request to the main repository
3. Describe your changes in detail in the pull request description
4. Wait for maintainers to review your pull request
5. Address any feedback or requested changes
6. Once approved, your changes will be merged

## Coding Standards

### Code Style

- Follow PEP 8 style guidelines
- Use descriptive variable names
- Write docstrings for all functions, classes, and modules
- Keep functions focused on a single responsibility
- Use type hints where appropriate

### Testing

- Write tests for all new features and bug fixes
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage

### Documentation

- Update documentation for any changes to the API
- Include examples for new features
- Keep the README.md up to date

## Feature Implementation Guidelines

### Adding New Feature Extractors

1. Create a new module in the appropriate directory (e.g., `raster_features/features/`)
2. Implement the feature extraction function with appropriate documentation
3. Add configuration options to `raster_features/core/config.py`
4. Update the CLI to include the new feature
5. Add tests for the new feature
6. Update documentation

### Modifying Existing Features

1. Ensure backward compatibility or provide migration path
2. Update tests to cover the changes
3. Update documentation to reflect the changes

## Release Process

1. Update version number in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a new release on GitHub
4. Publish to PyPI

## Questions?

If you have any questions about contributing, please open an issue or contact the maintainers.

Thank you for your contributions!
