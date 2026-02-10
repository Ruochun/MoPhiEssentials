# Contributing to MoPhiEssentials

This is a private repository. Contributions are limited to authorized collaborators.

## Development Workflow

1. **Branching**: Create a feature branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Coding Standards**:
   - Follow C++17 standard
   - Use consistent naming conventions (see existing code)
   - Add comments for complex logic
   - Update tests for new features

3. **Testing**: Ensure all tests pass before submitting
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ctest
   ```

4. **Documentation**: Update documentation for API changes

5. **Pull Requests**: Submit PR with clear description of changes

## Code Structure

- Keep header-only implementations where possible for easy integration
- Separate CPU and GPU implementations when necessary
- Use templates for generic implementations
- Maintain backward compatibility

## Adding New Components

1. Create header file in appropriate directory
2. Add to corresponding CMakeLists.txt
3. Create test in `tests/` directory
4. Create example in `examples/` directory
5. Update documentation

## Questions?

Contact the repository maintainers for guidance.
