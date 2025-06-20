---
applyTo: "**"
---

# Python Code Guidelines and Review Standards

## 1. Code Style and Formatting (PEP 8)

### Essential Style Rules
- **Indentation:** Always use 4 spaces per level (never tabs)
- **Line Length:** Maximum 200 characters per line
- **Naming Conventions:**
  - Variables/functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
- **Blank Lines:** 
  - Two blank lines between top-level functions and classes
  - One blank line between methods within a class
- **Import Organization:**
  1. Standard library imports
  2. Third-party library imports
  3. Local application imports
  - Separate each group with blank lines

### Documentation Standards
- **Docstrings:** Use Google, NumPy, or Sphinx style for all functions, classes, and modules
- **Comments:** Explain WHY, not WHAT. Focus on business logic and non-obvious decisions
- **Comment When:**
  - Complex business logic
  - Performance optimizations
  - Third-party library workarounds
  - Regex patterns
  - Security considerations
  - Error handling strategies

### Code Organization
- **Single Responsibility:** Functions and classes should have one clear purpose
- **Module Structure:** Group related functionality logically
- **Package Structure:** Use `__init__.py` files properly
- **Avoid Deep Nesting:** Keep indentation levels minimal

## 2. Correctness and Logic

### Functional Requirements
- **Requirements Compliance:** Code must correctly implement all specified requirements
- **Edge Case Handling:** Handle null inputs, empty collections, boundary conditions, and invalid inputs
- **Data Validation:** Validate all user input and external data

### Error Handling Best Practices
- **Specific Exceptions:** Catch specific exceptions (`FileNotFoundError`, `ValueError`, `TypeError`) rather than broad `except Exception:` blocks
- **Graceful Degradation:** Provide informative error messages and handle expected failures gracefully
- **Resource Management:** Use `with` statements for files, network connections, and other resources

### Algorithm Efficiency
- **Algorithm Choice:** Select appropriate algorithms considering time/space complexity (Big O notation)
- **Data Structures:** Use appropriate data structures:
  - `set` for fast lookups
  - `list` for ordered collections
  - `dict` for key-value pairs
  - `collections.deque` for efficient appends/pops from both ends
- **Performance:** Avoid redundant operations and unnecessary iterations over large datasets

## 3. Design Principles and Architecture

### SOLID Principles
- **Single Responsibility Principle (SRP):** Each class should have one reason to change
- **Open/Closed Principle (OCP):** Open for extension, closed for modification
- **Liskov Substitution Principle (LSP):** Subtypes must be substitutable for their base types
- **Interface Segregation Principle (ISP):** Clients shouldn't depend on unused interfaces
- **Dependency Inversion Principle (DIP):** Depend on abstractions, not concretions

### Key Design Patterns
- **Creational Patterns:** 
  - Factory Method: Create objects without specifying exact class
  - Singleton: Ensure only one instance exists (use with caution)
  - Builder: Construct complex objects step by step
- **Structural Patterns:** 
  - Adapter: Allow incompatible interfaces to work together
  - Decorator: Add new behaviors without modifying structure
  - Facade: Provide simplified interface to complex subsystem
  - Composite: Treat individual objects and compositions uniformly
  - Proxy: Provide placeholder/surrogate for another object
- **Behavioral Patterns:** 
  - Strategy: Define family of algorithms and make them interchangeable
  - Observer: Define one-to-many dependency between objects
  - Command: Encapsulate requests as objects
  - Iterator: Access elements sequentially without exposing underlying representation
  - Template Method: Define skeleton of algorithm, defer steps to subclasses

### Core Design Principles
- **Don't Repeat Yourself (DRY):** Extract common logic into reusable functions, classes, or modules. If you find yourself writing the same logic more than once, abstract it to reduce maintenance overhead and likelihood of bugs
- **Keep It Simple, Stupid (KISS):** Always strive for the simplest possible solution that meets requirements. Avoid unnecessary complexity, fancy patterns, or over-engineering. Simplicity is the best path to reliability and maintainability
- **You Aren't Gonna Need It (YAGNI):** Don't implement functionality that is not currently required, no matter how certain you are it will be needed in the future. Focus on immediate needs and build incrementally
- **Composition Over Inheritance:** Prefer composing objects with desired behaviors rather than relying on deep inheritance hierarchies. This leads to more flexible and robust designs, avoiding the "fragile base class" problem

### Architecture Guidelines
- **Separation of Concerns:** High cohesion within modules, low coupling between modules
- **Dependency Injection:** Pass dependencies rather than creating them internally

## 4. Pythonic Practices and Idioms

### Core Python Features
- **List Comprehensions/Generator Expressions:** Use for transformations and filtering where they improve readability
- **Context Managers (`with` statements):** Use for managing resources (files, locks, database connections)
- **enumerate() for Loops:** Use when both index and value are needed in a loop
- **zip() for Parallel Iteration:** Use when iterating over multiple iterables in parallel
- **dict.get() for Dictionary Access:** Use when a default value is preferable to raising a KeyError
- **f-strings for String Formatting:** Preferred for clear and concise string formatting

### Advanced Python Features
- **collections Module:** Use appropriate data structures (defaultdict, Counter, deque) where they provide advantages
- **Custom Context Managers:** Use for managing application-specific resources
- **Generators/Iterators:** Use for producing large sequences efficiently when memory is a concern
- **Decorators:** Use for adding cross-cutting concerns (logging, authentication, timing) without modifying original function code

## 5. Code Quality and Maintenance

### Code Smells to Avoid
- **Magic Numbers/Strings:** Replace hardcoded literal values with named constants
- **Long Parameter Lists:** Group related parameters into objects
- **Feature Envy:** Methods should focus on their own class's data
- **Shotgun Surgery:** Single changes shouldn't require modifications in many places
- **Duplicated Code:** Extract common logic into reusable functions
- **Dead Code:** Remove unreachable or unused code
- **Over/Under-engineering:** Balance complexity with current and future needs

### Testing and Quality Assurance
- **Unit Tests:** Write comprehensive tests covering critical functionality and edge cases
- **Test Coverage:** Focus on critical paths rather than just high percentage coverage
- **Clear Assertions:** Make test assertions specific and meaningful
- **Mocking:** Mock external dependencies (databases, APIs) appropriately in tests

### Security Considerations
- **Input Validation:** Validate and sanitize all user input and external data
- **Sensitive Data Handling:** Never hardcode secrets; use environment variables or secret management
- **Least Privilege:** Operate with minimum necessary privileges

## 6. Tools and Automation

### Development Tools
- **Linters:** Use flake8, pylint, black (auto-formatter) to enforce style and catch basic errors
- **Type Hinting:** Use type hints (mypy for static analysis) to improve code clarity and catch type-related bugs
- **IDE Features:** Leverage refactoring tools, debugger, and code analysis features
- **Version Control:** Ensure proper Git practices with meaningful commit messages and branching strategy
- **CI/CD Integration:** Automate testing, linting, and deployment processes

### Dependency Management
- **Virtual Environments:** Always use virtual environments (venv, conda, pipenv, poetry) to prevent dependency conflicts
- **Requirements Management:** Pin all dependencies in requirements.txt or pyproject.toml for reproducibility

## 7. Architectural Guidelines

### Architectural Patterns
- **Choose Appropriate Style:**
  - **Monolith:** Simple, single deployment unit (good for small projects)
  - **Layered Architecture:** Separate presentation, application, business logic, and data access layers
  - **Hexagonal Architecture:** Separate business logic from external concerns (UI, databases, APIs)
  - **Event-Driven Architecture:** Components communicate via events for decoupling

### Design Considerations
- **Separation of Concerns:** High cohesion within modules, low coupling between modules
- **Scalability Planning:** Identify potential bottlenecks and consider asynchronous programming for I/O-bound tasks
- **Security by Design:** Implement input validation, authentication, authorization, and secret management
- **Observability:** Use structured logging, metrics, and tracing for monitoring and debugging

### Core Design Principles
- **Single Responsibility Principle:** A class or module should have only one reason to change
- **Open/Closed Principle:** Open for extension, closed for modification
- **Liskov Substitution Principle:** Subtypes must be substitutable for their base types
- **Interface Segregation Principle:** Create fine-grained, specific interfaces
- **Dependency Inversion Principle:** Depend on abstractions, not concretions
- **Composition Over Inheritance:** Prefer composing objects over deep inheritance hierarchies


### Note
- **PROJECT.md**: We have to reffer this file to get updated information for the project structure, dependencies, and architecture.
- **Update PROJECT.md**: Ensure that any changes to project structure, dependencies, or architecture are reflected in the `PROJECT.md` file.
- **requirements.txt**: Always keep the `requirements.txt` file up-to-date with all dependencies, including development and production requirements.
- **README.md**: Maintain an up-to-date `README.md` file with project overview, setup instructions, and usage examples.
- **Changelog**: Document all changes in a `CHANGELOG.md` file to track project evolution and updates.
- **Contributing Guidelines**: Ensure that the `CONTRIBUTING.md` file is clear and provides all necessary information for contributors to follow the coding standards and practices outlined here.
- **Code Reviews**: All code changes must be reviewed by at least one other developer to ensure adherence to these standards and improve code quality.