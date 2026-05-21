# psnailder

An implementation of the phase spiral fitting algorithm described in
[Alinder et al. 2023](https://scixplorer.org/abs/2023A%26A...678A..46A/abstract).

my-project/
├── Cargo.toml                  # Workspace root
├── pyproject.toml              # Python project metadata (for the whole repo)
├── Makefile                    # Convenience build commands
│
├── crates/                     # All pure-Rust crates
│   ├── my-project-core/        # Core logic, no Python deps
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs
│   ├── my-project-ops/         # Operations built on core
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs
│   └── my-project-error/       # Shared error types
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs
│
└── py-my-project/              # The Python-facing crate (maturin lives here)
    ├── Cargo.toml              # Depends on your crates + pyo3
    ├── pyproject.toml          # maturin config
    ├── Makefile
    ├── src/
    │   └── lib.rs              # PyO3 bindings only — thin glue layer
    └── my_project/             # Python package
        ├── __init__.py
        ├── _internal.pyi       # Type stubs for the Rust extension
        └── submodule.py        # Pure-Python convenience wrappers
