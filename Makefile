.PHONY: help sync build build-sse build-avx2 build-native clean

# Default target
.DEFAULT_GOAL := help

help:
	@echo "psnailder build targets:"
	@echo ""
	@echo "  make sync              - Run uv sync to install dependencies"
	@echo "  make build             - Default build (SSE/SSE2 only)"
	@echo "  make build-sse         - Explicit SSE/SSE2 build"
	@echo "  make build-avx2        - Build with AVX2 support"
	@echo "  make build-native      - Build with native CPU features (auto-detect)"
	@echo "  make clean             - Clean build artifacts"
	@echo ""

# Sync dependencies
sync:
	cd py-psnailder && uv sync

# Default build (no special flags)
build: sync
	cd py-psnailder && uv run maturin develop --release

# Explicit SSE build
build-sse: sync
	cd py-psnailder && RUSTFLAGS="-C target-feature=+sse,+sse2" uv run maturin develop --release

# AVX2 build
build-avx2: sync
	cd py-psnailder && RUSTFLAGS="-C target-feature=+sse,+sse2,+avx,+avx2" uv run maturin develop --release

# Native build (auto-detects CPU features)
build-native: sync
	cd py-psnailder && RUSTFLAGS="-C target-cpu=native" uv run maturin develop --release

# Clean build artifacts
clean:
	rm -rf py-psnailder/target target
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
