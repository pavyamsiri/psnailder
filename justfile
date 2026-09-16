sync:
    cd py-psnailder && uv sync

benchmark:
	cd py-psnailder && uv run psnailder/__init__.py

build: sync
	cd py-psnailder && uv run maturin develop --release

build-sse: sync
	cd py-psnailder && RUSTFLAGS="-C target-feature=+sse,+sse2" uv run maturin develop --release

build-avx2: sync
	cd py-psnailder && RUSTFLAGS="-C target-feature=+sse,+sse2,+avx,+avx2" uv run maturin develop --release

build-native: sync
	cd py-psnailder && RUSTFLAGS="-C target-cpu=native" uv run maturin develop --release

clean:
	rm -rf py-psnailder/target target
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	
