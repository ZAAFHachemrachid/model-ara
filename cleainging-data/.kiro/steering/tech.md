# Tech Stack

## Language & Edition
- Rust 2021 edition

## Dependencies
| Crate | Version | Purpose |
|-------|---------|---------|
| csv | 1.3 | CSV parsing and writing |
| serde | 1.0 | Serialization/deserialization with `derive` feature |
| thiserror | 1.0 | Error type derivation |
| rand | 0.8 | Random sampling for dataset balancing |

## Dev Dependencies
| Crate | Version | Purpose |
|-------|---------|---------|
| proptest | 1.4 | Property-based testing |
| tempfile | 3.10 | Temporary files for tests |

## Common Commands

```bash
# Build
cargo build
cargo build --release

# Run
cargo run

# Test
cargo test

# Check/lint
cargo check
cargo clippy

# Format
cargo fmt
```

## Build Output
- Debug: `target/debug/nlp-fack`
- Release: `target/release/nlp-fack`
