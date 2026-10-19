#!/usr/bin/env bash
set -e

# Default configuration
BUILD_DIR="build"
BUILD_TYPE="Release"
ENABLE_CUDA="ON"
BUILD_TESTS="ON"
BUILD_BENCHMARKS="ON"
CLEAN_BUILD=false
PARALLEL_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

show_help() {
    echo "Usage: ./build.sh [options]"
    echo ""
    echo "Options:"
    echo "  --cpu, -c            Disable CUDA and build CPU test mode (-DENABLE_CUDA=OFF)"
    echo "  --cuda, -g           Enable CUDA support (-DENABLE_CUDA=ON) [default]"
    echo "  --debug              Build with Debug configuration (-DCMAKE_BUILD_TYPE=Debug)"
    echo "  --release            Build with Release configuration (-DCMAKE_BUILD_TYPE=Release) [default]"
    echo "  --clean              Clean build directory before building"
    echo "  --jobs N, -j N       Number of parallel build jobs [default: $PARALLEL_JOBS]"
    echo "  --no-tests           Disable building unit tests"
    echo "  --no-benchmarks      Disable building benchmarks"
    echo "  --help, -h           Show this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu|-c)
            ENABLE_CUDA="OFF"
            shift
            ;;
        --cuda|-g)
            ENABLE_CUDA="ON"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --jobs|-j)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --no-tests)
            BUILD_TESTS="OFF"
            shift
            ;;
        --no-benchmarks)
            BUILD_BENCHMARKS="OFF"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning build directory $BUILD_DIR..."
    rm -rf "$BUILD_DIR"
fi

echo "Configuring Mini-SGLang build:"
echo "  Build Directory : $BUILD_DIR"
echo "  Build Type      : $BUILD_TYPE"
echo "  CUDA Enabled    : $ENABLE_CUDA"
echo "  Build Tests     : $BUILD_TESTS"
echo "  Build Benchmarks: $BUILD_BENCHMARKS"
echo "  Parallel Jobs   : $PARALLEL_JOBS"

cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DENABLE_CUDA="$ENABLE_CUDA" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -DBUILD_BENCHMARKS="$BUILD_BENCHMARKS"

echo "Building Mini-SGLang..."
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j"$PARALLEL_JOBS"

echo "Build complete! Binary located at $BUILD_DIR/minisgl"
