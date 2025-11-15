#!/bin/bash
# Initialize git submodules for external dependencies
# This script sets up the TrustEval-toolkit as a git submodule

set -e  # Exit on error

echo "🔧 Setting up git submodules for compliance-mapper..."
echo

# Navigate to project root
cd "$(dirname "$0")"

# Check if external directory exists
if [ ! -d "external" ]; then
    echo "📁 Creating external directory..."
    mkdir -p external
fi

# Remove existing TrustEval-toolkit if it's not a submodule
if [ -d "external/TrustEval-toolkit" ] && [ ! -f "external/TrustEval-toolkit/.git" ]; then
    echo "🗑️  Removing existing TrustEval-toolkit directory (not a submodule)..."
    rm -rf external/TrustEval-toolkit
fi

# Try to add TrustEval-toolkit as a submodule
echo "📥 Adding TrustEval-toolkit as a git submodule..."
echo

# Try different possible repository URLs
REPOS=(
    "https://github.com/TrustAI-laboratory/TrustEval-toolkit.git"
    "https://github.com/TrustGen/TrustEval-toolkit.git"
    "https://github.com/trusteval/TrustEval-toolkit.git"
)

SUCCESS=false
for REPO in "${REPOS[@]}"; do
    echo "Trying: $REPO"
    if git submodule add "$REPO" external/TrustEval-toolkit 2>/dev/null; then
        SUCCESS=true
        echo "✅ Successfully added submodule from: $REPO"
        break
    else
        echo "❌ Failed to add from: $REPO"
    fi
done

if [ "$SUCCESS" = false ]; then
    echo
    echo "⚠️  Could not automatically add TrustEval-toolkit as a submodule."
    echo "Please manually add it using:"
    echo "  git submodule add <correct-repo-url> external/TrustEval-toolkit"
    echo
    echo "Check SUBMODULE_SETUP.md for more information."
    exit 1
fi

# Initialize and update submodules
echo
echo "🔄 Initializing and updating submodules..."
git submodule init
git submodule update

echo
echo "✅ Submodule setup complete!"
echo
echo "Next steps:"
echo "  1. Install the package: pip install -e ."
echo "  2. Set up API keys in .env file"
echo "  3. Run benchmarks: python src/run_benchmarks.py"
echo

