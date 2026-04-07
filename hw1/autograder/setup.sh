#!/usr/bin/env bash
# Gradescope autograder setup script.
# Runs once when the autograder Docker image is built.
# All packages installed here are available for every submission run.

set -e  # Exit immediately on error

apt-get update -y
apt-get install -y python3 python3-pip

# Install PyTorch (CPU-only build — no GPU needed for grading)
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies that student code may rely on.
# scipy must be pinned <1.14: cvxpy 1.2.1 relies on spmatrix.__div__,
# which was removed in scipy 1.14.
pip3 install \
    numpy \
    "scipy==1.10.1" \
    "cvxpy==1.2.1" \
    osqp \
    ecos \
    scs
