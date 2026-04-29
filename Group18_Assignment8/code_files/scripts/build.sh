#!/usr/bin/env bash
# ================================================================
#  build.sh — Compile Assignment 08 hybrid MPI + OpenMP code
# ================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${SCRIPT_DIR}/../src"
BIN="${SCRIPT_DIR}/../bin"

mkdir -p "${BIN}"

CC=gcc
CXXFLAGS="-O3 -march=native -fopenmp -funroll-loops -ffast-math -fno-signed-zeros -std=c++17"
LDFLAGS="-lm -fopenmp"
LOCAL_MPI_ROOT="${SCRIPT_DIR}/../../local_mpi/root"
LOCAL_OPENMPI_ROOT="${SCRIPT_DIR}/../../local_openmpi/root"

if [ -d "${LOCAL_OPENMPI_ROOT}/usr/lib/x86_64-linux-gnu/openmpi" ]; then
    CXX=g++
    MPI_INC="${LOCAL_OPENMPI_ROOT}/usr/lib/x86_64-linux-gnu/openmpi/include"
    MPI_INC2="${LOCAL_OPENMPI_ROOT}/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi"
    MPI_LIB="${LOCAL_OPENMPI_ROOT}/usr/lib/x86_64-linux-gnu"
    MPI_LIB2="${LOCAL_OPENMPI_ROOT}/usr/lib/x86_64-linux-gnu/openmpi/lib"
    CXXFLAGS="${CXXFLAGS} -DOMPI_SKIP_MPICXX=1 -I${MPI_INC} -I${MPI_INC2}"
    LDFLAGS="-L${MPI_LIB2} -L${MPI_LIB} -Wl,-rpath,${MPI_LIB2} -Wl,-rpath,${MPI_LIB} -Wl,-rpath-link,${MPI_LIB} ${LDFLAGS} -lmpi"
elif [ -d "${LOCAL_MPI_ROOT}/usr/lib/x86_64-linux-gnu/mpich" ]; then
    CXX=g++
    MPI_INC="${LOCAL_MPI_ROOT}/usr/lib/x86_64-linux-gnu/mpich/include"
    MPI_LIB="${LOCAL_MPI_ROOT}/usr/lib/x86_64-linux-gnu"
    CXXFLAGS="${CXXFLAGS} -I${MPI_INC}"
    LDFLAGS="-L${MPI_LIB} -Wl,-rpath,${MPI_LIB} -Wl,-rpath-link,${MPI_LIB} ${LDFLAGS} -lmpichcxx -lmpich"
elif command -v mpic++ >/dev/null 2>&1; then
    CXX=mpic++
elif command -v mpicxx >/dev/null 2>&1; then
    CXX=mpicxx
else
    echo "[ERROR] mpic++/mpicxx not found. Load an MPI module on the cluster first." >&2
    exit 1
fi

echo "[build] Compiling hybrid MPI+OpenMP pipeline ..."
${CXX} ${CXXFLAGS} \
    "${SRC}/main_mpi.cpp" \
    "${SRC}/init.cpp" \
    "${SRC}/utils.cpp" \
    "${SRC}/mover.cpp" \
    -o "${BIN}/pipeline_mpi" \
    ${LDFLAGS}

echo "[build] Compiling input_fileMaker ..."
${CC} -O2 -o "${BIN}/input_fileMaker" "${SRC}/input_fileMaker.c" -lm

echo "[build] Done. Binaries in ${BIN}/"
ls -lh "${BIN}/"
