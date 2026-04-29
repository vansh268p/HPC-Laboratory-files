#!/usr/bin/env bash
# ================================================================
#  run_all.sh — Run all 5 configs × core counts for Assignment 08
#
#  Produces data_cluster/summary.csv with per-iteration timing:
#    config,variant,mpi_ranks,omp_threads,total_cores,Nx,Ny,
#    N_particles,local_particles,Maxiter,iteration,
#    t_interp_s,t_comm_s,t_norm_s,t_mover_s,t_denorm_s,t_total_s,n_active
# ================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SCRIPT_DIR}/../.."
BIN="${ROOT}/code_files/bin"
DATA="${ROOT}/code_files/data"
RAW="${ROOT}/data_cluster/raw"
SUMMARY="${ROOT}/data_cluster/summary.csv"

mkdir -p "${DATA}" "${RAW}" "${ROOT}/results"

# ── Build first ──────────────────────────────────────────────────
bash "${SCRIPT_DIR}/build.sh"

FILEGEN="${BIN}/input_fileMaker"
PIPELINE="${BIN}/pipeline_mpi"

# ── Configurations ───────────────────────────────────────────────
CONFIGS=(
    "a 250  100  900000   10"
    "b 250  100  5000000  10"
    "c 500  200  3600000  10"
    "d 500  200  20000000 10"
    "e 1000 400  14000000 10"
)

# total_cores mpi_ranks omp_threads
# For 8+ total cores this keeps all four course nodes active.
LAYOUTS=(
    "1  1 1"
    "2  2 1"
    "4  4 1"
    "8  4 2"
    "16 4 4"
    "32 4 8"
    "64 4 16"
)

BEST_VARIANT="private_reduction"

MPI_HOSTS="${MPI_HOSTS:-gics1,gics2,gics3,gics4}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:-}"
LOCAL_MPI_ROOT="${ROOT}/local_mpi/root"
LOCAL_OPENMPI_ROOT="${ROOT}/local_openmpi/root"
MPIRUN="${MPIRUN:-mpirun}"
if [ -x "${LOCAL_OPENMPI_ROOT}/usr/bin/mpirun.openmpi" ]; then
    MPIRUN="${LOCAL_OPENMPI_ROOT}/usr/bin/mpirun.openmpi"
    export LD_LIBRARY_PATH="${LOCAL_OPENMPI_ROOT}/usr/lib/x86_64-linux-gnu:${LOCAL_OPENMPI_ROOT}/usr/lib/x86_64-linux-gnu/openmpi/lib:${LD_LIBRARY_PATH:-}"
    export OPAL_PREFIX="${LOCAL_OPENMPI_ROOT}/usr"
    export OMPI_MCA_btl="${OMPI_MCA_btl:-self,vader}"
    export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
    export PMIX_MCA_pcompress="${PMIX_MCA_pcompress:-^zlib}"
    MPI_HOSTS=""
elif [ -x "${LOCAL_MPI_ROOT}/usr/bin/mpirun.mpich" ]; then
    MPIRUN="${LOCAL_MPI_ROOT}/usr/bin/mpirun.mpich"
    export LD_LIBRARY_PATH="${LOCAL_MPI_ROOT}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    export UCX_TLS="${UCX_TLS:-self,shm}"
    MPI_HOSTS=""
fi
HOST_ARGS=()
if [ -n "${MPI_HOSTS}" ]; then
    HOST_ARGS=(--host "${MPI_HOSTS}")
fi

# ── Write CSV header ─────────────────────────────────────────────
echo "config,variant,mpi_ranks,omp_threads,total_cores,Nx,Ny,N_particles,local_particles,Maxiter,iteration,t_interp_s,t_comm_s,t_norm_s,t_mover_s,t_denorm_s,t_total_s,n_active" \
    > "${SUMMARY}"

for cfg_line in "${CONFIGS[@]}"; do
    read -r label nx ny npart maxiter <<< "${cfg_line}"
    INPUT="${DATA}/input_${label}.bin"

    echo ""
    echo "========================================"
    echo " Config ${label}: Nx=${nx} Ny=${ny} N=${npart}"
    echo "========================================"

    # Generate input data
    "${FILEGEN}" "${nx}" "${ny}" "${npart}" "${maxiter}" "${INPUT}"

    for layout in "${LAYOUTS[@]}"; do
        read -r cores ranks omp_threads <<< "${layout}"
        var="${BEST_VARIANT}"
        if [ "${cores}" -eq 1 ]; then
            var="serial"
        fi

        echo "[run] Config=${label}  Cores=${cores}  MPI=${ranks}  OMP=${omp_threads}  Variant=${var}"
        OUTCSV="${RAW}/run_${label}_${var}_c${cores}_r${ranks}_t${omp_threads}.csv"

        OMP_NUM_THREADS=${omp_threads} \
        OMP_PROC_BIND=close \
        OMP_PLACES=cores \
        "${MPIRUN}" -np "${ranks}" "${HOST_ARGS[@]}" ${MPI_EXTRA_ARGS} \
            "${PIPELINE}" "${INPUT}" "${omp_threads}" "${var}" \
            > "${OUTCSV}" 2>/dev/null

        tail -n +2 "${OUTCSV}" | while IFS= read -r line; do
            echo "${label},${line}" >> "${SUMMARY}"
        done
    done
done

echo ""
echo "════════════════════════════════════════"
echo " All runs complete.  Summary: ${SUMMARY}"
echo "════════════════════════════════════════"
echo ""
echo "Run plot_results.py to generate graphs:"
echo "  python3 ${SCRIPT_DIR}/plot_results_mpi.py"
