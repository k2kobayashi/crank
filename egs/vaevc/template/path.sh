# cuda related
export CUDA_HOME=/usr/local/cuda-10.0
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# virtualenv related
export PRJ_ROOT="${PWD}/../../.."
if [ -e "${PRJ_ROOT}/tools/venv/bin/activate" ]; then
    # shellcheck disable=SC1090
    . "${PRJ_ROOT}/tools/venv/bin/activate"
fi

# other important environment variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
export MPL_BACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=3

# check installation
if ! python -c "import crank" > /dev/null 2>&1 ; then
    echo "Error: It seems setup is not finished." >&2
    echo "Error: Please setup your environment by following README.md" >&2
    return 1
fi

# set default.yml as environment variable
export CRANK_DEFAULT_YAML="${PWD}"/conf/default.yml
