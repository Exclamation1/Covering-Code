#!/usr/bin/env bash
set -euo pipefail

# ========= CẤU HÌNH =========
SRC_CPP="covering_code_tabu.cpp"          # đổi nếu bạn đặt tên khác
BIN="./cc_tabu"                           # binary đầu ra
ITERS=2500
TABU=25
SAMPLE=120
SEED=123
N_START=2
N_END=8
R_LIST=(1 2 3)                            # 3 giá trị r sẽ chạy song song cho mỗi n

# Thư mục output + log (đặt theo timestamp)
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results_cc_tabu_${STAMP}"
LOG_FILE="${OUT_DIR}/run.log"
PID_FILE="${OUT_DIR}/run.pid"

# ========= HỖ TRỢ =========
compile_if_needed() {
  mkdir -p "${OUT_DIR}"
  if [[ ! -x "${BIN}" ]]; then
    echo "[build] Compile ${SRC_CPP} -> ${BIN}" | tee -a "${LOG_FILE}"
    g++ -O3 -march=native -std=c++20 "${SRC_CPP}" -o "${BIN}"
  else
    echo "[build] Found existing ${BIN}, skip compile" | tee -a "${LOG_FILE}"
  fi
}

# Truyền SIGTERM xuống con để dừng sạch
cleanup() {
  echo "[trap] Caught signal, killing children..." | tee -a "${LOG_FILE}"
  pkill -P $$ || true
}
trap cleanup SIGINT SIGTERM

run_all() {
  echo "[run] Output dir: ${OUT_DIR}" | tee -a "${LOG_FILE}"

  for (( n=${N_START}; n<=${N_END}; n++ )); do
    echo "[batch] n=${n} -> launching r in parallel..." | tee -a "${LOG_FILE}"
    pids=()

    for r in "${R_LIST[@]}"; do
      # bỏ qua r>n (không cần thiết & tránh lỗi)
      if (( r > n )); then
        echo "[skip] n=${n} r=${r} (r>n)" | tee -a "${LOG_FILE}"
        continue
      fi

      out="${OUT_DIR}/n${n}_r${r}.txt"
      pidfile="${OUT_DIR}/n${n}_r${r}.pid"

      echo "[run] n=${n} r=${r} -> ${out} (background)" | tee -a "${LOG_FILE}"
      # chạy nền từng r; ghi riêng ra file out, lỗi cũng đưa vào file out
      (
        echo "[start] $(date '+%F %T') n=${n} r=${r}" 
        "${BIN}" "${n}" "${r}" \
          --iters "${ITERS}" --tabu "${TABU}" --sample "${SAMPLE}" --seed "${SEED}"
        echo "[done]  $(date '+%F %T') n=${n} r=${r}"
      ) > "${out}" 2>&1 &

      child=$!
      echo "${child}" > "${pidfile}"
      echo "[pid] n=${n} r=${r} pid=${child}" | tee -a "${LOG_FILE}"
      pids+=("${child}")
    done

    # Đợi xong tất cả r của cùng một n
    if ((${#pids[@]} > 0)); then
      echo "[wait] n=${n} waiting for ${#pids[@]} job(s)..." | tee -a "${LOG_FILE}"
      for pid in "${pids[@]}"; do
        if wait "${pid}"; then
          echo "[ok] pid=${pid}" | tee -a "${LOG_FILE}"
        else
          echo "[err] pid=${pid} exited non-zero" | tee -a "${LOG_FILE}"
        fi
      done
      echo "[batch] n=${n} completed." | tee -a "${LOG_FILE}"
    else
      echo "[batch] n=${n} had nothing to run (all r skipped)." | tee -a "${LOG_FILE}"
    fi
  done

  echo "[done] All jobs finished." | tee -a "${LOG_FILE}"
}

# ========= TỰ TÁCH NỀN BẰNG NOHUP =========
if [[ "${1-}" != "--detached" ]]; then
  mkdir -p "${OUT_DIR}"
  nohup "$0" --detached > "${LOG_FILE}" 2>&1 < /dev/null &
  echo $! > "${PID_FILE}"
  echo "Started background job."
  echo "  PID: $(cat "${PID_FILE}")"
  echo "  OUT: ${OUT_DIR}"
  echo "  LOG: ${LOG_FILE}"
  exit 0
fi

# ========= LUỒNG THỰC THI NỀN =========
echo "[init] Detached mode. PID=$$"
echo "[init] Logs at ${LOG_FILE}"
compile_if_needed
run_all
