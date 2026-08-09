#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr double AIF_PCIE_MIB_PER_SECOND = 8000000000.0 / (1024.0 * 1024.0);

struct aif_csv_row {
    std::uint64_t call_index = 0;
    std::string phase = "unknown";
    std::uint64_t decode_call_index = 0;
    std::uint64_t batch_tokens = 0;
    std::uint64_t n_past_before = 0;
    bool success = false;
    bool stats_valid = false;
    std::string execution_mode = "sequential";
    std::string tensor_name;
    std::uint64_t tensor_id = 0;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t matrix_nbytes = 0;
    std::uint64_t full_matrix_nbytes = 0;
    std::uint64_t input_nbytes = 0;
    std::uint64_t output_nbytes = 0;
    std::uint64_t output_mismatches = 0;
    std::uint64_t ioctl_elapsed_ns = 0;
    std::uint64_t device_post_count = 0;
    std::uint64_t device_gemv_count = 0;
    std::uint64_t device_tensor_count = 0;
    std::uint64_t device_delay_ns = 0;
    std::uint64_t input_pcie_ns = 0;
    std::uint64_t matrix_ns = 0;
    std::uint64_t output_pcie_ns = 0;
    std::uint64_t layout_hit_count = 0;
    std::uint64_t layout_miss_count = 0;
    std::uint64_t mapped_pages = 0;
    std::uint64_t units_used = 0;
    std::uint64_t max_unit_pages = 0;
    std::uint64_t host_matrix_nbytes = 0;
    std::uint64_t host_modeled_ns = 0;
    std::uint64_t host_actual_ns = 0;
    std::uint64_t head_first_ready_wait_ns = 0;
    std::uint64_t dependency_wait_ns = 0;
    std::uint64_t parallel_elapsed_ns = 0;
    std::uint64_t overlap_ns = 0;
    std::uint64_t qkv_group_id = 0;
    std::string qkv_schedule;
    std::uint64_t q_heads = 0;
    std::uint64_t kv_heads = 0;
    bool qkv_group_summary = false;
    std::uint64_t qkv_group_modeled_ns = 0;
    std::uint64_t qkv_first_head_ready_ns = 0;
    std::uint64_t qkv_last_head_ready_ns = 0;
    std::uint64_t host_attention_actual_ns = 0;
    std::uint64_t host_idle_wait_ns = 0;
    std::uint64_t aif_idle_wait_ns = 0;
    std::uint64_t head_pipeline_finish_ns = 0;
    std::uint64_t head_pipeline_tail_wait_ns = 0;
};

struct tensor_summary {
    std::string tensor_name;
    std::uint64_t tensor_id = 0;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t matrix_nbytes = 0;
    std::uint64_t calls = 0;
    std::uint64_t successes = 0;
    std::uint64_t output_mismatches = 0;
    std::uint64_t input_nbytes = 0;
    std::uint64_t output_nbytes = 0;
    std::uint64_t device_delay_ns = 0;
    std::uint64_t input_pcie_ns = 0;
    std::uint64_t matrix_ns = 0;
    std::uint64_t output_pcie_ns = 0;
    std::uint64_t ioctl_elapsed_ns = 0;
};

struct phase_summary {
    std::uint64_t calls = 0;
    std::uint64_t successes = 0;
    std::uint64_t device_delay_ns = 0;
    std::uint64_t input_pcie_ns = 0;
    std::uint64_t matrix_ns = 0;
    std::uint64_t output_pcie_ns = 0;
    std::uint64_t ioctl_elapsed_ns = 0;
    std::uint64_t min_decode_call_index = 0;
    std::uint64_t max_decode_call_index = 0;
};

struct decode_step_summary {
    std::uint64_t decode_call_index = 0;
    std::uint64_t n_past_before = 0;
    std::uint64_t batch_tokens = 0;
    std::uint64_t calls = 0;
    std::uint64_t successes = 0;
    std::uint64_t output_mismatches = 0;
    std::uint64_t input_nbytes = 0;
    std::uint64_t output_nbytes = 0;
    std::uint64_t device_delay_ns = 0;
    std::uint64_t input_pcie_ns = 0;
    std::uint64_t matrix_ns = 0;
    std::uint64_t output_pcie_ns = 0;
    std::uint64_t ioctl_elapsed_ns = 0;
    std::uint64_t memory_ssd_matrix_pcie_ns = 0;
    std::uint64_t memory_ssd_read_ns = 0;
};

struct baseline_eval_summary {
    bool valid = false;
    std::string path;
    std::string line;
    double total_ms = 0.0;
    std::uint64_t runs = 0;
    double avg_us = 0.0;
    double tps = 0.0;
};

struct hybrid_estimate_config {
    bool has_cpu_offloadable_us = false;
    bool has_cpu_offloadable_frac = false;
    double cpu_offloadable_us = 0.0;
    double cpu_offloadable_frac = 0.0;
};

static void usage(const char * prog) {
    std::cerr << "Usage: " << prog
              << " CSV [--top N] [--steps N] [--baseline-log FILE] [--aif-log FILE]"
              << " [--pcie-mib-s MiB/s (default: AiF 8.0 GB/s)]"
              << " [--cpu-offloadable-us US | --cpu-offloadable-frac F]\n";
}

static std::vector<std::string> split_csv_line(const std::string & line) {
    std::vector<std::string> fields;
    std::string cur;
    bool quoted = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];

        if (quoted) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur.push_back('"');
                    ++i;
                } else {
                    quoted = false;
                }
            } else {
                cur.push_back(c);
            }
            continue;
        }

        if (c == ',') {
            fields.push_back(cur);
            cur.clear();
        } else if (c == '"') {
            quoted = true;
        } else {
            cur.push_back(c);
        }
    }

    fields.push_back(cur);
    return fields;
}

static std::uint64_t parse_u64(const std::string & value, const std::string & column, int line_no) {
    if (value.empty()) {
        return 0;
    }

    std::size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &consumed, 0);
    } catch (const std::exception & e) {
        std::ostringstream oss;
        oss << "line " << line_no << ": invalid integer in column '" << column << "': " << value;
        throw std::runtime_error(oss.str());
    }

    if (consumed != value.size()) {
        std::ostringstream oss;
        oss << "line " << line_no << ": trailing characters in column '" << column << "': " << value;
        throw std::runtime_error(oss.str());
    }

    return static_cast<std::uint64_t>(parsed);
}

static std::string get_field(
        const std::vector<std::string> & fields,
        const std::unordered_map<std::string, std::size_t> & columns,
        const std::string & name) {
    const auto it = columns.find(name);
    if (it == columns.end() || it->second >= fields.size()) {
        return "";
    }

    return fields[it->second];
}

static std::uint64_t get_u64(
        const std::vector<std::string> & fields,
        const std::unordered_map<std::string, std::size_t> & columns,
        const std::string & name,
        int line_no) {
    return parse_u64(get_field(fields, columns, name), name, line_no);
}

static bool has_column(const std::unordered_map<std::string, std::size_t> & columns, const std::string & name) {
    return columns.find(name) != columns.end();
}

static void require_columns(const std::unordered_map<std::string, std::size_t> & columns) {
    static const char * required[] = {
        "call_index",
        "success",
        "tensor_name",
        "tensor_id",
        "rows",
        "cols",
        "matrix_nbytes",
        "input_nbytes",
        "output_nbytes",
        "output_mismatches",
        "ioctl_elapsed_ns",
        "stats_valid",
        "device_delay_ns",
        "input_pcie_ns",
        "matrix_ns",
        "output_pcie_ns",
        "layout_hit_count",
        "layout_miss_count",
    };

    for (const char * name : required) {
        if (!has_column(columns, name)) {
            throw std::runtime_error(std::string("missing required CSV column: ") + name);
        }
    }
}

static std::vector<aif_csv_row> read_csv(const std::string & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open CSV: " + path);
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("empty CSV: " + path);
    }

    const std::vector<std::string> header = split_csv_line(line);
    std::unordered_map<std::string, std::size_t> columns;
    for (std::size_t i = 0; i < header.size(); ++i) {
        columns[header[i]] = i;
    }
    require_columns(columns);

    std::vector<aif_csv_row> rows;
    int line_no = 1;
    while (std::getline(input, line)) {
        ++line_no;
        if (line.empty()) {
            continue;
        }

        const std::vector<std::string> fields = split_csv_line(line);
        aif_csv_row row;
        row.call_index = get_u64(fields, columns, "call_index", line_no);
        if (has_column(columns, "phase")) {
            row.phase = get_field(fields, columns, "phase");
        }
        if (has_column(columns, "decode_call_index")) {
            row.decode_call_index = get_u64(fields, columns, "decode_call_index", line_no);
        }
        if (has_column(columns, "batch_tokens")) {
            row.batch_tokens = get_u64(fields, columns, "batch_tokens", line_no);
        }
        if (has_column(columns, "n_past_before")) {
            row.n_past_before = get_u64(fields, columns, "n_past_before", line_no);
        }
        row.success = get_u64(fields, columns, "success", line_no) != 0;
        if (has_column(columns, "execution_mode")) {
            row.execution_mode = get_field(fields, columns, "execution_mode");
        }
        row.stats_valid = get_u64(fields, columns, "stats_valid", line_no) != 0;
        row.tensor_name = get_field(fields, columns, "tensor_name");
        row.tensor_id = get_u64(fields, columns, "tensor_id", line_no);
        row.rows = get_u64(fields, columns, "rows", line_no);
        row.cols = get_u64(fields, columns, "cols", line_no);
        row.matrix_nbytes = get_u64(fields, columns, "matrix_nbytes", line_no);
        row.full_matrix_nbytes = has_column(columns, "full_matrix_nbytes") ?
                get_u64(fields, columns, "full_matrix_nbytes", line_no) : row.matrix_nbytes;
        row.input_nbytes = get_u64(fields, columns, "input_nbytes", line_no);
        row.output_nbytes = get_u64(fields, columns, "output_nbytes", line_no);
        row.output_mismatches = get_u64(fields, columns, "output_mismatches", line_no);
        row.ioctl_elapsed_ns = get_u64(fields, columns, "ioctl_elapsed_ns", line_no);
        row.device_delay_ns = get_u64(fields, columns, "device_delay_ns", line_no);
        row.input_pcie_ns = get_u64(fields, columns, "input_pcie_ns", line_no);
        row.matrix_ns = get_u64(fields, columns, "matrix_ns", line_no);
        row.output_pcie_ns = get_u64(fields, columns, "output_pcie_ns", line_no);
        row.layout_hit_count = get_u64(fields, columns, "layout_hit_count", line_no);
        row.layout_miss_count = get_u64(fields, columns, "layout_miss_count", line_no);

        if (has_column(columns, "device_post_count")) {
            row.device_post_count = get_u64(fields, columns, "device_post_count", line_no);
        }
        if (has_column(columns, "device_gemv_count")) {
            row.device_gemv_count = get_u64(fields, columns, "device_gemv_count", line_no);
        }
        if (has_column(columns, "device_tensor_count")) {
            row.device_tensor_count = get_u64(fields, columns, "device_tensor_count", line_no);
        }
        if (has_column(columns, "mapped_pages")) {
            row.mapped_pages = get_u64(fields, columns, "mapped_pages", line_no);
        }
        if (has_column(columns, "units_used")) {
            row.units_used = get_u64(fields, columns, "units_used", line_no);
        }
        if (has_column(columns, "max_unit_pages")) {
            row.max_unit_pages = get_u64(fields, columns, "max_unit_pages", line_no);
        }
        if (has_column(columns, "host_matrix_nbytes")) {
            row.host_matrix_nbytes = get_u64(fields, columns, "host_matrix_nbytes", line_no);
        }
        if (has_column(columns, "host_modeled_ns")) {
            row.host_modeled_ns = get_u64(fields, columns, "host_modeled_ns", line_no);
        }
        if (has_column(columns, "host_actual_ns")) {
            row.host_actual_ns = get_u64(fields, columns, "host_actual_ns", line_no);
        }
        if (has_column(columns, "head_first_ready_wait_ns")) {
            row.head_first_ready_wait_ns = get_u64(fields, columns, "head_first_ready_wait_ns", line_no);
        }
        if (has_column(columns, "dependency_wait_ns")) {
            row.dependency_wait_ns = get_u64(fields, columns, "dependency_wait_ns", line_no);
        }
        if (has_column(columns, "parallel_elapsed_ns")) {
            row.parallel_elapsed_ns = get_u64(fields, columns, "parallel_elapsed_ns", line_no);
        }
        if (has_column(columns, "overlap_ns")) {
            row.overlap_ns = get_u64(fields, columns, "overlap_ns", line_no);
        }
        if (has_column(columns, "qkv_group_id")) {
            row.qkv_group_id = get_u64(fields, columns, "qkv_group_id", line_no);
        }
        if (has_column(columns, "qkv_schedule")) {
            row.qkv_schedule = get_field(fields, columns, "qkv_schedule");
        }
        if (has_column(columns, "q_heads")) {
            row.q_heads = get_u64(fields, columns, "q_heads", line_no);
        }
        if (has_column(columns, "kv_heads")) {
            row.kv_heads = get_u64(fields, columns, "kv_heads", line_no);
        }
        if (has_column(columns, "qkv_group_summary")) {
            row.qkv_group_summary = get_u64(fields, columns, "qkv_group_summary", line_no) != 0;
        }
        if (has_column(columns, "qkv_group_modeled_ns")) {
            row.qkv_group_modeled_ns = get_u64(fields, columns, "qkv_group_modeled_ns", line_no);
        }
        if (has_column(columns, "qkv_first_head_ready_ns")) {
            row.qkv_first_head_ready_ns = get_u64(fields, columns, "qkv_first_head_ready_ns", line_no);
        }
        if (has_column(columns, "qkv_last_head_ready_ns")) {
            row.qkv_last_head_ready_ns = get_u64(fields, columns, "qkv_last_head_ready_ns", line_no);
        }
        if (has_column(columns, "host_attention_actual_ns")) {
            row.host_attention_actual_ns = get_u64(fields, columns, "host_attention_actual_ns", line_no);
        }
        if (has_column(columns, "host_idle_wait_ns")) {
            row.host_idle_wait_ns = get_u64(fields, columns, "host_idle_wait_ns", line_no);
        }
        if (has_column(columns, "aif_idle_wait_ns")) {
            row.aif_idle_wait_ns = get_u64(fields, columns, "aif_idle_wait_ns", line_no);
        }
        if (has_column(columns, "head_pipeline_finish_ns")) {
            row.head_pipeline_finish_ns = get_u64(fields, columns, "head_pipeline_finish_ns", line_no);
        }
        if (has_column(columns, "head_pipeline_tail_wait_ns")) {
            row.head_pipeline_tail_wait_ns = get_u64(fields, columns, "head_pipeline_tail_wait_ns", line_no);
        }

        rows.push_back(std::move(row));
    }

    return rows;
}

static double ns_to_us(std::uint64_t ns) {
    return static_cast<double>(ns) / 1000.0;
}

static double bytes_to_mib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static std::uint64_t bytes_to_pcie_ns(std::uint64_t bytes, double mib_per_sec) {
    if (bytes == 0 || mib_per_sec <= 0.0) {
        return 0;
    }

    const long double kib = std::ceil(static_cast<long double>(bytes) / 1024.0L);
    const long double ns = std::ceil(kib * 1000000000.0L / (static_cast<long double>(mib_per_sec) * 1024.0L));
    if (ns >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        return std::numeric_limits<std::uint64_t>::max();
    }

    return static_cast<std::uint64_t>(ns);
}

static baseline_eval_summary read_eval_log(const std::string & path, const std::string & label) {
    baseline_eval_summary result;
    result.path = path;

    if (path.empty()) {
        return result;
    }

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + label + " log: " + path);
    }

    std::string line;
    while (std::getline(input, line)) {
        if (line.find("prompt eval time") != std::string::npos) {
            continue;
        }

        const std::size_t eval_pos = line.find("eval time");
        if (eval_pos == std::string::npos) {
            continue;
        }

        const std::size_t eq_pos = line.find('=', eval_pos);
        if (eq_pos == std::string::npos) {
            continue;
        }

        std::istringstream iss(line.substr(eq_pos + 1));
        double total_ms = 0.0;
        std::string ms_label;
        char slash = '\0';
        std::uint64_t runs = 0;
        std::string runs_label;

        if (!(iss >> total_ms >> ms_label >> slash >> runs >> runs_label)) {
            continue;
        }
        if (ms_label != "ms" || slash != '/' || runs == 0) {
            continue;
        }

        result.valid = true;
        result.line = line;
        result.total_ms = total_ms;
        result.runs = runs;
        result.avg_us = total_ms * 1000.0 / static_cast<double>(runs);
        result.tps = result.avg_us > 0.0 ? 1000000.0 / result.avg_us : 0.0;
    }

    if (!result.valid) {
        throw std::runtime_error(label + " log does not contain a parseable non-prompt 'eval time =' line: " + path);
    }

    return result;
}

static void print_time_line(const std::string & label, std::uint64_t total_ns, std::uint64_t count, std::uint64_t max_ns) {
    const double avg_ns = count == 0 ? 0.0 : static_cast<double>(total_ns) / static_cast<double>(count);

    std::cout << std::left << std::setw(24) << label
              << " total=" << std::fixed << std::setprecision(3) << ns_to_us(total_ns) << " us"
              << " avg=" << std::fixed << std::setprecision(3) << avg_ns / 1000.0 << " us"
              << " max=" << std::fixed << std::setprecision(3) << ns_to_us(max_ns) << " us"
              << '\n';
}

static int run_report(
        const std::string & path,
        std::size_t top_n,
        std::size_t step_top_n,
        const baseline_eval_summary & baseline,
        const baseline_eval_summary & aif_eval,
        const hybrid_estimate_config & hybrid_config,
        double pcie_mib_s) {
    const std::vector<aif_csv_row> rows = read_csv(path);

    std::uint64_t success_count = 0;
    std::uint64_t stats_valid_count = 0;
    std::uint64_t mismatch_rows = 0;
    std::uint64_t input_bytes = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t device_delay_ns = 0;
    std::uint64_t input_pcie_ns = 0;
    std::uint64_t matrix_ns = 0;
    std::uint64_t output_pcie_ns = 0;
    std::uint64_t ioctl_elapsed_ns = 0;
    std::uint64_t matrix_read_bytes = 0;
    std::uint64_t memory_ssd_matrix_pcie_ns = 0;
    std::uint64_t memory_ssd_read_ns = 0;
    std::uint64_t max_device_delay_ns = 0;
    std::uint64_t max_input_pcie_ns = 0;
    std::uint64_t max_matrix_ns = 0;
    std::uint64_t max_output_pcie_ns = 0;
    std::uint64_t max_ioctl_elapsed_ns = 0;
    std::uint64_t max_memory_ssd_read_ns = 0;
    std::uint64_t parallel_rows = 0;
    std::uint64_t host_matrix_bytes = 0;
    std::uint64_t host_modeled_ns = 0;
    std::uint64_t host_actual_ns = 0;
    std::uint64_t head_first_ready_wait_ns = 0;
    std::uint64_t dependency_wait_ns = 0;
    std::uint64_t parallel_elapsed_ns = 0;
    std::uint64_t overlap_ns = 0;
    std::uint64_t qkv_groups = 0;
    std::uint64_t qkv_group_modeled_ns = 0;
    std::uint64_t qkv_first_head_ready_ns = 0;
    std::uint64_t qkv_last_head_ready_ns = 0;
    std::uint64_t host_attention_actual_ns = 0;
    std::uint64_t host_idle_wait_ns = 0;
    std::uint64_t aif_idle_wait_ns = 0;
    std::uint64_t head_pipeline_finish_ns = 0;
    std::uint64_t head_pipeline_tail_wait_ns = 0;

    std::map<std::string, tensor_summary> by_tensor;
    std::map<std::string, phase_summary> by_phase;
    std::map<std::uint64_t, decode_step_summary> by_decode_step;

    for (const auto & row : rows) {
        success_count += row.success ? 1 : 0;
        stats_valid_count += row.stats_valid ? 1 : 0;
        mismatch_rows += row.output_mismatches != 0 ? 1 : 0;
        input_bytes += row.input_nbytes;
        output_bytes += row.output_nbytes;
        device_delay_ns += row.device_delay_ns;
        input_pcie_ns += row.input_pcie_ns;
        matrix_ns += row.matrix_ns;
        output_pcie_ns += row.output_pcie_ns;
        ioctl_elapsed_ns += row.ioctl_elapsed_ns;
        matrix_read_bytes += row.full_matrix_nbytes;
        const std::uint64_t row_memory_ssd_matrix_pcie_ns = bytes_to_pcie_ns(row.full_matrix_nbytes, pcie_mib_s);
        const std::uint64_t row_memory_ssd_read_ns = row.matrix_ns + row_memory_ssd_matrix_pcie_ns;
        memory_ssd_matrix_pcie_ns += row_memory_ssd_matrix_pcie_ns;
        memory_ssd_read_ns += row_memory_ssd_read_ns;
        max_device_delay_ns = std::max(max_device_delay_ns, row.device_delay_ns);
        max_input_pcie_ns = std::max(max_input_pcie_ns, row.input_pcie_ns);
        max_matrix_ns = std::max(max_matrix_ns, row.matrix_ns);
        max_output_pcie_ns = std::max(max_output_pcie_ns, row.output_pcie_ns);
        max_ioctl_elapsed_ns = std::max(max_ioctl_elapsed_ns, row.ioctl_elapsed_ns);
        max_memory_ssd_read_ns = std::max(max_memory_ssd_read_ns, row_memory_ssd_read_ns);
        if (row.execution_mode != "sequential" && row.execution_mode != "shadow") {
            ++parallel_rows;
        }
        host_matrix_bytes += row.host_matrix_nbytes;
        host_modeled_ns += row.host_modeled_ns;
        host_actual_ns += row.host_actual_ns;
        head_first_ready_wait_ns += row.head_first_ready_wait_ns;
        dependency_wait_ns += row.dependency_wait_ns;
        parallel_elapsed_ns += row.parallel_elapsed_ns;
        overlap_ns += row.overlap_ns;
        if (row.qkv_group_summary) {
            ++qkv_groups;
            qkv_group_modeled_ns += row.qkv_group_modeled_ns;
            qkv_first_head_ready_ns += row.qkv_first_head_ready_ns;
            qkv_last_head_ready_ns += row.qkv_last_head_ready_ns;
            host_attention_actual_ns += row.host_attention_actual_ns;
            host_idle_wait_ns += row.host_idle_wait_ns;
            aif_idle_wait_ns += row.aif_idle_wait_ns;
            head_pipeline_finish_ns += row.head_pipeline_finish_ns;
            head_pipeline_tail_wait_ns += row.head_pipeline_tail_wait_ns;
        }

        auto & phase = by_phase[row.phase.empty() ? "unknown" : row.phase];
        ++phase.calls;
        phase.successes += row.success ? 1 : 0;
        phase.device_delay_ns += row.device_delay_ns;
        phase.input_pcie_ns += row.input_pcie_ns;
        phase.matrix_ns += row.matrix_ns;
        phase.output_pcie_ns += row.output_pcie_ns;
        phase.ioctl_elapsed_ns += row.ioctl_elapsed_ns;
        if (row.decode_call_index != 0) {
            if (phase.min_decode_call_index == 0 || row.decode_call_index < phase.min_decode_call_index) {
                phase.min_decode_call_index = row.decode_call_index;
            }
            phase.max_decode_call_index = std::max(phase.max_decode_call_index, row.decode_call_index);
        }

        if (row.phase == "decode" && row.decode_call_index != 0) {
            auto & step = by_decode_step[row.decode_call_index];
            if (step.calls == 0) {
                step.decode_call_index = row.decode_call_index;
                step.n_past_before = row.n_past_before;
                step.batch_tokens = row.batch_tokens;
            }

            ++step.calls;
            step.successes += row.success ? 1 : 0;
            step.output_mismatches += row.output_mismatches;
            step.input_nbytes += row.input_nbytes;
            step.output_nbytes += row.output_nbytes;
            step.device_delay_ns += row.device_delay_ns;
            step.input_pcie_ns += row.input_pcie_ns;
            step.matrix_ns += row.matrix_ns;
            step.output_pcie_ns += row.output_pcie_ns;
            step.ioctl_elapsed_ns += row.ioctl_elapsed_ns;
            step.memory_ssd_matrix_pcie_ns += row_memory_ssd_matrix_pcie_ns;
            step.memory_ssd_read_ns += row_memory_ssd_read_ns;
        }

        auto & summary = by_tensor[row.tensor_name];
        if (summary.calls == 0) {
            summary.tensor_name = row.tensor_name;
            summary.tensor_id = row.tensor_id;
            summary.rows = row.rows;
            summary.cols = row.cols;
            summary.matrix_nbytes = row.matrix_nbytes;
        }

        ++summary.calls;
        summary.successes += row.success ? 1 : 0;
        summary.output_mismatches += row.output_mismatches;
        summary.input_nbytes += row.input_nbytes;
        summary.output_nbytes += row.output_nbytes;
        summary.device_delay_ns += row.device_delay_ns;
        summary.input_pcie_ns += row.input_pcie_ns;
        summary.matrix_ns += row.matrix_ns;
        summary.output_pcie_ns += row.output_pcie_ns;
        summary.ioctl_elapsed_ns += row.ioctl_elapsed_ns;
    }

    std::cout << "AIF GEMV report: " << path << '\n';
    std::cout << "calls: " << rows.size()
              << " success: " << success_count
              << " stats_valid: " << stats_valid_count
              << " mismatch_rows: " << mismatch_rows << '\n';

    if (!rows.empty()) {
        const auto & last = rows.back();
        std::cout << "final device counters:"
                  << " post=" << last.device_post_count
                  << " gemv=" << last.device_gemv_count
                  << " tensors=" << last.device_tensor_count
                  << " layout_hit=" << last.layout_hit_count
                  << " layout_miss=" << last.layout_miss_count
                  << '\n';
    }

    std::cout << "total transfer:"
              << " input=" << std::fixed << std::setprecision(3) << bytes_to_mib(input_bytes) << " MiB"
              << " output=" << std::fixed << std::setprecision(3) << bytes_to_mib(output_bytes) << " MiB"
              << " memory_ssd_matrix_read=" << std::fixed << std::setprecision(3) << bytes_to_mib(matrix_read_bytes) << " MiB"
              << '\n';

    print_time_line("device modeled", device_delay_ns, rows.size(), max_device_delay_ns);
    print_time_line("  input PCIe", input_pcie_ns, rows.size(), max_input_pcie_ns);
    print_time_line("  SSD internal pipeline", matrix_ns, rows.size(), max_matrix_ns);
    print_time_line("  output PCIe", output_pcie_ns, rows.size(), max_output_pcie_ns);
    print_time_line("ioctl elapsed", ioctl_elapsed_ns, rows.size(), max_ioctl_elapsed_ns);
    print_time_line("Memory+SSD read LB", memory_ssd_read_ns, rows.size(), max_memory_ssd_read_ns);
    std::cout << "  Memory+SSD matrix PCIe total=" << std::fixed << std::setprecision(3) << ns_to_us(memory_ssd_matrix_pcie_ns)
              << " us at " << std::fixed << std::setprecision(1) << pcie_mib_s << " MiB/s\n";

    if (ioctl_elapsed_ns > 0) {
        const double ratio = static_cast<double>(device_delay_ns) / static_cast<double>(ioctl_elapsed_ns);
        std::cout << "device/ioctl ratio: " << std::fixed << std::setprecision(3) << ratio << '\n';
    }

    if (parallel_rows != 0) {
        std::cout << "parallel execution: rows=" << parallel_rows
                  << " host_matrix=" << std::fixed << std::setprecision(3) << bytes_to_mib(host_matrix_bytes) << " MiB"
                  << " host_modeled=" << std::fixed << std::setprecision(3) << ns_to_us(host_modeled_ns) << " us"
                  << " host_actual=" << std::fixed << std::setprecision(3) << ns_to_us(host_actual_ns) << " us"
                  << " first_ready_wait=" << std::fixed << std::setprecision(3) << ns_to_us(head_first_ready_wait_ns) << " us"
                  << " dependency_wait=" << std::fixed << std::setprecision(3) << ns_to_us(dependency_wait_ns) << " us"
                  << " component_elapsed=" << std::fixed << std::setprecision(3) << ns_to_us(parallel_elapsed_ns) << " us"
                  << " recorded_overlap=" << std::fixed << std::setprecision(3) << ns_to_us(overlap_ns) << " us"
                  << '\n';
        std::cout << "  note: summed device/parallel times are component totals; llama eval time is the end-to-end critical path\n";
        if (qkv_groups != 0) {
            std::cout << "head pipeline: schedule=head_major"
                      << " groups=" << qkv_groups
                      << " qkv_modeled=" << std::fixed << std::setprecision(3) << ns_to_us(qkv_group_modeled_ns) << " us"
                      << " first_head_ready=" << std::fixed << std::setprecision(3) << ns_to_us(qkv_first_head_ready_ns) << " us"
                      << " last_head_ready=" << std::fixed << std::setprecision(3) << ns_to_us(qkv_last_head_ready_ns) << " us"
                      << " host_attention=" << std::fixed << std::setprecision(3) << ns_to_us(host_attention_actual_ns) << " us"
                      << " host_idle=" << std::fixed << std::setprecision(3) << ns_to_us(host_idle_wait_ns) << " us"
                      << " aif_idle=" << std::fixed << std::setprecision(3) << ns_to_us(aif_idle_wait_ns) << " us"
                      << " pipeline_finish=" << std::fixed << std::setprecision(3) << ns_to_us(head_pipeline_finish_ns) << " us"
                      << " tail_wait=" << std::fixed << std::setprecision(3) << ns_to_us(head_pipeline_tail_wait_ns) << " us"
                      << '\n';
        }
    }

    std::cout << "\nby phase:\n";
    std::cout << std::left
              << std::setw(12) << "phase"
              << std::right
              << std::setw(8)  << "calls"
              << std::setw(10) << "success"
              << std::setw(14) << "device_us"
              << std::setw(12) << "avg_us"
              << std::setw(14) << "ioctl_us"
              << std::setw(12) << "decode_min"
              << std::setw(12) << "decode_max"
              << '\n';

    for (const auto & [phase_name, summary] : by_phase) {
        const double avg_us = summary.calls == 0 ? 0.0 : ns_to_us(summary.device_delay_ns) / static_cast<double>(summary.calls);
        std::cout << std::left
                  << std::setw(12) << phase_name
                  << std::right
                  << std::setw(8)  << summary.calls
                  << std::setw(10) << summary.successes
                  << std::setw(14) << std::fixed << std::setprecision(3) << ns_to_us(summary.device_delay_ns)
                  << std::setw(12) << std::fixed << std::setprecision(3) << avg_us
                  << std::setw(14) << std::fixed << std::setprecision(3) << ns_to_us(summary.ioctl_elapsed_ns)
                  << std::setw(12) << summary.min_decode_call_index
                  << std::setw(12) << summary.max_decode_call_index
                  << '\n';
    }

    std::vector<decode_step_summary> decode_steps;
    decode_steps.reserve(by_decode_step.size());

    std::uint64_t decode_calls = 0;
    std::uint64_t decode_successes = 0;
    std::uint64_t decode_output_mismatches = 0;
    std::uint64_t decode_device_delay_ns = 0;
    std::uint64_t decode_input_pcie_ns = 0;
    std::uint64_t decode_matrix_ns = 0;
    std::uint64_t decode_output_pcie_ns = 0;
    std::uint64_t decode_ioctl_elapsed_ns = 0;
    std::uint64_t decode_memory_ssd_matrix_pcie_ns = 0;
    std::uint64_t decode_memory_ssd_read_ns = 0;
    std::uint64_t decode_min_step_device_ns = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t decode_max_step_device_ns = 0;
    std::uint64_t decode_min_step_memory_ssd_read_ns = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t decode_max_step_memory_ssd_read_ns = 0;

    for (const auto & [_, step] : by_decode_step) {
        decode_steps.push_back(step);
        decode_calls += step.calls;
        decode_successes += step.successes;
        decode_output_mismatches += step.output_mismatches;
        decode_device_delay_ns += step.device_delay_ns;
        decode_input_pcie_ns += step.input_pcie_ns;
        decode_matrix_ns += step.matrix_ns;
        decode_output_pcie_ns += step.output_pcie_ns;
        decode_ioctl_elapsed_ns += step.ioctl_elapsed_ns;
        decode_memory_ssd_matrix_pcie_ns += step.memory_ssd_matrix_pcie_ns;
        decode_memory_ssd_read_ns += step.memory_ssd_read_ns;
        decode_min_step_device_ns = std::min(decode_min_step_device_ns, step.device_delay_ns);
        decode_max_step_device_ns = std::max(decode_max_step_device_ns, step.device_delay_ns);
        decode_min_step_memory_ssd_read_ns = std::min(decode_min_step_memory_ssd_read_ns, step.memory_ssd_read_ns);
        decode_max_step_memory_ssd_read_ns = std::max(decode_max_step_memory_ssd_read_ns, step.memory_ssd_read_ns);
    }

    std::cout << "\ndecode-step summary:\n";
    if (decode_steps.empty()) {
        std::cout << "no phase=decode rows with decode_call_index were found\n";
    } else {
        const double n_steps = static_cast<double>(decode_steps.size());
        const double avg_calls_per_step = static_cast<double>(decode_calls) / n_steps;
        const double avg_device_step_ns = static_cast<double>(decode_device_delay_ns) / n_steps;
        const double avg_ioctl_step_ns = static_cast<double>(decode_ioctl_elapsed_ns) / n_steps;
        const double avg_memory_ssd_step_ns = static_cast<double>(decode_memory_ssd_read_ns) / n_steps;
        const double gemv_only_tps = avg_device_step_ns > 0.0 ? 1000000000.0 / avg_device_step_ns : 0.0;
        const double ioctl_tps = avg_ioctl_step_ns > 0.0 ? 1000000000.0 / avg_ioctl_step_ns : 0.0;
        const double memory_ssd_read_tps = avg_memory_ssd_step_ns > 0.0 ? 1000000000.0 / avg_memory_ssd_step_ns : 0.0;

        std::cout << "steps=" << decode_steps.size()
                  << " gemv_calls=" << decode_calls
                  << " success=" << decode_successes
                  << " output_mismatches=" << decode_output_mismatches
                  << " avg_gemv_calls_per_step=" << std::fixed << std::setprecision(3) << avg_calls_per_step
                  << '\n';
        std::cout << "GEMV-only modeled decode latency:"
                  << " total=" << std::fixed << std::setprecision(3) << ns_to_us(decode_device_delay_ns) << " us"
                  << " avg=" << std::fixed << std::setprecision(3) << avg_device_step_ns / 1000.0 << " us/step"
                  << " min=" << std::fixed << std::setprecision(3) << ns_to_us(decode_min_step_device_ns) << " us"
                  << " max=" << std::fixed << std::setprecision(3) << ns_to_us(decode_max_step_device_ns) << " us"
                  << " estimated=" << std::fixed << std::setprecision(2) << gemv_only_tps << " tok/s"
                  << '\n';
        std::cout << "  input PCIe total=" << std::fixed << std::setprecision(3) << ns_to_us(decode_input_pcie_ns) << " us"
                  << " SSD internal total=" << std::fixed << std::setprecision(3) << ns_to_us(decode_matrix_ns) << " us"
                  << " output PCIe total=" << std::fixed << std::setprecision(3) << ns_to_us(decode_output_pcie_ns) << " us"
                  << '\n';
        std::cout << "ioctl-observed decode path:"
                  << " total=" << std::fixed << std::setprecision(3) << ns_to_us(decode_ioctl_elapsed_ns) << " us"
                  << " avg=" << std::fixed << std::setprecision(3) << avg_ioctl_step_ns / 1000.0 << " us/step"
                  << " estimated=" << std::fixed << std::setprecision(2) << ioctl_tps << " tok/s"
                  << '\n';
        std::cout << "Memory+SSD matrix-read lower bound:"
                  << " total=" << std::fixed << std::setprecision(3) << ns_to_us(decode_memory_ssd_read_ns) << " us"
                  << " avg=" << std::fixed << std::setprecision(3) << avg_memory_ssd_step_ns / 1000.0 << " us/step"
                  << " min=" << std::fixed << std::setprecision(3) << ns_to_us(decode_min_step_memory_ssd_read_ns) << " us"
                  << " max=" << std::fixed << std::setprecision(3) << ns_to_us(decode_max_step_memory_ssd_read_ns) << " us"
                  << " estimated=" << std::fixed << std::setprecision(2) << memory_ssd_read_tps << " tok/s"
                  << '\n';
        std::cout << "  baseline matrix flash total=" << std::fixed << std::setprecision(3) << ns_to_us(decode_matrix_ns) << " us"
                  << " matrix PCIe total=" << std::fixed << std::setprecision(3) << ns_to_us(decode_memory_ssd_matrix_pcie_ns) << " us"
                  << " at " << std::fixed << std::setprecision(1) << pcie_mib_s << " MiB/s"
                  << '\n';

        if (aif_eval.valid) {
            std::cout << "\nAIF replacement end-to-end eval:\n";
            std::cout << "AIF log: " << aif_eval.path << '\n';
            std::cout << "AIF eval:"
                      << " total=" << std::fixed << std::setprecision(3) << aif_eval.total_ms << " ms"
                      << " runs=" << aif_eval.runs
                      << " avg=" << std::fixed << std::setprecision(3) << aif_eval.avg_us << " us/token"
                      << " tps=" << std::fixed << std::setprecision(2) << aif_eval.tps
                      << '\n';
        }

        if (baseline.valid) {
            const double aif_gemv_avg_us = avg_device_step_ns / 1000.0;
            const double aif_ioctl_avg_us = avg_ioctl_step_ns / 1000.0;
            const double memory_ssd_read_avg_us = avg_memory_ssd_step_ns / 1000.0;
            const double memory_ssd_full_avg_us = baseline.avg_us + memory_ssd_read_avg_us;
            const double memory_ssd_full_tps = memory_ssd_full_avg_us > 0.0 ? 1000000.0 / memory_ssd_full_avg_us : 0.0;
            const double speedup_gemv_only = aif_gemv_avg_us > 0.0 ? baseline.avg_us / aif_gemv_avg_us : 0.0;
            const double speedup_ioctl = aif_ioctl_avg_us > 0.0 ? baseline.avg_us / aif_ioctl_avg_us : 0.0;
            const double matrix_read_speedup = aif_gemv_avg_us > 0.0 ? memory_ssd_read_avg_us / aif_gemv_avg_us : 0.0;

            std::cout << "\nbaseline comparison:\n";
            std::cout << "baseline log: " << baseline.path << '\n';
            std::cout << "CPU baseline eval:"
                      << " total=" << std::fixed << std::setprecision(3) << baseline.total_ms << " ms"
                      << " runs=" << baseline.runs
                      << " avg=" << std::fixed << std::setprecision(3) << baseline.avg_us << " us/token"
                      << " tps=" << std::fixed << std::setprecision(2) << baseline.tps
                      << '\n';
            std::cout << "AiF GEMV-only modeled:"
                      << " avg=" << std::fixed << std::setprecision(3) << aif_gemv_avg_us << " us/decode-step"
                      << " tps=" << std::fixed << std::setprecision(2) << gemv_only_tps
                      << " speedup_vs_cpu_eval=" << std::fixed << std::setprecision(3) << speedup_gemv_only << "x"
                      << '\n';
            std::cout << "AIF ioctl-observed:"
                      << " avg=" << std::fixed << std::setprecision(3) << aif_ioctl_avg_us << " us/decode-step"
                      << " tps=" << std::fixed << std::setprecision(2) << ioctl_tps
                      << " speedup_vs_cpu_eval=" << std::fixed << std::setprecision(3) << speedup_ioctl << "x"
                      << '\n';
            std::cout << "Memory+SSD estimated full decode:"
                      << " CPU_eval=" << std::fixed << std::setprecision(3) << baseline.avg_us << " us/token"
                      << " + matrix_read_LB=" << std::fixed << std::setprecision(3) << memory_ssd_read_avg_us << " us/token"
                      << " => avg=" << std::fixed << std::setprecision(3) << memory_ssd_full_avg_us << " us/token"
                      << " tps=" << std::fixed << std::setprecision(2) << memory_ssd_full_tps
                      << '\n';
            std::cout << "AiF GEMV-only speedup_vs_Memory+SSD_matrix_read_LB="
                      << std::fixed << std::setprecision(3) << matrix_read_speedup << "x"
                      << '\n';
            if (aif_eval.valid) {
                const double aif_e2e_speedup_vs_cpu = aif_eval.avg_us > 0.0 ? baseline.avg_us / aif_eval.avg_us : 0.0;
                const double aif_e2e_speedup_vs_memory_ssd = aif_eval.avg_us > 0.0 ? memory_ssd_full_avg_us / aif_eval.avg_us : 0.0;
                std::cout << "AIF replacement end-to-end:"
                          << " speedup_vs_CPU_eval=" << std::fixed << std::setprecision(3) << aif_e2e_speedup_vs_cpu << "x"
                          << " speedup_vs_Memory+SSD_full_LB=" << std::fixed << std::setprecision(3) << aif_e2e_speedup_vs_memory_ssd << "x"
                          << '\n';
            }

            const double break_even_frac = baseline.avg_us > 0.0 ? aif_gemv_avg_us / baseline.avg_us : 0.0;
            std::cout << "hybrid break-even:"
                      << " CPU offloadable GEMV must exceed " << std::fixed << std::setprecision(3) << aif_gemv_avg_us << " us/token"
                      << " (" << std::fixed << std::setprecision(1) << break_even_frac * 100.0 << "% of CPU eval)"
                      << '\n';

            bool have_hybrid = false;
            double cpu_offloadable_us = 0.0;
            const char * source = "";
            if (hybrid_config.has_cpu_offloadable_us) {
                have_hybrid = true;
                cpu_offloadable_us = hybrid_config.cpu_offloadable_us;
                source = "--cpu-offloadable-us";
            } else if (hybrid_config.has_cpu_offloadable_frac) {
                have_hybrid = true;
                cpu_offloadable_us = baseline.avg_us * hybrid_config.cpu_offloadable_frac;
                source = "--cpu-offloadable-frac";
            }

            if (have_hybrid) {
                if (cpu_offloadable_us > baseline.avg_us) {
                    cpu_offloadable_us = baseline.avg_us;
                }
                if (cpu_offloadable_us < 0.0) {
                    cpu_offloadable_us = 0.0;
                }

                const double cpu_remaining_us = baseline.avg_us - cpu_offloadable_us;
                const double hybrid_avg_us = cpu_remaining_us + aif_gemv_avg_us;
                const double hybrid_tps = hybrid_avg_us > 0.0 ? 1000000.0 / hybrid_avg_us : 0.0;
                const double hybrid_speedup = hybrid_avg_us > 0.0 ? baseline.avg_us / hybrid_avg_us : 0.0;

                std::cout << "estimated AiF hybrid end-to-end (" << source << "):"
                          << " CPU_remaining=" << std::fixed << std::setprecision(3) << cpu_remaining_us << " us/token"
                          << " + AiF_GEMV=" << std::fixed << std::setprecision(3) << aif_gemv_avg_us << " us/token"
                          << " => avg=" << std::fixed << std::setprecision(3) << hybrid_avg_us << " us/token"
                          << " tps=" << std::fixed << std::setprecision(2) << hybrid_tps
                          << " speedup_vs_cpu_eval=" << std::fixed << std::setprecision(3) << hybrid_speedup << "x"
                          << '\n';
            } else {
                std::cout << "estimated AiF hybrid end-to-end: provide --cpu-offloadable-us or --cpu-offloadable-frac to compute CPU_remaining + AiF_GEMV\n";
            }

            std::cout << "note: Memory+SSD full decode is a lower-bound estimate: CPU in-memory eval plus modeled SSD matrix-read cost, excluding extra software/cache effects\n";
            std::cout << "note: AiF GEMV-only excludes CPU non-GEMV work; hybrid estimate replaces only the CPU offloadable GEMV portion\n";
        }

        std::sort(decode_steps.begin(), decode_steps.end(), [](const decode_step_summary & lhs, const decode_step_summary & rhs) {
            if (lhs.device_delay_ns != rhs.device_delay_ns) {
                return lhs.device_delay_ns > rhs.device_delay_ns;
            }
            return lhs.decode_call_index < rhs.decode_call_index;
        });

        if (step_top_n == 0 || step_top_n > decode_steps.size()) {
            step_top_n = decode_steps.size();
        }

        std::cout << "\ntop decode steps by modeled delay:\n";
        std::cout << std::left
                  << std::setw(8)  << "decode"
                  << std::right
                  << std::setw(10) << "n_past"
                  << std::setw(8)  << "batch"
                  << std::setw(8)  << "calls"
                  << std::setw(14) << "device_us"
                  << std::setw(12) << "avg_us"
                  << std::setw(14) << "ioctl_us"
                  << '\n';

        for (std::size_t i = 0; i < step_top_n; ++i) {
            const auto & step = decode_steps[i];
            const double avg_gemv_us = step.calls == 0 ? 0.0 : ns_to_us(step.device_delay_ns) / static_cast<double>(step.calls);
            std::cout << std::left
                      << std::setw(8) << step.decode_call_index
                      << std::right
                      << std::setw(10) << step.n_past_before
                      << std::setw(8)  << step.batch_tokens
                      << std::setw(8)  << step.calls
                      << std::setw(14) << std::fixed << std::setprecision(3) << ns_to_us(step.device_delay_ns)
                      << std::setw(12) << std::fixed << std::setprecision(3) << avg_gemv_us
                      << std::setw(14) << std::fixed << std::setprecision(3) << ns_to_us(step.ioctl_elapsed_ns)
                      << '\n';
        }
    }

    std::vector<tensor_summary> summaries;
    summaries.reserve(by_tensor.size());
    for (const auto & [_, summary] : by_tensor) {
        summaries.push_back(summary);
    }

    std::sort(summaries.begin(), summaries.end(), [](const tensor_summary & lhs, const tensor_summary & rhs) {
        if (lhs.device_delay_ns != rhs.device_delay_ns) {
            return lhs.device_delay_ns > rhs.device_delay_ns;
        }
        if (lhs.calls != rhs.calls) {
            return lhs.calls > rhs.calls;
        }
        return lhs.tensor_name < rhs.tensor_name;
    });

    if (top_n == 0 || top_n > summaries.size()) {
        top_n = summaries.size();
    }

    std::cout << "\ntop tensors by modeled delay:\n";
    std::cout << std::left
              << std::setw(6)  << "rank"
              << std::setw(32) << "tensor"
              << std::right
              << std::setw(8)  << "calls"
              << std::setw(14) << "device_us"
              << std::setw(12) << "avg_us"
              << std::setw(14) << "ioctl_us"
              << std::setw(10) << "rows"
              << std::setw(10) << "cols"
              << std::setw(12) << "MiB"
              << '\n';

    for (std::size_t i = 0; i < top_n; ++i) {
        const auto & summary = summaries[i];
        const double avg_us = summary.calls == 0 ? 0.0 : ns_to_us(summary.device_delay_ns) / static_cast<double>(summary.calls);
        std::cout << std::left
                  << std::setw(6) << (i + 1)
                  << std::setw(32) << summary.tensor_name.substr(0, 31)
                  << std::right
                  << std::setw(8) << summary.calls
                  << std::setw(14) << std::fixed << std::setprecision(3) << ns_to_us(summary.device_delay_ns)
                  << std::setw(12) << std::fixed << std::setprecision(3) << avg_us
                  << std::setw(14) << std::fixed << std::setprecision(3) << ns_to_us(summary.ioctl_elapsed_ns)
                  << std::setw(10) << summary.rows
                  << std::setw(10) << summary.cols
                  << std::setw(12) << std::fixed << std::setprecision(3) << bytes_to_mib(summary.matrix_nbytes)
                  << '\n';
    }

    return rows.empty() ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    std::string path;
    std::string baseline_log_path;
    std::string aif_log_path;
    std::size_t top_n = 10;
    std::size_t step_top_n = 10;
    double pcie_mib_s = AIF_PCIE_MIB_PER_SECOND;
    hybrid_estimate_config hybrid_config;

    path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--top") {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 2;
            }
            const auto parsed = std::stoull(argv[++i]);
            if (parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
                std::cerr << "--top is too large\n";
                return 2;
            }
            top_n = static_cast<std::size_t>(parsed);
        } else if (arg == "--steps") {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 2;
            }
            const auto parsed = std::stoull(argv[++i]);
            if (parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
                std::cerr << "--steps is too large\n";
                return 2;
            }
            step_top_n = static_cast<std::size_t>(parsed);
        } else if (arg == "--baseline-log") {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 2;
            }
            baseline_log_path = argv[++i];
        } else if (arg == "--aif-log") {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 2;
            }
            aif_log_path = argv[++i];
        } else if (arg == "--pcie-mib-s") {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 2;
            }
            pcie_mib_s = std::stod(argv[++i]);
            if (pcie_mib_s <= 0.0) {
                std::cerr << "--pcie-mib-s must be positive\n";
                return 2;
            }
        } else if (arg == "--cpu-offloadable-us") {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 2;
            }
            hybrid_config.cpu_offloadable_us = std::stod(argv[++i]);
            if (hybrid_config.cpu_offloadable_us < 0.0) {
                std::cerr << "--cpu-offloadable-us must be non-negative\n";
                return 2;
            }
            hybrid_config.has_cpu_offloadable_us = true;
        } else if (arg == "--cpu-offloadable-frac") {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 2;
            }
            hybrid_config.cpu_offloadable_frac = std::stod(argv[++i]);
            if (hybrid_config.cpu_offloadable_frac < 0.0 || hybrid_config.cpu_offloadable_frac > 1.0) {
                std::cerr << "--cpu-offloadable-frac must be in [0, 1]\n";
                return 2;
            }
            hybrid_config.has_cpu_offloadable_frac = true;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (hybrid_config.has_cpu_offloadable_us && hybrid_config.has_cpu_offloadable_frac) {
        std::cerr << "use only one of --cpu-offloadable-us or --cpu-offloadable-frac\n";
        return 2;
    }

    try {
        const baseline_eval_summary baseline = read_eval_log(baseline_log_path, "baseline");
        const baseline_eval_summary aif_eval = read_eval_log(aif_log_path, "AIF");
        if ((hybrid_config.has_cpu_offloadable_us || hybrid_config.has_cpu_offloadable_frac) && !baseline.valid) {
            throw std::runtime_error("--cpu-offloadable-* requires --baseline-log");
        }
        return run_report(path, top_n, step_top_n, baseline, aif_eval, hybrid_config, pcie_mib_s);
    } catch (const std::exception & e) {
        std::cerr << "llama-aif-report: " << e.what() << '\n';
        return 1;
    }
}
