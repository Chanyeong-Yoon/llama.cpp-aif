#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "aif-client.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "unicode.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <locale>
#include <windows.h>
#include <string.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sys/types.h>
#include <pwd.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);

common_time_meas::common_time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

common_time_meas::~common_time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}

//
// CPU utils
//

int32_t common_cpu_get_num_physical_cores() {
#ifdef __linux__
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    unsigned int n_threads_win = std::thread::hardware_concurrency();
    unsigned int default_threads = n_threads_win > 0 ? (n_threads_win <= 4 ? n_threads_win : n_threads_win / 2) : 4;

    DWORD buffer_size = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return default_threads;
        }
    }

    std::vector<char> buffer(buffer_size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &buffer_size)) {
        return default_threads;
    }

    int32_t num_physical_cores = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    while (buffer_size > 0) {
        if (info->Relationship == RelationProcessorCore) {
            num_physical_cores += info->Processor.GroupCount;
        }
        buffer_size -= info->Size;
        info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<char*>(info) + info->Size);
    }

    return num_physical_cores > 0 ? num_physical_cores : default_threads;
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>

static void cpuid(unsigned leaf, unsigned subleaf,
                  unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx) {
    __asm__("movq\t%%rbx,%%rsi\n\t"
            "cpuid\n\t"
            "xchgq\t%%rbx,%%rsi"
            : "=a"(*eax), "=S"(*ebx), "=c"(*ecx), "=d"(*edx)
            : "0"(leaf), "2"(subleaf));
}

static int pin_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

static bool is_hybrid_cpu(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return !!(edx & (1u << 15));
}

static bool is_running_on_efficiency_core(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(0x1a, 0, &eax, &ebx, &ecx, &edx);
    int intel_atom = 0x20;
    int core_type = (eax & 0xff000000u) >> 24;
    return core_type == intel_atom;
}

static int cpu_count_math_cpus(int n_cpu) {
    int result = 0;
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        if (pin_cpu(cpu)) {
            return -1;
        }
        if (is_running_on_efficiency_core()) {
            continue; // efficiency cores harm lockstep threading
        }
        ++cpu; // hyperthreading isn't useful for linear algebra
        ++result;
    }
    return result;
}

#endif // __x86_64__ && __linux__

/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t common_cpu_get_num_math() {
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    int n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpu < 1) {
        return common_cpu_get_num_physical_cores();
    }
    if (is_hybrid_cpu()) {
        cpu_set_t affinity;
        if (!pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity)) {
            int result = cpu_count_math_cpus(n_cpu);
            pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            if (result > 0) {
                return result;
            }
        }
    }
#endif
    return common_cpu_get_num_physical_cores();
}

// Helper for setting process priority

#if defined(_WIN32)

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    DWORD p = NORMAL_PRIORITY_CLASS;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = BELOW_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_NORMAL:   p = NORMAL_PRIORITY_CLASS;       break;
        case GGML_SCHED_PRIO_MEDIUM:   p = ABOVE_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_HIGH:     p = HIGH_PRIORITY_CLASS;         break;
        case GGML_SCHED_PRIO_REALTIME: p = REALTIME_PRIORITY_CLASS;     break;
    }

    if (!SetPriorityClass(GetCurrentProcess(), p)) {
        LOG_WRN("failed to set process priority class %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#else // MacOS and POSIX
#include <sys/types.h>
#include <sys/resource.h>

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    int p = 0;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p =  5;  break;
        case GGML_SCHED_PRIO_NORMAL:   p =  0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   p = -5;  break;
        case GGML_SCHED_PRIO_HIGH:     p = -10; break;
        case GGML_SCHED_PRIO_REALTIME: p = -20; break;
    }

    if (setpriority(PRIO_PROCESS, 0, p) != 0) {
        LOG_WRN("failed to set process priority %d : %s (%d)\n", prio, strerror(errno), errno);
        return false;
    }
    return true;
}

#endif

//
// CLI argument parsing
//


void postprocess_cpu_params(common_cpu_params & cpuparams, const common_cpu_params * role_model) {
    int32_t n_set = 0;

    if (cpuparams.n_threads < 0) {
        // Assuming everything about cpuparams is invalid
        if (role_model != nullptr) {
            cpuparams = *role_model;
        } else {
            cpuparams.n_threads = common_cpu_get_num_math();
        }
    }

    for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (cpuparams.cpumask[i]) {
            n_set++;
        }
    }

    if (n_set && n_set < cpuparams.n_threads) {
        // Not enough set bits, may experience performance issues.
        LOG_WRN("Not enough set bits in CPU mask (%d) to satisfy requested thread count: %d\n", n_set, cpuparams.n_threads);
    }
}

bool parse_cpu_range(const std::string & range, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    size_t dash_loc = range.find('-');
    if (dash_loc == std::string::npos) {
        LOG_ERR("Format of CPU range is invalid! Expected [<start>]-[<end>].\n");
        return false;
    }

    size_t start_i;
    size_t end_i;

    if (dash_loc == 0) {
        start_i = 0;
    } else {
        start_i = std::stoull(range.substr(0, dash_loc));
        if (start_i >= GGML_MAX_N_THREADS) {
            LOG_ERR("Start index out of bounds!\n");
            return false;
        }
    }

    if (dash_loc == range.length() - 1) {
        end_i = GGML_MAX_N_THREADS - 1;
    } else {
        end_i = std::stoull(range.substr(dash_loc + 1));
        if (end_i >= GGML_MAX_N_THREADS) {
            LOG_ERR("End index out of bounds!\n");
            return false;
        }
    }

    for (size_t i = start_i; i <= end_i; i++) {
        boolmask[i] = true;
    }

    return true;
}

bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    // Discard potential 0x prefix
    size_t start_i = 0;
    if (mask.length() >= 2 && mask.substr(0, 2) == "0x") {
        start_i = 2;
    }

    size_t num_digits = mask.length() - start_i;
    if (num_digits > 128) num_digits = 128;

    size_t end_i = num_digits + start_i;

    for (size_t i = start_i, n = (num_digits*4 - 1); i < end_i; i++, n-=4) {
        char c = mask.at(i);
        int8_t id = c;

        if ((c >= '0' && c <= '9')) {
            id -= '0';
        } else if (c >= 'a' && c <= 'f') {
            id -= 'a' - 10;
        } else if (c >= 'A' && c <= 'F') {
            id -= 'A' - 10;
        } else {
            LOG_ERR("Invalid hex character '%c' at position %d\n", c, int32_t(i));
            return false;
        }

        boolmask[  n  ] = boolmask[  n  ] || ((id & 8) != 0);
        boolmask[n - 1] = boolmask[n - 1] || ((id & 4) != 0);
        boolmask[n - 2] = boolmask[n - 2] || ((id & 2) != 0);
        boolmask[n - 3] = boolmask[n - 3] || ((id & 1) != 0);
    }

    return true;
}

void common_init() {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), true);

    llama_log_set(common_log_default_callback, NULL);
}

void common_params_print_info(const common_params & params, bool print_devices) {
#ifdef NDEBUG
    const char * build_type = "";
#else
    const char * build_type = " (debug)";
#endif
    LOG_TRC("%s: build %d (%s) with %s for %s%s\n", __func__, llama_build_number(), llama_commit(), llama_compiler(), llama_build_target(), build_type);

    LOG_INF("log_info: verbosity = %d (adjust with the `-lv N` CLI arg)\n", common_log_get_verbosity_thold());

    // device enumeration creates a primary context on CUDA backends, skip it when the caller does not own any device
    if (print_devices) {
        LOG_INF("device_info:\n");
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            size_t free, total;
            ggml_backend_dev_memory(dev, &free, &total);
            LOG_INF("  - %-8s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
        }
    }
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
}

std::string common_params_get_system_info(const common_params & params) {
    std::ostringstream os;

    os << "system_info: n_threads = " << params.cpuparams.n_threads;
    if (params.cpuparams_batch.n_threads != -1) {
        os << " (n_threads_batch = " << params.cpuparams_batch.n_threads << ")";
    }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    DWORD logicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    os << " / " << logicalProcessorCount << " | " << llama_print_system_info();
#else
    os << " / " << std::thread::hardware_concurrency() << " | " << llama_print_system_info();
#endif

    return os.str();
}

//
// String utils
//

std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string string_strip(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && std::isspace(str[start])) {
        start++;
    }
    while (end > start && std::isspace(str[end - 1])) {
        end--;
    }
    return str.substr(start, end - start);
}

std::string string_lcs(std::string_view a, std::string_view b) {
    if (a.empty() || b.empty()) return {};

    std::vector<std::vector<size_t>> dp(a.size() + 1, std::vector<size_t>(b.size() + 1, 0));
    size_t best_len = 0;
    size_t best_end_a = 0;

    for (size_t i = 1; i <= a.size(); ++i) {
        for (size_t j = 1; j <= b.size(); ++j) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                if (dp[i][j] > best_len) {
                    best_len = dp[i][j];
                    best_end_a = i;
                }
            }
        }
    }
    return std::string(a.substr(best_end_a - best_len, best_len));
}

std::string string_get_sortable_timestamp() {
    using clock = std::chrono::system_clock;

    const clock::time_point current_time = clock::now();
    const time_t as_time_t = clock::to_time_t(current_time);
    char timestamp_no_ns[100];
    std::strftime(timestamp_no_ns, 100, "%Y_%m_%d-%H_%M_%S", std::localtime(&as_time_t));

    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        current_time.time_since_epoch() % 1000000000).count();
    char timestamp_ns[11];
    snprintf(timestamp_ns, 11, "%09" PRId64, ns);

    return std::string(timestamp_no_ns) + "." + std::string(timestamp_ns);
}

void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string regex_escape(const std::string & s) {
    static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
    return std::regex_replace(s, special_chars, "\\$&");
}

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

std::string string_from(bool value) {
    return value ? "true" : "false";
}

std::string string_from(const std::vector<int> & values) {
    std::stringstream buf;

    buf << "[ ";
    bool first = true;
    for (auto e : values) {
        if (first) {
            first = false;
        } else {
            buf << ", ";
        }
        buf << std::to_string(e);
    }
    buf << " ]";

    return buf.str();
}

std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (const auto & token : tokens) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, token);

        buf << "'" << detokenized << "'"
            << ":" << std::to_string(token);
    }

    buf << " ]";

    return buf.str();
}

std::string string_from(const struct llama_context * ctx, const struct llama_batch & batch) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, batch.token[i]);

        buf << "\n"          << std::to_string(i)
            << ", token '"   << detokenized << "'"
            << ", pos "      << std::to_string(batch.pos[i])
            << ", n_seq_id " << std::to_string(batch.n_seq_id[i])
            << ", seq_id "   << std::to_string(batch.seq_id[i][0])
            << ", logits "   << std::to_string(batch.logits[i]);
    }

    buf << " ]";

    return buf.str();
}

void string_process_escapes(std::string & input) {
    std::size_t input_len = input.length();
    std::size_t output_idx = 0;

    for (std::size_t input_idx = 0; input_idx < input_len; ++input_idx) {
        if (input[input_idx] == '\\' && input_idx + 1 < input_len) {
            switch (input[++input_idx]) {
                case 'n':  input[output_idx++] = '\n'; break;
                case 'r':  input[output_idx++] = '\r'; break;
                case 't':  input[output_idx++] = '\t'; break;
                case '\'': input[output_idx++] = '\''; break;
                case '\"': input[output_idx++] = '\"'; break;
                case '\\': input[output_idx++] = '\\'; break;
                case 'x':
                    // Handle \x12, etc
                    if (input_idx + 2 < input_len) {
                        const char x[3] = { input[input_idx + 1], input[input_idx + 2], 0 };
                        char *err_p = nullptr;
                        const long val = std::strtol(x, &err_p, 16);
                        if (err_p == x + 2) {
                            input_idx += 2;
                            input[output_idx++] = char(val);
                            break;
                        }
                    }
                    // fall through
                default:   input[output_idx++] = '\\';
                           input[output_idx++] = input[input_idx]; break;
            }
        } else {
            input[output_idx++] = input[input_idx];
        }
    }

    input.resize(output_idx);
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides) {
    const char * sep = strchr(data, '=');
    if (sep == nullptr || sep - data >= 128) {
        LOG_ERR("%s: malformed KV override '%s'\n", __func__, data);
        return false;
    }
    llama_model_kv_override kvo;
    std::strncpy(kvo.key, data, sep - data);
    kvo.key[sep - data] = 0;
    sep++;
    if (strncmp(sep, "int:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        kvo.val_i64 = std::atol(sep);
    } else if (strncmp(sep, "float:", 6) == 0) {
        sep += 6;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        kvo.val_f64 = std::atof(sep);
    } else if (strncmp(sep, "bool:", 5) == 0) {
        sep += 5;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        if (std::strcmp(sep, "true") == 0) {
            kvo.val_bool = true;
        } else if (std::strcmp(sep, "false") == 0) {
            kvo.val_bool = false;
        } else {
            LOG_ERR("%s: invalid boolean value for KV override '%s'\n", __func__, data);
            return false;
        }
    } else if (strncmp(sep, "str:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        if (strlen(sep) > 127) {
            LOG_ERR("%s: malformed KV override '%s', value cannot exceed 127 chars\n", __func__, data);
            return false;
        }
        strncpy(kvo.val_str, sep, 127);
        kvo.val_str[127] = '\0';
    } else {
        LOG_ERR("%s: invalid type for KV override '%s'\n", __func__, data);
        return false;
    }
    overrides.emplace_back(std::move(kvo));
    return true;
}

static inline bool glob_class_match(const char c, const char * pattern, const char * class_end) {
    const char * class_start = pattern;
    bool negated = false;

    if (*class_start == '!') {
        negated = true;
        class_start++;
    }

    // If first character after negation is ']' or '-', treat it as literal
    if (*class_start == ']' || *class_start == '-') {
        if (class_start < class_end && *class_start == c) {
            return !negated;
        }
        class_start++;
    }

    bool matched = false;

    while (class_start < class_end) {
        if (class_start + 2 < class_end && class_start[1] == '-' && class_start[2] != ']') {
            char start_char = *class_start;
            char end_char = class_start[2];
            if (c >= start_char && c <= end_char) {
                matched = true;
                break;
            }
            class_start += 3;
        } else {
            if (*class_start == c) {
                matched = true;
                break;
            }
            class_start++;
        }
    }

    return negated ? !matched : matched;
}

// simple glob: * matches non-/ chars, ** matches anything including /, [] matches character class
static inline bool glob_match(const char * pattern, const char * str) {
    if (*pattern == '\0') {
        return *str == '\0';
    }
    if (pattern[0] == '*' && pattern[1] == '*') {
        const char * p = pattern + 2;
        if (glob_match(p, str)) return true;
        if (*str != '\0') return glob_match(pattern, str + 1);
        return false;
    }
    if (*pattern == '*') {
        const char * p = pattern + 1;
        for (; *str != '\0' && *str != '/'; str++) {
            if (glob_match(p, str)) return true;
        }
        return glob_match(p, str);
    }
    if (*pattern == '?' && *str != '\0' && *str != '/') {
        return glob_match(pattern + 1, str + 1);
    }
    if (*pattern == '[') {
        const char * class_end = pattern + 1;
        // If first character after '[' is ']' or '-', treat it as literal
        if (*class_end == ']' || *class_end == '-') {
            class_end++;
        }
        while (*class_end != '\0' && *class_end != ']') {
            class_end++;
        }
        if (*class_end == ']') {
            if (*str == '\0') return false;
            bool matched = glob_class_match(*str, pattern + 1, class_end);
            return matched && glob_match(class_end + 1, str + 1);
        } else {
            if (*str == '[') {
                return glob_match(pattern + 1, str + 1);
            }
            return false;
        }
    }
    if (*pattern == *str) {
        return glob_match(pattern + 1, str + 1);
    }
    return false;
}

bool glob_match(const std::string & pattern, const std::string & str) {
    return glob_match(pattern.c_str(), str.c_str());
}

//
// Filesystem utils
//

// Validate if a filename is safe to use
// To validate a full path, split the path by the OS-specific path separator, and validate each part with this function
bool fs_validate_filename(const std::string & filename, bool allow_subdirs) {
    if (!filename.length()) {
        // Empty filename invalid
        return false;
    }
    if (filename.length() > 255) {
        // Limit at common largest possible filename on Linux filesystems
        // to avoid unnecessary further validation
        // (On systems with smaller limits it will be caught by the OS)
        return false;
    }

    size_t offset = 0;
    while (offset < filename.size()) {
        utf8_parse_result result = common_parse_utf8_codepoint(filename, offset);

        if (result.status != utf8_parse_result::SUCCESS) {
            return false;
        }
        uint32_t c = result.codepoint;

        if ((result.bytes_consumed == 2 && c < 0x80) ||
            (result.bytes_consumed == 3 && c < 0x800) ||
            (result.bytes_consumed == 4 && c < 0x10000)) {
            return false;
        }

        // Check for forbidden codepoints:
        // - Control characters
        // - Unicode equivalents of illegal characters
        // - UTF-16 surrogate pairs
        // - UTF-8 replacement character
        // - Byte order mark (BOM)
        // - Illegal characters: / \ : * ? " < > |
        if (c <= 0x1F // Control characters (C0)
            || c == 0x7F // Control characters (DEL)
            || (c >= 0x80 && c <= 0x9F) // Control characters (C1)
            || c == 0xFF0E // Fullwidth Full Stop (period equivalent)
            || c == 0x2215 // Division Slash (forward slash equivalent)
            || c == 0x2216 // Set Minus (backslash equivalent)
            || (c >= 0xD800 && c <= 0xDFFF) // UTF-16 surrogate pairs
            || c > 0x10FFFF // Max Unicode limit
            || c == 0xFFFD // Replacement Character (UTF-8)
            || c == 0xFEFF // Byte Order Mark (BOM)
            || c == ':' || c == '*' // Illegal characters
            || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            return false;
        }
        if (!allow_subdirs && (c == '/' || c == '\\')) {
            // Subdirectories not allowed, reject path separators
            return false;
        }
        offset += result.bytes_consumed;
    }

    // Reject any leading or trailing ' ', or any trailing '.', these are stripped on Windows and will cause a different filename
    // Unicode and other whitespace is not affected, only 0x20 space
    if (filename.front() == ' ' || filename.back() == ' ' || filename.back() == '.') {
        return false;
    }

    // Reject any ".." (currently stricter than necessary, it should be fine to just check for == ".." instead)
    if (filename.find("..") != std::string::npos) {
        return false;
    }

    // Reject "."
    if (filename == ".") {
        return false;
    }

    return true;
}

#include <iostream>


#ifdef _WIN32
static std::wstring utf8_to_wstring(const std::string & str) {
    if (str.empty()) {
        return std::wstring();
    }

    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);

    if (size <= 0) {
        return std::wstring();
    }

    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size);

    return wstr;
}
#endif

// returns true if successful, false otherwise
bool fs_create_directory_with_parents(const std::string & path) {
#ifdef _WIN32
    std::wstring wpath = utf8_to_wstring(path);

    // if the path already exists, check whether it's a directory
    const DWORD attributes = GetFileAttributesW(wpath.c_str());
    if ((attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return true;
    }

    size_t pos_slash = 0;

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('\\', pos_slash)) != std::string::npos) {
        const std::wstring subpath = wpath.substr(0, pos_slash);

        pos_slash += 1;

        // skip the drive letter, in some systems it can return an access denied error
        if (subpath.length() == 2 && subpath[1] == ':') {
            continue;
        }

        const bool success = CreateDirectoryW(subpath.c_str(), NULL);

        if (!success) {
            const DWORD error = GetLastError();

            // if the path already exists, ensure that it's a directory
            if (error == ERROR_ALREADY_EXISTS) {
                const DWORD attributes = GetFileAttributesW(subpath.c_str());
                if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    return true;
#else
    // if the path already exists, check whether it's a directory
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }

    size_t pos_slash = 1; // skip leading slashes for directory creation

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
        const std::string subpath = path.substr(0, pos_slash);
        struct stat info;

        // if the path already exists, ensure that it's a directory
        if (stat(subpath.c_str(), &info) == 0) {
            if (!S_ISDIR(info.st_mode)) {
                return false;
            }
        } else {
            // create parent directories
            const int ret = mkdir(subpath.c_str(), 0755);
            if (ret != 0) {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#endif // _WIN32
}

bool fs_is_directory(const std::string & path) {
    std::filesystem::path dir(path);
    return std::filesystem::exists(dir) && std::filesystem::is_directory(dir);
}

std::string fs_get_cache_directory() {
    std::string cache_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        // Make sure to add trailing slash
        if (p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
    if (getenv("LLAMA_CACHE")) {
        cache_directory = std::getenv("LLAMA_CACHE");
    } else {
#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX) || \
        defined(__OpenBSD__) || defined(__NetBSD__)
        if (std::getenv("XDG_CACHE_HOME")) {
            cache_directory = std::getenv("XDG_CACHE_HOME");
        } else if (std::getenv("HOME")) {
            cache_directory = std::getenv("HOME") + std::string("/.cache/");
        } else {
#if defined(__linux__)
            /* no $HOME is defined, fallback to getpwuid */
            struct passwd *pw = getpwuid(getuid());
            if ((!pw) || (!pw->pw_dir)) {
                throw std::runtime_error("Failed to find $HOME directory");
            }

            cache_directory = std::string(pw->pw_dir) + std::string("/.cache/");
#else /* defined(__linux__) */
            throw std::runtime_error("Failed to find $HOME directory");
#endif /* defined(__linux__) */
        }
#elif defined(__APPLE__)
        cache_directory = std::getenv("HOME") + std::string("/Library/Caches/");
#elif defined(_WIN32)
        cache_directory = std::getenv("LOCALAPPDATA");
#elif defined(__EMSCRIPTEN__)
        GGML_ABORT("not implemented on this platform");
#else
#  error Unknown architecture
#endif
        cache_directory = ensure_trailing_slash(cache_directory);
        cache_directory += "llama.cpp";
    }
    return ensure_trailing_slash(cache_directory);
}

std::string fs_get_cache_file(const std::string & filename) {
    GGML_ASSERT(filename.find(DIRECTORY_SEPARATOR) == std::string::npos);
    std::string cache_directory = fs_get_cache_directory();
    const bool success = fs_create_directory_with_parents(cache_directory);
    if (!success) {
        throw std::runtime_error("failed to create cache directory: " + cache_directory);
    }
    return cache_directory + filename;
}

std::vector<common_file_info> fs_list(const std::string & path, bool include_directories) {
    std::vector<common_file_info> files;
    if (path.empty()) return files;

    std::filesystem::path dir(path);
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return files;
    }

    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        try {
            // Only include regular files (skip directories)
            const auto & p = entry.path();
            if (std::filesystem::is_regular_file(p)) {
                common_file_info info;
                info.path   = p.string();
                info.name   = p.filename().string();
                info.is_dir = false;
                try {
                    info.size = static_cast<size_t>(std::filesystem::file_size(p));
                } catch (const std::filesystem::filesystem_error &) {
                    info.size = 0;
                }
                files.push_back(std::move(info));
            } else if (include_directories && std::filesystem::is_directory(p)) {
                common_file_info info;
                info.path   = p.string();
                info.name   = p.filename().string();
                info.size   = 0; // Directories have no size
                info.is_dir = true;
                files.push_back(std::move(info));
            }
        } catch (const std::filesystem::filesystem_error &) {
            // skip entries we cannot inspect
            continue;
        }
    }

    return files;
}

std::ifstream fs_open_ifstream(const std::string & fname, std::ios_base::openmode mode) {
#ifdef _WIN32
    int wlen = MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, NULL, 0);
    if (!wlen) { return std::ifstream(); }
    std::vector<wchar_t> wfname(wlen);
    (void)MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, wfname.data(), wlen);
    return std::ifstream(wfname.data(), mode);
#else
    return std::ifstream(fname, mode);
#endif
}

//
// TTY utils
//

bool tty_can_use_colors() {
    // Check NO_COLOR environment variable (https://no-color.org/)
    if (const char * no_color = std::getenv("NO_COLOR")) {
        if (no_color[0] != '\0') {
            return false;
        }
    }

    // Check TERM environment variable
    if (const char * term = std::getenv("TERM")) {
        if (std::strcmp(term, "dumb") == 0) {
            return false;
        }
    }

    // Check if stdout and stderr are connected to a terminal
    // We check both because log messages can go to either
    bool stdout_is_tty = isatty(fileno(stdout));
    bool stderr_is_tty = isatty(fileno(stderr));

    return stdout_is_tty || stderr_is_tty;
}

//
// Model utils
//

// TODO: move to common/sampling
static void common_init_sampler_from_model(
    const llama_model * model,
    common_params_sampling & sparams) {

    const uint64_t config = sparams.user_sampling_config;

    auto get_int32 = [&](const char * key, int32_t & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[64] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            int32_t v = strtol(buf, &end, 10);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    auto get_float = [&](const char * key, float & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[128] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            float v = strtof(buf, &end);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    // Sampling sequence
    if (!(config & common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS)) {
        char buf[512] = {0};
        if (llama_model_meta_val_str(model, llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE), buf, sizeof(buf)) > 0) {
            const std::vector<std::string> sampler_names = string_split<std::string>(std::string(buf), ';');
            if (!sampler_names.empty()) {
                sparams.samplers = common_sampler_types_from_names(sampler_names);
            }
        }
    }

    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_K),           sparams.top_k,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_P),           sparams.top_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIN_P),           sparams.min_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY), sparams.xtc_probability, common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD),   sparams.xtc_threshold,   common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TEMP),            sparams.temp,            common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N),  sparams.penalty_last_n,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT),  sparams.penalty_repeat,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT),        sparams.mirostat,        common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU),    sparams.mirostat_tau,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA),    sparams.mirostat_eta,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA);
}

static uint64_t common_aif_align_up(uint64_t value, uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

static constexpr uint64_t COMMON_AIF_SECTOR_SIZE = 512;
static constexpr uint64_t COMMON_AIF_PAGE_SIZE   = 16 * 1024;
static constexpr uint64_t COMMON_AIF_LBAS_PER_PAGE = COMMON_AIF_PAGE_SIZE / COMMON_AIF_SECTOR_SIZE;
static constexpr uint64_t COMMON_AIF_CHANNELS = 8;
static constexpr uint64_t COMMON_AIF_CHIPS = 16;
static constexpr uint64_t COMMON_AIF_PLANES_PER_CHIP = 4;

static bool common_aif_validate_device_profile(const aif_stats_resp & stats, std::string & error) {
    if (stats.config_channels != COMMON_AIF_CHANNELS ||
        stats.config_chips != COMMON_AIF_CHIPS ||
        stats.config_planes_per_chip != COMMON_AIF_PLANES_PER_CHIP ||
        stats.config_page_size != COMMON_AIF_PAGE_SIZE) {
        error = string_format(
                "incompatible AIF SSD profile: got channels=%" PRIu64
                ", chips=%" PRIu64 ", planes/chip=%" PRIu64 ", page=%" PRIu64
                "; expected 8, 16, 4, 16384",
                stats.config_channels,
                stats.config_chips,
                stats.config_planes_per_chip,
                stats.config_page_size);
        return false;
    }
    if (stats.config_pcie_bytes_per_sec == 0 ||
        stats.config_onfi_bytes_per_sec == 0 ||
        stats.config_chip_bytes_per_sec == 0 ||
        stats.config_cr_read_ns == 0) {
        error = "AIF SSD profile reports a zero timing parameter";
        return false;
    }

    return true;
}

static uint64_t common_aif_tensor_id(const std::string & name) {
    uint64_t hash = 1469598103934665603ull;

    for (const unsigned char c : name) {
        hash ^= c;
        hash *= 1099511628211ull;
    }

    return hash == 0 ? 1 : hash;
}

static bool common_aif_u64_to_u32(uint64_t value, uint32_t & out) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    out = static_cast<uint32_t>(value);
    return true;
}

enum class common_aif_tensor_stage {
    other,
    attn_q,
    attn_k,
    attn_v,
    attn_output,
    ffn_gate,
    ffn_up,
    ffn_down,
    output,
};

struct common_aif_tensor_identity {
    common_aif_tensor_stage stage = common_aif_tensor_stage::other;
    int32_t layer = -1;
};

static common_aif_tensor_identity common_aif_parse_tensor_identity(const std::string & name) {
    if (name == "output.weight") {
        return { common_aif_tensor_stage::output, -1 };
    }
    if (name.rfind("blk.", 0) != 0) {
        return {};
    }

    const size_t layer_end = name.find('.', 4);
    if (layer_end == std::string::npos) {
        return {};
    }

    int32_t layer = -1;
    try {
        const std::string layer_text = name.substr(4, layer_end - 4);
        size_t consumed = 0;
        const long parsed = std::stol(layer_text, &consumed, 10);
        if (consumed != layer_text.size() || parsed < 0 || parsed > std::numeric_limits<int32_t>::max()) {
            return {};
        }
        layer = static_cast<int32_t>(parsed);
    } catch (const std::exception &) {
        return {};
    }

    const std::string suffix = name.substr(layer_end + 1);
    if (suffix == "attn_q.weight")      return { common_aif_tensor_stage::attn_q, layer };
    if (suffix == "attn_k.weight")      return { common_aif_tensor_stage::attn_k, layer };
    if (suffix == "attn_v.weight")      return { common_aif_tensor_stage::attn_v, layer };
    if (suffix == "attn_output.weight") return { common_aif_tensor_stage::attn_output, layer };
    if (suffix == "ffn_gate.weight")    return { common_aif_tensor_stage::ffn_gate, layer };
    if (suffix == "ffn_up.weight")      return { common_aif_tensor_stage::ffn_up, layer };
    if (suffix == "ffn_down.weight")    return { common_aif_tensor_stage::ffn_down, layer };

    return {};
}

static bool common_aif_is_qkv_stage(common_aif_tensor_stage stage) {
    return stage == common_aif_tensor_stage::attn_q ||
           stage == common_aif_tensor_stage::attn_k ||
           stage == common_aif_tensor_stage::attn_v;
}

static bool common_aif_is_ffn_stage(common_aif_tensor_stage stage) {
    return stage == common_aif_tensor_stage::ffn_gate ||
           stage == common_aif_tensor_stage::ffn_up ||
           stage == common_aif_tensor_stage::ffn_down;
}

struct common_aif_posted_tensor {
    std::string name;
    aif_post_req req;
    common_aif_tensor_identity identity;
    uint32_t full_rows = 0;
    uint64_t full_matrix_nbytes = 0;
    uint32_t host_rows = 0;
    uint64_t host_matrix_nbytes = 0;
};

struct common_aif_gemv_result {
    bool success = false;
    bool stats_valid = false;

    std::string phase = "unknown";
    uint64_t decode_call_index = 0;
    int32_t batch_tokens = 0;
    int32_t n_past_before = 0;

    uint32_t input_nbytes = 0;
    uint32_t output_nbytes = 0;
    uint64_t input_checksum = 0;
    uint32_t output_mismatches = 0;
    int64_t ioctl_elapsed_ns = 0;

    uint64_t call_index = 0;
    uint64_t async_task_id = 0;
    std::string execution_mode = "sequential";
    int32_t layer = -1;
    uint64_t submit_offset_ns = 0;
    uint64_t device_start_offset_ns = 0;
    uint64_t device_end_offset_ns = 0;
    uint64_t predicted_device_ns = 0;
    uint64_t head_first_ready_wait_ns = 0;
    uint64_t dependency_wait_ns = 0;
    uint64_t host_matrix_nbytes = 0;
    uint64_t host_modeled_ns = 0;
    uint64_t host_actual_ns = 0;
    uint64_t parallel_elapsed_ns = 0;
    uint64_t overlap_ns = 0;

    uint64_t qkv_group_id = 0;
    std::string qkv_schedule;
    uint32_t q_heads = 0;
    uint32_t kv_heads = 0;
    bool qkv_group_summary = false;
    uint64_t qkv_group_modeled_ns = 0;
    uint64_t qkv_first_head_ready_ns = 0;
    uint64_t qkv_last_head_ready_ns = 0;
    uint64_t host_attention_actual_ns = 0;
    uint64_t host_idle_wait_ns = 0;
    uint64_t aif_idle_wait_ns = 0;
    uint64_t head_pipeline_finish_ns = 0;
    uint64_t head_pipeline_tail_wait_ns = 0;

    aif_stats_resp stats {};
};

struct common_aif_decode_scope_state {
    std::string phase = "unknown";
    uint64_t decode_call_index = 0;
    int32_t batch_tokens = 0;
    int32_t n_past_before = 0;
};

struct common_aif_device_prediction {
    uint64_t input_prefix_ns = 0;
    uint64_t stream_ns = 0;
    uint64_t total_ns = 0;
};

struct common_aif_async_task {
    uint64_t id = 0;
    uint64_t ticket = 0;
    uint64_t call_index = 0;
    common_aif_posted_tensor posted;
    common_aif_decode_scope_state scope;
    std::string execution_mode;
    std::vector<uint8_t> input;
    common_aif_gemv_result result;
    std::promise<bool> completion;
    std::shared_future<bool> done;

    std::mutex start_mutex;
    std::condition_variable start_cv;
    bool started = false;
    int64_t submit_us = 0;
    int64_t start_us = 0;
    int64_t end_us = 0;
    uint64_t predicted_device_ns = 0;
    common_aif_device_prediction prediction;
    uint64_t head_first_ready_wait_ns = 0;
    uint64_t qkv_group_id = 0;
    uint32_t q_heads = 0;
    uint32_t kv_heads = 0;
};

struct common_aif_qkv_group {
    uint64_t id = 0;
    int32_t layer = -1;
    std::vector<std::shared_ptr<common_aif_async_task>> tasks;
    std::vector<uint64_t> head_ready_ns;
    uint64_t modeled_start_ns = 0;
    uint64_t host_start_ns = 0;
    uint64_t modeled_ns = 0;
    uint64_t first_head_ready_ns = 0;
    uint64_t last_head_ready_ns = 0;
    bool timeline_ready = false;
};

struct common_aif_runtime {
    ~common_aif_runtime();

    common_aif_client client;
    std::vector<common_aif_posted_tensor> posted;
    std::ofstream shadow_log;
    std::string shadow_log_path;
    std::mutex shadow_log_mutex;
    uint64_t shadow_log_rows = 0;
    int32_t shadow_gemv_max = 1;
    int32_t shadow_gemv_count = 0;
    bool replace_gemv = false;
    bool parallel = false;
    bool graph_decode_only = true;
    bool log_gemv = false;

    uint64_t start_us = 0;
    uint64_t async_task_next = 0;
    uint64_t device_ticket_next = 0;
    uint64_t qkv_group_next = 0;
    std::mutex device_queue_mutex;
    std::condition_variable device_queue_cv;
    std::deque<std::shared_ptr<common_aif_async_task>> device_queue;
    std::thread device_worker;
    bool device_worker_stop = false;
    std::unordered_map<int32_t, common_aif_qkv_group> pending_qkv;

    uint64_t host_budget_bytes = 0;
    uint64_t kv_cache_bytes = 0;
    uint64_t base_host_model_bytes = 0;
    uint64_t total_ffn_bytes = 0;
    uint64_t host_ffn_bytes = 0;
    double host_fraction = 0.0;
    uint64_t host_bandwidth_bytes_s = 86500000000ull;
    uint32_t n_heads = 1;
    uint32_t n_heads_kv = 1;

    aif_stats_resp device_config {};
};

static std::mutex common_aif_decode_scope_mutex;
static uint64_t common_aif_decode_scope_next = 0;
static common_aif_decode_scope_state common_aif_decode_scope_current;

struct common_aif_decode_scope {
    common_aif_decode_scope(const char * phase, int32_t batch_tokens, int32_t n_past_before) {
        std::lock_guard<std::mutex> lock(common_aif_decode_scope_mutex);
        common_aif_decode_scope_current.phase = phase != nullptr ? phase : "unknown";
        common_aif_decode_scope_current.decode_call_index = ++common_aif_decode_scope_next;
        common_aif_decode_scope_current.batch_tokens = batch_tokens;
        common_aif_decode_scope_current.n_past_before = n_past_before;
    }

    ~common_aif_decode_scope() {
        std::lock_guard<std::mutex> lock(common_aif_decode_scope_mutex);
        common_aif_decode_scope_current = {};
    }
};

static common_aif_decode_scope_state common_aif_decode_scope_get() {
    std::lock_guard<std::mutex> lock(common_aif_decode_scope_mutex);
    return common_aif_decode_scope_current;
}

static bool common_aif_graph_phase_is_allowed(const common_aif_runtime * runtime) {
    if (runtime == nullptr || !runtime->graph_decode_only) {
        return true;
    }

    const auto scope = common_aif_decode_scope_get();
    return scope.phase == "decode";
}

static int common_aif_decode_with_scope(
        llama_context * ctx,
        llama_batch batch,
        const char * phase,
        int32_t batch_tokens,
        int32_t n_past_before) {
    common_aif_decode_scope scope(phase, batch_tokens, n_past_before);
    return llama_decode(ctx, batch);
}

static std::string common_aif_csv_escape(const std::string & value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char c : value) {
        if (c == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(c);
    }
    escaped.push_back('"');
    return escaped;
}

static void common_aif_write_shadow_log_header(std::ostream & out) {
    out << "call_index"
        << ",phase"
        << ",decode_call_index"
        << ",batch_tokens"
        << ",n_past_before"
        << ",success"
        << ",execution_mode"
        << ",layer"
        << ",async_task_id"
        << ",tensor_name"
        << ",tensor_id"
        << ",rows"
        << ",cols"
        << ",ggml_type"
        << ",matrix_nbytes"
        << ",full_rows"
        << ",full_matrix_nbytes"
        << ",host_rows"
        << ",host_matrix_nbytes"
        << ",lba_start"
        << ",lba_count"
        << ",input_nbytes"
        << ",output_nbytes"
        << ",input_checksum"
        << ",output_mismatches"
        << ",ioctl_elapsed_ns"
        << ",submit_offset_ns"
        << ",device_start_offset_ns"
        << ",device_end_offset_ns"
        << ",predicted_device_ns"
        << ",head_first_ready_wait_ns"
        << ",dependency_wait_ns"
        << ",host_modeled_ns"
        << ",host_actual_ns"
        << ",parallel_elapsed_ns"
        << ",overlap_ns"
        << ",qkv_group_id"
        << ",qkv_schedule"
        << ",q_heads"
        << ",kv_heads"
        << ",qkv_group_summary"
        << ",qkv_group_modeled_ns"
        << ",qkv_first_head_ready_ns"
        << ",qkv_last_head_ready_ns"
        << ",host_attention_actual_ns"
        << ",host_idle_wait_ns"
        << ",aif_idle_wait_ns"
        << ",head_pipeline_finish_ns"
        << ",head_pipeline_tail_wait_ns"
        << ",stats_valid"
        << ",device_post_count"
        << ",device_gemv_count"
        << ",device_tensor_count"
        << ",device_delay_ns"
        << ",input_pcie_ns"
        << ",matrix_ns"
        << ",output_pcie_ns"
        << ",input_onfi_ns"
        << ",chip_compute_ns"
        << ",output_onfi_ns"
        << ",active_chips"
        << ",tail_chip"
        << ",layout_hit_count"
        << ",layout_miss_count"
        << ",layout_slot"
        << ",layout_pages"
        << ",layout_stripes"
        << ",layout_units"
        << ",mapped_pages"
        << ",units_used"
        << ",max_unit_pages"
        << ",layout_chips"
        << ",ifp_blocks"
        << ",reserved_ifp_lines"
        << ",max_chip_pages"
        << ",lsb_pages"
        << ",non_lsb_pages"
        << ",first_ppa"
        << ",last_ppa"
        << '\n';
}

static bool common_aif_open_shadow_log(common_aif_runtime & runtime, const std::string & path) {
    runtime.shadow_log_path = path;
    runtime.shadow_log.open(path, std::ios::out | std::ios::trunc);
    if (!runtime.shadow_log) {
        LOG_ERR("%s: failed to open AIF shadow GEMV CSV log '%s'\n", __func__, path.c_str());
        return false;
    }

    common_aif_write_shadow_log_header(runtime.shadow_log);
    runtime.shadow_log.flush();
    return true;
}

static void common_aif_write_shadow_log_row(
        common_aif_runtime & runtime,
        const common_aif_posted_tensor & posted,
        const common_aif_gemv_result & result) {
    if (!runtime.shadow_log.is_open()) {
        return;
    }

    const aif_stats_resp empty_stats {};
    const aif_stats_resp & stats = result.stats_valid ? result.stats : empty_stats;

    std::lock_guard<std::mutex> lock(runtime.shadow_log_mutex);
    runtime.shadow_log
        << result.call_index
        << ',' << common_aif_csv_escape(result.phase)
        << ',' << result.decode_call_index
        << ',' << result.batch_tokens
        << ',' << result.n_past_before
        << ',' << (result.success ? 1 : 0)
        << ',' << common_aif_csv_escape(result.execution_mode)
        << ',' << result.layer
        << ',' << result.async_task_id
        << ',' << common_aif_csv_escape(posted.name)
        << ',' << posted.req.tensor_id
        << ',' << posted.req.rows
        << ',' << posted.req.cols
        << ',' << posted.req.ggml_type
        << ',' << posted.req.matrix_nbytes
        << ',' << posted.full_rows
        << ',' << posted.full_matrix_nbytes
        << ',' << posted.host_rows
        << ',' << posted.host_matrix_nbytes
        << ',' << posted.req.lba_start
        << ',' << posted.req.lba_count
        << ',' << result.input_nbytes
        << ',' << result.output_nbytes
        << ',' << result.input_checksum
        << ',' << result.output_mismatches
        << ',' << result.ioctl_elapsed_ns
        << ',' << result.submit_offset_ns
        << ',' << result.device_start_offset_ns
        << ',' << result.device_end_offset_ns
        << ',' << result.predicted_device_ns
        << ',' << result.head_first_ready_wait_ns
        << ',' << result.dependency_wait_ns
        << ',' << result.host_modeled_ns
        << ',' << result.host_actual_ns
        << ',' << result.parallel_elapsed_ns
        << ',' << result.overlap_ns
        << ',' << result.qkv_group_id
        << ',' << common_aif_csv_escape(result.qkv_schedule)
        << ',' << result.q_heads
        << ',' << result.kv_heads
        << ',' << (result.qkv_group_summary ? 1 : 0)
        << ',' << result.qkv_group_modeled_ns
        << ',' << result.qkv_first_head_ready_ns
        << ',' << result.qkv_last_head_ready_ns
        << ',' << result.host_attention_actual_ns
        << ',' << result.host_idle_wait_ns
        << ',' << result.aif_idle_wait_ns
        << ',' << result.head_pipeline_finish_ns
        << ',' << result.head_pipeline_tail_wait_ns
        << ',' << (result.stats_valid ? 1 : 0)
        << ',' << stats.post_count
        << ',' << stats.gemv_count
        << ',' << stats.tensor_count
        << ',' << stats.last_delay_ns
        << ',' << stats.last_input_pcie_ns
        << ',' << stats.last_matrix_compute_ns
        << ',' << stats.last_output_pcie_ns
        << ',' << stats.last_input_onfi_ns
        << ',' << stats.last_chip_compute_ns
        << ',' << stats.last_output_onfi_ns
        << ',' << stats.last_active_chips
        << ',' << stats.last_tail_chip
        << ',' << stats.layout_hit_count
        << ',' << stats.layout_miss_count
        << ',' << stats.last_layout_slot
        << ',' << stats.last_layout_pages
        << ',' << stats.last_layout_stripes
        << ',' << stats.last_layout_parallel_units
        << ',' << stats.last_mapped_pages
        << ',' << stats.last_units_used
        << ',' << stats.last_max_pages_per_unit
        << ',' << stats.last_layout_chip_count
        << ',' << stats.last_layout_ifp_blocks
        << ',' << stats.last_reserved_ifp_lines
        << ',' << stats.last_max_chip_pages
        << ',' << stats.last_lsb_pages
        << ',' << stats.last_non_lsb_pages
        << ',' << stats.last_first_ppa
        << ',' << stats.last_last_ppa
        << '\n';
    runtime.shadow_log_rows++;
    if (runtime.shadow_log_rows % 256 == 0) {
        runtime.shadow_log.flush();
    }
}

static bool common_aif_fill_post_req(
        const ggml_tensor * tensor,
        uint64_t lba_start,
        uint32_t selected_rows,
        aif_post_req & req,
        std::string & error) {
    if ((lba_start % COMMON_AIF_LBAS_PER_PAGE) != 0) {
        error = "AIF LBA start must be 16 KiB aligned";
        return false;
    }

    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t elem_size = 0;
    uint32_t row_stride = 0;

    if (!common_aif_u64_to_u32(static_cast<uint64_t>(tensor->ne[1]), rows) ||
        !common_aif_u64_to_u32(static_cast<uint64_t>(tensor->ne[0]), cols) ||
        !common_aif_u64_to_u32(static_cast<uint64_t>(ggml_type_size(tensor->type)), elem_size) ||
        !common_aif_u64_to_u32(static_cast<uint64_t>(tensor->nb[1]), row_stride)) {
        error = "tensor metadata does not fit AIF protocol fields";
        return false;
    }
    if (selected_rows == 0 || selected_rows > rows) {
        error = "selected AIF row count is invalid";
        return false;
    }

    const uint64_t full_matrix_nbytes = ggml_nbytes(tensor);
    const uint64_t matrix_nbytes = selected_rows == rows ?
            full_matrix_nbytes : static_cast<uint64_t>(row_stride) * selected_rows;
    const uint64_t matrix_pages = common_aif_align_up(matrix_nbytes, COMMON_AIF_PAGE_SIZE) / COMMON_AIF_PAGE_SIZE;

    if (matrix_pages == 0 || matrix_nbytes > full_matrix_nbytes) {
        error = "selected tensor storage range is invalid";
        return false;
    }

    std::memset(&req, 0, sizeof(req));
    req.magic = AIF_MAGIC;
    req.version = AIF_VERSION;
    req.flags = AIF_POST_F_LAYOUT_ONLY;
    req.tensor_id = common_aif_tensor_id(ggml_get_name(tensor));
    req.lba_start = lba_start;
    req.lba_count = matrix_pages * COMMON_AIF_LBAS_PER_PAGE;
    req.rows = selected_rows;
    req.cols = cols;
    req.ggml_type = static_cast<uint32_t>(tensor->type);
    req.elem_size = elem_size;
    req.matrix_nbytes = matrix_nbytes;
    req.row_stride = row_stride;
    std::snprintf(req.tensor_name, sizeof(req.tensor_name), "%s", ggml_get_name(tensor));

    return true;
}

static uint64_t common_aif_checksum_bytes(const uint8_t * buf, uint32_t len) {
    uint64_t checksum = 0;

    for (uint32_t i = 0; i < len; ++i) {
        checksum = checksum * 131 + buf[i];
    }

    return checksum;
}

static bool common_aif_alloc_page_aligned(size_t size, std::unique_ptr<uint8_t, decltype(&std::free)> & ptr) {
    void * raw = nullptr;

    if (posix_memalign(&raw, 4096, size) != 0) {
        return false;
    }

    std::memset(raw, 0, size);
    ptr.reset(static_cast<uint8_t *>(raw));
    return true;
}

static bool common_aif_smoke_gemv(
        const common_aif_client & client,
        const aif_post_req & posted,
        const std::string & tensor_name,
        const char * label,
        common_aif_gemv_result * result = nullptr,
        const ggml_tensor * input_tensor = nullptr,
        bool log_result = true,
        const std::vector<uint8_t> * captured_input = nullptr,
        const common_aif_decode_scope_state * scope_override = nullptr,
        uint64_t call_index = 0) {
    constexpr uint32_t activation_elem_size = sizeof(float);
    // The controller MDTS permits 1 MiB, but this host exposes max_segments=128. With 4 KiB
    // pages, passthrough buffers above 512 KiB cannot be mapped into one NVMe command.
    constexpr uint32_t max_ioctl_payload = 512 * 1024;

    if (result != nullptr) {
        *result = {};
    }

    uint32_t input_nbytes = 0;
    uint32_t output_nbytes = 0;

    if (!common_aif_u64_to_u32(static_cast<uint64_t>(posted.cols) * activation_elem_size, input_nbytes) ||
        !common_aif_u64_to_u32(static_cast<uint64_t>(posted.rows) * activation_elem_size, output_nbytes)) {
        LOG_ERR("%s: %s buffer size overflow for tensor '%s'\n", __func__, label, tensor_name.c_str());
        return false;
    }

    const uint32_t input_offset = sizeof(aif_gemv_req);
    const uint32_t output_offset = input_offset + input_nbytes;
    const uint64_t full_total_u64 = static_cast<uint64_t>(output_offset) + output_nbytes;
    const bool skip_output_copy = full_total_u64 > max_ioctl_payload;
    const uint64_t total_u64 = skip_output_copy ? static_cast<uint64_t>(input_offset) + input_nbytes : full_total_u64;

    uint32_t total = 0;
    if (!common_aif_u64_to_u32(total_u64, total)) {
        LOG_ERR("%s: %s command buffer too large for tensor '%s'\n", __func__, label, tensor_name.c_str());
        return false;
    }

    std::unique_ptr<uint8_t, decltype(&std::free)> buf(nullptr, &std::free);
    if (!common_aif_alloc_page_aligned(total, buf)) {
        LOG_ERR("%s: failed to allocate %" PRIu32 " bytes for %s\n", __func__, total, label);
        return false;
    }

    auto * req = reinterpret_cast<aif_gemv_req *>(buf.get());
    req->magic = AIF_MAGIC;
    req->version = AIF_VERSION;
    req->flags = AIF_GEMV_F_DUMMY_OUTPUT | AIF_GEMV_F_CHECK_INPUT;
    if (skip_output_copy) {
        req->flags |= AIF_GEMV_F_SKIP_OUTPUT_COPY;
    }
    req->tensor_id = posted.tensor_id;
    req->lba_start = posted.lba_start;
    req->lba_count = posted.lba_count;
    req->input_dim = posted.cols;
    req->output_dim = posted.rows;
    req->input_offset = input_offset;
    req->input_nbytes = input_nbytes;
    req->output_offset = skip_output_copy ? 0 : output_offset;
    req->output_nbytes = output_nbytes;
    req->ggml_type = posted.ggml_type;

    bool input_from_tensor = false;
    if (captured_input != nullptr && captured_input->size() >= input_nbytes) {
        std::memcpy(buf.get() + input_offset, captured_input->data(), input_nbytes);
        input_from_tensor = true;
    } else if (input_tensor != nullptr && ggml_is_contiguous(input_tensor) && ggml_nbytes(input_tensor) >= input_nbytes) {
        const ggml_backend_buffer_t input_buffer = input_tensor->view_src ? input_tensor->view_src->buffer : input_tensor->buffer;
        if (input_buffer != nullptr && input_tensor->data != nullptr) {
            ggml_backend_tensor_get(input_tensor, buf.get() + input_offset, 0, input_nbytes);
            input_from_tensor = true;
        }
    }

    if (!input_from_tensor) {
        for (uint32_t i = 0; i < input_nbytes; ++i) {
            buf.get()[input_offset + i] = static_cast<uint8_t>(i & 0xff);
        }
    }

    const uint64_t input_checksum = common_aif_checksum_bytes(buf.get() + input_offset, input_nbytes);
    const int64_t start_us = ggml_time_us();

    if (result != nullptr) {
        result->input_nbytes = input_nbytes;
        result->output_nbytes = output_nbytes;
        result->input_checksum = input_checksum;

        const auto scope = scope_override != nullptr ? *scope_override : common_aif_decode_scope_get();
        result->call_index = call_index;
        result->phase = scope.phase;
        result->decode_call_index = scope.decode_call_index;
        result->batch_tokens = scope.batch_tokens;
        result->n_past_before = scope.n_past_before;
    }

    std::string error;
    if (!client.gemv(buf.get(), total, error)) {
        LOG_ERR("%s: AIF_OP_GEMV failed for tensor '%s': %s\n", __func__, tensor_name.c_str(), error.c_str());
        return false;
    }

    const int64_t elapsed_ns = (ggml_time_us() - start_us) * 1000;
    if (result != nullptr) {
        result->ioctl_elapsed_ns = elapsed_ns;
    }

    uint32_t output_mismatches = 0;
    if (!skip_output_copy) {
        for (uint32_t i = 0; i < output_nbytes; ++i) {
            const auto expected = static_cast<uint8_t>(i + posted.tensor_id + input_checksum);
            const auto actual = buf.get()[output_offset + i];

            if (actual != expected) {
                ++output_mismatches;
                if (output_mismatches <= 8) {
                    LOG_WRN("%s: %s output mismatch at byte %" PRIu32 ": got=0x%02x expected=0x%02x\n",
                            __func__, label, i, actual, expected);
                }
            }
        }
    }

    if (result != nullptr) {
        result->output_mismatches = output_mismatches;
        result->success = output_mismatches == 0;

        std::string stats_error;
        result->stats_valid = client.stats(result->stats, stats_error);
        if (!result->stats_valid) {
            LOG_WRN("%s: AIF_OP_STATS failed after %s for tensor '%s': %s\n",
                    __func__, label, tensor_name.c_str(), stats_error.c_str());
        }
    }

    if (log_result) {
        LOG_INF("%s: %s tensor '%s': input=%" PRIu32 " bytes, output=%" PRIu32 " bytes%s, checksum=%" PRIu64 ", mismatches=%" PRIu32 ", ioctl_elapsed_ns=%" PRId64 "\n",
                __func__, label, tensor_name.c_str(), input_nbytes, output_nbytes,
                skip_output_copy ? " (modeled, not copied)" : "",
                input_checksum, output_mismatches, elapsed_ns);
    }

    return output_mismatches == 0;
}

static uint64_t common_aif_bytes_to_ns(uint64_t bytes, uint64_t bytes_per_second) {
    if (bytes == 0 || bytes_per_second == 0) {
        return 0;
    }

    const uint64_t whole_seconds = bytes / bytes_per_second;
    const uint64_t remaining = bytes % bytes_per_second;
    return whole_seconds * 1000000000ull +
           (remaining * 1000000000ull + bytes_per_second - 1) / bytes_per_second;
}

static common_aif_device_prediction common_aif_predict_device(
        const common_aif_runtime & runtime,
        const aif_post_req & req) {
    const aif_stats_resp & cfg = runtime.device_config;
    const uint64_t channels = cfg.config_channels != 0 ? cfg.config_channels : 8;
    const uint64_t chips = cfg.config_chips != 0 ? cfg.config_chips : 16;
    const uint64_t planes = cfg.config_planes_per_chip != 0 ? cfg.config_planes_per_chip : 4;
    const uint64_t page_size = cfg.config_page_size != 0 ? cfg.config_page_size : COMMON_AIF_PAGE_SIZE;
    const uint64_t pcie_bps = cfg.config_pcie_bytes_per_sec != 0 ? cfg.config_pcie_bytes_per_sec : 8000000000ull;
    const uint64_t onfi_bps = cfg.config_onfi_bytes_per_sec != 0 ? cfg.config_onfi_bytes_per_sec : 2000000000ull;
    const uint64_t chip_bps = cfg.config_chip_bytes_per_sec != 0 ? cfg.config_chip_bytes_per_sec : 6400000000ull;
    const uint64_t cr_read_ns = cfg.config_cr_read_ns != 0 ? cfg.config_cr_read_ns : 9700;
    const uint64_t input_nbytes = static_cast<uint64_t>(req.cols) * sizeof(float);
    const uint64_t output_nbytes = static_cast<uint64_t>(req.rows) * sizeof(float);
    const uint64_t pages = (req.matrix_nbytes + page_size - 1) / page_size;
    const uint64_t max_chip_pages = (pages + chips - 1) / chips;
    const uint64_t max_plane_pages = (max_chip_pages + planes - 1) / planes;
    const uint64_t chip_matrix_bytes = (req.matrix_nbytes + chips - 1) / chips;
    const uint64_t chips_per_channel = (chips + channels - 1) / channels;
    const uint64_t input_pcie_ns = common_aif_bytes_to_ns(input_nbytes, pcie_bps);
    const uint64_t input_onfi_tail_ns = chips_per_channel * common_aif_bytes_to_ns(input_nbytes, onfi_bps);
    const uint64_t plane_read_ns = max_plane_pages * cr_read_ns;
    const uint64_t chip_bandwidth_ns = common_aif_bytes_to_ns(chip_matrix_bytes, chip_bps);
    const uint64_t chip_compute_ns = std::max(plane_read_ns, chip_bandwidth_ns);
    const uint64_t output_chip_nbytes = (output_nbytes + chips - 1) / chips;
    const uint64_t output_tail_ns = common_aif_bytes_to_ns(output_chip_nbytes, onfi_bps);
    const uint64_t output_pcie_ns = common_aif_bytes_to_ns(output_nbytes, pcie_bps);

    common_aif_device_prediction prediction;
    prediction.input_prefix_ns = input_pcie_ns + input_onfi_tail_ns;
    prediction.stream_ns = chip_compute_ns + output_tail_ns + output_pcie_ns;
    prediction.total_ns = prediction.input_prefix_ns + prediction.stream_ns;
    return prediction;
}

static void common_aif_capture_input(
        const aif_post_req & posted,
        const ggml_tensor * input_tensor,
        std::vector<uint8_t> & captured) {
    const uint64_t input_nbytes_u64 = static_cast<uint64_t>(posted.cols) * sizeof(float);
    uint32_t input_nbytes = 0;
    if (!common_aif_u64_to_u32(input_nbytes_u64, input_nbytes)) {
        captured.clear();
        return;
    }

    captured.resize(input_nbytes);
    bool copied = false;
    if (input_tensor != nullptr && ggml_is_contiguous(input_tensor) && ggml_nbytes(input_tensor) >= input_nbytes) {
        const ggml_backend_buffer_t input_buffer = input_tensor->view_src ? input_tensor->view_src->buffer : input_tensor->buffer;
        if (input_buffer != nullptr && input_tensor->data != nullptr) {
            ggml_backend_tensor_get(input_tensor, captured.data(), 0, input_nbytes);
            copied = true;
        }
    }

    if (!copied) {
        for (uint32_t i = 0; i < input_nbytes; ++i) {
            captured[i] = static_cast<uint8_t>(i & 0xff);
        }
    }
}

static uint64_t common_aif_offset_ns(const common_aif_runtime & runtime, int64_t time_us) {
    if (time_us <= 0 || static_cast<uint64_t>(time_us) < runtime.start_us) {
        return 0;
    }
    return (static_cast<uint64_t>(time_us) - runtime.start_us) * 1000;
}

static void common_aif_device_worker(common_aif_runtime * runtime) {
    for (;;) {
        std::shared_ptr<common_aif_async_task> task;
        {
            std::unique_lock<std::mutex> lock(runtime->device_queue_mutex);
            runtime->device_queue_cv.wait(lock, [runtime]() {
                return runtime->device_worker_stop || !runtime->device_queue.empty();
            });
            if (runtime->device_worker_stop && runtime->device_queue.empty()) {
                return;
            }
            task = runtime->device_queue.front();
            runtime->device_queue.pop_front();
        }

        {
            std::lock_guard<std::mutex> lock(task->start_mutex);
            task->started = true;
            task->start_us = ggml_time_us();
        }
        task->start_cv.notify_all();

        bool ok = false;
        try {
            ok = common_aif_smoke_gemv(
                    runtime->client,
                    task->posted.req,
                    task->posted.name,
                    "AIF parallel GEMV",
                    &task->result,
                    nullptr,
                    runtime->log_gemv,
                    &task->input,
                    &task->scope,
                    task->call_index);
        } catch (const std::exception & e) {
            LOG_ERR("%s: asynchronous AIF task failed for tensor '%s': %s\n",
                    __func__, task->posted.name.c_str(), e.what());
            ok = false;
        }

        task->end_us = ggml_time_us();
        task->result.async_task_id = task->id;
        task->result.execution_mode = task->execution_mode;
        task->result.layer = task->posted.identity.layer;
        task->result.submit_offset_ns = common_aif_offset_ns(*runtime, task->submit_us);
        task->result.device_start_offset_ns = common_aif_offset_ns(*runtime, task->start_us);
        task->result.device_end_offset_ns = common_aif_offset_ns(*runtime, task->end_us);
        task->result.predicted_device_ns = task->predicted_device_ns;
        task->result.qkv_group_id = task->qkv_group_id;
        task->result.qkv_schedule = task->qkv_group_id != 0 ? "head_major" : "";
        task->result.q_heads = task->q_heads;
        task->result.kv_heads = task->kv_heads;
        task->completion.set_value(ok);
    }
}

static void common_aif_start_device_worker(common_aif_runtime & runtime) {
    if (!runtime.device_worker.joinable()) {
        runtime.device_worker_stop = false;
        runtime.device_worker = std::thread(common_aif_device_worker, &runtime);
    }
}

static std::shared_ptr<common_aif_async_task> common_aif_submit_async(
        common_aif_runtime & runtime,
        const common_aif_posted_tensor & posted,
        const ggml_tensor * input_tensor,
        const std::string & execution_mode) {
    auto task = std::make_shared<common_aif_async_task>();
    task->id = ++runtime.async_task_next;
    task->call_index = static_cast<uint64_t>(++runtime.shadow_gemv_count);
    task->posted = posted;
    task->scope = common_aif_decode_scope_get();
    task->execution_mode = execution_mode;
    task->submit_us = ggml_time_us();
    task->prediction = common_aif_predict_device(runtime, posted.req);
    task->predicted_device_ns = task->prediction.total_ns;
    common_aif_capture_input(posted.req, input_tensor, task->input);

    task->done = task->completion.get_future().share();

    task->result.call_index = task->call_index;
    task->result.async_task_id = task->id;
    task->result.execution_mode = execution_mode;
    task->result.layer = posted.identity.layer;
    task->result.submit_offset_ns = common_aif_offset_ns(runtime, task->submit_us);
    task->result.predicted_device_ns = task->predicted_device_ns;

    common_aif_start_device_worker(runtime);
    {
        std::lock_guard<std::mutex> lock(runtime.device_queue_mutex);
        task->ticket = runtime.device_ticket_next++;
        runtime.device_queue.push_back(task);
    }
    runtime.device_queue_cv.notify_one();

    return task;
}

static bool common_aif_wait_task(
        common_aif_runtime & runtime,
        const std::shared_ptr<common_aif_async_task> & task,
        bool write_log) {
    const int64_t wait_start_us = ggml_time_us();
    const bool ok = task->done.get();
    const uint64_t wait_ns = static_cast<uint64_t>(std::max<int64_t>(0, ggml_time_us() - wait_start_us)) * 1000;
    task->result.dependency_wait_ns += wait_ns;
    task->result.head_first_ready_wait_ns = task->head_first_ready_wait_ns;
    task->result.parallel_elapsed_ns = static_cast<uint64_t>(std::max<int64_t>(0, task->end_us - task->submit_us)) * 1000;

    if (write_log) {
        common_aif_write_shadow_log_row(runtime, task->posted, task->result);
    }
    return ok;
}

static uint64_t common_aif_now_ns() {
    return static_cast<uint64_t>(std::max<int64_t>(0, ggml_time_us())) * 1000;
}

static uint64_t common_aif_partition_ns(uint64_t total_ns, uint32_t index, uint32_t count) {
    if (count == 0 || index >= count) {
        return 0;
    }

    const uint64_t base = total_ns / count;
    const uint64_t remainder = total_ns % count;
    return base + (index < remainder ? 1 : 0);
}

static std::shared_ptr<common_aif_async_task> common_aif_find_qkv_task(
        const common_aif_qkv_group & group,
        common_aif_tensor_stage stage) {
    for (const auto & task : group.tasks) {
        if (task->posted.identity.stage == stage) {
            return task;
        }
    }
    return nullptr;
}

static bool common_aif_add_qkv_task(
        common_aif_runtime & runtime,
        const std::shared_ptr<common_aif_async_task> & task) {
    const int32_t layer = task->posted.identity.layer;
    auto [it, inserted] = runtime.pending_qkv.try_emplace(layer);
    common_aif_qkv_group & group = it->second;
    if (inserted) {
        group.id = ++runtime.qkv_group_next;
        group.layer = layer;
    }

    if (group.timeline_ready || common_aif_find_qkv_task(group, task->posted.identity.stage) != nullptr) {
        LOG_ERR("%s: duplicate or late QKV task for layer %d, tensor '%s'\n",
                __func__, layer, task->posted.name.c_str());
        return false;
    }

    task->qkv_group_id = group.id;
    task->q_heads = runtime.n_heads;
    task->kv_heads = runtime.n_heads_kv;
    group.tasks.push_back(task);
    return true;
}

static bool common_aif_start_qkv_head_pipeline(
        common_aif_runtime & runtime,
        int32_t layer) {
    auto it = runtime.pending_qkv.find(layer);
    if (it == runtime.pending_qkv.end()) {
        return false;
    }

    common_aif_qkv_group & group = it->second;
    if (group.timeline_ready) {
        return true;
    }
    if (group.tasks.size() != 3) {
        LOG_ERR("%s: layer %d has %zu QKV tasks, expected 3\n", __func__, layer, group.tasks.size());
        return false;
    }

    const auto q_task = common_aif_find_qkv_task(group, common_aif_tensor_stage::attn_q);
    const auto k_task = common_aif_find_qkv_task(group, common_aif_tensor_stage::attn_k);
    const auto v_task = common_aif_find_qkv_task(group, common_aif_tensor_stage::attn_v);
    if (q_task == nullptr || k_task == nullptr || v_task == nullptr) {
        LOG_ERR("%s: layer %d QKV group is incomplete\n", __func__, layer);
        return false;
    }

    const uint32_t q_heads = std::max<uint32_t>(1, runtime.n_heads);
    const uint32_t kv_heads = std::max<uint32_t>(1, std::min(runtime.n_heads_kv, q_heads));
    const uint64_t input_prefix_ns = q_task->prediction.input_prefix_ns +
            k_task->prediction.input_prefix_ns + v_task->prediction.input_prefix_ns;

    group.head_ready_ns.clear();
    group.head_ready_ns.reserve(q_heads);
    uint64_t cursor_ns = input_prefix_ns;
    uint32_t previous_kv_head = std::numeric_limits<uint32_t>::max();
    // Figure 15(c) exposes Q/K/V as head-sized sets. For GQA, emit K/V once when the
    // first query head that shares that KV head is reached.
    for (uint32_t q_head = 0; q_head < q_heads; ++q_head) {
        cursor_ns += common_aif_partition_ns(q_task->prediction.stream_ns, q_head, q_heads);

        const uint32_t kv_head = static_cast<uint32_t>(
                (static_cast<uint64_t>(q_head) * kv_heads) / q_heads);
        if (kv_head != previous_kv_head) {
            cursor_ns += common_aif_partition_ns(k_task->prediction.stream_ns, kv_head, kv_heads);
            cursor_ns += common_aif_partition_ns(v_task->prediction.stream_ns, kv_head, kv_heads);
            previous_kv_head = kv_head;
        }
        group.head_ready_ns.push_back(cursor_ns);
    }

    group.modeled_ns = q_task->prediction.total_ns +
            k_task->prediction.total_ns + v_task->prediction.total_ns;
    if (cursor_ns != group.modeled_ns) {
        LOG_ERR("%s: layer %d QKV timeline mismatch: heads=%" PRIu64 ", commands=%" PRIu64 "\n",
                __func__, layer, cursor_ns, group.modeled_ns);
        return false;
    }

    const int64_t wait_start_us = ggml_time_us();
    {
        std::unique_lock<std::mutex> lock(q_task->start_mutex);
        q_task->start_cv.wait(lock, [&q_task]() { return q_task->started; });
        group.modeled_start_ns = static_cast<uint64_t>(std::max<int64_t>(0, q_task->start_us)) * 1000;
    }

    group.first_head_ready_ns = group.head_ready_ns.front();
    group.last_head_ready_ns = group.head_ready_ns.back();
    const uint64_t first_ready_at_ns = group.modeled_start_ns + group.first_head_ready_ns;
    const uint64_t now_ns = common_aif_now_ns();
    if (now_ns < first_ready_at_ns) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(first_ready_at_ns - now_ns));
    }

    group.host_start_ns = common_aif_now_ns();
    group.timeline_ready = true;
    v_task->head_first_ready_wait_ns =
            static_cast<uint64_t>(std::max<int64_t>(0, ggml_time_us() - wait_start_us)) * 1000;
    return true;
}

static bool common_aif_drain_qkv_layer(common_aif_runtime & runtime, int32_t layer) {
    auto it = runtime.pending_qkv.find(layer);
    if (it == runtime.pending_qkv.end()) {
        return true;
    }

    common_aif_qkv_group & group = it->second;
    if (!group.timeline_ready && !common_aif_start_qkv_head_pipeline(runtime, layer)) {
        return false;
    }

    const uint64_t barrier_arrival_ns = common_aif_now_ns();
    const uint64_t host_attention_actual_ns = barrier_arrival_ns > group.host_start_ns ?
            barrier_arrival_ns - group.host_start_ns : 0;

    uint64_t host_finish_ns = 0;
    uint64_t host_idle_wait_ns = 0;
    const uint32_t q_heads = static_cast<uint32_t>(group.head_ready_ns.size());
    for (uint32_t head = 0; head < q_heads; ++head) {
        const uint64_t host_start_ns = std::max(group.head_ready_ns[head], host_finish_ns);
        host_idle_wait_ns += host_start_ns - host_finish_ns;
        host_finish_ns = host_start_ns +
                common_aif_partition_ns(host_attention_actual_ns, head, q_heads);
    }

    const uint64_t aif_idle_wait_ns = host_finish_ns > group.last_head_ready_ns ?
            host_finish_ns - group.last_head_ready_ns : 0;
    const uint64_t target_finish_ns = group.modeled_start_ns + host_finish_ns;

    bool ok = true;
    uint64_t group_end_ns = 0;
    for (const auto & task : group.tasks) {
        ok = common_aif_wait_task(runtime, task, false) && ok;
        group_end_ns = std::max(group_end_ns,
                static_cast<uint64_t>(std::max<int64_t>(0, task->end_us)) * 1000);
    }

    const uint64_t tail_wait_start_ns = common_aif_now_ns();
    if (tail_wait_start_ns < target_finish_ns) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(target_finish_ns - tail_wait_start_ns));
    }
    const uint64_t final_ns = common_aif_now_ns();
    const uint64_t tail_wait_ns = final_ns > tail_wait_start_ns ? final_ns - tail_wait_start_ns : 0;

    const uint64_t overlap_start_ns = group.host_start_ns;
    const uint64_t overlap_end_ns = std::min(group_end_ns, barrier_arrival_ns);
    const uint64_t overlap_ns = overlap_end_ns > overlap_start_ns ?
            overlap_end_ns - overlap_start_ns : 0;

    const auto summary_task = common_aif_find_qkv_task(group, common_aif_tensor_stage::attn_v);
    if (summary_task != nullptr) {
        summary_task->result.qkv_group_summary = true;
        summary_task->result.qkv_group_modeled_ns = group.modeled_ns;
        summary_task->result.qkv_first_head_ready_ns = group.first_head_ready_ns;
        summary_task->result.qkv_last_head_ready_ns = group.last_head_ready_ns;
        summary_task->result.host_attention_actual_ns = host_attention_actual_ns;
        summary_task->result.host_idle_wait_ns = host_idle_wait_ns;
        summary_task->result.aif_idle_wait_ns = aif_idle_wait_ns;
        summary_task->result.head_pipeline_finish_ns = host_finish_ns;
        summary_task->result.head_pipeline_tail_wait_ns = tail_wait_ns;
        summary_task->result.overlap_ns = overlap_ns;
        summary_task->result.parallel_elapsed_ns = final_ns > group.modeled_start_ns ?
                final_ns - group.modeled_start_ns : 0;
    }

    for (const auto & task : group.tasks) {
        common_aif_write_shadow_log_row(runtime, task->posted, task->result);
    }
    runtime.pending_qkv.erase(it);
    return ok;
}

static bool common_aif_drain_all_qkv(common_aif_runtime & runtime) {
    bool ok = true;
    while (!runtime.pending_qkv.empty()) {
        const int32_t layer = runtime.pending_qkv.begin()->first;
        if (!common_aif_drain_qkv_layer(runtime, layer)) {
            runtime.pending_qkv.erase(layer);
            ok = false;
        }
    }
    return ok;
}

common_aif_runtime::~common_aif_runtime() {
    common_aif_drain_all_qkv(*this);
    {
        std::lock_guard<std::mutex> lock(device_queue_mutex);
        device_worker_stop = true;
    }
    device_queue_cv.notify_all();
    if (device_worker.joinable()) {
        device_worker.join();
    }
    if (shadow_log.is_open()) {
        shadow_log.flush();
    }
}

static const common_aif_posted_tensor * common_aif_find_posted_tensor(
        const common_aif_runtime * runtime,
        const char * name) {
    if (runtime == nullptr || name == nullptr) {
        return nullptr;
    }

    for (const auto & posted : runtime->posted) {
        if (posted.name == name) {
            return &posted;
        }
    }

    return nullptr;
}

static bool common_aif_is_gemv_node(const ggml_tensor * t) {
    if (t == nullptr || t->op != GGML_OP_MUL_MAT || t->src[0] == nullptr || t->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * src1 = t->src[1];

    return src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
}

static bool common_aif_shadow_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * runtime = static_cast<common_aif_runtime *>(user_data);

    if (runtime == nullptr || !common_aif_is_gemv_node(t)) {
        return ask ? false : true;
    }
    if (!common_aif_graph_phase_is_allowed(runtime)) {
        return ask ? false : true;
    }
    if (runtime->shadow_gemv_max > 0 && runtime->shadow_gemv_count >= runtime->shadow_gemv_max) {
        return ask ? false : true;
    }

    const auto * posted = common_aif_find_posted_tensor(runtime, ggml_get_name(t->src[0]));
    if (posted == nullptr) {
        return ask ? false : true;
    }

    if (ask) {
        return true;
    }

    const uint64_t call_index = static_cast<uint64_t>(++runtime->shadow_gemv_count);
    common_aif_gemv_result result;
    const bool ok = common_aif_smoke_gemv(
            runtime->client, posted->req, posted->name, "AIF shadow GEMV", &result,
            nullptr, runtime->log_gemv, nullptr, nullptr, call_index);
    result.execution_mode = "shadow";
    result.layer = posted->identity.layer;
    common_aif_write_shadow_log_row(*runtime, *posted, result);
    return ok;
}

static bool common_aif_replace_output_matches(const ggml_tensor * t, const common_aif_posted_tensor & posted) {
    if (t == nullptr || t->type != GGML_TYPE_F32) {
        return false;
    }

    return ggml_nbytes(t) == static_cast<size_t>(posted.full_rows) * sizeof(float);
}

static bool common_aif_zero_tensor(ggml_tensor * t) {
    if (t == nullptr || (t->view_src ? t->view_src->buffer : t->buffer) == nullptr || t->data == nullptr) {
        return false;
    }

    std::vector<uint8_t> zeros(ggml_nbytes(t), 0);
    ggml_backend_tensor_set(t, zeros.data(), 0, zeros.size());
    return true;
}

static ggml_backend_buffer_t common_aif_tensor_buffer(const ggml_tensor * t) {
    while (t != nullptr && t->buffer == nullptr) {
        t = t->view_src;
    }

    return t != nullptr ? t->buffer : nullptr;
}

static bool common_aif_alias_preallocated_tensor(ggml_tensor * alias, const ggml_tensor * source) {
    if (alias == nullptr || source == nullptr || source->data == nullptr) {
        return false;
    }

    ggml_backend_buffer_t buffer = common_aif_tensor_buffer(source);
    if (buffer == nullptr) {
        return false;
    }

    alias->buffer = buffer;
    alias->data = source->data;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        alias->nb[i] = source->nb[i];
    }
    return true;
}

static bool common_aif_compute_host_submatrix(
        ggml_backend_t backend,
        ggml_tensor * t,
        uint32_t host_rows,
        uint64_t & elapsed_ns) {
    elapsed_ns = 0;
    if (host_rows == 0) {
        return true;
    }
    if (backend == nullptr || t == nullptr || t->src[0] == nullptr || t->src[1] == nullptr ||
        host_rows > static_cast<uint64_t>(t->src[0]->ne[1])) {
        return false;
    }
    if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        LOG_ERR("%s: AiF host submatrix execution requires a CPU backend, got '%s'\n",
                __func__, ggml_backend_name(backend));
        return false;
    }

    constexpr size_t graph_size = 8;
    ggml_init_params init_params {
        /*.mem_size   =*/ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(init_params);
    if (ctx == nullptr) {
        return false;
    }

    // The source tensors belong to the decode graph and still carry their parent links. Use leaf aliases
    // so this small graph computes only the host-resident rows instead of recursively importing that graph.
    ggml_tensor * weight_alias = ggml_new_tensor_4d(
            ctx,
            t->src[0]->type,
            t->src[0]->ne[0],
            host_rows,
            t->src[0]->ne[2],
            t->src[0]->ne[3]);
    ggml_tensor * input_alias = ggml_dup_tensor(ctx, t->src[1]);
    if (!common_aif_alias_preallocated_tensor(weight_alias, t->src[0]) ||
        !common_aif_alias_preallocated_tensor(input_alias, t->src[1])) {
        LOG_ERR("%s: failed to alias host submatrix inputs\n", __func__);
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * host_output = ggml_mul_mat(ctx, weight_alias, input_alias);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(graph, host_output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        LOG_ERR("%s: failed to allocate host submatrix output on backend '%s'\n",
                __func__, ggml_backend_name(backend));
        ggml_free(ctx);
        return false;
    }

    const int64_t start_us = ggml_time_us();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    elapsed_ns = static_cast<uint64_t>(std::max<int64_t>(0, ggml_time_us() - start_us)) * 1000;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("%s: host submatrix GEMV failed on backend '%s' with status %d\n",
                __func__, ggml_backend_name(backend), static_cast<int>(status));
        return false;
    }
    return true;
}

static bool common_aif_replace_cb_eval(ggml_backend_t backend, ggml_tensor * t, bool ask, void * user_data) {
    (void) backend;
    auto * runtime = static_cast<common_aif_runtime *>(user_data);

    if (runtime == nullptr || !common_aif_is_gemv_node(t)) {
        return ask ? false : true;
    }
    if (!common_aif_graph_phase_is_allowed(runtime)) {
        return ask ? false : true;
    }
    if (runtime->shadow_gemv_max > 0 && runtime->shadow_gemv_count >= runtime->shadow_gemv_max) {
        return ask ? false : true;
    }

    const auto * posted = common_aif_find_posted_tensor(runtime, ggml_get_name(t->src[0]));
    if (posted == nullptr || !common_aif_replace_output_matches(t, *posted)) {
        return ask ? false : true;
    }

    if (ask) {
        return true;
    }

    const uint64_t call_index = static_cast<uint64_t>(++runtime->shadow_gemv_count);
    common_aif_gemv_result result;
    const bool ok = common_aif_smoke_gemv(
            runtime->client, posted->req, posted->name, "AIF replacement GEMV", &result,
            t->src[1], runtime->log_gemv, nullptr, nullptr, call_index);
    result.execution_mode = "sequential";
    result.layer = posted->identity.layer;
    common_aif_write_shadow_log_row(*runtime, *posted, result);
    if (!ok) {
        return false;
    }

    if (!common_aif_zero_tensor(t)) {
        LOG_ERR("%s: failed to write dummy output for tensor '%s'\n", __func__, ggml_get_name(t));
        return false;
    }

    return true;
}

static bool common_aif_parallel_cb_eval(ggml_backend_t backend, ggml_tensor * t, bool ask, void * user_data) {
    auto * runtime = static_cast<common_aif_runtime *>(user_data);

    if (runtime == nullptr || !common_aif_is_gemv_node(t)) {
        return ask ? false : true;
    }
    if (!common_aif_graph_phase_is_allowed(runtime)) {
        return ask ? false : true;
    }

    const auto * posted = common_aif_find_posted_tensor(runtime, ggml_get_name(t->src[0]));
    if (posted == nullptr || !common_aif_replace_output_matches(t, *posted)) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }

    if (posted->identity.stage == common_aif_tensor_stage::attn_output) {
        if (!common_aif_drain_qkv_layer(*runtime, posted->identity.layer)) {
            return false;
        }
    } else if (posted->identity.stage == common_aif_tensor_stage::output) {
        if (!common_aif_drain_all_qkv(*runtime)) {
            return false;
        }
    }

    if (common_aif_is_qkv_stage(posted->identity.stage)) {
        auto task = common_aif_submit_async(*runtime, *posted, t->src[1], "head_async");
        if (!common_aif_add_qkv_task(*runtime, task)) {
            return false;
        }

        if (!common_aif_zero_tensor(t)) {
            LOG_ERR("%s: failed to write QKV dummy output for tensor '%s'\n", __func__, ggml_get_name(t));
            return false;
        }

        const auto group_it = runtime->pending_qkv.find(posted->identity.layer);
        if (group_it != runtime->pending_qkv.end() && group_it->second.tasks.size() == 3 &&
            !common_aif_start_qkv_head_pipeline(*runtime, posted->identity.layer)) {
            return false;
        }
        return true;
    }

    auto task = common_aif_submit_async(
            *runtime,
            *posted,
            t->src[1],
            common_aif_is_ffn_stage(posted->identity.stage) ? "tensor_parallel" : "parallel_sync");

    const int64_t parallel_start_us = ggml_time_us();
    uint64_t host_ns = 0;
    uint64_t host_actual_ns = 0;
    bool host_ok = true;
    if (common_aif_is_ffn_stage(posted->identity.stage)) {
        host_ns = common_aif_bytes_to_ns(posted->host_matrix_nbytes, runtime->host_bandwidth_bytes_s);
        host_ok = common_aif_compute_host_submatrix(backend, t, posted->host_rows, host_actual_ns);
    }

    const bool ok = common_aif_wait_task(*runtime, task, false);
    task->result.host_matrix_nbytes = posted->host_matrix_nbytes;
    task->result.host_modeled_ns = host_ns;
    task->result.host_actual_ns = host_actual_ns;
    task->result.parallel_elapsed_ns =
            static_cast<uint64_t>(std::max<int64_t>(0, ggml_time_us() - parallel_start_us)) * 1000;
    task->result.overlap_ns = std::min<uint64_t>(host_actual_ns, static_cast<uint64_t>(std::max<int64_t>(0, task->result.ioctl_elapsed_ns)));
    common_aif_write_shadow_log_row(*runtime, *posted, task->result);
    if (!ok || !host_ok) {
        return false;
    }

    if (!common_aif_zero_tensor(t)) {
        LOG_ERR("%s: failed to write parallel dummy output for tensor '%s'\n", __func__, ggml_get_name(t));
        return false;
    }
    return true;
}

static bool common_aif_post_model_tensors(const common_params & params, const llama_model * model, common_aif_runtime * runtime) {
    if (params.aif.device.empty()) {
        return true;
    }

    LOG_INF("%s: AIF tensor posting enabled: dev=%s, post_max=%d, lba_base=%" PRIu64 "\n",
            __func__, params.aif.device.c_str(), params.aif.post_max, params.aif.lba_base);

    std::regex tensor_filter;
    try {
        tensor_filter = std::regex(params.aif.tensor_filter);
    } catch (const std::regex_error & e) {
        LOG_ERR("%s: invalid --aif-tensor-filter regex: %s\n", __func__, e.what());
        return false;
    }

    const auto & tensors = llama_internal_get_tensor_map(model);
    struct post_candidate {
        std::string name;
        ggml_tensor * tensor = nullptr;
        common_aif_tensor_identity identity;
    };

    std::vector<post_candidate> candidates;
    std::unordered_set<const void *> counted_storage;
    uint64_t total_model_bytes = 0;
    uint64_t ffn_bytes = 0;

    for (const auto & [name, tensor] : tensors) {
        if (tensor == nullptr) {
            continue;
        }

        const void * storage_key = tensor->data != nullptr ? tensor->data : static_cast<const void *>(tensor);
        if (counted_storage.insert(storage_key).second) {
            total_model_bytes += ggml_nbytes(tensor);
        }

        if (params.aif.post_max > 0 && static_cast<int32_t>(candidates.size()) >= params.aif.post_max) {
            continue;
        }
        if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            continue;
        }
        if (!std::regex_match(name, tensor_filter)) {
            continue;
        }

        const common_aif_tensor_identity identity = common_aif_parse_tensor_identity(name);
        candidates.push_back({ name, tensor, identity });
        if (common_aif_is_ffn_stage(identity.stage)) {
            ffn_bytes += ggml_nbytes(tensor);
        }
    }

    if (candidates.empty()) {
        LOG_ERR("%s: no model tensors matched --aif-tensor-filter: %s\n", __func__, params.aif.tensor_filter.c_str());
        return false;
    }

    common_aif_client local_client;
    common_aif_client * client = runtime != nullptr ? &runtime->client : &local_client;
    std::string error;
    if (!client->is_open() && !client->open(params.aif.device, error)) {
        LOG_ERR("%s: failed to open AIF device '%s': %s\n", __func__, params.aif.device.c_str(), error.c_str());
        return false;
    }

    aif_stats_resp device_config {};
    if (!client->stats(device_config, error)) {
        LOG_ERR("%s: failed to query AIF device '%s': %s\n", __func__, params.aif.device.c_str(), error.c_str());
        return false;
    }
    if (!common_aif_validate_device_profile(device_config, error)) {
        LOG_ERR("%s: %s\n", __func__, error.c_str());
        return false;
    }
    if (runtime != nullptr) {
        runtime->device_config = device_config;
    }
    const uint64_t lbas_per_page = device_config.config_page_size / COMMON_AIF_SECTOR_SIZE;

    std::unordered_set<std::string> candidate_names;
    for (const auto & candidate : candidates) {
        candidate_names.insert(candidate.name);
    }
    std::unordered_set<const void *> base_host_storage;
    uint64_t base_host_model_bytes = 0;
    for (const auto & [name, tensor] : tensors) {
        if (tensor == nullptr || candidate_names.find(name) != candidate_names.end()) {
            continue;
        }
        const void * storage_key = tensor->data != nullptr ? tensor->data : static_cast<const void *>(tensor);
        if (base_host_storage.insert(storage_key).second) {
            base_host_model_bytes += ggml_nbytes(tensor);
        }
    }

    double host_fraction = 0.0;
    if (params.aif.parallel && runtime != nullptr && ffn_bytes != 0) {
        const long double aif_bandwidth_bytes_s =
                static_cast<long double>(device_config.config_chips) *
                static_cast<long double>(device_config.config_chip_bytes_per_sec);
        const uint64_t host_budget_bytes = params.aif.host_budget_mib * 1024ull * 1024ull;
        const uint64_t kv_cache_bytes = params.aif.kv_cache_mib * 1024ull * 1024ull;
        const uint64_t required_base_bytes = base_host_model_bytes + kv_cache_bytes;
        const uint64_t available_ffn_bytes = host_budget_bytes > required_base_bytes ?
                host_budget_bytes - required_base_bytes : 0;
        const long double host_bandwidth_bytes_s = params.aif.host_bandwidth_gbps * 1000000000.0L;
        const long double ideal_fraction = host_bandwidth_bytes_s /
                (host_bandwidth_bytes_s + aif_bandwidth_bytes_s);
        const long double capacity_fraction = std::min<long double>(
                1.0L, static_cast<long double>(available_ffn_bytes) / static_cast<long double>(ffn_bytes));
        host_fraction = static_cast<double>(std::min(ideal_fraction, capacity_fraction));

        runtime->host_budget_bytes = host_budget_bytes;
        runtime->kv_cache_bytes = kv_cache_bytes;
        runtime->base_host_model_bytes = base_host_model_bytes;
        runtime->total_ffn_bytes = ffn_bytes;
        runtime->host_fraction = host_fraction;
        runtime->host_bandwidth_bytes_s = static_cast<uint64_t>(host_bandwidth_bytes_s);

        LOG_INF("%s: AIF parallel placement: model=%.2f GiB, base_host=%.2f GiB, KV=%.2f GiB, FFN=%.2f GiB, host_fraction=%.4f, host_bw=%.1f GB/s\n",
                __func__,
                static_cast<double>(total_model_bytes) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(base_host_model_bytes) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(kv_cache_bytes) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(ffn_bytes) / (1024.0 * 1024.0 * 1024.0),
                host_fraction,
                params.aif.host_bandwidth_gbps);
    }

    uint64_t next_lba = common_aif_align_up(params.aif.lba_base, lbas_per_page);
    uint64_t posted_bytes = 0;
    uint64_t retained_host_bytes = 0;
    int32_t posted_count = 0;
    int32_t skipped_count = 0;
    bool have_first_posted = false;
    aif_post_req first_posted_req {};
    std::string first_posted_name;

    for (const auto & candidate : candidates) {
        const std::string & name = candidate.name;
        ggml_tensor * tensor = candidate.tensor;
        uint32_t full_rows = 0;
        uint32_t host_rows = 0;
        if (!common_aif_u64_to_u32(static_cast<uint64_t>(tensor->ne[1]), full_rows)) {
            LOG_WRN("%s: skipping AIF tensor '%s': row count overflow\n", __func__, name.c_str());
            ++skipped_count;
            continue;
        }
        if (params.aif.parallel && common_aif_is_ffn_stage(candidate.identity.stage)) {
            host_rows = static_cast<uint32_t>(std::floor(static_cast<double>(full_rows) * host_fraction));
            if (host_rows >= full_rows) {
                host_rows = full_rows - 1;
            }
        }
        const uint32_t ssd_rows = full_rows - host_rows;

        aif_post_req req;
        if (!common_aif_fill_post_req(tensor, next_lba, ssd_rows, req, error)) {
            LOG_WRN("%s: skipping AIF tensor '%s': %s\n", __func__, name.c_str(), error.c_str());
            ++skipped_count;
            continue;
        }

        const uint64_t full_matrix_nbytes = ggml_nbytes(tensor);
        const uint64_t host_matrix_nbytes = host_rows == 0 ? 0 :
                static_cast<uint64_t>(tensor->nb[1]) * host_rows;
        common_aif_posted_tensor posted {
            name,
            req,
            candidate.identity,
            full_rows,
            full_matrix_nbytes,
            host_rows,
            host_matrix_nbytes,
        };

        if (!client->post(req, error)) {
            LOG_ERR("%s: AIF_OP_POST failed for tensor '%s': %s\n", __func__, name.c_str(), error.c_str());
            return false;
        }

        posted_bytes += req.matrix_nbytes;
        retained_host_bytes += host_matrix_nbytes;
        ++posted_count;
        if (!have_first_posted) {
            have_first_posted = true;
            first_posted_req = req;
            first_posted_name = name;
        }
        if (runtime != nullptr) {
            runtime->posted.push_back(posted);
            runtime->host_ffn_bytes += host_matrix_nbytes;
        }
        LOG_INF("%s: posted tensor '%s': id=%" PRIu64 ", full_rows=%" PRIu32 ", host_rows=%" PRIu32 ", aif_rows=%" PRIu32 ", aif_bytes=%" PRIu64 ", lba=%" PRIu64 "+%" PRIu64 "\n",
                __func__, name.c_str(), req.tensor_id, full_rows, host_rows, req.rows,
                req.matrix_nbytes, req.lba_start, req.lba_count);

        next_lba = common_aif_align_up(req.lba_start + req.lba_count, lbas_per_page);
    }

    if (posted_count == 0) {
        LOG_ERR("%s: no matching model tensor could be posted\n", __func__);
        return false;
    }

    if (params.aif.smoke_gemv) {
        if (!common_aif_smoke_gemv(*client, first_posted_req, first_posted_name, "AIF smoke GEMV")) {
            return false;
        }
    }

    aif_stats_resp stats;
    if (client->stats(stats, error)) {
        if (runtime != nullptr) {
            runtime->device_config = stats;
        }
        LOG_INF("%s: posted %d tensor(s), AIF=%.2f MiB, host_FFN=%.2f MiB; device post_count=%" PRIu64 ", tensor_count=%" PRIu64 ", last_mapped_pages=%" PRIu64 ", last_units_used=%" PRIu64 "\n",
                __func__,
                posted_count,
                static_cast<double>(posted_bytes) / (1024.0 * 1024.0),
                static_cast<double>(retained_host_bytes) / (1024.0 * 1024.0),
                stats.post_count,
                stats.tensor_count,
                stats.last_mapped_pages,
                stats.last_units_used);
    } else {
        LOG_WRN("%s: AIF_OP_STATS failed after posting tensors: %s\n", __func__, error.c_str());
        LOG_INF("%s: posted %d tensor(s), %.2f MiB total\n",
                __func__, posted_count, static_cast<double>(posted_bytes) / (1024.0 * 1024.0));
    }

    if (skipped_count > 0) {
        LOG_WRN("%s: skipped %d matching tensor(s) because their metadata was not postable\n", __func__, skipped_count);
    }

    return true;
}

struct common_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top

    llama_model_ptr   model;
    std::unique_ptr<common_aif_runtime> aif_runtime;
    llama_context_ptr context;

    std::vector<llama_adapter_lora_ptr> lora;

    std::vector<common_sampler_ptr> samplers;
    std::vector<llama_sampler_seq_config> samplers_seq_config;
};

common_init_result::common_init_result(common_params & params, bool model_only) :
    pimpl(new impl{}) {
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (params.fit_params) {
        LOG_INF("%s: fitting params to device memory ...\n", __func__);
        LOG_INF("%s: (for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on)\n", __func__);
        common_fit_params(params.model.path.c_str(), &mparams, &cparams,
            params.tensor_split,
            params.tensor_buft_overrides.data(),
            params.fit_params_target.data(),
            params.fit_params_min_ctx,
            params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
    }

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == NULL) {
        return;
    }

    pimpl->model.reset(model);

    std::unique_ptr<common_aif_runtime> aif_runtime;
    if (params.aif.shadow_gemv && params.aif.replace_gemv) {
        LOG_ERR("%s: --aif-shadow-gemv and --aif-replace-gemv cannot be combined\n", __func__);
        pimpl->model.reset();
        return;
    }

    if (params.aif.shadow_gemv || params.aif.replace_gemv) {
        if (params.aif.shadow_gemv && params.cb_eval != nullptr) {
            LOG_ERR("%s: --aif-shadow-gemv cannot be combined with another eval callback\n", __func__);
            pimpl->model.reset();
            return;
        }
        if (params.aif.replace_gemv && params.cb_node_override != nullptr) {
            LOG_ERR("%s: --aif-replace-gemv cannot be combined with another node override callback\n", __func__);
            pimpl->model.reset();
            return;
        }

        aif_runtime = std::make_unique<common_aif_runtime>();
        aif_runtime->replace_gemv = params.aif.replace_gemv;
        aif_runtime->parallel = params.aif.parallel;
        aif_runtime->shadow_gemv_max = params.aif.parallel ? 0 :
                (params.aif.replace_gemv ? params.aif.replace_gemv_max : params.aif.shadow_gemv_max);
        aif_runtime->graph_decode_only = params.aif.graph_decode_only;
        aif_runtime->log_gemv = params.aif.log_gemv;
        aif_runtime->start_us = static_cast<uint64_t>(ggml_time_us());
        aif_runtime->n_heads = static_cast<uint32_t>(std::max<int32_t>(1, llama_model_n_head(model)));
        aif_runtime->n_heads_kv = static_cast<uint32_t>(std::max<int32_t>(1, llama_model_n_head_kv(model)));
    }

    if (!common_aif_post_model_tensors(params, model, aif_runtime.get())) {
        pimpl->model.reset();
        return;
    }

    if (aif_runtime) {
        std::string error;
        if (!aif_runtime->client.is_open() && !aif_runtime->client.open(params.aif.device, error)) {
            LOG_ERR("%s: failed to open AIF device '%s' for graph GEMV: %s\n", __func__, params.aif.device.c_str(), error.c_str());
            pimpl->model.reset();
            return;
        }
        if (!params.aif.shadow_gemv_log.empty() && !common_aif_open_shadow_log(*aif_runtime, params.aif.shadow_gemv_log)) {
            pimpl->model.reset();
            return;
        }

        const char * aif_mode = aif_runtime->parallel ? "parallel replacement" :
                (aif_runtime->replace_gemv ? "replacement" : "shadow");
        LOG_INF("%s: AIF %s GEMV enabled: posted_tensors=%zu, max_calls=%d, phases=%s, per_call_log=%s, qkv_schedule=%s\n",
                __func__, aif_mode,
                aif_runtime->posted.size(), aif_runtime->shadow_gemv_max,
                aif_runtime->graph_decode_only ? "decode-only" : "all",
                aif_runtime->log_gemv ? "on" : "off",
                aif_runtime->parallel ? "head_major" : "none");
        if (!params.aif.shadow_gemv_log.empty()) {
            LOG_INF("%s: AIF graph GEMV CSV log: %s\n", __func__, params.aif.shadow_gemv_log.c_str());
        }
        if (aif_runtime->replace_gemv) {
            params.cb_node_override = aif_runtime->parallel ? common_aif_parallel_cb_eval : common_aif_replace_cb_eval;
            params.cb_node_override_user_data = aif_runtime.get();
        } else {
            params.cb_eval = common_aif_shadow_cb_eval;
            params.cb_eval_user_data = aif_runtime.get();
        }
        pimpl->aif_runtime = std::move(aif_runtime);
        cparams.cb_eval = params.cb_eval;
        cparams.cb_eval_user_data = params.cb_eval_user_data;
        cparams.cb_node_override = params.cb_node_override;
        cparams.cb_node_override_user_data = params.cb_node_override_user_data;
    }

    if (model_only) {
        return;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // load and optionally apply lora adapters
    for (auto & la : params.lora_adapters) {
        llama_adapter_lora_ptr lora;
        lora.reset(llama_adapter_lora_init(model, la.path.c_str()));
        if (lora == nullptr) {
            LOG_ERR("%s: failed to load lora adapter '%s'\n", __func__, la.path.c_str());
            pimpl->model.reset(model);
            return;
        }

        char buf[1024];
        la.ptr = lora.get();
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.task_name", buf, sizeof(buf));
        la.task_name = buf;
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.prompt_prefix", buf, sizeof(buf));
        la.prompt_prefix = buf;
        pimpl->lora.emplace_back(std::move(lora)); // copy to list of loaded adapters
    }

    // updates params.sampling
    // TODO: fix naming
    common_init_sampler_from_model(model, params.sampling);

    if (params.sampling.ignore_eos && llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
        LOG_WRN("%s: warning: vocab does not have an EOS token, ignoring --ignore-eos\n", __func__);
        params.sampling.ignore_eos = false;
    }

    // initialize once
    for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
        if (llama_vocab_is_eog(vocab, i)) {
            LOG_TRC("%s: added %s logit bias = %f\n", __func__, common_token_to_piece(vocab, i).c_str(), -INFINITY);
            params.sampling.logit_bias_eog.push_back({i, -INFINITY});
        }
    }

    if (params.sampling.ignore_eos) {
        // add EOG biases to the active set of logit biases
        params.sampling.logit_bias.insert(
                params.sampling.logit_bias.end(),
                params.sampling.logit_bias_eog.begin(), params.sampling.logit_bias_eog.end());
    }

    //if (params.sampling.penalty_last_n == -1) {
    //    LOG_TRC("%s: setting penalty_last_n to ctx_size = %d\n", __func__, llama_n_ctx(lctx));
    //    params.sampling.penalty_last_n = llama_n_ctx(lctx);
    //}

    //if (params.sampling.dry_penalty_last_n == -1) {
    //    LOG_TRC("%s: setting dry_penalty_last_n to ctx_size = %d\n", __func__, llama_n_ctx(lctx));
    //    params.sampling.dry_penalty_last_n = llama_n_ctx(lctx);
    //}

    // init the backend samplers as part of the context creation
    pimpl->samplers.resize(cparams.n_seq_max);
    pimpl->samplers_seq_config.resize(cparams.n_seq_max);

    for (int i = 0; i < (int) cparams.n_seq_max; ++i) {
        pimpl->samplers[i].reset(common_sampler_init(model, params.sampling));
        pimpl->samplers_seq_config[i] = { i, common_sampler_get(pimpl->samplers[i].get()) };
    }

    if (params.sampling.backend_sampling) {
        cparams.samplers   = pimpl->samplers_seq_config.data();
        cparams.n_samplers = pimpl->samplers_seq_config.size();
    }

    llama_context * lctx = llama_init_from_model(model, cparams);
    if (lctx == NULL) {
        LOG_ERR("%s: failed to create context with model '%s'\n", __func__, params.model.path.c_str());
        return;
    }

    pimpl->context.reset(lctx);
}

llama_model * common_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_init_result::context() {
    return pimpl->context.get();
}

common_sampler * common_init_result::sampler(llama_seq_id seq_id) {
    if (seq_id < 0 || seq_id >= (int) pimpl->samplers.size()) {
        return nullptr;
    }
    return pimpl->samplers[seq_id].get();
}

void common_init_result::reset_samplers() {
    for (int i = 0; i < (int) pimpl->samplers.size(); ++i) {
        llama_sampler_reset(common_sampler_get(pimpl->samplers[i].get()));
    }
}

std::vector<llama_adapter_lora_ptr> & common_init_result::lora() {
    return pimpl->lora;
}

common_init_result_ptr common_init_from_params(common_params & params, bool model_only) {
    common_init_result_ptr res(new common_init_result(params, model_only));

    llama_model * model = res->model();
    if (model == NULL) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return res;
    }

    if (model_only) {
        return res;
    }

    llama_context * lctx = res->context();
    if (lctx == NULL) {
        LOG_ERR("%s: failed to create context with model '%s'\n", __func__, params.model.path.c_str());
        return res;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (params.ctx_shift && !llama_memory_can_shift(llama_get_memory(lctx))) {
        LOG_WRN("%s: KV cache shifting is not supported for this context, disabling KV cache shifting\n", __func__);
        params.ctx_shift = false;
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_model_n_layer(model);

        const auto cvec = common_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            return res;
        }

        int err = llama_set_adapter_cvec(
                lctx,
                cvec.data.data(),
                cvec.data.size(),
                cvec.n_embd,
                params.control_vector_layer_start,
                params.control_vector_layer_end);
        if (err) {
            return res;
        }
    }

    if (llama_pooling_type(lctx) == LLAMA_POOLING_TYPE_RANK) {
        bool ok = true;

        if (llama_vocab_bos(vocab) == LLAMA_TOKEN_NULL) {
            LOG_WRN("%s: warning: vocab does not have a  BOS token, reranking will not work\n", __func__);
            ok = false;
        }

        bool has_eos = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;
        bool has_sep = llama_vocab_sep(vocab) != LLAMA_TOKEN_NULL;
        bool has_rerank_prompt = llama_model_chat_template(model, "rerank") != NULL;

        if (!has_eos && !has_sep && !has_rerank_prompt) {
            LOG_WRN("%s: warning: vocab does not have an EOS token, SEP token, or rerank prompt. Reranking will not work\n", __func__);
            ok = false;
        } else if (!has_eos) {
            LOG_WRN("%s: warning: vocab does not have an EOS token, using SEP token as fallback\n", __func__);
        }

        if (!ok) {
            return res;
        }
    }

    if (!params.lora_init_without_apply) {
        common_set_adapter_lora(lctx, params.lora_adapters);
    }

    if (params.warmup) {
        LOG_INF("%s: warming up the model with an empty run - please wait ... (--no-warmup to disable)\n", __func__);

        std::vector<llama_token> tmp;
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);

        // some models (e.g. T5) don't have a BOS token
        if (bos != LLAMA_TOKEN_NULL) {
            tmp.push_back(bos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            tmp.push_back(eos);
        }
        if (tmp.empty()) {
            tmp.push_back(0);
        }

        if (llama_model_has_encoder(model)) {
            llama_encode(lctx, llama_batch_get_one(tmp.data(), tmp.size()));
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = bos;
            }
            tmp.clear();
            tmp.push_back(decoder_start_token_id);
        }
        if (llama_model_has_decoder(model)) {
            llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch)));
        }
        llama_memory_clear(llama_get_memory(lctx), true);
        llama_synchronize(lctx);
        llama_perf_context_reset(lctx);

        // reset samplers to reset RNG state after warmup to the seeded state
        res->reset_samplers();
    }

    return res;
}

common_init_result::~common_init_result() = default;

std::string common_get_model_endpoint() {
    const char * model_endpoint_env = getenv("MODEL_ENDPOINT");
    // We still respect the use of environment-variable "HF_ENDPOINT" for backward-compatibility.
    const char * hf_endpoint_env = getenv("HF_ENDPOINT");
    const char * endpoint_env = model_endpoint_env ? model_endpoint_env : hf_endpoint_env;
    std::string model_endpoint = "https://huggingface.co/";
    if (endpoint_env) {
        model_endpoint = endpoint_env;
        if (model_endpoint.back() != '/') {
            model_endpoint += '/';
        }
    }
    return model_endpoint;
}

common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx) {
    auto * mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    }

    common_context_seq_rm_type res = COMMON_CONTEXT_SEQ_RM_TYPE_PART;

    llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret = llama_decode(ctx, llama_batch_get_one(tmp.data(), tmp.size()));
    if (ret != 0) {
        LOG_ERR("%s: llama_decode() failed: %d\n", __func__, ret);
        res = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
        goto done;
    }

    if (llama_n_rs_seq(ctx) > 0) {
        LOG_INF("%s: the context supports bounded partial sequence removal\n", __func__);
        res = COMMON_CONTEXT_SEQ_RM_TYPE_RS;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
        LOG_TRC("%s: the context does not support partial sequence removal\n", __func__);
        res = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        goto done;
    }

done:
    llama_memory_clear(mem, true);
    llama_synchronize(ctx);

    return res;
}

void common_context_seq_rm(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    auto * mem = llama_get_memory(ctx);
    if (!llama_memory_seq_rm(mem, seq_id, p0, p1)) {
        GGML_ABORT("%s", string_format("failed to remove sequence %d with p0=%d, p1=%d\n", seq_id, p0, p1).c_str());
    }
}

void common_context_seq_cp(llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1);
}

void common_context_seq_add(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_seq_add(mem, seq_id, p0, p1, delta);
}

void common_set_adapter_lora(struct llama_context * ctx, std::vector<common_adapter_lora_info> & lora) {
    std::vector<llama_adapter_lora *> loras;
    std::vector<float> scales;

    for (auto & la: lora) {
        loras.push_back(la.ptr);
        scales.push_back(la.scale);
    }

    llama_set_adapters_lora(ctx, loras.data(), loras.size(), scales.data());
}

struct llama_model_params common_model_params_to_llama(common_params & params) {
    auto mparams = llama_model_default_params();

    if (!params.devices.empty()) {
        mparams.devices = params.devices.data();
    }

    mparams.n_gpu_layers    = params.n_gpu_layers;
    mparams.main_gpu        = params.main_gpu;
    mparams.split_mode      = params.split_mode;
    mparams.tensor_split    = params.tensor_split;
    mparams.use_mmap        = params.use_mmap;
    mparams.use_direct_io   = params.use_direct_io;
    mparams.use_mlock       = params.use_mlock;
    mparams.check_tensors   = params.check_tensors;
    mparams.use_extra_bufts = !params.no_extra_bufts;
    mparams.no_host         = params.no_host;

    if (params.kv_overrides.empty()) {
        mparams.kv_overrides = NULL;
    } else {
        GGML_ASSERT(params.kv_overrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
        mparams.kv_overrides = params.kv_overrides.data();
    }

    if (params.tensor_buft_overrides.empty()) {
        mparams.tensor_buft_overrides = NULL;
    } else {
        GGML_ASSERT(params.tensor_buft_overrides.back().pattern == nullptr && "Tensor buffer overrides not terminated with empty pattern");
        mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
    }

    mparams.progress_callback           = params.load_progress_callback;
    mparams.progress_callback_user_data = params.load_progress_callback_user_data;
    mparams.no_alloc                    = params.no_alloc;

    return mparams;
}

struct llama_context_params common_context_params_to_llama(const common_params & params) {
    auto cparams = llama_context_default_params();

    cparams.n_ctx             = params.n_ctx;
    cparams.n_seq_max         = params.n_parallel;
    cparams.n_rs_seq          = params.speculative.need_n_rs_seq();
    cparams.n_outputs_max     = std::max(params.n_outputs_max, 0);
    cparams.n_batch           = params.n_batch;
    cparams.n_ubatch          = params.n_ubatch;
    cparams.n_threads         = params.cpuparams.n_threads;
    cparams.n_threads_batch   = params.cpuparams_batch.n_threads == -1 ?
                                params.cpuparams.n_threads : params.cpuparams_batch.n_threads;
    cparams.embeddings        = params.embedding;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.rope_freq_base    = params.rope_freq_base;
    cparams.rope_freq_scale   = params.rope_freq_scale;
    cparams.yarn_ext_factor   = params.yarn_ext_factor;
    cparams.yarn_attn_factor  = params.yarn_attn_factor;
    cparams.yarn_beta_fast    = params.yarn_beta_fast;
    cparams.yarn_beta_slow    = params.yarn_beta_slow;
    cparams.yarn_orig_ctx     = params.yarn_orig_ctx;
    cparams.pooling_type      = params.pooling_type;
    cparams.attention_type    = params.attention_type;
    cparams.flash_attn_type   = params.flash_attn_type;
    cparams.cb_eval                 = params.cb_eval;
    cparams.cb_eval_user_data       = params.cb_eval_user_data;
    cparams.cb_node_override        = params.cb_node_override;
    cparams.cb_node_override_user_data = params.cb_node_override_user_data;
    cparams.offload_kqv       = !params.no_kv_offload;
    cparams.no_perf           = params.no_perf;
    cparams.op_offload        = !params.no_op_offload;
    cparams.swa_full          = params.swa_full;
    cparams.kv_unified        = params.kv_unified;

    cparams.type_k = params.cache_type_k;
    cparams.type_v = params.cache_type_v;

    return cparams;
}

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params) {
    struct ggml_threadpool_params tpp;

    ggml_threadpool_params_init(&tpp, params.n_threads); // setup the defaults

    if (params.mask_valid) {
        std::memcpy(&tpp.cpumask, &params.cpumask, GGML_MAX_N_THREADS);
    }

    tpp.prio       = params.priority;
    tpp.poll       = params.poll;
    tpp.strict_cpu = params.strict_cpu;

    return tpp;
}

//
// Batch utils
//

void common_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

void common_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

//
// Vocab utils
//

std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_tokenize(vocab, text, add_special, parse_special);
}

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
    }
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string common_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_token_to_piece(vocab, token, special);
}

std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

std::string common_detokenize(const struct llama_context * ctx, const std::vector<llama_token> & tokens, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_detokenize(vocab, tokens, special);
}

std::string common_detokenize(const struct llama_vocab * vocab, const std::vector<llama_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

//
// Embedding utils
//

void common_embd_normalize(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) {
                    sum = std::abs(inp[i]);
                }
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

float common_embd_similarity_cos(const float * embd1, const float * embd2, int n){
    double sum  = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < n; i++) {
        sum  += embd1[i] * embd2[i];
        sum1 += embd1[i] * embd1[i];
        sum2 += embd2[i] * embd2[i];
    }

    // Handle the case where one or both vectors are zero vectors
    if (sum1 == 0.0 || sum2 == 0.0) {
        if (sum1 == 0.0 && sum2 == 0.0) {
            return 1.0f; // two zero vectors are similar
        }
        return 0.0f;
    }

    return sum / (sqrt(sum1) * sqrt(sum2));
}

//
// Control vector utils
//

static common_control_vector_data common_control_vector_load_one(const common_control_vector_load_info & load_info) {
    common_control_vector_data result = { -1, {} };

    ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(load_info.fname.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        LOG_ERR("%s: failed to load control vector file from %s\n", __func__, load_info.fname.c_str());
        return result;
    }

    int32_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors == 0) {
        LOG_WRN("%s: no direction tensors found in %s\n", __func__, load_info.fname.c_str());
    }

    for (int i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);

        int layer_idx = -1;

        // split on '.'
        size_t dotpos = name.find('.');
        if (dotpos != std::string::npos && name.substr(0, dotpos) == "direction") {
            try {
                layer_idx = std::stoi(name.substr(dotpos + 1));
            } catch (...) {
                layer_idx = -1;
            }
        }
        if (layer_idx < 0) {
            LOG_ERR("%s: invalid/unparsable direction tensor layer index in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        } else if (layer_idx == 0) {
            LOG_ERR("%s: invalid (zero) direction tensor layer index in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
        if (tensor->type != GGML_TYPE_F32) {
            LOG_ERR("%s: invalid (non-F32) direction tensor type in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }
        if (ggml_n_dims(tensor) != 1) {
            LOG_ERR("%s: invalid (non-1D) direction tensor shape in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result.n_embd = ggml_nelements(tensor);
        } else if (ggml_nelements(tensor) != result.n_embd) {
            LOG_ERR("%s: direction tensor in %s does not match previous dimensions\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        // extend if necessary - do not store data for layer 0 (it's not used)
        result.data.resize(std::max(result.data.size(), static_cast<size_t>(result.n_embd * layer_idx)), 0.0f);

        const float * src = (const float *) tensor->data;
        float * dst = result.data.data() + result.n_embd * (layer_idx - 1);  // layer 1 at [0]
        for (int j = 0; j < result.n_embd; j++) {
            dst[j] += src[j] * load_info.strength;  // allows multiple directions for same layer in same file
        }

    }

    if (result.n_embd == -1) {
        LOG_WRN("%s: skipping %s due to invalid direction tensors\n", __func__, load_info.fname.c_str());
        result.data.clear();
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return result;
}

common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos) {
    common_control_vector_data result = { -1, {} };

    for (const auto & info : load_infos) {
        auto cur = common_control_vector_load_one(info);

        if (cur.n_embd == -1) {
            result.n_embd = -1;
            break;
        }
        if (result.n_embd != -1 && result.n_embd != cur.n_embd) {
            LOG_ERR("%s: control vectors in %s does not match previous dimensions\n", __func__, info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result = std::move(cur);
        } else {
            result.data.resize(std::max(result.data.size(), cur.data.size()), 0.0f);  // extend if necessary
            for (size_t i = 0; i < cur.data.size(); i++) {
                result.data[i] += cur.data[i];
            }
        }
    }

    if (result.n_embd == -1) {
        LOG_ERR("%s: no valid control vector files passed\n", __func__);
        result.data.clear();
    }

    return result;
}

ggml_opt_dataset_t common_opt_dataset_init(struct llama_context * ctx, const std::vector<llama_token> & tokens, int64_t stride) {
    const int64_t ne_datapoint = llama_n_ctx(ctx);
    const int64_t ndata        = (tokens.size() - ne_datapoint - 1) / stride;
    ggml_opt_dataset_t result = ggml_opt_dataset_init(
        GGML_TYPE_I32, GGML_TYPE_I32, ne_datapoint, ne_datapoint, ndata, /*ndata_shard =*/ 1);

    llama_token * data   = (llama_token *) ggml_opt_dataset_data(result)->data;
    llama_token * labels = (llama_token *) ggml_opt_dataset_labels(result)->data;

    for (int64_t idata = 0; idata < ndata; ++idata) {
        memcpy(data   + idata*ne_datapoint, tokens.data() + idata*stride + 0, ne_datapoint*sizeof(llama_token));
        memcpy(labels + idata*ne_datapoint, tokens.data() + idata*stride + 1, ne_datapoint*sizeof(llama_token));
    }

    return result;
}

ggml_opt_optimizer_params common_opt_lr_pars(void * userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
    const lr_opt &            d      = *(lr_opt *) userdata;
    result.adamw.alpha = result.sgd.alpha = d.get_lr(d.epoch);
    result.sgd.wd = result.adamw.wd = d.wd;
    return result;
}

// TODO make all command line args case-insensitive
static inline bool eq_case_insensitive(char const* a, char const* b) {
    return !
#if defined(_MSC_VER)
        _stricmp
#else
        strcasecmp
#endif // defined(_MSC_VER)
        (a, b);
}

enum ggml_opt_optimizer_type common_opt_get_optimizer(const char * n) {
    if (eq_case_insensitive("adamw", n)) {
        return GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    }
    if (eq_case_insensitive("sgd", n)) {
        return GGML_OPT_OPTIMIZER_TYPE_SGD;
    }
    return GGML_OPT_OPTIMIZER_TYPE_COUNT;
}

// TODO simplify to use just log and exp
static float const k_log_2 = std::log(2.f);

void lr_opt::init() {
    if (lr_min > 0 && lr_min < lr0) {
        float nhalf = std::log(lr0 / lr_min) / k_log_2;
        float e     = epochs;
        if (decay_epochs > 0 && decay_epochs < e) {
            e = decay_epochs;
        } else {
            decay_epochs = e;
        }
        scale_epoch = nhalf / e;
    }
}

float lr_opt::get_lr(float epoch) const {
    float r = lr_min <= 0 ? lr0 :
        epoch >= decay_epochs ? lr_min :
        lr0 * std::pow(0.5f, epoch * scale_epoch);
    LOG_INF("epoch %.2g lr=%.2g\n", epoch, r);
    return r;
}

bool common_replay_last_token(struct llama_context * ctx, llama_token last_token, int32_t pos) {
    llama_batch batch = llama_batch_get_one(&last_token, 1);
    batch.pos = &pos;
    if (common_aif_decode_with_scope(ctx, batch, "replay", 1, pos)) {
        LOG_ERR("%s: failed to replay last token\n", __func__);
        return false;
    }
    return true;
}

bool common_prompt_batch_decode(
              struct llama_context * ctx,
    const std::vector<llama_token> & all_tokens,
                               int   n_new,
                               int & n_past,
                               int   n_batch,
                  std::string_view   state_path,
                              bool   save_state) {
    if (n_new == 0) {
        return true;
    }
    const int offset = all_tokens.size() - n_new;

    if (save_state && n_new > 1) {
        const int n_tokens_before_last = n_new - 1;

        GGML_ASSERT(n_new <= n_batch);

        // Decode all but the last token so we can save the memory state before decoding the last token.
        // This is done so we can restore the session state later and replay the last token.
        // Memory implementations in recurrent/hybrid models don't support removing tokens from their
        // memory, so we can't just remove the last token from the memory and replay the last token which
        // is the reason for this logic.
        if (common_aif_decode_with_scope(
                    ctx,
                    llama_batch_get_one(const_cast<llama_token*>(all_tokens.data() + offset), n_tokens_before_last),
                    "prompt",
                    n_tokens_before_last,
                    n_past)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        n_past += n_tokens_before_last;

        llama_state_save_file(ctx, state_path.data(), all_tokens.data(), all_tokens.size());
        LOG_INF("saved session before last token to %s, n_new = %zu\n", state_path.data(), all_tokens.size());

        llama_token last_token = all_tokens.back();
        llama_batch batch = llama_batch_get_one(&last_token, 1);
        int32_t pos = n_past;
        batch.pos = &pos;

        if (common_aif_decode_with_scope(ctx, batch, "prompt", 1, n_past)) {
            LOG_ERR("%s : failed to eval last token\n", __func__);
            return false;
        }
        n_past++;
    } else {
        const char * phase = (n_past == 0 || n_new > 1) ? "prompt" : "decode";
        if (common_aif_decode_with_scope(
                    ctx,
                    llama_batch_get_one(const_cast<llama_token*>(all_tokens.data() + offset), n_new),
                    phase,
                    n_new,
                    n_past)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        n_past += n_new;
    }

    return true;
}

size_t common_prompt_checkpoint::size() const {
    return data_tgt.size() + data_dft.size() + data_spec.size();
}

bool common_prompt_checkpoint::empty() const {
    return data_tgt.empty();
}

void common_prompt_checkpoint::clear() {
    n_tokens = 0;

    pos_min = 0;
    pos_max = 0;

    data_tgt.clear();
    data_dft.clear();
    data_spec.clear();
}

void common_prompt_checkpoint::update_pos(
        int64_t n_tokens,
        llama_pos pos_min,
        llama_pos pos_max) {
    this->n_tokens = n_tokens;
    this->pos_min  = pos_min;
    this->pos_max  = pos_max;
}

void common_prompt_checkpoint::update_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return;
    }

    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);

    data_tgt.resize(ckpt_size);

    const size_t n = llama_state_seq_get_data_ext(ctx, data_tgt.data(), ckpt_size, seq_id, flags);
    if (n != ckpt_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
    }
}

void common_prompt_checkpoint::update_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return;
    }

    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);

    data_dft.resize(ckpt_size);

    const size_t n = llama_state_seq_get_data_ext(ctx, data_dft.data(), ckpt_size, seq_id, flags);
    if (n != ckpt_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
    }
}

void common_prompt_checkpoint::load_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return;
    }

    if (data_tgt.empty()) {
        return;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_tgt.data(), data_tgt.size(), seq_id, flags);
    if (n != data_tgt.size()) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", data_tgt.size(), n);
    }
}

void common_prompt_checkpoint::load_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return;
    }

    if (data_dft.empty()) {
        return;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_dft.data(), data_dft.size(), seq_id, flags);
    if (n != data_dft.size()) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", data_dft.size(), n);
    }
}

void common_prompt_checkpoint::clear_tgt() {
    data_tgt.clear();
}

void common_prompt_checkpoint::clear_dft() {
    data_dft.clear();
    data_spec.clear();
}
