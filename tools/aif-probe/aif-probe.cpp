#include "aif-client.h"

#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void usage(const char * prog) {
    std::fprintf(stderr, "Usage: %s /dev/nvmeXnY [stats|reset|post|gemv|all]\n", prog);
}

static void * alloc_page_aligned(std::size_t size) {
    void * ptr = nullptr;

    if (posix_memalign(&ptr, 4096, size) != 0) {
        return nullptr;
    }

    std::memset(ptr, 0, size);
    return ptr;
}

static std::uint64_t monotonic_ns() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

static void print_stats(const aif_stats_resp & stats) {
    std::printf("stats.magic             = 0x%08" PRIx32 "\n", stats.magic);
    std::printf("stats.version           = %" PRIu16 "\n", stats.version);
    std::printf("stats.post_count        = %" PRIu64 "\n", stats.post_count);
    std::printf("stats.gemv_count        = %" PRIu64 "\n", stats.gemv_count);
    std::printf("stats.tensor_count      = %" PRIu64 "\n", stats.tensor_count);
    std::printf("stats.last_tensor_id    = %" PRIu64 "\n", stats.last_tensor_id);
    std::printf("stats.last_input_nbytes = %" PRIu64 "\n", stats.last_input_nbytes);
    std::printf("stats.last_output_nbytes= %" PRIu64 "\n", stats.last_output_nbytes);
    std::printf("stats.last_matrix_nbytes= %" PRIu64 "\n", stats.last_matrix_nbytes);
    std::printf("stats.last_delay_ns     = %" PRIu64 "\n", stats.last_delay_ns);
    std::printf("stats.last_checksum     = %" PRIu64 "\n", stats.last_input_checksum);
    std::printf("stats.last_input_pcie_ns= %" PRIu64 "\n", stats.last_input_pcie_ns);
    std::printf("stats.last_internal_ns  = %" PRIu64 "\n", stats.last_matrix_compute_ns);
    std::printf("stats.last_output_pcie_ns= %" PRIu64 "\n", stats.last_output_pcie_ns);
    std::printf("stats.layout_hit_count  = %" PRIu64 "\n", stats.layout_hit_count);
    std::printf("stats.layout_miss_count = %" PRIu64 "\n", stats.layout_miss_count);
    std::printf("stats.last_layout_slot  = %" PRIu64 "\n", stats.last_layout_slot);
    std::printf("stats.last_layout_pages = %" PRIu64 "\n", stats.last_layout_pages);
    std::printf("stats.last_layout_stripes= %" PRIu64 "\n", stats.last_layout_stripes);
    std::printf("stats.last_layout_units = %" PRIu64 "\n", stats.last_layout_parallel_units);
    std::printf("stats.last_lba_start    = %" PRIu64 "\n", stats.last_lba_start);
    std::printf("stats.last_lba_count    = %" PRIu64 "\n", stats.last_lba_count);
    std::printf("stats.last_start_lpn    = %" PRIu64 "\n", stats.last_start_lpn);
    std::printf("stats.last_mapped_pages = %" PRIu64 "\n", stats.last_mapped_pages);
    std::printf("stats.last_units_used   = %" PRIu64 "\n", stats.last_units_used);
    std::printf("stats.last_max_unit_pages= %" PRIu64 "\n", stats.last_max_pages_per_unit);
    std::printf("stats.last_first_ppa    = 0x%016" PRIx64 "\n", stats.last_first_ppa);
    std::printf("stats.last_last_ppa     = 0x%016" PRIx64 "\n", stats.last_last_ppa);
    std::printf("stats.last_input_onfi_ns= %" PRIu64 "\n", stats.last_input_onfi_ns);
    std::printf("stats.last_chip_compute_ns= %" PRIu64 "\n", stats.last_chip_compute_ns);
    std::printf("stats.last_output_onfi_ns= %" PRIu64 "\n", stats.last_output_onfi_ns);
    std::printf("stats.last_active_chips = %" PRIu64 "\n", stats.last_active_chips);
    std::printf("stats.last_tail_chip    = %" PRIu64 "\n", stats.last_tail_chip);
    std::printf("stats.last_layout_chips = %" PRIu64 "\n", stats.last_layout_chip_count);
    std::printf("stats.last_ifp_blocks   = %" PRIu64 "\n", stats.last_layout_ifp_blocks);
    std::printf("stats.last_ifp_lines    = %" PRIu64 "\n", stats.last_reserved_ifp_lines);
    std::printf("stats.last_max_chip_pages= %" PRIu64 "\n", stats.last_max_chip_pages);
    std::printf("stats.last_lsb_pages    = %" PRIu64 "\n", stats.last_lsb_pages);
    std::printf("stats.last_non_lsb_pages= %" PRIu64 "\n", stats.last_non_lsb_pages);
    std::printf("config.channels         = %" PRIu64 "\n", stats.config_channels);
    std::printf("config.chips            = %" PRIu64 "\n", stats.config_chips);
    std::printf("config.planes_per_chip  = %" PRIu64 "\n", stats.config_planes_per_chip);
    std::printf("config.page_size        = %" PRIu64 "\n", stats.config_page_size);
    std::printf("config.pcie_bytes_per_sec= %" PRIu64 "\n", stats.config_pcie_bytes_per_sec);
    std::printf("config.onfi_bytes_per_sec= %" PRIu64 "\n", stats.config_onfi_bytes_per_sec);
    std::printf("config.chip_bytes_per_sec= %" PRIu64 "\n", stats.config_chip_bytes_per_sec);
    std::printf("config.cr_read_ns       = %" PRIu64 "\n", stats.config_cr_read_ns);

    for (std::uint32_t i = 0; i < stats.last_layout_chip_count && i < AIF_STATS_MAX_CHIPS; ++i) {
        const auto & chip = stats.last_chips[i];
        std::printf("chip[%02" PRIu32 "] ch=%" PRIu32 " lun=%" PRIu32
                    " rows=%" PRIu64 "+%" PRIu64 " pages=%" PRIu64
                    " planes=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
                    " ifp_blocks=%" PRIu64 " first=0x%016" PRIx64 " last=0x%016" PRIx64 "\n",
                i, chip.channel, chip.lun, chip.row_start, chip.row_count, chip.page_count,
                chip.plane_page_count[0], chip.plane_page_count[1], chip.plane_page_count[2],
                chip.plane_page_count[3], chip.ifp_block_count, chip.first_ppa, chip.last_ppa);
    }
}

static std::uint64_t checksum_bytes(const std::uint8_t * buf, std::uint32_t len) {
    std::uint64_t checksum = 0;

    for (std::uint32_t i = 0; i < len; ++i) {
        checksum = checksum * 131 + buf[i];
    }

    return checksum;
}

static int send_stats(const common_aif_client & client) {
    aif_stats_resp stats {};
    std::string error;

    if (!client.stats(stats, error)) {
        std::fprintf(stderr, "AIF_OP_STATS failed: %s\n", error.c_str());
        return 1;
    }

    print_stats(stats);
    return 0;
}

static int send_reset(const common_aif_client & client) {
    std::string error;

    if (!client.reset(error)) {
        std::fprintf(stderr, "AIF_OP_RESET failed: %s\n", error.c_str());
        return 1;
    }

    std::printf("AIF_OP_RESET sent\n");
    return 0;
}

static int send_post(const common_aif_client & client) {
    aif_post_req req {};
    std::string error;

    req.magic = AIF_MAGIC;
    req.version = AIF_VERSION;
    req.flags = AIF_POST_F_LAYOUT_ONLY;
    req.tensor_id = 1;
    req.lba_start = 0x100000;
    req.rows = 4096;
    req.cols = 4096;
    req.ggml_type = 0;
    req.elem_size = 1;
    req.matrix_nbytes = static_cast<std::uint64_t>(req.rows) * req.cols * req.elem_size;
    req.lba_count = (req.matrix_nbytes + 511) / 512;
    req.row_stride = req.cols * req.elem_size;
    std::snprintf(req.tensor_name, sizeof(req.tensor_name), "probe.blk0.ffn_up.weight");

    if (!client.post(req, error)) {
        std::fprintf(stderr, "AIF_OP_POST failed: %s\n", error.c_str());
        return 1;
    }

    std::printf("AIF_OP_POST sent: tensor_id=%" PRIu64 " matrix_nbytes=%" PRIu64 "\n",
            req.tensor_id, req.matrix_nbytes);
    return 0;
}

static int send_gemv(const common_aif_client & client) {
    constexpr std::uint32_t input_nbytes = 4096;
    constexpr std::uint32_t output_nbytes = 1024;
    constexpr std::uint32_t input_offset = sizeof(aif_gemv_req);
    constexpr std::uint32_t output_offset = input_offset + input_nbytes;
    constexpr std::size_t total = output_offset + output_nbytes;

    std::uint8_t * buf = static_cast<std::uint8_t *>(alloc_page_aligned(total));
    if (!buf) {
        std::fprintf(stderr, "failed to allocate gemv buffer\n");
        return 1;
    }

    aif_gemv_req * req = reinterpret_cast<aif_gemv_req *>(buf);
    req->magic = AIF_MAGIC;
    req->version = AIF_VERSION;
    req->flags = AIF_GEMV_F_DUMMY_OUTPUT | AIF_GEMV_F_CHECK_INPUT;
    req->tensor_id = 1;
    req->lba_start = 0x100000;
    req->lba_count = 32768;
    req->input_dim = 4096;
    req->output_dim = 1024;
    req->input_offset = input_offset;
    req->input_nbytes = input_nbytes;
    req->output_offset = output_offset;
    req->output_nbytes = output_nbytes;
    req->ggml_type = 0;

    for (std::uint32_t i = 0; i < input_nbytes; ++i) {
        buf[input_offset + i] = static_cast<std::uint8_t>(i & 0xff);
    }

    const std::uint64_t input_checksum = checksum_bytes(buf + input_offset, input_nbytes);

    std::string error;
    const std::uint64_t start_ns = monotonic_ns();
    const bool ok = client.gemv(buf, static_cast<std::uint32_t>(total), error);
    const std::uint64_t end_ns = monotonic_ns();
    if (!ok) {
        std::fprintf(stderr, "AIF_OP_GEMV failed: %s\n", error.c_str());
        std::free(buf);
        return 1;
    }

    std::uint32_t output_mismatches = 0;
    for (std::uint32_t i = 0; i < output_nbytes; ++i) {
        const auto expected = static_cast<std::uint8_t>(i + req->tensor_id + input_checksum);
        const auto actual = buf[output_offset + i];

        if (actual != expected) {
            ++output_mismatches;
            if (output_mismatches <= 8) {
                std::fprintf(stderr, "output mismatch at byte %" PRIu32 ": got=0x%02x expected=0x%02x\n",
                        i, actual, expected);
            }
        }
    }

    std::printf("AIF_OP_GEMV sent: tensor_id=%" PRIu64 " input=%" PRIu32 " output=%" PRIu32 "\n",
            req->tensor_id, req->input_nbytes, req->output_nbytes);
    std::printf("AIF_OP_GEMV input_checksum=%" PRIu64 " output_mismatches=%" PRIu32 "\n",
            input_checksum, output_mismatches);
    std::printf("AIF_OP_GEMV ioctl_elapsed_ns=%" PRIu64 "\n", end_ns - start_ns);

    std::free(buf);
    return output_mismatches == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        usage(argv[0]);
        return 2;
    }

    const char * mode = argc >= 3 ? argv[2] : "all";

    common_aif_client client;
    std::string error;
    if (!client.open(argv[1], error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    std::printf("device=%s nsid=%d mode=%s\n", client.device_path().c_str(), client.nsid(), mode);

    if (std::strcmp(mode, "stats") == 0) {
        return send_stats(client);
    }
    if (std::strcmp(mode, "reset") == 0) {
        return send_reset(client);
    }
    if (std::strcmp(mode, "post") == 0) {
        return send_post(client);
    }
    if (std::strcmp(mode, "gemv") == 0) {
        return send_gemv(client);
    }
    if (std::strcmp(mode, "all") == 0) {
        int ret = send_stats(client);
        if (ret == 0) {
            ret = send_post(client);
        }
        if (ret == 0) {
            ret = send_gemv(client);
        }
        if (ret == 0) {
            ret = send_stats(client);
        }
        return ret;
    }

    usage(argv[0]);
    return 2;
}
