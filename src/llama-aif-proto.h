#ifndef AIF_PROTO_H
#define AIF_PROTO_H

#ifdef __KERNEL__
#include <linux/types.h>
typedef __u8  aif_u8;
typedef __u16 aif_u16;
typedef __u32 aif_u32;
typedef __u64 aif_u64;
#else
#include <stdint.h>
typedef uint8_t  aif_u8;
typedef uint16_t aif_u16;
typedef uint32_t aif_u32;
typedef uint64_t aif_u64;
#endif

#define AIF_MAGIC   0x41494631u /* "AIF1" */
#define AIF_VERSION 2u

#define AIF_OP_POST  0xC1u
#define AIF_OP_STATS 0xC2u
#define AIF_OP_GEMV  0xC3u
#define AIF_OP_RESET 0xC4u

#define AIF_TENSOR_NAME_MAX 128u
#define AIF_LAYOUT_MAX_TENSORS 1024u
#define AIF_STATS_MAX_CHIPS 16u
#define AIF_STATS_MAX_PLANES 4u

#define AIF_POST_F_LAYOUT_ONLY  (1u << 0)
#define AIF_GEMV_F_DUMMY_OUTPUT (1u << 0)
#define AIF_GEMV_F_CHECK_INPUT  (1u << 1)
#define AIF_GEMV_F_SKIP_OUTPUT_COPY (1u << 2)

#if defined(__GNUC__) || defined(__clang__)
#define AIF_PACKED __attribute__((packed))
#else
#define AIF_PACKED
#endif

struct AIF_PACKED aif_post_req {
    aif_u32 magic;
    aif_u16 version;
    aif_u16 flags;

    aif_u64 tensor_id;
    aif_u64 lba_start;
    aif_u64 lba_count;

    aif_u32 rows;
    aif_u32 cols;
    aif_u32 ggml_type;
    aif_u32 elem_size;

    aif_u64 matrix_nbytes;
    aif_u32 row_stride;
    aif_u32 reserved0;

    char tensor_name[AIF_TENSOR_NAME_MAX];
};

struct AIF_PACKED aif_gemv_req {
    aif_u32 magic;
    aif_u16 version;
    aif_u16 flags;

    aif_u64 tensor_id;
    aif_u64 lba_start;
    aif_u64 lba_count;

    aif_u32 input_dim;
    aif_u32 output_dim;
    aif_u32 input_offset;
    aif_u32 input_nbytes;
    aif_u32 output_offset;
    aif_u32 output_nbytes;

    aif_u32 ggml_type;
    aif_u32 reserved0;
};

struct AIF_PACKED aif_chip_stats {
    aif_u32 chip_id;
    aif_u32 channel;
    aif_u32 lun;
    aif_u32 active;

    aif_u64 row_start;
    aif_u64 row_count;
    aif_u64 page_count;
    aif_u64 ifp_block_count;
    aif_u64 first_ppa;
    aif_u64 last_ppa;
    aif_u64 plane_page_count[AIF_STATS_MAX_PLANES];
};

struct AIF_PACKED aif_stats_resp {
    aif_u32 magic;
    aif_u16 version;
    aif_u16 reserved0;

    aif_u64 post_count;
    aif_u64 gemv_count;
    aif_u64 tensor_count;

    aif_u64 last_tensor_id;
    aif_u64 last_input_nbytes;
    aif_u64 last_output_nbytes;
    aif_u64 last_matrix_nbytes;
    aif_u64 last_delay_ns;
    aif_u64 last_input_checksum;
    aif_u64 last_input_pcie_ns;
    aif_u64 last_matrix_compute_ns;
    aif_u64 last_output_pcie_ns;

    aif_u64 layout_hit_count;
    aif_u64 layout_miss_count;
    aif_u64 last_layout_slot;
    aif_u64 last_layout_pages;
    aif_u64 last_layout_stripes;
    aif_u64 last_layout_parallel_units;
    aif_u64 last_lba_start;
    aif_u64 last_lba_count;
    aif_u64 last_start_lpn;
    aif_u64 last_mapped_pages;
    aif_u64 last_units_used;
    aif_u64 last_max_pages_per_unit;
    aif_u64 last_first_ppa;
    aif_u64 last_last_ppa;

    aif_u64 last_input_onfi_ns;
    aif_u64 last_chip_compute_ns;
    aif_u64 last_output_onfi_ns;
    aif_u64 last_active_chips;
    aif_u64 last_tail_chip;

    aif_u64 last_layout_chip_count;
    aif_u64 last_layout_ifp_blocks;
    aif_u64 last_reserved_ifp_lines;
    aif_u64 last_max_chip_pages;
    aif_u64 last_lsb_pages;
    aif_u64 last_non_lsb_pages;

    aif_u64 config_channels;
    aif_u64 config_chips;
    aif_u64 config_planes_per_chip;
    aif_u64 config_page_size;
    aif_u64 config_pcie_bytes_per_sec;
    aif_u64 config_onfi_bytes_per_sec;
    aif_u64 config_chip_bytes_per_sec;
    aif_u64 config_cr_read_ns;

    struct aif_chip_stats last_chips[AIF_STATS_MAX_CHIPS];
};

#define AIF_POST_REQ_ABI_SIZE   192u
#define AIF_GEMV_REQ_ABI_SIZE    64u
#define AIF_CHIP_STATS_ABI_SIZE  96u
#define AIF_STATS_RESP_ABI_SIZE 1904u

#if defined(__cplusplus)
static_assert(sizeof(aif_post_req) == AIF_POST_REQ_ABI_SIZE, "aif_post_req ABI changed");
static_assert(sizeof(aif_gemv_req) == AIF_GEMV_REQ_ABI_SIZE, "aif_gemv_req ABI changed");
static_assert(sizeof(aif_chip_stats) == AIF_CHIP_STATS_ABI_SIZE, "aif_chip_stats ABI changed");
static_assert(sizeof(aif_stats_resp) == AIF_STATS_RESP_ABI_SIZE, "aif_stats_resp ABI changed");
#else
_Static_assert(sizeof(struct aif_post_req) == AIF_POST_REQ_ABI_SIZE, "aif_post_req ABI changed");
_Static_assert(sizeof(struct aif_gemv_req) == AIF_GEMV_REQ_ABI_SIZE, "aif_gemv_req ABI changed");
_Static_assert(sizeof(struct aif_chip_stats) == AIF_CHIP_STATS_ABI_SIZE, "aif_chip_stats ABI changed");
_Static_assert(sizeof(struct aif_stats_resp) == AIF_STATS_RESP_ABI_SIZE, "aif_stats_resp ABI changed");
#endif

#undef AIF_PACKED

#endif /* AIF_PROTO_H */
