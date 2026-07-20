#include "aif-client.h"

#include <cerrno>
#include <cstring>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/nvme_ioctl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

common_aif_client::common_aif_client() = default;

common_aif_client::~common_aif_client() {
    close();
}

bool common_aif_client::is_open() const {
    return fd_ >= 0;
}

int common_aif_client::nsid() const {
    return nsid_;
}

const std::string & common_aif_client::device_path() const {
    return device_path_;
}

#if defined(__linux__)

static std::string common_aif_errno_message(const std::string & prefix, int err) {
    return prefix + ": " + std::strerror(err);
}

bool common_aif_client::open(const std::string & device_path, std::string & error) {
    close();

    fd_ = ::open(device_path.c_str(), O_RDWR);
    if (fd_ < 0) {
        error = common_aif_errno_message(device_path, errno);
        return false;
    }

    nsid_ = ::ioctl(fd_, NVME_IOCTL_ID);
    if (nsid_ < 0) {
        const int err = errno;
        close();
        error = common_aif_errno_message("NVME_IOCTL_ID", err);
        return false;
    }

    device_path_ = device_path;
    return true;
}

void common_aif_client::close() {
    if (fd_ >= 0) {
        ::close(fd_);
    }

    fd_ = -1;
    nsid_ = -1;
    device_path_.clear();
}

bool common_aif_client::ioctl_cmd(std::uint8_t opcode, void * data, std::uint32_t data_len, std::string & error) const {
    if (fd_ < 0 || nsid_ < 0) {
        error = "AIF NVMe device is not open";
        return false;
    }

    nvme_passthru_cmd64 cmd {};
    cmd.opcode = opcode;
    cmd.nsid = static_cast<std::uint32_t>(nsid_);
    cmd.addr = reinterpret_cast<std::uint64_t>(data);
    cmd.data_len = data_len;
    cmd.timeout_ms = 30000;

    if (::ioctl(fd_, NVME_IOCTL_IO64_CMD, &cmd) < 0) {
        error = common_aif_errno_message("NVME_IOCTL_IO64_CMD", errno);
        return false;
    }

    return true;
}

#else

bool common_aif_client::open(const std::string & device_path, std::string & error) {
    (void) device_path;
    error = "AIF NVMe passthrough requires Linux NVMe ioctl support";
    return false;
}

void common_aif_client::close() {
    fd_ = -1;
    nsid_ = -1;
    device_path_.clear();
}

bool common_aif_client::ioctl_cmd(std::uint8_t opcode, void * data, std::uint32_t data_len, std::string & error) const {
    (void) opcode;
    (void) data;
    (void) data_len;
    error = "AIF NVMe passthrough requires Linux NVMe ioctl support";
    return false;
}

#endif

bool common_aif_client::stats(aif_stats_resp & stats, std::string & error) const {
    std::memset(&stats, 0, sizeof(stats));
    if (!ioctl_cmd(AIF_OP_STATS, &stats, sizeof(stats), error)) {
        return false;
    }
    if (stats.magic != AIF_MAGIC) {
        error = "AIF protocol magic mismatch (device=" + std::to_string(stats.magic) +
                ", client=" + std::to_string(AIF_MAGIC) + ")";
        return false;
    }
    if (stats.version != AIF_VERSION) {
        error = "AIF protocol version mismatch (device=" + std::to_string(stats.version) +
                ", client=" + std::to_string(AIF_VERSION) + ")";
        return false;
    }

    return true;
}

bool common_aif_client::reset(std::string & error) const {
    aif_stats_resp scratch {};
    return ioctl_cmd(AIF_OP_RESET, &scratch, sizeof(scratch), error);
}

bool common_aif_client::post(const aif_post_req & req, std::string & error) const {
    aif_post_req req_copy = req;
    return ioctl_cmd(AIF_OP_POST, &req_copy, sizeof(req_copy), error);
}

bool common_aif_client::gemv(void * data, std::uint32_t data_len, std::string & error) const {
    return ioctl_cmd(AIF_OP_GEMV, data, data_len, error);
}
