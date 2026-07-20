#pragma once

#include "../src/llama-aif-proto.h"

#include <cstdint>
#include <string>

struct common_aif_client {
    common_aif_client();
    ~common_aif_client();

    common_aif_client(const common_aif_client &) = delete;
    common_aif_client & operator=(const common_aif_client &) = delete;

    bool open(const std::string & device_path, std::string & error);
    void close();

    bool is_open() const;
    int  nsid() const;

    const std::string & device_path() const;

    bool stats(aif_stats_resp & stats, std::string & error) const;
    bool reset(std::string & error) const;
    bool post(const aif_post_req & req, std::string & error) const;
    bool gemv(void * data, std::uint32_t data_len, std::string & error) const;

private:
    bool ioctl_cmd(std::uint8_t opcode, void * data, std::uint32_t data_len, std::string & error) const;

    int         fd_   = -1;
    int         nsid_ = -1;
    std::string device_path_;
};
