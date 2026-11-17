#pragma once
// The renderer's generated shaders (GLSL ES 3.00, gl_shader.cpp) for Vulkan: rewritten as Vulkan GLSL 4.50 with the
// loose uniforms in one block, samplers and varyings bound by number, then compiled to SPIR-V with glslang.
#include <cstdint>
#include <string>
#include <vector>

namespace vkglsl {

// set 0: the uniform blocks (dynamic offsets into the frame's uniform ring); set 1: the samplers (pushed)
constexpr uint32_t kVertexBlock = 0;    // the vertex shader's uniforms, std140
constexpr uint32_t kFragmentBlock = 1;  // the fragment shader's uniforms, std140
constexpr uint32_t kFirstSampler = 2;   // sampler N at binding 2 + N of set 1

// a uniform of the GLES source, now a member of its stage's block: byte offset (std140, all vec4 or vec4 arrays)
struct Member {
    std::string name;
    uint32_t offset = 0, count = 1;  // count: array length, 1 for a plain vec4
};
struct Converted {
    std::string source;              // Vulkan GLSL
    std::vector<Member> block;       // the uniform block's members in order
    uint32_t block_bytes = 0;
    uint32_t samplers = 0;           // bit N: u_sN is used
};

// vertex: true for a vertex shader (its clip z goes from GL's -w..w to Vulkan's 0..w)
Converted ToVulkan(const std::string& gles, bool vertex);

// SPIR-V of a Vulkan GLSL source, empty on failure (log has why)
std::vector<uint32_t> Compile(const std::string& source, bool vertex, std::string* log = nullptr);

}  // namespace vkglsl
