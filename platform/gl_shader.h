#pragma once
// GLSL ES 3.00 for the title's pixel stage: register combiners and D3D fixed function stages, written from the same
// definitions as the CPU reference renderer's Combiners and ShadeStages, which are what it is tested against.
#include <string>
#include <vector>

#include "gpu/shaders.h"

namespace glshader {

extern const char* kVertexSource;

// alpha_test: the shader discards by u_at (x function, y reference 0..255)
std::string FragmentSource(const gpu::PixelKey& key, const gpu::TexInfo info[4], bool alpha_test);

// the console's vertex program (microcode, 4 words an instruction) as a GLSL ES vertex shader with the outputs the
// fragment shaders read: the same decoding and the same order of writes as the CPU reference renderer's VertexProgram
// variants: 2 bits per input register, how the shader reads the attribute: 0 as a vec4, 1 bytes B G R A (D3DCOLOR),
// 2 a uint of 11:11:10 signed normals (NORMPACKED3), 3 x y w as 3 floats (FLOAT2H)
// w1_mask: bit per input register whose fourth component is 1 whatever the buffer holds (Vulkan reads such a register
// with four components, as three byte or short components are not vertex formats it has)
std::string VertexProgramSource(const std::vector<uint32_t>& program, uint32_t variants = 0, uint32_t w1_mask = 0);

// the bytes that decide the source: a cache key
std::string CacheKey(const gpu::PixelKey& key, const gpu::TexInfo info[4], bool alpha_test);
// the same bytes hashed, without building them up: looked up on every draw
uint64_t CacheHash(const gpu::PixelKey& key, const gpu::TexInfo info[4], bool alpha_test);

}  // namespace glshader
