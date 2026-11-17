#include "vk_glsl.h"

#include <cstdio>
#include <mutex>
#include <regex>
#include <sstream>

#include <SPIRV/GlslangToSpv.h>
#include <glslang/Public/ShaderLang.h>

namespace vkglsl {
namespace {

// the varyings the generated stages pass, by name: one location each
int VaryingLocation(const std::string& name) {
    static const char* kNames[] = {"v_v0", "v_v1", "v_tc0", "v_tc1", "v_tc2", "v_tc3", "v_fog"};
    for (int i = 0; i < 7; i++)
        if (name == kNames[i]) return i;
    return -1;
}

// the limits glslang checks a shader against: generous, the device's own limits are what really apply
const TBuiltInResource& Resources() {
    static const TBuiltInResource r = [] {
        TBuiltInResource x{};
        x.maxLights = 32;
        x.maxClipPlanes = 6;
        x.maxTextureUnits = 32;
        x.maxTextureCoords = 32;
        x.maxVertexAttribs = 64;
        x.maxVertexUniformComponents = 4096;
        x.maxVaryingFloats = 64;
        x.maxVertexTextureImageUnits = 32;
        x.maxCombinedTextureImageUnits = 80;
        x.maxTextureImageUnits = 32;
        x.maxFragmentUniformComponents = 4096;
        x.maxDrawBuffers = 32;
        x.maxVertexUniformVectors = 1024;
        x.maxVaryingVectors = 32;
        x.maxFragmentUniformVectors = 1024;
        x.maxVertexOutputVectors = 32;
        x.maxFragmentInputVectors = 32;
        x.minProgramTexelOffset = -8;
        x.maxProgramTexelOffset = 7;
        x.maxClipDistances = 8;
        x.maxCullDistances = 8;
        x.maxCombinedClipAndCullDistances = 8;
        x.maxSamples = 4;
        x.limits.nonInductiveForLoops = true;
        x.limits.whileLoops = true;
        x.limits.doWhileLoops = true;
        x.limits.generalUniformIndexing = true;
        x.limits.generalAttributeMatrixVectorIndexing = true;
        x.limits.generalVaryingIndexing = true;
        x.limits.generalSamplerIndexing = true;
        x.limits.generalVariableIndexing = true;
        x.limits.generalConstantMatrixVectorIndexing = true;
        return x;
    }();
    return r;
}

}  // namespace

Converted ToVulkan(const std::string& gles, bool vertex) {
    Converted out;
    std::istringstream in(gles);
    std::ostringstream body;
    static const std::regex kUniformVec4(R"(^\s*uniform\s+(?:highp\s+|mediump\s+|lowp\s+)?vec4\s+(\w+)\s*(?:\[\s*(\d+)\s*\])?\s*;.*$)");
    static const std::regex kSampler(R"(^\s*uniform\s+(?:highp\s+|mediump\s+|lowp\s+)?(sampler2D|samplerCube)\s+u_s(\d)\s*;.*$)");
    static const std::regex kVarying(R"(^\s*(in|out)\s+(vec4|float)\s+(v_\w+)\s*;.*$)");
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        std::smatch m;
        if (first) {
            first = false;
            if (line.rfind("#version", 0) == 0) {
                body << "#version 450\n";
                continue;
            }
        }
        if (std::regex_match(line, m, kUniformVec4)) {
            Member mem;
            mem.name = m[1];
            mem.count = m[2].matched ? uint32_t(std::stoul(m[2])) : 1;
            mem.offset = out.block_bytes;
            out.block_bytes += 16 * mem.count;
            out.block.push_back(mem);
            continue;
        }
        if (std::regex_match(line, m, kSampler)) {
            const uint32_t n = uint32_t(std::stoul(m[2]));
            out.samplers |= 1u << n;
            // the textures are 8 bits a channel: read at 16 bits in a fragment shader
            body << "layout(set = 1, binding = " << (kFirstSampler + n) << ") uniform " << (vertex ? "" : "mediump ") << m[1] << " u_s" << n
                 << ";\n";
            continue;
        }
        if (std::regex_match(line, m, kVarying)) {
            const int loc = VaryingLocation(m[3]);
            if (loc >= 0) {
                // a fragment shader's texture coordinates stay 32 bit (16 bit ones shake on large textures)
                const bool coords = !vertex && std::string(m[3]).rfind("v_tc", 0) == 0;
                body << "layout(location = " << loc << ") " << m[1] << " " << (coords ? "highp " : "") << m[2] << " " << m[3] << ";\n";
                continue;
            }
        }
        // the fragment shaders' colour arithmetic at 16 bits (the picture is 8 bits a channel): the GPU does it at twice
        // the rate for less power, at the screen's full resolution most of a frame's GPU work
        if (!vertex && line.find("precision highp float;") != std::string::npos) {
            body << "precision mediump float;\n";
            continue;
        }
        body << line << "\n";
    }
    std::string src = body.str();
    // the block goes right after the version line, before anything that names its members
    std::ostringstream block;
    if (!out.block.empty()) {
        block << "layout(std140, set = 0, binding = " << (vertex ? kVertexBlock : kFragmentBlock) << ") uniform "
              << (vertex ? "VertexUniforms" : "FragmentUniforms") << " {\n";
        for (const Member& mem : out.block) {
            block << "    vec4 " << mem.name;
            if (mem.count > 1) block << "[" << mem.count << "]";
            block << ";\n";
        }
        block << "};\n";
    }
    const size_t after_version = src.find('\n') + 1;
    src.insert(after_version, block.str());
    // Vulkan names the vertex index differently
    src = std::regex_replace(src, std::regex(R"(\bgl_VertexID\b)"), "gl_VertexIndex");
    if (vertex) {
        // GL's clip z runs from -w to w, Vulkan's from 0 to w: the same depth values come out
        src = std::regex_replace(src, std::regex(R"(\bvoid\s+main\s*\(\s*\))"), "void xr_main()");
        src += "void main() {\n    xr_main();\n    gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5;\n}\n";
    }
    out.source = std::move(src);
    return out;
}

std::vector<uint32_t> Compile(const std::string& source, bool vertex, std::string* log) {
    static std::once_flag once;
    std::call_once(once, [] { glslang::InitializeProcess(); });
    const EShLanguage stage = vertex ? EShLangVertex : EShLangFragment;
    glslang::TShader shader(stage);
    const char* text = source.c_str();
    shader.setStrings(&text, 1);
    shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_3);
    const EShMessages msgs = EShMessages(EShMsgSpvRules | EShMsgVulkanRules);
    if (!shader.parse(&Resources(), 450, false, msgs)) {
        if (log) *log = std::string(shader.getInfoLog()) + shader.getInfoDebugLog();
        return {};
    }
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(msgs)) {
        if (log) *log = program.getInfoLog();
        return {};
    }
    std::vector<uint32_t> spirv;
    glslang::SpvOptions opt;
    opt.disableOptimizer = false;
    glslang::GlslangToSpv(*program.getIntermediate(stage), spirv, &opt);
    return spirv;
}

}  // namespace vkglsl
