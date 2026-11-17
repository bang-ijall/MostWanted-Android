// audio output through SDL3: the SDK's software mixer is pulled from SDL's audio thread
#include <SDL3/SDL.h>
#include <functional>
#include <vector>

#include "audio/audio.h"

namespace audio {
namespace {
std::function<void(float*, uint32_t)> g_render;

void SDLCALL Pull(void*, SDL_AudioStream* stream, int additional, int) {
    std::vector<float> buf;
    while (additional > 0) {
        uint32_t frames = std::min<uint32_t>(uint32_t(additional) / (2 * sizeof(float)), 1024);
        if (!frames) break;
        buf.assign(size_t(frames) * 2, 0.0f);
        g_render(buf.data(), frames);
        SDL_PutAudioStreamData(stream, buf.data(), int(buf.size() * sizeof(float)));
        additional -= int(buf.size() * sizeof(float));
    }
}
}  // namespace

void StartOutput(std::function<void(float*, uint32_t)> render) {
    g_render = std::move(render);
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        return;
    }
    SDL_AudioSpec spec{SDL_AUDIO_F32, 2, int(kRate)};
    SDL_AudioStream* s = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, Pull, nullptr);
    if (!s) {
        return;
    }
    SDL_ResumeAudioStreamDevice(s);
}
}  // namespace audio
