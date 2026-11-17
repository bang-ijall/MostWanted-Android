// Android's performance hints (ADPF) for the game's and the renderer's threads: the system learns how long a frame's
// work takes and how often frames come, and sets the CPU clocks to just that instead of running them flat out
#pragma once
#include <cstdint>

namespace power {
enum Thread { kGame = 0, kRender = 1 };
void RegisterThread(Thread which);  // on that thread, once
void GameFrame(int64_t work_ns);    // game thread, each frame: the time it worked (not slept)
void RenderFrame(int64_t work_ns);  // render thread, each frame: the same
void Update(int frame_hz);          // window thread, each loop: the frame rate aimed at, and the work reported
}  // namespace power
