// an on-screen controller laid out like an Xbox pad: sticks, d-pad, A B X Y, bumpers, triggers, Back and Start.
// It shows while no controller is plugged in and hides while one is.
#pragma once
#include <SDL3/SDL.h>

#include "core/input.h"

namespace touchpad {
void Event(const SDL_Event& e);          // window thread: finger events
void Draw(SDL_Renderer* r, int w, int h);  // window thread: over the game picture, before the present
void Merge(int port, input::Pad& p);     // GameInfo::pad: the touches added to port 1
// a button's label as the system's font draws it (white, coverage per pixel), from the app's Java side; the bitmap
// font of SDL is the stand-in until it comes
void SetLabel(const char* text, int w, int h, const uint8_t* alpha);
}  // namespace touchpad
