#include "app/app.h"
#include "touch_pad.h"

// what the runtime asks of the game
const app::GameInfo app::kGame = {
    .title = L"Need for Speed: Most Wanted",
    .short_title = L"NFSMW",
    .name = L"NFSMW",
    .log = "nfsmw",
    .settings = "settings.toml",
    .splash = L"splash",
    .missing = L"Put default.xbe and the game's files in the game folder.",
    .images = {{"default.xbe", "NFSMW"}},
    .options = {},
    .load = nullptr,
    .save = nullptr,
    .init = nullptr,
    .loaded = nullptr,
    .window = nullptr,
    .message = nullptr,
    .ui = nullptr,
    .pad = touchpad::Merge,
    .rumble = nullptr,
};
