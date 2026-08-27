"""Play against the neural pilot in Forge's normal desktop GUI.

Launches the Forge GUI with the bridge classpath so every AI player in
your game is piloted by the chosen checkpoint. The human seat is
completely normal Forge - deck editor, priority stops, all of it.
Set up a Constructed game vs an AI opponent (give the AI fac_roaming
or any deck) and play.

Levels:
  builtin    stock Forge AI (bridge off)          ~36% vs gauntlet
  clone      behavior clone of the builtin        parity, casts RE
  champion   rl8, confirmed +11 pts over builtin
  generalist rl16 big net, trained on 150+ decks

Usage:
  python experiments/play_vs_pilot.py champion
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

LEVELS = {
    "builtin": None,
    "clone": ("pilot2.pt", None),
    "champion": ("pilot2_rl8.pt", None),
    "generalist": ("pilot2_rl16.pt", ("256", "4")),
}


def main():
    level = sys.argv[1] if len(sys.argv) > 1 else "champion"
    if level == "custom":
        # play_vs_pilot.py custom <ckpt-path> [D,layers]
        arch = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 \
            else ("192", "6")
        LEVELS["custom"] = (sys.argv[2], arch)
    if level not in LEVELS:
        print(f"levels: {', '.join(LEVELS)} | custom <ckpt> [D,layers]")
        return

    from improve_deck import FORGE_DIR
    ext = Path(__file__).resolve().parent / "forge_ext"
    jar = FORGE_DIR / "forge-gui-desktop-2.0.14-jar-with-dependencies.jar"
    env = dict(os.environ)

    if LEVELS[level] is not None:
        ckpt, arch = LEVELS[level]
        if arch:
            os.environ["PILOT_D"], os.environ["PILOT_LAYERS"] = arch
            env["PILOT_D"], env["PILOT_LAYERS"] = arch
        from pilot_bridge import ModelPolicy, start_server
        policy = ModelPolicy(
            ckpt=Path(__file__).resolve().parent / "output" / ckpt)
        srv = start_server(0, policy=policy)
        port = srv.server_address[1]
        env["FORGE_EXT_POLICY"] = str(port)
        env["FORGE_EXT_PLAYER"] = ""       # every AI seat -> the pilot
        env["FORGE_EXT_TARGETS"] = "1"
        env["FORGE_EXT_MULL"] = "1"
        print(f"[{level}] policy server on port {port} "
              f"({ckpt}) - AI opponents are the pilot")
    else:
        for k in ("FORGE_EXT_POLICY", "FORGE_EXT_TARGETS",
                  "FORGE_EXT_MULL"):
            env.pop(k, None)
        print("[builtin] stock Forge AI")

    print("launching Forge GUI... set up a Constructed match vs an AI "
          "opponent (e.g. give it fac_roaming). Close Forge to exit.")
    proc = subprocess.Popen(
        ["java", "-Xmx4g", "-Dio.netty.tryReflectionSetAccessible=true",
         "-Dfile.encoding=UTF-8",
         "-cp", f"{ext}{os.pathsep}{jar}", "forge.view.Main"],
        cwd=FORGE_DIR, env=env)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
    print("Forge closed.")
    time.sleep(0.5)


if __name__ == "__main__":
    main()
