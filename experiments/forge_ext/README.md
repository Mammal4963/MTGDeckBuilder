# forge_ext - external-policy pilot bridge

Classpath-shadow extension for Forge 2.0.14 (GPL; LobbyPlayerAi.java is
adapted from upstream). Compile against your install's jar:

    javac -cp <forge>/forge-gui-desktop-2.0.14-jar-with-dependencies.jar \
        forge/ai/*.java

Run a sim with the bridge active (see pilot_bridge.py for the server):

    FORGE_EXT_POLICY=8877 FORGE_EXT_PLAYER=mydeck \
    xvfb-run -a java -cp .:<jar> forge.view.Main sim -d mydeck.dck opp.dck -n 24

Protocol: one JSON line per decision out (turn, phase, life totals,
hand, battlefields, proposed casts); one line back: `ok` or
`veto\tCard A\tCard B`. Fail-open - bridge errors fall back to the
built-in AI choice.
