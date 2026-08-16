package forge.view;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;

import forge.deck.Deck;
import forge.deck.io.DeckSerializer;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.player.RegisteredPlayer;
import forge.gui.GuiBase;
import forge.localinstance.properties.ForgeConstants;
import forge.model.FModel;
import forge.player.GamePlayerUtil;

/**
 * Persistent simulation server: pay Forge's ~6s startup (GUI toolkit +
 * full card database load) ONCE, then accept match jobs on stdin
 * forever. Kills the per-invocation startup tax that dominates the
 * evolver's and the RL data collector's wall clock, and keeps the JIT
 * warm across jobs.
 *
 * Run (prepend forge_ext to the classpath, xvfb still required):
 *   xvfb-run -a java -cp forge_ext:forge-gui-desktop-...jar \
 *       forge.view.SimServer
 *
 * Protocol, one job per line on stdin:
 *   SIM <nGames> <quiet:0|1> <deckA> <deckB>
 * Deck names resolve in the constructed decks dir (".dck" optional).
 * Output: the same game log / "Game Result" lines the classic sim mode
 * prints (existing parsers keep working), then a sentinel line:
 *   ##JOB_DONE <ok|error> <detail>
 * EOF on stdin or the line "QUIT" shuts the server down.
 */
public final class SimServer {

    private SimServer() {
    }

    private static Deck loadDeck(String name) {
        if (!name.endsWith(".dck")) {
            name = name + ".dck";
        }
        File f = new File(name);
        if (!f.isFile()) {
            f = new File(ForgeConstants.DECK_CONSTRUCTED_DIR, name);
        }
        return f.isFile() ? DeckSerializer.fromFile(f) : null;
    }

    private static void runJob(int nGames, boolean quiet,
                               String deckA, String deckB) {
        Deck da = loadDeck(deckA);
        Deck db = loadDeck(deckB);
        if (da == null || db == null) {
            System.out.println("##JOB_DONE error could-not-load-deck");
            return;
        }
        GameRules rules = new GameRules(GameType.Constructed);
        rules.setAppliedVariants(EnumSet.of(GameType.Constructed));

        List<RegisteredPlayer> pp = new ArrayList<>();
        int i = 1;
        for (Deck d : new Deck[]{da, db}) {
            String name = "Ai(" + i + ")-" + d.getName();
            RegisteredPlayer rp = new RegisteredPlayer(d);
            rp.setPlayer(GamePlayerUtil.createAiPlayer(name, i - 1, ""));
            pp.add(rp);
            i++;
        }
        Match mc = new Match(rules, pp, "SimServer");
        for (int g = 0; g < nGames; g++) {
            SimulateMatch.simulateSingleMatch(mc, g, !quiet);
        }
        System.out.println("##JOB_DONE ok " + nGames);
    }

    public static void main(String[] args) throws Exception {
        System.setProperty("java.util.Arrays.useLegacyMergeSort", "true");
        System.setProperty("sun.java2d.d3d", "false");
        GuiBase.setInterface(new forge.GuiDesktop());
        long t0 = System.currentTimeMillis();
        FModel.initialize(null, null);
        System.out.println("##READY " + (System.currentTimeMillis() - t0) + "ms");
        System.out.flush();

        BufferedReader in = new BufferedReader(new InputStreamReader(
                System.in, StandardCharsets.UTF_8));
        String line;
        while ((line = in.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) {
                continue;
            }
            if (line.equals("QUIT")) {
                break;
            }
            try {
                String[] parts = line.split("\\s+", 4);
                if (parts.length == 4 && parts[0].equals("SIM")) {
                    // deck names may contain spaces: split the tail on tab
                    String[] decks = parts[3].split("\t");
                    if (decks.length != 2) {
                        System.out.println("##JOB_DONE error need-two-decks");
                    } else {
                        runJob(Integer.parseInt(parts[1]),
                               parts[2].equals("1"), decks[0], decks[1]);
                    }
                } else {
                    System.out.println("##JOB_DONE error bad-command");
                }
            } catch (Exception | StackOverflowError e) {
                System.out.println("##JOB_DONE error "
                        + e.getClass().getSimpleName());
            }
            System.out.flush();
        }
        System.exit(0);
    }
}
