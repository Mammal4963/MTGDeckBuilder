package forge.ai;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.List;

import forge.LobbyPlayer;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

/**
 * Rung 2 of the custom-pilot ladder: a PlayerControllerAi subclass that
 * lets an external policy (a Python process on localhost) veto the
 * built-in AI's cast decisions. The built-in AI still handles all
 * mechanics (mana, targeting, combat, triggers); the policy only sees
 * "the AI wants to play X" plus a compact state summary, and answers
 * which proposals to veto. Fail-open: any bridge error falls back to
 * the built-in choice, so a dead policy server never hangs a sim.
 *
 * Enabled when the env var FORGE_EXT_POLICY holds the policy port.
 */
public class PlayerControllerExt extends PlayerControllerAi {

    private static Socket sock;
    private static BufferedReader in;
    private static Writer out;

    public PlayerControllerExt(Game game, Player p, LobbyPlayer lp) {
        super(game, p, lp);
    }

    private static synchronized void ensureSocket() throws Exception {
        if (sock != null && sock.isConnected() && !sock.isClosed()) {
            return;
        }
        int port = Integer.parseInt(System.getenv("FORGE_EXT_POLICY"));
        sock = new Socket("127.0.0.1", port);
        sock.setSoTimeout(3000);
        in = new BufferedReader(new InputStreamReader(
                sock.getInputStream(), StandardCharsets.UTF_8));
        out = new OutputStreamWriter(
                sock.getOutputStream(), StandardCharsets.UTF_8);
    }

    private static String esc(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private void names(StringBuilder sb, Iterable<Card> cards) {
        boolean first = true;
        for (Card c : cards) {
            if (!first) {
                sb.append(",");
            }
            sb.append("\"").append(esc(c.getName())).append("\"");
            first = false;
        }
    }

    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        List<SpellAbility> def = super.chooseSpellAbilityToPlay();
        if (def == null || def.isEmpty()
                || System.getenv("FORGE_EXT_POLICY") == null) {
            return def;
        }
        try {
            ensureSocket();
            Player me = getPlayer();
            StringBuilder sb = new StringBuilder(512);
            sb.append("{\"turn\":").append(getGame().getPhaseHandler().getTurn());
            sb.append(",\"phase\":\"").append(
                    getGame().getPhaseHandler().getPhase()).append("\"");
            sb.append(",\"my_life\":").append(me.getLife());
            int oppLife = 0;
            for (Player o : me.getOpponents()) {
                oppLife = o.getLife();
            }
            sb.append(",\"opp_life\":").append(oppLife);
            sb.append(",\"my_hand\":[");
            names(sb, me.getCardsIn(ZoneType.Hand));
            sb.append("],\"my_battlefield\":[");
            names(sb, me.getCardsIn(ZoneType.Battlefield));
            sb.append("],\"opp_battlefield\":[");
            boolean first = true;
            for (Player o : me.getOpponents()) {
                for (Card c : o.getCardsIn(ZoneType.Battlefield)) {
                    if (!first) {
                        sb.append(",");
                    }
                    sb.append("\"").append(esc(c.getName())).append("\"");
                    first = false;
                }
            }
            sb.append("],\"proposed\":[");
            first = true;
            for (SpellAbility sa : def) {
                Card host = sa.getHostCard();
                if (!first) {
                    sb.append(",");
                }
                sb.append("{\"card\":\"").append(
                        esc(host == null ? "?" : host.getName()));
                sb.append("\",\"type\":\"").append(
                        esc(host == null ? "?" : host.getType().toString()));
                sb.append("\",\"desc\":\"").append(
                        esc(sa.toString())).append("\"}");
                first = false;
            }
            sb.append("]}\n");
            out.write(sb.toString());
            out.flush();
            String reply = in.readLine();
            if (reply == null) {
                return def;
            }
            // reply: veto\tCard A\tCard B   |   ok
            if (reply.startsWith("veto\t")) {
                String[] parts = reply.split("\t");
                for (int i = 1; i < parts.length; i++) {
                    final String banned = parts[i];
                    boolean anyMatch = false;
                    for (SpellAbility sa : def) {
                        Card host = sa.getHostCard();
                        if (host != null && host.getName().equals(banned)) {
                            anyMatch = true;
                            break;
                        }
                    }
                    if (anyMatch) {
                        return null;    // pass priority instead
                    }
                }
            }
            return def;
        } catch (Exception e) {
            try {
                if (sock != null) {
                    sock.close();
                }
            } catch (Exception ignored) {
            }
            sock = null;
            return def;
        }
    }
}
