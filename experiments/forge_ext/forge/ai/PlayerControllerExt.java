package forge.ai;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import forge.LobbyPlayer;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

/**
 * Bridge protocol v2 (rung 3): besides the built-in AI's proposal, every
 * decision now carries the full list of LEGAL candidate plays, and the
 * policy may FORCE one of them - including plays the built-in AI would
 * never choose (the whole point: Acorn Catapult pings under a Tainted
 * Aether lock). Reply verbs:
 *
 *   ok                        play the AI's own proposal
 *   veto\tCardA\tCardB        pass priority instead of the proposal
 *   force\t<idx>              play candidate idx, AI picks targets
 *   force\t<idx>\topponent    play candidate idx targeting the opponent
 *
 * Safety: a (turn, card) pair is forced at most once - if the engine
 * rejects the play and the AI returns to priority, we fall back to the
 * default instead of looping. All errors fail open to the built-in AI.
 */
public class PlayerControllerExt extends PlayerControllerAi {

    private static Socket sock;
    private static BufferedReader in;
    private static Writer out;
    private final Set<String> forcedThisTurn = new HashSet<>();
    private int lastSeenTurn = -1;

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
        return s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", " ");
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

    private List<SpellAbility> legalCandidates() {
        Player me = getPlayer();
        CardCollection pool = new CardCollection(me.getCardsIn(ZoneType.Hand));
        pool.addAll(me.getCardsIn(ZoneType.Battlefield));
        List<SpellAbility> result = new ArrayList<>();
        for (SpellAbility sa : ComputerUtilAbility.getSpellAbilities(pool, me)) {
            try {
                if (sa.isManaAbility()) {
                    continue;
                }
                sa.setActivatingPlayer(me);
                if (!sa.canPlay()) {
                    continue;
                }
                if (!ComputerUtilCost.canPayCost(sa, me, false)) {
                    continue;
                }
                result.add(sa);
            } catch (Exception ignored) {
                // a card whose canPlay probe explodes is not a candidate
            }
        }
        return result;
    }

    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        List<SpellAbility> def = super.chooseSpellAbilityToPlay();
        if (System.getenv("FORGE_EXT_POLICY") == null) {
            return def;
        }
        try {
            ensureSocket();
            Player me = getPlayer();
            int turn = getGame().getPhaseHandler().getTurn();
            if (turn != lastSeenTurn) {
                forcedThisTurn.clear();
                lastSeenTurn = turn;
            }
            List<SpellAbility> candidates = legalCandidates();

            StringBuilder sb = new StringBuilder(1024);
            sb.append("{\"turn\":").append(turn);
            sb.append(",\"phase\":\"").append(
                    getGame().getPhaseHandler().getPhase()).append("\"");
            sb.append(",\"my_life\":").append(me.getLife());
            int oppLife = 0;
            Player opp = null;
            for (Player o : me.getOpponents()) {
                oppLife = o.getLife();
                opp = o;
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
            if (def != null) {
                for (SpellAbility sa : def) {
                    Card host = sa.getHostCard();
                    if (!first) {
                        sb.append(",");
                    }
                    sb.append("\"").append(
                            esc(host == null ? "?" : host.getName())).append("\"");
                    first = false;
                }
            }
            sb.append("],\"candidates\":[");
            first = true;
            for (int i = 0; i < candidates.size(); i++) {
                SpellAbility sa = candidates.get(i);
                Card host = sa.getHostCard();
                if (!first) {
                    sb.append(",");
                }
                sb.append("{\"i\":").append(i);
                sb.append(",\"card\":\"").append(
                        esc(host == null ? "?" : host.getName()));
                sb.append("\",\"type\":\"").append(
                        esc(host == null ? "?" : host.getType().toString()));
                sb.append("\",\"zone\":\"").append(
                        host != null && host.isInPlay() ? "battlefield" : "hand");
                sb.append("\",\"targeted\":").append(sa.usesTargeting());
                sb.append(",\"desc\":\"").append(esc(sa.toString())).append("\"}");
                first = false;
            }
            sb.append("]}\n");
            out.write(sb.toString());
            out.flush();
            String reply = in.readLine();
            if (reply == null || reply.equals("ok")) {
                return def;
            }
            if (reply.startsWith("veto\t")) {
                if (def == null) {
                    return null;
                }
                for (String banned : reply.substring(5).split("\t")) {
                    for (SpellAbility sa : def) {
                        Card host = sa.getHostCard();
                        if (host != null && host.getName().equals(banned)) {
                            return null;
                        }
                    }
                }
                return def;
            }
            if (reply.startsWith("force\t")) {
                String[] parts = reply.split("\t");
                int idx = Integer.parseInt(parts[1]);
                if (idx < 0 || idx >= candidates.size()) {
                    return def;
                }
                SpellAbility sa = candidates.get(idx);
                Card host = sa.getHostCard();
                String key = host == null ? sa.toString() : host.getName();
                if (!forcedThisTurn.add(key)) {
                    return def;               // already forced this turn
                }
                sa.setActivatingPlayer(me);
                if (parts.length > 2 && parts[2].equals("opponent")
                        && sa.usesTargeting() && opp != null) {
                    sa.resetTargets();
                    sa.getTargets().add(opp);
                }
                return Collections.singletonList(sa);
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
