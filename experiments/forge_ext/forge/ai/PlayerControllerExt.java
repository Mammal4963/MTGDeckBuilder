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
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

/**
 * Bridge protocol v3 (rung 4): per-card board state and combat events.
 *
 * Cast decisions ("kind":"cast") now serialize each battlefield card as
 * {id, n, p, t, tapped, dmg, cr} instead of a bare name, so the policy
 * (and the set-transformer BC model) sees real board texture.
 *
 * Combat observation: declareAttackers / declareBlockers let the
 * built-in AI decide, then ship what it chose -
 *   {"kind":"attackers", my_creatures:[...], chosen:[ids...]}
 *   {"kind":"blockers", attackers:[...], my_creatures:[...],
 *    assignments:[[blockerId, attackerId]...]}
 * The reply is ignored for combat (observe-only this rung) - this is
 * the behavior-cloning feed for combat decisions. All errors fail open.
 */
public class PlayerControllerExt extends PlayerControllerAi {

    private static Socket sock;
    private static BufferedReader in;
    private static Writer out;
    private final Set<String> forcedThisTurn = new HashSet<>();
    private int lastSeenTurn = -1;

    public PlayerControllerExt(Game game, Player p, LobbyPlayer lp) {
        super(game, p, lp);
        // final-state feed: on game end, ship both players' REAL life
        // totals (can be negative) so the archive's last frame shows
        // the fatal blow. Reply is ignored; fail-open like everything.
        if (System.getenv("FORGE_EXT_POLICY") != null) {
            game.subscribeToEvents(new Object() {
                @com.google.common.eventbus.Subscribe
                public void onFinished(
                        forge.game.event.GameEventGameFinished ev) {
                    try {
                        Player me = getPlayer();
                        StringBuilder sb = new StringBuilder(256);
                        sb.append("{\"kind\":\"game_end\"");
                        sb.append(",\"player\":\"").append(esc(me.getName()))
                          .append("\"");
                        sb.append(",\"turn\":").append(
                                getGame().getPhaseHandler().getTurn());
                        sb.append(",\"my_life\":").append(me.getLife());
                        int oppLife = 0;
                        boolean oppLost = false;
                        for (Player o : me.getOpponents()) {
                            oppLife = o.getLife();
                            oppLost = o.hasLost();
                        }
                        sb.append(",\"opp_life\":").append(oppLife);
                        sb.append(",\"i_lost\":").append(me.hasLost());
                        sb.append(",\"opp_lost\":").append(oppLost)
                          .append("}\n");
                        roundTrip(sb.toString());
                    } catch (Throwable ignored) {
                    }
                }
            });
        }
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

    private static synchronized String roundTrip(String msg) {
        try {
            ensureSocket();
            out.write(msg);
            out.flush();
            return in.readLine();
        } catch (Exception e) {
            try {
                if (sock != null) {
                    sock.close();
                }
            } catch (Exception ignored) {
            }
            sock = null;
            return null;
        }
    }

    private static String esc(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", " ");
    }

    private static void cardObj(StringBuilder sb, Card c) {
        sb.append("{\"id\":").append(c.getId());
        sb.append(",\"n\":\"").append(esc(c.getName())).append("\"");
        sb.append(",\"p\":").append(c.isCreature() ? c.getNetPower() : 0);
        sb.append(",\"t\":").append(c.isCreature() ? c.getNetToughness() : 0);
        sb.append(",\"tapped\":").append(c.isTapped());
        sb.append(",\"dmg\":").append(c.getDamage());
        sb.append(",\"cr\":").append(c.isCreature()).append("}");
    }

    private static void cardObjs(StringBuilder sb, Iterable<Card> cards) {
        boolean first = true;
        for (Card c : cards) {
            if (!first) {
                sb.append(",");
            }
            cardObj(sb, c);
            first = false;
        }
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

    private void stateCommon(StringBuilder sb, String kind) {
        Player me = getPlayer();
        sb.append("{\"kind\":\"").append(kind).append("\"");
        // seat identity for both-sides trajectory separation
        sb.append(",\"player\":\"").append(esc(me.getName())).append("\"");
        sb.append(",\"turn\":").append(getGame().getPhaseHandler().getTurn());
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
        cardObjs(sb, me.getCardsIn(ZoneType.Battlefield));
        sb.append("],\"opp_battlefield\":[");
        boolean first = true;
        for (Player o : me.getOpponents()) {
            for (Card c : o.getCardsIn(ZoneType.Battlefield)) {
                if (!first) {
                    sb.append(",");
                }
                cardObj(sb, c);
                first = false;
            }
        }
        sb.append("],\"my_graveyard\":[");
        names(sb, me.getCardsIn(ZoneType.Graveyard));
        sb.append("],\"my_exile\":[");
        names(sb, me.getCardsIn(ZoneType.Exile));
        sb.append("],\"opp_graveyard\":[");
        first = true;
        for (Player o : me.getOpponents()) {
            for (Card c : o.getCardsIn(ZoneType.Graveyard)) {
                if (!first) {
                    sb.append(",");
                }
                sb.append("\"").append(esc(c.getName())).append("\"");
                first = false;
            }
        }
        sb.append("]");
    }

    private static String zoneOf(Card c) {
        if (c == null) {
            return "hand";
        }
        if (c.isInPlay()) {
            return "battlefield";
        }
        if (c.isInZone(ZoneType.Graveyard)) {
            return "graveyard";
        }
        if (c.isInZone(ZoneType.Exile)) {
            return "exile";
        }
        return "hand";
    }

    private List<SpellAbility> legalCandidates() {
        Player me = getPlayer();
        CardCollection pool = new CardCollection(me.getCardsIn(ZoneType.Hand));
        pool.addAll(me.getCardsIn(ZoneType.Battlefield));
        // flashback, escape, jump-start, foretell...: castable abilities
        // live on cards in the graveyard and exile
        pool.addAll(me.getCardsIn(ZoneType.Graveyard));
        pool.addAll(me.getCardsIn(ZoneType.Exile));
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
            }
        }
        return result;
    }

    private static boolean primaryMode() {
        // model-primary: skip the built-in AI's evaluation on casts and
        // combat (we pay it on every decision otherwise, then usually
        // discard its answer - it's the bulk of per-decision wall time)
        return System.getenv("FORGE_EXT_PRIMARY") != null;
    }

    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        if (System.getenv("FORGE_EXT_POLICY") == null) {
            return super.chooseSpellAbilityToPlay();
        }
        List<SpellAbility> def = primaryMode() ? null
                : super.chooseSpellAbilityToPlay();
        try {
            Player me = getPlayer();
            int turn = getGame().getPhaseHandler().getTurn();
            if (turn != lastSeenTurn) {
                forcedThisTurn.clear();
                lastSeenTurn = turn;
            }
            List<SpellAbility> candidates = legalCandidates();

            StringBuilder sb = new StringBuilder(1536);
            stateCommon(sb, "cast");
            sb.append(",\"proposed\":[");
            boolean first = true;
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
                sb.append("\",\"zone\":\"").append(zoneOf(host));
                sb.append("\",\"targeted\":").append(sa.usesTargeting());
                sb.append(",\"desc\":\"").append(esc(sa.toString())).append("\"}");
                first = false;
            }
            sb.append("]}\n");
            String reply = roundTrip(sb.toString());
            if (reply == null) {
                // dead bridge in primary mode must NOT become pass-forever
                return primaryMode() ? super.chooseSpellAbilityToPlay() : def;
            }
            if (reply.equals("ok")) {
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
                    return def;
                }
                sa.setActivatingPlayer(me);
                Player opp = null;
                for (Player o : me.getOpponents()) {
                    opp = o;
                }
                if (parts.length > 2 && parts[2].equals("opponent")
                        && sa.usesTargeting() && opp != null) {
                    sa.resetTargets();
                    sa.getTargets().add(opp);
                }
                return Collections.singletonList(sa);
            }
            return def;
        } catch (Exception e) {
            return def;
        }
    }

    /**
     * Protocol v4: trigger/spell target selection through the bridge.
     * Gated by FORGE_EXT_TARGETS=1 (in addition to FORGE_EXT_POLICY) so
     * v3 pipelines are byte-identical without it. The built-in AI picks
     * first; its choice ships as `proposed` (free behavior-cloning
     * labels in observe mode). Reply "target\t&lt;i&gt;" retargets to
     * candidate i; "ok"/null keeps the built-in choice. This rung
     * covers single-target, non-divided abilities only.
     */
    @Override
    public boolean chooseTargetsFor(SpellAbility sa) {
        boolean ok = super.chooseTargetsFor(sa);
        if (ok) {
            maybeRetarget(sa);
        }
        return ok;
    }

    /**
     * Triggered abilities never reach chooseTargetsFor: static triggers
     * route through playTrigger, and normal ones land on the stack as a
     * WrappedAbility whose resolve() calls playSpellAbilityNoStack -
     * which picks targets (AiController.doTrigger) and RESOLVES in one
     * breath. Override both, inserting the bridge between the AI's
     * target choice and resolution.
     */
    @Override
    public boolean playTrigger(Card host,
            forge.game.trigger.WrappedAbility wrapper, boolean isMandatory) {
        boolean play = super.playTrigger(host, wrapper, isMandatory);
        if (play) {
            try {
                SpellAbility inner = wrapper.getWrappedAbility();
                maybeRetarget(inner == null ? wrapper : inner);
            } catch (Throwable t) {
                if (System.getenv("FORGE_EXT_DEBUG") != null) {
                    System.err.println("EXTDBG playTrigger EX " + t);
                }
            }
        }
        return play;
    }

    @Override
    public void playSpellAbilityNoStack(SpellAbility sa, boolean mandatory) {
        if (System.getenv("FORGE_EXT_POLICY") == null
                || System.getenv("FORGE_EXT_TARGETS") == null) {
            super.playSpellAbilityNoStack(sa, mandatory);
            return;
        }
        // replicate super (doTrigger -> playNoStack), bridging between
        if (mandatory) {
            try {
                getAi().doTrigger(sa, true);
            } catch (Throwable t) {
                if (System.getenv("FORGE_EXT_DEBUG") != null) {
                    System.err.println("EXTDBG doTrigger EX " + t);
                }
            }
        }
        try {
            maybeRetarget(sa);
        } catch (Throwable ignored) {
        }
        ComputerUtil.playNoStack(getPlayer(), sa, getGame(), true);
    }

    private void maybeRetarget(SpellAbility sa) {
        if (System.getenv("FORGE_EXT_POLICY") == null
                || System.getenv("FORGE_EXT_TARGETS") == null) {
            return;
        }
        boolean dbg = System.getenv("FORGE_EXT_DEBUG") != null;
        try {
            if (dbg) {
                System.err.println("EXTDBG maybeRetarget cls="
                        + sa.getClass().getSimpleName()
                        + " host=" + (sa.getHostCard() == null ? "?"
                                : sa.getHostCard().getName())
                        + " usesTargeting=" + sa.usesTargeting());
            }
            if (!sa.usesTargeting()) {
                return;         // check first: the max-target accessors
            }                   // can throw on non-targeted subabilities
            if (sa.isDividedAsYouChoose() || sa.getMaxTargets() != 1) {
                return;
            }
            forge.game.spellability.TargetRestrictions tgt =
                    sa.getTargetRestrictions();
            if (tgt == null) {
                return;
            }
            List<forge.game.GameEntity> cands = tgt.getAllCandidates(sa, true);
            if (dbg) {
                System.err.println("EXTDBG cands=" + cands.size());
            }
            if (cands.size() < 2) {
                return;         // no real choice to make
            }
            Player me = getPlayer();
            Card host = sa.getHostCard();
            StringBuilder sb = new StringBuilder(1536);
            stateCommon(sb, "target");
            sb.append(",\"host\":\"").append(
                    esc(host == null ? "?" : host.getName())).append("\"");
            sb.append(",\"ability\":\"").append(esc(sa.toString())).append("\"");
            sb.append(",\"proposed\":[");
            boolean first = true;
            for (forge.game.GameObject o : sa.getTargets()) {
                int idx = cands.indexOf(o);
                if (idx >= 0) {
                    if (!first) {
                        sb.append(",");
                    }
                    sb.append(idx);
                    first = false;
                }
            }
            sb.append("],\"candidates\":[");
            first = true;
            for (int i = 0; i < cands.size(); i++) {
                forge.game.GameEntity e = cands.get(i);
                if (!first) {
                    sb.append(",");
                }
                if (e instanceof Card) {
                    Card c = (Card) e;
                    sb.append("{\"i\":").append(i);
                    sb.append(",\"kind\":\"card\",\"id\":").append(c.getId());
                    sb.append(",\"n\":\"").append(esc(c.getName())).append("\"");
                    sb.append(",\"mine\":").append(c.getController() == me);
                    sb.append(",\"cr\":").append(c.isCreature());
                    sb.append(",\"p\":").append(
                            c.isCreature() ? c.getNetPower() : 0);
                    sb.append(",\"t\":").append(
                            c.isCreature() ? c.getNetToughness() : 0)
                      .append("}");
                } else if (e instanceof Player) {
                    Player p = (Player) e;
                    sb.append("{\"i\":").append(i);
                    sb.append(",\"kind\":\"player\",\"n\":\"")
                      .append(esc(p.getName())).append("\"");
                    sb.append(",\"mine\":").append(p == me);
                    sb.append(",\"life\":").append(p.getLife()).append("}");
                } else {
                    sb.append("{\"i\":").append(i)
                      .append(",\"kind\":\"other\",\"n\":\"")
                      .append(esc(e.toString())).append("\"}");
                }
                first = false;
            }
            sb.append("]}\n");
            String reply = roundTrip(sb.toString());
            if (reply == null || !reply.startsWith("target\t")) {
                return;
            }
            int idx = Integer.parseInt(reply.substring(7).trim().split(",")[0]);
            if (idx < 0 || idx >= cands.size()) {
                return;
            }
            sa.resetTargets();
            sa.getTargets().add(cands.get(idx));
        } catch (Exception e) {
            // fail open: builtin targets stand
        }
    }

    /**
     * Mulligan through the bridge (gated by FORGE_EXT_MULL=1): the
     * built-in AI decides first (free BC label in `proposed`); reply
     * "keep" or "mull" overrides it. tuckCardsViaMulligan (London
     * bottoming) is observe-only this rung.
     */
    @Override
    public boolean mulliganKeepHand(Player player, int cardsToReturn) {
        boolean keep = super.mulliganKeepHand(player, cardsToReturn);
        if (System.getenv("FORGE_EXT_POLICY") == null
                || System.getenv("FORGE_EXT_MULL") == null) {
            return keep;
        }
        try {
            StringBuilder sb = new StringBuilder(1024);
            stateCommon(sb, "mulligan");
            sb.append(",\"cards_to_return\":").append(cardsToReturn);
            sb.append(",\"proposed\":\"").append(keep ? "keep" : "mull")
              .append("\"}\n");
            String reply = roundTrip(sb.toString());
            if ("keep".equals(reply)) {
                return true;
            }
            if ("mull".equals(reply)) {
                return false;
            }
        } catch (Exception ignored) {
        }
        return keep;
    }

    @Override
    public forge.game.card.CardCollectionView tuckCardsViaMulligan(
            forge.game.card.CardCollectionView cards, int amount) {
        forge.game.card.CardCollectionView chosen =
                super.tuckCardsViaMulligan(cards, amount);
        if (System.getenv("FORGE_EXT_POLICY") == null
                || System.getenv("FORGE_EXT_MULL") == null) {
            return chosen;
        }
        try {
            StringBuilder sb = new StringBuilder(1024);
            stateCommon(sb, "mulligan_tuck");
            sb.append(",\"amount\":").append(amount);
            sb.append(",\"options\":[");
            boolean first = true;
            for (Card c : cards) {
                if (!first) {
                    sb.append(",");
                }
                sb.append("{\"id\":").append(c.getId());
                sb.append(",\"n\":\"").append(esc(c.getName())).append("\"}");
                first = false;
            }
            sb.append("],\"proposed\":[");
            names(sb, chosen);
            sb.append("]}\n");
            String reply = roundTrip(sb.toString());
            // write path: "tuck\tid,id,..." picks WHICH cards go to the
            // bottom (must be exactly `amount` valid ids; else builtin)
            if (reply != null && reply.startsWith("tuck\t")) {
                forge.game.card.CardCollection pick =
                        new forge.game.card.CardCollection();
                for (String s : reply.substring(5).split(",")) {
                    if (s.isEmpty()) {
                        continue;
                    }
                    int id = Integer.parseInt(s.trim());
                    for (Card c : cards) {
                        if (c.getId() == id && !pick.contains(c)) {
                            pick.add(c);
                        }
                    }
                }
                if (pick.size() == amount) {
                    return pick;
                }
            }
        } catch (Exception ignored) {
        }
        return chosen;
    }

    private Card findMyCard(int id) {
        for (Card c : getPlayer().getCardsIn(ZoneType.Battlefield)) {
            if (c.getId() == id) {
                return c;
            }
        }
        return null;
    }

    @Override
    public void declareAttackers(Player attacker, Combat combat) {
        if (System.getenv("FORGE_EXT_POLICY") == null) {
            super.declareAttackers(attacker, combat);
            return;
        }
        boolean ranSuper = !primaryMode();
        if (ranSuper) {
            super.declareAttackers(attacker, combat);
        }
        try {
            StringBuilder sb = new StringBuilder(1024);
            stateCommon(sb, "attackers");
            sb.append(",\"chosen\":[");
            boolean first = true;
            for (Card c : combat.getAttackers()) {
                if (!first) {
                    sb.append(",");
                }
                sb.append(c.getId());
                first = false;
            }
            sb.append("]}\n");
            String reply = roundTrip(sb.toString());
            if (reply == null && !ranSuper) {
                super.declareAttackers(attacker, combat);   // fail open
                return;
            }
            // write path: "attack\tid,id,..." replaces the attack set
            // (legality-checked per card; illegal requests are skipped)
            if (reply != null && reply.startsWith("attack\t")) {
                Set<Integer> want = new HashSet<>();
                for (String s : reply.substring(7).split(",")) {
                    if (!s.isEmpty()) {
                        want.add(Integer.parseInt(s.trim()));
                    }
                }
                for (Card c : new CardCollection(combat.getAttackers())) {
                    if (!want.contains(c.getId())) {
                        combat.removeFromCombat(c);
                    }
                }
                forge.game.GameEntity defender = null;
                for (forge.game.GameEntity d : combat.getDefenders()) {
                    defender = d;
                    break;
                }
                if (defender != null) {
                    for (int id : want) {
                        Card c = findMyCard(id);
                        if (c != null && !combat.isAttacking(c)
                                && forge.game.combat.CombatUtil
                                        .canAttack(c, defender)) {
                            combat.addAttacker(c, defender);
                        }
                    }
                }
            }
        } catch (Exception ignored) {
        }
    }

    @Override
    public void declareBlockers(Player defender, Combat combat) {
        if (System.getenv("FORGE_EXT_POLICY") == null) {
            super.declareBlockers(defender, combat);
            return;
        }
        boolean ranSuper = !primaryMode();
        if (ranSuper) {
            super.declareBlockers(defender, combat);
        }
        try {
            StringBuilder sb = new StringBuilder(1024);
            stateCommon(sb, "blockers");
            sb.append(",\"attackers\":[");
            cardObjs(sb, combat.getAttackers());
            sb.append("],\"assignments\":[");
            boolean first = true;
            for (Card a : combat.getAttackers()) {
                for (Card b : combat.getBlockers(a)) {
                    if (!first) {
                        sb.append(",");
                    }
                    sb.append("[").append(b.getId()).append(",")
                      .append(a.getId()).append("]");
                    first = false;
                }
            }
            sb.append("]}\n");
            String reply = roundTrip(sb.toString());
            if (reply == null && !ranSuper) {
                super.declareBlockers(defender, combat);    // fail open
                return;
            }
            // write path: "block\tblockerId:attackerId,..." replaces MY
            // block assignments (legality-checked; illegal pairs skipped)
            if (reply != null && reply.startsWith("block\t")) {
                for (Card b : new CardCollection(combat.getAllBlockers())) {
                    if (b.getController() == defender) {
                        combat.removeFromCombat(b);
                    }
                }
                for (String pair : reply.substring(6).split(",")) {
                    if (pair.isEmpty() || !pair.contains(":")) {
                        continue;
                    }
                    String[] ba = pair.split(":");
                    Card b = findMyCard(Integer.parseInt(ba[0].trim()));
                    Card a = null;
                    for (Card c : combat.getAttackers()) {
                        if (c.getId() == Integer.parseInt(ba[1].trim())) {
                            a = c;
                        }
                    }
                    if (a != null && b != null
                            && forge.game.combat.CombatUtil
                                    .canBlock(a, b, combat)) {
                        combat.addBlocker(a, b);
                    }
                }
            }
        } catch (Exception ignored) {
        }
    }
}
