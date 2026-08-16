package forge.ai;

import java.util.Set;

import forge.LobbyPlayer;
import forge.game.Game;
import forge.game.player.IGameEntitiesFactory;
import forge.game.player.Player;
import forge.game.player.PlayerController;

/**
 * Classpath-shadow of Forge's LobbyPlayerAi (GPL, adapted from the
 * upstream source). Identical behavior, except: when the env var
 * FORGE_EXT_POLICY is set AND this lobby player's name matches
 * FORGE_EXT_PLAYER (or FORGE_EXT_PLAYER is unset = all AI players),
 * the in-game controller is PlayerControllerExt, which consults the
 * external policy bridge. Prepend the compiled classes to the
 * classpath so this class wins over the jar's copy:
 *
 *   java -cp forge_ext_classes:forge-gui-desktop-...-jar-with-dependencies.jar \
 *        forge.view.Main sim -d deckA.dck deckB.dck -n 10
 */
public class LobbyPlayerAi extends LobbyPlayer implements IGameEntitiesFactory {

    private String aiProfile = "";
    private boolean rotateProfileEachGame;
    private AIOption option;

    public LobbyPlayerAi(String name, Set<AIOption> options) {
        super(name);
        if (options != null && !options.isEmpty()) {
            option = options.iterator().next();
        }
    }

    public void setAiProfile(String profileName) {
        aiProfile = profileName;
    }

    public String getAiProfile() {
        return aiProfile;
    }

    public void setRotateProfileEachGame(boolean rotate) {
        this.rotateProfileEachGame = rotate;
    }

    private boolean useExtPolicy() {
        if (System.getenv("FORGE_EXT_POLICY") == null) {
            return false;
        }
        String target = System.getenv("FORGE_EXT_PLAYER");
        return target == null || target.isEmpty()
                || getName().contains(target);
    }

    private PlayerControllerAi createControllerFor(Player ai) {
        PlayerControllerAi result = useExtPolicy()
                ? new PlayerControllerExt(ai.getGame(), ai, this)
                : new PlayerControllerAi(ai.getGame(), ai, this);
        result.getAi().setUseSimulation(option);
        return result;
    }

    @Override
    public PlayerController createMindSlaveController(Player master, Player slave) {
        return createControllerFor(slave);
    }

    @Override
    public Player createIngamePlayer(Game game, final int id) {
        Player ai = new Player(getName(), game, id);
        ai.setFirstController(createControllerFor(ai));

        if (rotateProfileEachGame) {
            setAiProfile(AiProfileUtil.getRandomProfile());
        }
        return ai;
    }

    @Override
    public void hear(LobbyPlayer player, String message) { /* Local AI is deaf. */ }
}
