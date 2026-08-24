# Deploying the site (for the session holding CLOUDFLARE_API_TOKEN)

This folder is the complete static bundle for the `mtgdeckbuilder`
Worker (mtgdeckbuilder.abe141516.workers.dev). It is regenerated on
the training PC at every milestone (dashboard.html carries the latest
training snapshot; the map pages only change when re-generated).

To publish, from a checkout of `claude/embeddings-research`:

    git pull
    npx wrangler deploy --assets experiments/site --name mtgdeckbuilder

Contents:
- index.html / card-map.html — card embedding map (+ nav link to the
  training dashboard)
- deck-seeker.html, deck-map.html — as before
- dashboard.html — NEW: neural-pilot training dashboard snapshot
  (charts, confirmation arms, Random Encounter stats, ramp curve)

builder.html / evolver.html were already serving empty bodies on the
live site and are intentionally absent.
