# TwinRouterBench Leaderboard

Static leaderboard site for GitHub Pages.

## Local preview

```bash
cd leaderboard
python3 -m http.server 8000
```

Open `http://127.0.0.1:8000/`.

## GitHub Pages

Use one of these deployment layouts:

1. Point GitHub Pages at the `leaderboard/` directory if the repository UI allows it.
2. Rename or copy this directory to `docs/` and select `main` / `docs` in GitHub Pages.
3. Use a GitHub Actions workflow to publish `leaderboard/` to Pages.

The site is fully static. Updating `data/leaderboard.json` updates the rendered table.
