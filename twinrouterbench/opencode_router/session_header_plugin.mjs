// Optional OpenCode plugin: attach session identity without sending its raw ID
// to the upstream model service. Verified against @opencode-ai/plugin 1.19.2 types.
export const TwinRouterSession = async () => ({
  "chat.headers": async (input, output) => {
    if (input.model.providerID === "twinrouter") {
      output.headers["x-twinrouter-session"] = input.sessionID;
    }
  },
});
